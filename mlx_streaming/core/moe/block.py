"""MoE 热路径块：路由 → 选专家 → 取专家权重 → 计算 → 加权合并。

包含两种块：
- `StreamingMoeBlock`：包住原生 MoE 块，专家权重常驻（switch_mlp），只激活选中专家。
- `FileStreamingMoeBlock`：专家权重从磁盘按需加载（流式），是低内存推理的核心热路径，
  集成常驻池 acquire、native-fused-prefetch 预取、共享专家叠加等逻辑。
"""
import os
import time

import mlx.core as mx

from mlx_streaming import config
from mlx_streaming.core.moe import native_moe
from mlx_streaming.core import route_trace
from mlx_streaming.core.profiling import (
    PROF, WINDOW_PROF, PREDICT_RECALL_PROF, MISS_ATTRIB, note_miss_attrib,
    note_tprof, TPROF_ON, _PROF_ON, _tick, note_union, UNION_ON)

# decode/verify 热路径判据:seq 短(单 token decode=1、MTP verify=K≤几)，与 prefill 长 seq 区分。
_DECODE_SEQ_MAX = 8
from mlx_streaming.core.moe.gate import _effective_top_k
from mlx_streaming.core.moe.compute import (
    streaming_switch_glu_forward, PersistentSubGLU)


class StreamingMoeBlock:
    """包住原 Qwen3MoeSparseMoeBlock：路由器常驻，专家计算改为只算选中专家。

    若提供 store（LruExpertStore），可进一步把专家权重从磁盘按需加载；否则直接在
    常驻的 switch_mlp 上做 uniq 切片计算（仍只激活少数专家）。
    """

    def __init__(self, orig_block, layer_idx: int, store=None):
        self.gate = orig_block.gate            # 路由器常驻（很小）
        self.top_k = orig_block.top_k
        self.norm_topk_prob = orig_block.norm_topk_prob
        self.switch_mlp = orig_block.switch_mlp
        self.store = store
        self.layer_idx = layer_idx

    def __call__(self, x: mx.array) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = _effective_top_k(self.top_k)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        mx.eval(inds)                          # 先物化路由结果，才能按结果取专家
        y = streaming_switch_glu_forward(self.switch_mlp, x, inds)
        return (y * scores[..., None]).sum(axis=-2)


class FileStreamingMoeBlock:
    """文件后端流式 MoE 块：路由器常驻，专家权重从磁盘按需加载（不持有堆叠 switch_mlp）。"""

    def __init__(self, gate, top_k, norm_topk_prob, store, layer_idx,
                 hidden, moe_inter, group_size, bits,
                 proj_bits: dict | None = None,
                 shared_expert=None, shared_expert_gate=None):
        self.gate = gate
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.store = store
        self.layer_idx = layer_idx
        self.hidden = hidden
        self.moe_inter = moe_inter
        self.group_size = group_size
        self.bits = bits
        # Qwen3-Next 等带共享专家的模型：共享专家恒激活、必须常驻（不流式），
        # 输出叠加 sigmoid(shared_expert_gate(x)) * shared_expert(x)。None 则退化为纯路由（如 Qwen3-MoE）。
        self.shared_expert = shared_expert
        self.shared_expert_gate = shared_expert_gate
        # 可选全流式 blob 源（STREAM_BLOB=1 时由 model_builder 注入），见 core/blob_loader.py。
        self._blob = None
        # 持久化子模块：跨 token 复用，避免每调用重建 QSL。
        # proj_bits 非空时走混合精度（逐 proj 不同 bit）。
        self._sub = PersistentSubGLU(hidden, moe_inter, group_size, bits,
                                     proj_bits=proj_bits, layer_idx=layer_idx)

    def __call__(self, x: mx.array) -> mx.array:
        if _PROF_ON:
            return self._call_prof(x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = _effective_top_k(self.top_k)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        if config.predict_recall_prof():
            ps = getattr(self, "_predicted_set", None)
            if ps is not None:
                act = {int(i) for i in inds.reshape(-1).tolist()}
                PREDICT_RECALL_PROF["hit"] += len(ps & act)
                PREDICT_RECALL_PROF["routed"] += len(act)
                PREDICT_RECALL_PROF["n"] += 1
        if config.native_moe():
            native_y = self._try_native_forward(x, inds, scores, gates.shape[-1])
            if native_y is not None:
                y = native_y
                if self.shared_expert is not None:
                    y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
                return y
        if self._blob is not None and config.stream_blob():
            # 全流式 blob 路径：每层按需并行读专家 → 复用 _sub.forward(MLX quantized_matmul)。
            flat = [int(i) for i in inds.reshape(-1).tolist()]
            pool_arrays, slots = self._blob.acquire(self.layer_idx, flat)
            local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
            y = self._sub.forward(pool_arrays, len(set(flat)), x, local)
            y = (y * scores[..., None]).sum(axis=-2)
            if self.shared_expert is not None:
                y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
            return y
        if config.probe_perlayer_sync():
            # 诊断：每层强制一次 host 同步（模拟预测 barrier），不做任何预取。
            # 若单这个就拖慢 ~30% → 税是 eval barrier，C++ 读统一内存也救不了。
            _ = int(inds.reshape(-1)[:1].item())
        if (config.stream_blob_bg()
                and getattr(self.store, "_bg", None) is not None):
            # 后台预取已物化的专家在此（主线程、acquire 前）写进常驻池槽 → 转 miss 为 hit。
            if config.window_prof():
                t0 = getattr(self, "_submit_t", None)
                if t0 is not None:
                    WINDOW_PROF["sum_s"] += time.perf_counter() - t0
                    WINDOW_PROF["n"] += 1
            self.store.promote_prefetched(self.layer_idx)
        # native-fused-prefetch 的 promote（把回调预读好的专家写进池槽）下移到各 acquire 分支前：
        # host/verify 路径已算出真实路由 uniq_set，可"只 promote 命中本层路由的专家"，零额外同步地
        # 丢弃假阳性 → 省掉无用 scatter + 池污染（这是 promote -3.2% 开销的主因）。
        _stg_mgr = getattr(self.store, "_staging", None)
        # 双源：投机专家留池侧区由 acquire_gpu_dual 取，不 promote、不驱逐 → 关 promote。
        _do_promote = (_stg_mgr is not None and not config.native_no_promote()
                       and not config.zerocopy_dual_source())
        if config.route_trace_enabled():
            flat_trace = [int(i) for i in inds.reshape(-1).tolist()]
            resident = (self.store.resident_experts(self.layer_idx)
                        if hasattr(self.store, "resident_experts") else set())
            rank = (self.store.resident_lru_scores(self.layer_idx)
                    if hasattr(self.store, "resident_lru_scores") else {})
            route_trace.record(
                self.layer_idx, flat_trace, set(flat_trace) - resident, resident, rank)
        # native-fused-prefetch：在 seq 分支**之前**提交，使 decode(seq=1) 与 MTP verify(seq=K)
        # 两条路径都触发预取。dummy 折进 inds(加 0)：GPU 路径靠 acquire_gpu 的 n_miss eval、
        # host 路径靠后面的 .tolist() eval，都会触发完成回调里的 pread。verify 时 x 为 K 个
        # token（verify_in=[x, d_1..d_{K-1}]），预测的是"下一层这 K 个 token 的专家并集"（recall≈0.96 的口径）。
        # 双源双缓冲：每块开头先推进/检测前向边界（必须在本块预取提交之前，保证同前向内 fill_gen/read_gen 恒定）。
        if config.zerocopy_dual_source() and getattr(self, "_vpool", None) is not None:
            self._vpool.begin_forward(self.layer_idx)
        if (getattr(self.store, "_staging", None) is not None
                and not config.native_no_submit()):
            _dummy = self._native_fused_prefetch(x)
            if _dummy is not None:
                inds = inds + (_dummy.reshape(()).astype(inds.dtype) * 0)
        layer_cap = self.store.cap_for(self.layer_idx)
        # verify(小 seq)可走 GPU 重映射;prefill(大 seq)唯一专家可能超 cap,须留在 host 路径(有超容量 fetch 回退)。
        if config.zerocopy_dual_source() and getattr(self, "_vpool", None) is not None:
            # 双源两级缓存本就是为 MTP verify 建的:acquire_gpu_dual 用全专家宽表寻址,任意并集都安全
            # (miss 走 demand 回退),有效容量 = cap + 侧区行。verify 默认走 dual 路径读侧区,否则
            # 落 host 路径会白填侧区(预取填了却不读)→ 命中骤降、读盘翻倍。判据用 dual 有效容量,
            # 避免更大 K 时 K×top_k 误伤 cap。
            rp = self.store._resident
            _dual_cap = layer_cap + rp.spec_gens * rp.spec_slots
            _verify_gpu = (x.shape[1] * k <= _dual_cap)
        else:
            _verify_gpu = (config.verify_gpu_remap() and x.shape[1] * k <= layer_cap)
        if (config.resident_pool_enabled()
                and config.gpu_remap_enabled()
                and (x.shape[1] == 1 or _verify_gpu)):
            # decode 热路径:GPU 侧 slot 重映射,命中层零 host 往返(消除每层 .tolist 同步),
            # 仅一次 miss 标志同步;真 miss 才回退读盘。decode top-k(≤cap)恒装得下。
            # 专家总数 = gate 输出维(gates 末维),无需依赖 gate.weight。
            if UNION_ON:
                # GPU 路径本不 .tolist 真实路由;仅 UNION_PROF=1 时付一次同步取并集大小。
                note_union(x.shape[1], len({int(i) for i in inds.reshape(-1).tolist()}),
                           self.layer_idx)
            # decode/verify GPU 重映射路径:host 无现成真实路由。开关开时用 GPU membership 仅对
            # 预读候选现算 used(drain≤budget)过滤假阳性;关时退回整批写入(回退/对照基线)。
            if _do_promote:
                if config.gpu_remap_promote_filter():
                    _stg_mgr.promote(self.layer_idx, self.store,
                                     route_inds=inds, num_experts=gates.shape[-1])
                else:
                    _stg_mgr.promote(self.layer_idx, self.store)
            if config.miss_attrib():
                # 诊断专用:GPU remap 路径本不 .tolist 真实路由(那是它消栅栏的关键),
                # 仅 MISS_ATTRIB=1 时付一次 .tolist 取真实路由,统计 decode 热路径 A/B 构成。
                _uniq = {int(i) for i in inds.reshape(-1).tolist()}
                _ps = getattr(self, "_predicted_set", None)
                _res = (self.store.resident_experts(self.layer_idx)
                        if hasattr(self.store, "resident_experts") else set())
                _rdy = _stg_mgr.last_ready.get(self.layer_idx) if _stg_mgr else None
                note_miss_attrib(_uniq, _ps, _res, x.shape[1] <= _DECODE_SEQ_MAX, _rdy)
            if config.zerocopy_dual_source():
                # 双源双缓冲：统一走 VirtualPool.acquire（呈现「所有专家都在」视角，
                # 内部真实区表 ∪ 侧区(读代) 单次 gather，返回 (pool, local, n_experts)）。
                pool_arrays, local, n_experts = self._vpool.acquire(
                    self.layer_idx, inds, gates.shape[-1],
                    seq_len=x.shape[1], layer_cap=layer_cap)
                y = self._sub.forward(pool_arrays, n_experts, x, local)
            else:
                pool_arrays, local = self.store.acquire_gpu(
                    self.layer_idx, inds, gates.shape[-1])
                if config.stg_verify():
                    # 诊断：消费侧字节校验（默认 off），定位混合 ahead 下池槽损坏。
                    self.store._resident.verify_acquire_bytes(
                        self.layer_idx, inds, _stg_mgr)
                y = self._sub.forward(pool_arrays, layer_cap, x, local)
        else:
            # prefill/大批量(seq>1)或显式关闭 GPU_REMAP:走 host 路径。
            # 只在此处对 inds 做一次 .tolist() 同步，uniq/local 全在 Python 里算。
            flat = [int(i) for i in inds.reshape(-1).tolist()]
            uniq_set = set(flat)
            if UNION_ON:
                note_union(x.shape[1], len(uniq_set), self.layer_idx)  # 本层路由专家并集(零额外同步,uniq_set 已算)
            # promote 只写"预读好 ∩ 本层真实路由"的专家：复用已算的 uniq_set，零额外同步，
            # 假阳性不进池 → 省 scatter、不污染池（acquire 前完成，命中转化生效）。
            if _do_promote:
                _stg_mgr.promote(self.layer_idx, self.store, used=uniq_set)
            if config.miss_attrib():
                # 在 promote 之后、acquire(读盘)之前统计：本层真实路由的命中构成。
                # miss 拆成 A(预测到却没进池：budget丢/时序/驱逐) 与 B(没预测到：召回缺口)。
                _ps = getattr(self, "_predicted_set", None)
                _res = (self.store.resident_experts(self.layer_idx)
                        if hasattr(self.store, "resident_experts") else set())
                _rdy = _stg_mgr.last_ready.get(self.layer_idx) if _stg_mgr else None
                note_miss_attrib(uniq_set, _ps, _res, x.shape[1] <= _DECODE_SEQ_MAX, _rdy)
            # 池只能同时容纳 ≤该层容量 个唯一专家；prefill 唯一数超容量时回退 stack。
            # dual 模式（_vpool 存在）统一走 VirtualPool.acquire_host 收口 host+fetch；
            # 非 dual（_vpool 为 None）保留内联逻辑（非目标路径，避免额外依赖）。
            if (config.resident_pool_enabled()
                    and getattr(self, "_vpool", None) is not None):
                pool_arrays, local, n_experts = self._vpool.acquire_host(
                    self.layer_idx, flat, inds.shape, inds.dtype, layer_cap)
                y = self._sub.forward(pool_arrays, n_experts, x, local)
            elif (config.resident_pool_enabled()
                    and len(uniq_set) <= layer_cap):
                pool_arrays, slots = self.store.acquire(self.layer_idx, flat)
                local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
                y = self._sub.forward(pool_arrays, layer_cap, x, local)
            else:
                uniq_sorted = sorted(uniq_set)
                remap = {g: i for i, g in enumerate(uniq_sorted)}
                local = mx.array([remap[i] for i in flat], dtype=inds.dtype).reshape(inds.shape)
                fetched = self.store.fetch(self.layer_idx, uniq_sorted)
                y = self._sub.forward(fetched, len(uniq_sorted), x, local)
        y = (y * scores[..., None]).sum(axis=-2)
        if self.shared_expert is not None:        # 共享专家（常驻）叠加
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return y

    def _native_fused_prefetch(self, x: mx.array):
        """搭车式预取：用下 AHEAD 层 gate 对 x 算预测 inds(lazy)，挂 GPU 完成回调，
        在 C++ 里(GPU 算完后)读 id + pread 预热下层专家字节。返回 dummy 张量(需被 eval 才触发)。
        全程不加主线程 host 同步——靠折进 inds、搭 acquire_gpu 的 n_miss eval。
        """
        model = getattr(self, "_prefetch_model_ref", None)
        bl = getattr(self.store, "_blob_loader", None)
        if model is None or bl is None:
            return None
        layers = getattr(model, "layers", [])
        vp = getattr(self, "_vpool", None)
        if vp is not None:
            tgt = vp.target_for(self.layer_idx)              # per-layer cutoff ahead
            if not (self.layer_idx < tgt < len(layers)):     # 0/越界/无前瞻 → 跳过
                return None
        else:
            ahead = max(1, config.cross_layer_ahead(default=1))
            tgt = self.layer_idx + ahead
            if not (0 <= tgt < len(layers)):
                return None
        tmlp = getattr(layers[tgt], "mlp", None)
        if not isinstance(tmlp, FileStreamingMoeBlock):
            return None
        try:
            from mlx_streaming import native_moe_ext as _N
        except Exception:
            return None
        # 方案B：预测"宽集合"(top-N, N=predict_width)只为高 recall，不占内存（仅一次 gate argpartition）；
        # 真正占 staging 的是 C++ 回调过滤常驻后、按分截到 staging budget 的"缺口"子集。
        _tp0 = time.perf_counter() if TPROF_ON else 0.0
        predict_width = config.cross_layer_predict_width()
        # 默认(PREDICT_USE_X=1)：用本层 MoE 输入 x 喂目标层 gate。x 含了本层 attention，离 L+1
        # 近半层 → 比未归一化输入更新鲜，实测 recall 0.812→0.847(+3.6pp)、还省一次 norm。
        # PREDICT_USE_X=0 回退旧路径：目标层 post_attention_layernorm 作用在本层未归一化输入上。
        h = getattr(self, "_unnormed_input", None)
        if config.predict_use_x():
            g = tmlp.gate(x)
        elif h is None:
            h = x  # 退化：无未归一化输入时用 x（norm 偏差，覆盖会差）
            g = tmlp.gate(h)
        else:
            g = tmlp.gate(layers[tgt].post_attention_layernorm(h))
        # 按 seq 维聚合成"该批 K 个 token 的专家并集"近似。聚合方式 PREDICT_AGG：
        # - max（默认）：任一 token 强烈想要即入选；mean：各 token 平均偏好；
        # - union：每 token 各取 top-kk 再并集（与真实路由"并集"结构最一致，候选更多，由 handler 去重+截断）。
        _agg = config.predict_agg()
        kk = min(g.shape[-1], predict_width)
        if g.ndim == 3 and _agg == "union":
            # union 候选数 = seq × union_k。每 token 只取 top-union_k（独立于 predict_width），
            # 调小 union_k 缩候选 → 减少缺口、缓解 staging/读盘洪水（每 token 真实只路由 top_k 个）。
            uk = min(g.shape[-1], config.predict_union_k())
            pred = mx.argpartition(g, kth=-uk, axis=-1)[..., -uk:].reshape(-1).astype(mx.uint32)
        else:
            if g.ndim == 3:
                g = g.mean(axis=1) if _agg == "mean" else g.max(axis=1)
            # 取 top-kk（argpartition，O(E) 不全排序）。实测：加宽 kk 不提升命中——staging 更多 = 后台
            # pread 更多、到位更晚 + 抢带宽 → 反而更低；故 width=16 即最优，cap 截断极少触发、无需排序。
            pred = mx.argpartition(g, kth=-kk, axis=-1)[..., -kk:].reshape(-1).astype(mx.uint32)
        if config.predict_recall_prof() or config.miss_attrib():
            try:
                tmlp._predicted_set = {int(e) for e in pred.tolist()}
            except Exception:
                pass
        if TPROF_ON:
            note_tprof("predict_s", time.perf_counter() - _tp0, count_key="predict_n")
        stg = getattr(self.store, "_staging", None)
        if stg is not None:
            resident = (self.store.resident_experts(tgt)
                        if hasattr(self.store, "resident_experts") else None)
            if config.zerocopy_dual_source():
                # 双源双缓冲：经 VirtualPool 向填代 submit,C++ 回调把预读段散写进该代侧区行。
                rp = self.store._resident
                if tgt not in rp._pools:
                    return None                          # 目标层池未建(首 token 预热) → 跳过本次
                segs = stg.src._segs                     # (proj, tensor, dt, shape, nb)，与池 key 顺序一致
                pool_list = [rp._pools[tgt][f"{p}.{t}"] for p, t, *_ in segs]
                return self._vpool.prefetch(tgt, pred, resident, pool_list)
            # miss→hit:回调按目标层常驻快照过滤，只把缺口 pread 进 staging（≤budget 行），promote 时写池。
            if TPROF_ON:
                _ts0 = time.perf_counter()
                _r = stg.submit(tgt, pred, resident)
                note_tprof("submit_s", time.perf_counter() - _ts0, count_key="submit_n")
                return _r
            return stg.submit(tgt, pred, resident)
        # 仅预热字节（page cache）的轻量版。
        path = os.path.join(bl.dir, f"layer{tgt:02d}.blob")
        return _N.prefetch_on_complete(pred, path, int(bl.stride), True)

    def _try_native_forward(self, x: mx.array, inds: mx.array, scores: mx.array, num_experts: int):
        # native 第一版只支持三投影同 bit 的专家；其他情况保持现有 MLX 回退。
        proj_bits = getattr(self._sub, "proj_bits", {})
        bits_set = {int(proj_bits.get(name, self.bits)) for name in ("gate_proj", "up_proj", "down_proj")}
        if len(bits_set) != 1:
            return None
        bits = bits_set.pop()
        if not native_moe.can_native_moe(
                self.layer_idx, self.hidden, self.moe_inter,
                self.group_size, bits, num_experts):
            return None
        t_sync = time.perf_counter()
        mx.eval(inds)
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        native_moe.note_route_sync(time.perf_counter() - t_sync)
        return native_moe.try_native_moe(
            self.layer_idx, flat, x, scores,
            self.hidden, self.moe_inter, self.group_size, bits, num_experts)

    def _call_prof(self, x: mx.array) -> mx.array:
        t = time.perf_counter()
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = _effective_top_k(self.top_k)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        mx.eval(inds, scores)
        _tick("route", t); t = time.perf_counter()

        flat = [int(i) for i in inds.reshape(-1).tolist()]
        uniq_set = set(flat)
        layer_cap = self.store.cap_for(self.layer_idx)
        use_pool = (config.resident_pool_enabled()
                    and len(uniq_set) <= layer_cap)
        if not use_pool:
            uniq_sorted = sorted(uniq_set)
            remap = {g: i for i, g in enumerate(uniq_sorted)}
            local = mx.array([remap[i] for i in flat], dtype=inds.dtype).reshape(inds.shape)
            mx.eval(local)
        _tick("pyremap", t); t = time.perf_counter()

        if use_pool:
            pool_arrays, slots = self.store.acquire(self.layer_idx, flat)
            local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
            mx.eval(local)
            fetched, n_experts = pool_arrays, layer_cap
        else:
            fetched = self.store.fetch(self.layer_idx, uniq_sorted)
            n_experts = len(uniq_sorted)
        mx.eval(list(fetched.values()))
        _tick("fetch", t); t = time.perf_counter()

        y = self._sub.forward(fetched, n_experts, x, local)
        mx.eval(y)
        _tick("matmul", t); t = time.perf_counter()

        y = (y * scores[..., None]).sum(axis=-2)
        if self.shared_expert is not None:
            y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        mx.eval(y)
        _tick("combine", t)
        PROF["n_calls"] += 1
        return y
