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
    note_tprof, note_rerank, TPROF_ON, _PROF_ON, _tick, note_union, UNION_ON)

# decode/verify 热路径判据:seq 短(单 token decode=1、MTP verify=K≤几)，与 prefill 长 seq 区分。
_DECODE_SEQ_MAX = 8
_SHARED_EXPERT_STREAM = None


def _shared_expert_stream():
    global _SHARED_EXPERT_STREAM
    if _SHARED_EXPERT_STREAM is None:
        _SHARED_EXPERT_STREAM = mx.new_stream(mx.default_device())
    return _SHARED_EXPERT_STREAM


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
        raw_gates = self.gate(x)
        gates = mx.softmax(raw_gates, axis=-1, precise=True)
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

    def _shared_forward(self, x: mx.array) -> mx.array:
        """Consume an exact adjacent precompute, or run the shared expert."""
        reuse = getattr(self, "_prefetch_reuse_shared", None)
        if reuse is not None:
            object.__setattr__(self, "_prefetch_reuse_shared", None)
            if reuse[0] is x:
                return reuse[1]
        return mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    def __call__(self, x: mx.array) -> mx.array:
        if _PROF_ON:
            return self._call_prof(x)
        # An exact adjacent-layer target-cache refinement may already have
        # computed this gate from the identical post-attention input.  Consume
        # it once instead of immediately repeating the router matmul.
        raw_gates = getattr(self, "_prefetch_reuse_raw_gates", None)
        if raw_gates is None:
            raw_gates = self.gate(x)
        else:
            object.__setattr__(self, "_prefetch_reuse_raw_gates", None)
        gate_history = getattr(self, "_prefetch_gate_history", None)
        if gate_history is not None:
            # A long layer-0 call starts a new request/prefill.  Clear prior
            # request history before any source callback can consume it.
            if self.layer_idx == 0 and int(raw_gates.shape[1]) > _DECODE_SEQ_MAX:
                gate_history.clear()
            gate_history[self.layer_idx] = (
                raw_gates
                if int(raw_gates.shape[1]) <= _DECODE_SEQ_MAX
                else raw_gates[:, -1:, :]
            )
        gates = mx.softmax(raw_gates, axis=-1, precise=True)
        k = _effective_top_k(self.top_k)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        if config.transition_trace():
            from mlx_streaming.core.prefetch import transition_trace
            transition_trace.record_target(layer=self.layer_idx, target_routes=inds)
        route_delta_trace = None
        route_forward_id = None
        route_target_inds = None
        if config.route_delta_trace():
            from mlx_streaming.core.prefetch import route_delta_trace
            route_forward_id = route_delta_trace.current_forward_id()
            route_target_inds = inds
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
                    y = y + self._shared_forward(x)
                if route_forward_id is not None:
                    route_delta_trace.record_target(
                        forward_id=route_forward_id,
                        layer=self.layer_idx,
                        target_routes=route_target_inds,
                        actual_logits=raw_gates,
                        future_hidden=x,
                    )
                return y
        if self._blob is not None and config.stream_blob():
            # 全流式 blob 路径：每层按需并行读专家 → 复用 _sub.forward(MLX quantized_matmul)。
            flat = [int(i) for i in inds.reshape(-1).tolist()]
            pool_arrays, slots = self._blob.acquire(self.layer_idx, flat)
            local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
            y = self._sub.forward(pool_arrays, len(set(flat)), x, local)
            y = (y * scores[..., None]).sum(axis=-2)
            if self.shared_expert is not None:
                y = y + self._shared_forward(x)
            if route_forward_id is not None:
                route_delta_trace.record_target(
                    forward_id=route_forward_id,
                    layer=self.layer_idx,
                    target_routes=route_target_inds,
                    actual_logits=raw_gates,
                    future_hidden=x,
                )
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
        # unified 模式由 VirtualPool 在 demand 边界把全局 staging 晋升进主池；
        # 不走旧 Python promote 路径。
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
        # 每块开头推进逻辑 forward id，供 progressive 状态与审计精确配对。
        if config.zerocopy_dual_source() and getattr(self, "_vpool", None) is not None:
            self._vpool.begin_forward(self.layer_idx)
        if (getattr(self.store, "_staging", None) is not None
                and not config.native_no_submit()):
            _vpool = getattr(self, "_vpool", None)
            _targets = (
                _vpool.targets_for(self.layer_idx)
                if _vpool is not None and hasattr(_vpool, "targets_for")
                else (None,)
            )
            for _target in _targets:
                if config.prefetch_async_predict() and _vpool is not None:
                    # The target predictor is speculative and independent of
                    # this layer's expert output.  Launch it on the auxiliary
                    # stream so the current MoE is not serialized behind a
                    # full 2048->512 gate.  Target demand remains the sole
                    # correctness join for pending direct rows.
                    with mx.stream(_vpool.progressive_stream()):
                        _dummy = self._native_fused_prefetch(
                            x, source_routes=inds, source_scores=scores,
                            source_logits=raw_gates,
                            target_layer=_target,
                        )
                        if _dummy is not None:
                            mx.async_eval(_dummy)
                else:
                    _dummy = self._native_fused_prefetch(
                        x, source_routes=inds, source_scores=scores,
                        source_logits=raw_gates,
                        target_layer=_target,
                    )
                    if _dummy is not None:
                        # 多个 dummy 都折进当前层真实路由图；同一次 demand eval 会触发全部
                        # target callback，不增加 host 同步，也不把任一提交挪到 source MoE 后。
                        inds = inds + (_dummy.reshape(()).astype(inds.dtype) * 0)
        # route-delta 会物化路由与 raw logits，必须在当前块预取提交构造完成后执行。
        if route_forward_id is not None:
            route_delta_trace.record_target(
                forward_id=route_forward_id,
                layer=self.layer_idx,
                target_routes=route_target_inds,
                actual_logits=raw_gates,
                future_hidden=x,
            )
        # The shared expert is resident and independent of expert->slot remap.
        # Submit it only after every prefetch dummy has been attached, so this
        # cannot delay the original callback, then overlap it with pending SSD
        # reads on a separate device stream. Exact adjacent moved output is
        # consumed here rather than recomputed.
        _shared_y = None
        if (
            self.shared_expert is not None
            and config.shared_expert_overlap()
            and int(x.shape[1]) <= _DECODE_SEQ_MAX
        ):
            with mx.stream(_shared_expert_stream()):
                _shared_y = self._shared_forward(x)
                mx.async_eval(_shared_y)
        layer_cap = self.store.cap_for(self.layer_idx)
        # verify(小 seq)可走 GPU 重映射;prefill(大 seq)唯一专家可能超 cap,须留在 host 路径(有超容量 fetch 回退)。
        if config.zerocopy_dual_source() and getattr(self, "_vpool", None) is not None:
            # 旧侧区容量已经合并进 layer_cap；全局 staging 只是搬运缓冲，
            # 不能再重复计入可寻址专家行数。
            _verify_gpu = (x.shape[1] * k <= layer_cap)
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
                # PIN_HOT 校准是显式诊断阶段；只在 record=True 时允许这次
                # GPU→host 路由同步，正常 decode/MTP 热路径仍没有 ids.eval/tolist。
                if getattr(self.store, "record", False):
                    self.store.note(
                        self.layer_idx,
                        [int(i) for i in inds.reshape(-1).tolist()],
                    )
                pool_arrays, local, n_experts = self._vpool.acquire(
                    self.layer_idx, inds, gates.shape[-1],
                    seq_len=x.shape[1], layer_cap=layer_cap)
                if isinstance(local, tuple):
                    entry_local, final_local = local
                    entry_hit = entry_local >= 0
                    hit_local = mx.maximum(entry_local, 0)
                    # The remap command buffer is committed before either
                    # branch.  Hit compute is encoded before the final-local
                    # SharedEvent wait, allowing useful MoE work to overlap
                    # the CPU/SSD fallback callback.  The second pass is
                    # selected only at entry misses; this generic MLX version
                    # intentionally precedes a masked fused-kernel variant.
                    if config.custom_fused_moe():
                        hit_y = self._sub.forward_masked(
                            pool_arrays, n_experts, x, hit_local, entry_hit,
                        )
                        # MLX is lazy: without an explicit submission point the
                        # hit and miss graphs are first evaluated together at
                        # ``hit_y + miss_y``.  That lets the final-local event
                        # wait enter the command stream before useful hit work,
                        # defeating the intended SSD/compute overlap.
                        mx.async_eval(hit_y)
                        miss_y = self._sub.forward_masked(
                            pool_arrays, n_experts, x, final_local, ~entry_hit,
                        )
                        y = hit_y + miss_y
                    else:
                        hit_y = self._sub.forward(
                            pool_arrays, n_experts, x, hit_local,
                        )
                        miss_y = self._sub.forward(
                            pool_arrays, n_experts, x, final_local,
                        )
                        y = mx.where(entry_hit[..., None], hit_y, miss_y)
                else:
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
                    and config.zerocopy_dual_source()
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
            y = y + (
                _shared_y if _shared_y is not None else self._shared_forward(x)
            )
        return y

    def _native_fused_prefetch(
        self,
        x: mx.array,
        source_routes: mx.array | None = None,
        source_scores: mx.array | None = None,
        source_logits: mx.array | None = None,
        target_layer: int | None = None,
    ):
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
        if target_layer is not None:
            tgt = int(target_layer)
            if not (self.layer_idx < tgt < len(layers)):
                return None
        elif vp is not None:
            tgt = vp.target_for(self.layer_idx)              # 兼容单目标调用
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
        if config.prefetch_adaptive():
            target_cap = int(self.store.cap_for(tgt))
            minimum = max(1, int(target_cap * config.prefetch_adaptive_fill()))
            if not _N.real_should_predict(
                int(tgt), minimum, config.prefetch_adaptive_cooldown(),
            ):
                return None
        # 方案B：预测"宽集合"(top-N, N=predict_width)只为高 recall，不占内存（仅一次 gate argpartition）；
        # 真正占 staging 的是 C++ 回调过滤常驻后、按分截到 staging budget 的"缺口"子集。
        _tp0 = time.perf_counter() if TPROF_ON else 0.0
        predict_width = config.cross_layer_predict_width()
        # 默认(PREDICT_USE_X=1)：用本层 MoE 输入 x 喂目标层 gate。x 含了本层 attention，离 L+1
        # 近半层 → 比未归一化输入更新鲜，实测 recall 0.812→0.847(+3.6pp)、还省一次 norm。
        # PREDICT_USE_X=0 回退旧路径：目标层 post_attention_layernorm 作用在本层未归一化输入上。
        h = getattr(self, "_unnormed_input", None)
        predictor_gate = getattr(tmlp, "_prefetch_predict_gate", tmlp.gate)
        if config.predict_use_x():
            predictor_input = x
        elif h is None:
            h = x  # 退化：无未归一化输入时用 x（norm 偏差，覆盖会差）
            predictor_input = h
        else:
            predictor_input = layers[tgt].post_attention_layernorm(h)
        proxy_g = predictor_gate(predictor_input)
        rerank_gate = getattr(tmlp, "_prefetch_rerank_gate", None)
        source_correction = getattr(
            self, "_prefetch_source_correction_profile", {},
        ).get(tgt)
        forecast_g = (
            rerank_gate(predictor_input)
            if rerank_gate is not None and source_correction is None
            else proxy_g
        )
        raw_proxy_logits = proxy_g
        cache_logits = getattr(self, "_target_cache_logits", {}).get(tgt)
        target_cache_profile = getattr(
            self, "_prefetch_target_cache_profile", {},
        ).get(tgt)
        profile_applied = False
        if source_correction is not None:
            if int(source_correction.source_layer) != self.layer_idx:
                raise ValueError(
                    f"source correction L{tgt} 要求 source "
                    f"L{source_correction.source_layer}，实际从 L{self.layer_idx} 提交",
                )
            if source_logits is None:
                raise ValueError("source correction 缺少 source gate logits")
            from mlx_streaming.core.prefetch.source_correction_runtime import (
                apply_source_correction,
            )
            g = apply_source_correction(
                proxy_g, source_logits, source_correction,
            )
        elif target_cache_profile is not None:
            if int(target_cache_profile.source_layer) != self.layer_idx:
                raise ValueError(
                    f"target-cache profile L{tgt} 要求 source "
                    f"L{target_cache_profile.source_layer}，实际从 L{self.layer_idx} 提交",
                )
            if (
                cache_logits is not None
                and source_routes is not None
                and source_scores is not None
            ):
                from mlx_streaming.core.prefetch.target_cache_correction_runtime import (
                    apply_target_cache_correction,
                )
                g = apply_target_cache_correction(
                    proxy_g,
                    cache_logits,
                    source_routes,
                    source_scores,
                    target_cache_profile,
                )
                profile_applied = True
            else:
                # 首个 cache 尚未建立的前向保持 main proxy；不能套用基于 cache
                # 校准的 ranking/width 参数。正常 decode/verify 在 prefill 后会进入上支。
                g = proxy_g
        elif config.prefetch_target_cache() and cache_logits is not None:
            ahead = tgt - self.layer_idx
            alpha = (
                config.prefetch_target_cache_alpha_lo()
                if ahead <= config.cross_layer_ahead_lo()
                else config.prefetch_target_cache_alpha_hi()
            )
            g = proxy_g * (1.0 - alpha) + cache_logits * alpha
        else:
            g = forecast_g
        history_beta = config.prefetch_rerank_history_beta_overrides().get(
            int(tgt), 0.0,
        )
        previous_target_logits = getattr(
            self, "_prefetch_gate_history", {},
        ).get(tgt)
        if history_beta > 0 and previous_target_logits is not None:
            from mlx_streaming.core.prefetch.history_rerank import (
                blend_previous_target_logits,
            )
            g = blend_previous_target_logits(
                g, previous_target_logits, history_beta,
            )
        transition_profile = getattr(
            self, "_prefetch_transition_profile", None,
        )
        if (
            transition_profile
            and tgt in transition_profile
            and source_routes is not None
            and source_scores is not None
        ):
            from mlx_streaming.core.prefetch.source_transition_runtime import (
                apply_source_transition,
            )
            g = apply_source_transition(
                g, source_routes, source_scores, transition_profile[tgt],
            )
        stg = getattr(self.store, "_staging", None)
        resident = (self.store.resident_experts(tgt)
                    if stg is not None and hasattr(self.store, "resident_experts") else None)
        if (config.transition_trace()
                and source_routes is not None
                and source_scores is not None):
            from mlx_streaming.core.prefetch import transition_trace
            transition_trace.record_prediction(
                source_layer=self.layer_idx,
                target_layer=tgt,
                source_routes=source_routes,
                source_scores=source_scores,
                proxy_logits=g,
                resident=resident or (),
                width=config.transition_trace_width(),
                hidden_states=x if config.residual_hidden_trace() else None,
            )
        route_delta_trace = None
        route_forward_id = None
        if (config.route_delta_trace()
                and source_routes is not None
                and source_scores is not None):
            from mlx_streaming.core.prefetch import route_delta_trace
            route_forward_id = route_delta_trace.current_forward_id()
        token_count = int(g.size) // int(g.shape[-1])
        prefetch_budget = int(getattr(stg, "budget", 0))
        # 单 token 且候选宽度不超过 I/O budget 时，Noisy-OR 与原 gate 排序等价，
        # 额外 softmax/prod/sort 只有开销；仅多 token 或确需 budget 内重排时启用。
        rerank_width_policy = (
            "predicted_route_union"
            if source_correction is not None or profile_applied
            else config.prefetch_rerank_width_policy()
        )
        rerank_ranking_policy = (
            source_correction.ranking_policy
            if source_correction is not None
            else target_cache_profile.ranking_policy
            if profile_applied
            else config.prefetch_rerank_ranking_policy_overrides().get(
                int(tgt), config.prefetch_rerank_ranking_policy(),
            )
        )
        rerank_union_margin = (
            int(source_correction.union_margin)
            if source_correction is not None
            else int(target_cache_profile.union_margin)
            if profile_applied
            else config.prefetch_rerank_union_margin_overrides().get(
                int(tgt), config.prefetch_rerank_union_margin(),
            )
        )
        rerank_candidate_width = (
            int(source_correction.candidate_width)
            if source_correction is not None
            else int(target_cache_profile.candidate_width)
            if profile_applied
            else config.prefetch_rerank_candidate_width()
        )
        use_rerank = (
            source_correction is not None
            or profile_applied
            or config.prefetch_rerank() == "noisy_or"
            and (
                token_count > 1
                or rerank_candidate_width > prefetch_budget
                # 单 token 时排序与原 gate 相同，但 predicted-route-union 仍必须
                # 把输出压到 1.5×top_k；跳过会让 W=24 直接违反 K=1 的 15 上限。
                or rerank_width_policy != "mass"
            )
        )
        progressive = config.prefetch_progressive_for(tgt)
        if progressive:
            # Keep main's first source callback unchanged.  Only a conservative
            # core is irreversible here; the same-forward T-1 decoder boundary
            # later fills the remaining legal slots from a moved exact gate.
            from mlx_streaming.core.prefetch.progressive import initial_core_ids

            pred = initial_core_ids(
                raw_proxy_logits,
                top_k=_effective_top_k(tmlp.top_k),
                core_width=config.prefetch_progressive_core_for(tgt),
                resident=resident or (),
                candidate_width=config.prefetch_rerank_candidate_width(),
            )
        elif use_rerank:
            from mlx_streaming.core.prefetch.rerank import rerank_prefetch_candidates
            # Candidate membership is the raw per-token top64 union.  Output
            # width is a separate physical contract and can never exceed the
            # side-region rows available to this submission.
            # Pool capacity and per-occurrence prediction width are separate
            # contracts.  Keeping 32 persistent side rows must not silently
            # turn a K=3 rerank into W32 (or K=1 into W24/32): that increases
            # SSD traffic and evicts useful rows without improving the target
            # route budget.  The raw candidate membership remains top64.
            logical_width = config.prefetch_rerank_max_width(
                default=15 if token_count == 1 else 26,
            )
            output_width = min(rerank_candidate_width, logical_width)
            if prefetch_budget > 0:
                output_width = min(output_width, prefetch_budget)
            _audit_rerank = (
                config.prefetch_audit_prof()
                and vp is not None
                and config.zerocopy_dual_source()
            )
            _rerank_result = rerank_prefetch_candidates(
                g,
                top_k=_effective_top_k(tmlp.top_k),
                max_width=output_width,
                retained_mass=config.prefetch_rerank_mass(),
                min_width=config.prefetch_rerank_min_width(
                    default=_effective_top_k(tmlp.top_k)),
                resident=resident or (),
                width_policy=rerank_width_policy,
                ranking_policy=rerank_ranking_policy,
                union_margin=rerank_union_margin,
                # 原 proxy 固定每 token raw top64 union；resident 在候选
                # 生成后过滤且不补位。correction 绝不能从完整 512 轴补入。
                candidate_logits=raw_proxy_logits,
                candidate_ranking_policy="max",
                candidate_width=rerank_candidate_width,
                return_candidate_ids=_audit_rerank,
            )
            pred, _rerank_scores, _rerank_keep = _rerank_result[:3]
            if _audit_rerank:
                vp.record_rerank_acceptance(
                    tgt,
                    candidate_ids=_rerank_result[3],
                    selected_ids=pred,
                    online_width=mx.sum(_rerank_keep.astype(mx.int32)),
                    resident=resident,
                )
            if config.prefetch_rerank_prof():
                note_rerank(_rerank_scores, _rerank_keep, len(resident or ()))
        # 按 seq 维聚合成"该批 K 个 token 的专家并集"近似。聚合方式 PREDICT_AGG：
        # - max（默认）：任一 token 强烈想要即入选；mean：各 token 平均偏好；
        # - union：每 token 各取 top-kk 再并集（与真实路由"并集"结构最一致，候选更多，由 handler 去重+截断）。
        else:
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
        if stg is not None:
            if config.zerocopy_dual_source():
                # unified allocation：默认直接写最终 GPU-addressable rows；
                # 仅 PREFETCH_DIRECT_SLOTS=0 的兼容路径使用全局 staging/promote。
                rp = self.store._resident
                if tgt not in rp._pools:
                    return None                          # 目标层池未建(首 token 预热) → 跳过本次
                segs = stg.src._segs                     # (proj, tensor, dt, shape, nb)，与池 key 顺序一致
                pool_list = [rp._pools[tgt][f"{p}.{t}"] for p, t, *_ in segs]
                import os as _os
                if _os.environ.get("POOL_PTR_TRACE") is not None and tgt == int(_os.environ["POOL_PTR_TRACE"]):
                    try:
                        import mlx_streaming.native_moe_ext as _N
                        _fk = f"{segs[0][0]}.{segs[0][1]}"
                        print(f"[POOL_PTR] submit  layer={tgt} key={_fk} obj_id={id(rp._pools[tgt][_fk])} "
                              f"ptr={hex(_N.array_data_ptr(rp._pools[tgt][_fk]))}", flush=True)
                    except Exception as _e:
                        print(f"[POOL_PTR] err {_e}", flush=True)
                result = self._vpool.prefetch(
                    tgt, pred, resident, pool_list,
                    source_layer=self.layer_idx,
                )
                if progressive:
                    self._vpool.record_progressive(
                        tgt,
                        candidate_logits=raw_proxy_logits,
                        early_ids=pred,
                        resident=resident,
                        pool_list=pool_list,
                        top_k=_effective_top_k(tmlp.top_k),
                        candidate_width=config.prefetch_rerank_candidate_width(),
                        early_dummy=result,
                    )
                if route_forward_id is not None:
                    route_delta_trace.record_prediction(
                        forward_id=route_forward_id,
                        source_layer=self.layer_idx,
                        target_layer=tgt,
                        source_routes=source_routes,
                        source_scores=source_scores,
                        proxy_logits=raw_proxy_logits,
                        hidden_states=x,
                        resident=tuple(resident or ()),
                        width=config.route_delta_trace_width(),
                    )
                return result
            # miss→hit:回调按目标层常驻快照过滤，只把缺口 pread 进 staging（≤budget 行），promote 时写池。
            if TPROF_ON:
                _ts0 = time.perf_counter()
                result = stg.submit(tgt, pred, resident)
                note_tprof("submit_s", time.perf_counter() - _ts0, count_key="submit_n")
            else:
                result = stg.submit(tgt, pred, resident)
            if route_forward_id is not None:
                route_delta_trace.record_prediction(
                    forward_id=route_forward_id,
                    source_layer=self.layer_idx,
                    target_layer=tgt,
                    source_routes=source_routes,
                    source_scores=source_scores,
                    proxy_logits=raw_proxy_logits,
                    hidden_states=x,
                    resident=tuple(resident or ()),
                    width=config.route_delta_trace_width(),
                )
            return result
        # 仅预热字节（page cache）的轻量版。
        path = os.path.join(bl.dir, f"layer{tgt:02d}.blob")
        result = _N.prefetch_on_complete(pred, path, int(bl.stride), True)
        if route_forward_id is not None:
            route_delta_trace.record_prediction(
                forward_id=route_forward_id,
                source_layer=self.layer_idx,
                target_layer=tgt,
                source_routes=source_routes,
                source_scores=source_scores,
                proxy_logits=raw_proxy_logits,
                hidden_states=x,
                resident=tuple(resident or ()),
                width=config.route_delta_trace_width(),
            )
        return result

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
        raw_gates = self.gate(x)
        gates = mx.softmax(raw_gates, axis=-1, precise=True)
        k = _effective_top_k(self.top_k)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        if config.route_delta_trace():
            from mlx_streaming.core.prefetch import route_delta_trace
            forward_id = route_delta_trace.current_forward_id()
            if forward_id is not None:
                route_delta_trace.record_target(
                    forward_id=forward_id,
                    layer=self.layer_idx,
                    target_routes=inds,
                    actual_logits=raw_gates,
                    future_hidden=x,
                )
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
            y = y + self._shared_forward(x)
        mx.eval(y)
        _tick("combine", t)
        PROF["n_calls"] += 1
        return y
