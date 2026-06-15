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
    PROF, WINDOW_PROF, PREDICT_RECALL_PROF, _PROF_ON, _tick)
from mlx_streaming.core.moe.gate import _effective_top_k
from mlx_streaming.core.moe.compute import (
    streaming_switch_glu_forward, PersistentSubGLU, RotatedSubGLU)


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
                 hidden, moe_inter, group_size, bits, rotated: bool = False,
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
        # rotated=True 时用 Hadamard 旋转前向（配合旋转重量化的专家权重）。
        # proj_bits 非空时走混合精度（逐 proj 不同 bit），与 rotated 互斥。
        if rotated:
            self._sub = RotatedSubGLU(hidden, moe_inter, group_size, bits)
        else:
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
        _stg_mgr = getattr(self.store, "_staging", None)
        if _stg_mgr is not None and not config.native_no_promote():
            # native-fused-prefetch miss→hit:把回调预读好的本层专家写进池槽（acquire 前）。
            _stg_mgr.promote(self.layer_idx, self.store)
        if config.route_trace_enabled():
            flat_trace = [int(i) for i in inds.reshape(-1).tolist()]
            resident = (self.store.resident_experts(self.layer_idx)
                        if hasattr(self.store, "resident_experts") else set())
            rank = (self.store.resident_lru_scores(self.layer_idx)
                    if hasattr(self.store, "resident_lru_scores") else {})
            route_trace.record(
                self.layer_idx, flat_trace, set(flat_trace) - resident, resident, rank)
        # native-fused-prefetch：在 seq 分支**之前**提交，使 decode(seq=1) 与 MTP verify(seq=K+1)
        # 两条路径都触发预取。dummy 折进 inds(加 0)：GPU 路径靠 acquire_gpu 的 n_miss eval、
        # host 路径靠后面的 .tolist() eval，都会触发完成回调里的 pread。verify 时 x 为 K+1 个
        # token，预测的是"下一层这 K+1 token 的专家并集"（recall≈0.96 的口径）。
        if (getattr(self.store, "_staging", None) is not None
                and not config.native_no_submit()):
            _dummy = self._native_fused_prefetch(x)
            if _dummy is not None:
                inds = inds + (_dummy.reshape(()).astype(inds.dtype) * 0)
        layer_cap = self.store.cap_for(self.layer_idx)
        if (config.resident_pool_enabled()
                and config.gpu_remap_enabled()
                and x.shape[1] == 1):
            # decode 热路径:GPU 侧 slot 重映射,命中层零 host 往返(消除每层 .tolist 同步),
            # 仅一次 miss 标志同步;真 miss 才回退读盘。decode top-k(≤cap)恒装得下。
            # 专家总数 = gate 输出维(gates 末维),无需依赖 gate.weight。
            pool_arrays, local = self.store.acquire_gpu(
                self.layer_idx, inds, gates.shape[-1])
            y = self._sub.forward(pool_arrays, layer_cap, x, local)
        else:
            # prefill/大批量(seq>1)或显式关闭 GPU_REMAP:走 host 路径。
            # 只在此处对 inds 做一次 .tolist() 同步，uniq/local 全在 Python 里算。
            flat = [int(i) for i in inds.reshape(-1).tolist()]
            uniq_set = set(flat)
            # 池只能同时容纳 ≤该层容量 个唯一专家；prefill 唯一数超容量时回退 stack
            if (config.resident_pool_enabled()
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
        ahead = max(1, config.cross_layer_ahead(default=1))
        tgt = self.layer_idx + ahead
        layers = getattr(model, "layers", [])
        if not (0 <= tgt < len(layers)):
            return None
        tmlp = getattr(layers[tgt], "mlp", None)
        if not isinstance(tmlp, FileStreamingMoeBlock):
            return None
        try:
            from mlx_streaming import native_moe_ext as _N
        except Exception:
            return None
        budget = config.stream_blob_bg_budget(default=4)
        # 对齐 0.95-recall 探针：目标层 post_attention_layernorm + gate，作用在本层**未归一化**输入。
        h = getattr(self, "_unnormed_input", None)
        if h is None:
            h = x  # 退化：无未归一化输入时用 x（norm 偏差，覆盖会差）
            g = tmlp.gate(h)
        else:
            g = tmlp.gate(layers[tgt].post_attention_layernorm(h))
        # 按 seq 维聚合成"该批 K+1 token 的专家并集"近似（max），再取 top-budget。
        # 必须 ≤ budget，否则 verify(seq=K+1) pred 超 staging 行数 → 越界崩。
        if g.ndim == 3:
            g = g.max(axis=1)
        kk = min(g.shape[-1], budget)
        pred = mx.argpartition(g, kth=-kk, axis=-1)[..., -kk:].reshape(-1).astype(mx.uint32)
        if config.predict_recall_prof():
            try:
                tmlp._predicted_set = {int(e) for e in pred.tolist()}
            except Exception:
                pass
        stg = getattr(self.store, "_staging", None)
        if stg is not None:
            # miss→hit:pread 进 staging（回调），promote 时写进池。
            return stg.submit(tgt, pred)
        # 仅预热字节（page cache）的轻量版。
        path = os.path.join(bl.dir, f"layer{tgt:02d}.blob")
        return _N.prefetch_on_complete(pred, path, int(bl.stride), True)

    def _try_native_forward(self, x: mx.array, inds: mx.array, scores: mx.array, num_experts: int):
        # native 第一版只支持未旋转、三投影同 bit 的专家；其他情况保持现有 MLX 回退。
        if isinstance(self._sub, RotatedSubGLU):
            return None
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
