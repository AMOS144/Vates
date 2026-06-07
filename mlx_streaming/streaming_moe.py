"""路线 B：流式 MoE。

核心思路：MoE 每个 token 只激活 top-k 个专家。先求出本批涉及的「唯一专家集合」，
只在这些专家上做 gather 计算（权重切片到 uniq），用「全局专家 id → 本地下标」的
重映射喂给 gather。数值上与 mlx-lm 的 SwitchGLU 完全等价，但参与计算的专家从
全部 E 个降到实际激活的少数几个 —— 这是把常驻压到正比于活跃专家的前提。
"""
import math
import os
import time
from typing import Tuple

import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear, SwitchGLU

# 细粒度热路径计时(env STREAM_PROF=1 开启)。累加各段秒数,probe 读取。
_PROF_ON = os.environ.get("STREAM_PROF", "0") == "1"
PROF = {"route": 0.0, "pyremap": 0.0, "fetch": 0.0, "matmul": 0.0, "combine": 0.0,
        "n_calls": 0}


def _effective_top_k(default_k: int) -> int:
    """实验开关：降低每 token 激活专家数以测速度/质量取舍。默认不改变模型。"""
    override = os.environ.get("MOE_TOPK_OVERRIDE")
    if not override:
        return default_k
    return max(1, min(default_k, int(override)))


def prof_reset():
    for k in PROF:
        PROF[k] = 0.0


def _tick(seg, t0):
    mx.eval  # noqa  (占位,避免误用)
    PROF[seg] += time.perf_counter() - t0


def _unique_and_local(inds: mx.array) -> Tuple[mx.array, mx.array]:
    """返回 (uniq, local)：uniq 是排序后的唯一全局专家 id，local 是与 inds 同形状的本地下标。"""
    flat = [int(i) for i in inds.reshape(-1).tolist()]
    uniq_sorted = sorted(set(flat))
    remap = {g: i for i, g in enumerate(uniq_sorted)}
    local = mx.array([remap[i] for i in flat], dtype=inds.dtype).reshape(inds.shape)
    uniq = mx.array(uniq_sorted, dtype=inds.dtype)
    return uniq, local


def _slice_switch_linear(lin, uniq: mx.array):
    """把一个 SwitchLinear/QuantizedSwitchLinear 沿专家维切到 uniq，返回新的小 linear。"""
    n = int(uniq.shape[0])
    has_bias = "bias" in lin
    if isinstance(lin, QuantizedSwitchLinear):
        new = QuantizedSwitchLinear(
            lin.input_dims, lin.output_dims, n, bias=has_bias,
            group_size=lin.group_size, bits=lin.bits, mode=lin.mode,
        )
    else:
        new = SwitchLinear(lin.input_dims, lin.output_dims, n, bias=has_bias)

    sliced = {}
    for name, p in lin.parameters().items():
        if isinstance(p, mx.array) and p.ndim >= 1 and p.shape[0] == lin.num_experts:
            sliced[name] = p[uniq]   # 第 0 维是专家维，按 uniq 取子集
        else:
            sliced[name] = p
    new.update(sliced)
    return new


def streaming_switch_glu_forward(glu: SwitchGLU, x: mx.array, inds: mx.array) -> mx.array:
    """只在 inds 涉及到的唯一专家上做 SwitchGLU 等价计算。"""
    uniq, local = _unique_and_local(inds)
    sub = SwitchGLU(glu.gate_proj.input_dims, glu.gate_proj.output_dims, int(uniq.shape[0]))
    sub.gate_proj = _slice_switch_linear(glu.gate_proj, uniq)
    sub.up_proj = _slice_switch_linear(glu.up_proj, uniq)
    sub.down_proj = _slice_switch_linear(glu.down_proj, uniq)
    sub.activation = glu.activation
    return sub(x, local)


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


def _build_qsl(prefix: str, fetched: dict, in_dims: int, out_dims: int, n: int,
               group_size: int, bits: int):
    """用 fetched 里以 prefix.* 为键的堆叠参数构造一个 QuantizedSwitchLinear。"""
    lin = QuantizedSwitchLinear(in_dims, out_dims, n, bias=False,
                                group_size=group_size, bits=bits, mode="affine")
    sub = {name.split(".", 1)[1]: v for name, v in fetched.items()
           if name.startswith(prefix + ".")}
    lin.update(sub)
    return lin


def streaming_switch_glu_forward_from_store(store, layer, x, inds, hidden, moe_inter,
                                            group_size, bits):
    """文件后端版：从 store 只取选中专家、构造小 SwitchGLU 计算（等价于全量 SwitchGLU）。

    无状态版本（每次新建 sub），保留给单测使用；端到端走 PersistentSubGLU。
    """
    uniq, local = _unique_and_local(inds)
    fetched = store.fetch(layer, [int(i) for i in uniq.tolist()])
    n = int(uniq.shape[0])
    sub = SwitchGLU(hidden, moe_inter, n)
    sub.gate_proj = _build_qsl("gate_proj", fetched, hidden, moe_inter, n, group_size, bits)
    sub.up_proj = _build_qsl("up_proj", fetched, hidden, moe_inter, n, group_size, bits)
    sub.down_proj = _build_qsl("down_proj", fetched, moe_inter, hidden, n, group_size, bits)
    return sub(x, local)


def _update_qsl(lin, prefix: str, fetched: dict):
    """原地更新一个 QuantizedSwitchLinear 的参数（只换数组引用，不重建对象）。"""
    sub = {name.split(".", 1)[1]: v for name, v in fetched.items()
           if name.startswith(prefix + ".")}
    lin.update(sub)


class PersistentSubGLU:
    """按专家数 n 缓存一套 SwitchGLU + 3×QuantizedSwitchLinear，跨调用复用。

    解码 batch=1 时每层唯一专家数恒为 top_k，n 不变 → 只在首次/n 变化时构造一次，
    之后每个 token 仅原地 update 三组权重，省掉每调用重建对象与初始化随机量化权重的开销。
    """

    def __init__(self, hidden: int, moe_inter: int, group_size: int, bits: int,
                 proj_bits: dict | None = None):
        self.hidden = hidden
        self.moe_inter = moe_inter
        self.group_size = group_size
        self.bits = bits
        # 混合精度：每个 proj 用各自 bit；None 时三 proj 统一用 bits。
        self.proj_bits = proj_bits or {
            "gate_proj": bits, "up_proj": bits, "down_proj": bits}
        self._glu = None
        self._n = None

    def _ensure(self, n: int):
        if self._glu is not None and self._n == n:
            return
        pb = self.proj_bits
        glu = SwitchGLU(self.hidden, self.moe_inter, n)
        glu.gate_proj = QuantizedSwitchLinear(
            self.hidden, self.moe_inter, n, bias=False,
            group_size=self.group_size, bits=pb["gate_proj"], mode="affine")
        glu.up_proj = QuantizedSwitchLinear(
            self.hidden, self.moe_inter, n, bias=False,
            group_size=self.group_size, bits=pb["up_proj"], mode="affine")
        glu.down_proj = QuantizedSwitchLinear(
            self.moe_inter, self.hidden, n, bias=False,
            group_size=self.group_size, bits=pb["down_proj"], mode="affine")
        self._glu = glu
        self._n = n

    def forward(self, fetched: dict, n: int, x: mx.array, local: mx.array) -> mx.array:
        self._ensure(n)
        _update_qsl(self._glu.gate_proj, "gate_proj", fetched)
        _update_qsl(self._glu.up_proj, "up_proj", fetched)
        _update_qsl(self._glu.down_proj, "down_proj", fetched)
        return self._glu(x, local)


class RotatedSubGLU(PersistentSubGLU):
    """旋转版 PersistentSubGLU：复用对象缓存/原地 update，但前向在 gate/up 前旋转输入、
    在 down 前旋转中间激活。配合 rotate_requantize_experts 产出的旋转权重，数学等价 W·x。
    """

    def forward(self, fetched: dict, n: int, x: mx.array, local: mx.array) -> mx.array:
        self._ensure(n)
        _update_qsl(self._glu.gate_proj, "gate_proj", fetched)
        _update_qsl(self._glu.up_proj, "up_proj", fetched)
        _update_qsl(self._glu.down_proj, "down_proj", fetched)
        return self._rotated_call(self._glu, x, local)

    def _rotated_call(self, glu, x, indices):
        # 镜像 mlx_lm SwitchGLU.__call__，仅在两处插入 Hadamard（沿特征维，与 token 维排序正交）
        hs = self.hidden ** -0.5
        ms = self.moe_inter ** -0.5
        x = mx.hadamard_transform(x, scale=hs)        # 输入旋转（gate/up 共用）
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            from mlx_lm.models.switch_layers import _gather_sort
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = glu.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = glu.gate_proj(x, idx, sorted_indices=do_sort)
        a = glu.activation(x_up, x_gate)              # 融合激活（原 SwitchGLU 约定）
        a = mx.hadamard_transform(a, scale=ms)        # 中间激活旋转（down 前）
        x = glu.down_proj(a, idx, sorted_indices=do_sort)
        if do_sort:
            from mlx_lm.models.switch_layers import _scatter_unsort
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


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
        # 持久化子模块：跨 token 复用，避免每调用重建 QSL。
        # rotated=True 时用 Hadamard 旋转前向（配合旋转重量化的专家权重）。
        # proj_bits 非空时走混合精度（逐 proj 不同 bit），与 rotated 互斥。
        if rotated:
            self._sub = RotatedSubGLU(hidden, moe_inter, group_size, bits)
        else:
            self._sub = PersistentSubGLU(hidden, moe_inter, group_size, bits,
                                         proj_bits=proj_bits)

    def __call__(self, x: mx.array) -> mx.array:
        if _PROF_ON:
            return self._call_prof(x)
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = _effective_top_k(self.top_k)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        # 热路径：只在此处对 inds 做一次 .tolist() 同步，uniq/local 全在 Python 里算，
        # 避免再对 uniq 做第二次 .tolist()（以及冗余的 mx.eval(inds)）。
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        uniq_set = set(flat)
        # 池只能同时容纳 ≤该层容量 个唯一专家；prefill/大批量 verify 唯一数超容量时回退 stack
        layer_cap = self.store.cap_for(self.layer_idx)
        if (os.environ.get("RESIDENT_POOL", "1") == "1"
                and len(uniq_set) <= layer_cap):
            # slots 与 flat 一一对应(含重复专家)，直接 reshape 成 routing 的 slot 索引
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
        use_pool = (os.environ.get("RESIDENT_POOL", "1") == "1"
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


def patch_model_filebacked(model, store, hidden, moe_inter, group_size, bits,
                           rotated: bool = False, proj_bits: dict | None = None,
                           layer_proj_bits: dict | None = None):
    """把每个 MoE 块替换为 FileStreamingMoeBlock，并丢弃常驻的堆叠 switch_mlp。

    store：FileExpertStore（所有 MoE 层共用，按 (layer,expert) 缓存）。
    rotated：是否用 Hadamard 旋转前向（专家权重须为 rotate_requantize 产出的旋转版）。
    proj_bits：非空时走混合精度（逐 proj 不同 bit），专家须为对应混合重量化产出。
    layer_proj_bits：{绝对层号: {proj:bits}}，非空时逐层用各自 proj_bits（优先于 proj_bits），
        对应 requantize_dir_layered 产出。各层 QSL 用该层 bit，与流式存盘文件一一对应。
    返回被替换的层数。被替换后原 switch_mlp 不再被引用，惰性权重不会被物化。
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            pb = layer_proj_bits.get(i, proj_bits) if layer_proj_bits else proj_bits
            # 捕获共享专家引用使其常驻（Qwen3-Next 有，Qwen3-MoE 无）
            layer.mlp = FileStreamingMoeBlock(
                gate=mlp.gate, top_k=mlp.top_k, norm_topk_prob=mlp.norm_topk_prob,
                store=store, layer_idx=i, hidden=hidden, moe_inter=moe_inter,
                group_size=group_size, bits=bits, rotated=rotated, proj_bits=pb,
                shared_expert=getattr(mlp, "shared_expert", None),
                shared_expert_gate=getattr(mlp, "shared_expert_gate", None),
            )
            patched += 1
    return patched


def patch_model(model, store_factory=None):
    """把模型里每个 MoE 块（含 switch_mlp 与 gate 的块）替换成 StreamingMoeBlock。

    store_factory(layer_idx)->LruExpertStore，可为 None（仅做 uniq 切片、不接磁盘后端）。
    返回被替换的层数，便于校验确实命中了 MoE 层。
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            store = store_factory(i) if store_factory is not None else None
            layer.mlp = StreamingMoeBlock(mlp, layer_idx=i, store=store)
            patched += 1
    return patched
