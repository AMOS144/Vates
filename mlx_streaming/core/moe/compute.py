"""MoE 专家计算：把选中专家切片成小 SwitchGLU 并前向（数值等价全量 SwitchGLU）。

设计要点：
- `_unique_and_local`：把 top-k 路由的全局专家 id 压成「唯一集合 + 本地下标」，
  只在实际激活的少数专家上 gather 计算。
- `PersistentSubGLU`：跨调用复用一套 SwitchGLU + 3×QuantizedSwitchLinear 对象，
  解码 batch=1 时唯一专家数恒为 top_k，只首次构造、之后原地 update 权重，省重建开销。
"""
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchLinear, QuantizedSwitchLinear, SwitchGLU

from mlx_streaming import config
from mlx_streaming.core.moe.custom_kernel import (
    _custom_qproj_enabled, _custom_fused_moe_enabled, _custom_qproj_targets,
    _custom_qlinear_indexed, _custom_fused_moe_indexed)


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
                 proj_bits: dict | None = None, layer_idx: int | None = None,
                 quant_mode: str = "affine", swiglu_limit: float = 0.0):
        self.hidden = hidden
        self.moe_inter = moe_inter
        self.group_size = group_size
        self.bits = bits
        self.layer_idx = layer_idx
        # 混合精度：每个 proj 用各自 bit；None 时三 proj 统一用 bits。
        self.proj_bits = proj_bits or {
            "gate_proj": bits, "up_proj": bits, "down_proj": bits}
        # quant_mode：量化模式（affine / mxfp4）；swiglu_limit>0 时走 DeepSeek 手动 clip 路径。
        self.quant_mode = quant_mode
        self.swiglu_limit = swiglu_limit
        self._glu = None
        self._n = None

    def _ensure(self, n: int):
        if self._glu is not None and self._n == n:
            return
        pb = self.proj_bits
        glu = SwitchGLU(self.hidden, self.moe_inter, n)
        glu.gate_proj = QuantizedSwitchLinear(
            self.hidden, self.moe_inter, n, bias=False,
            group_size=self.group_size, bits=pb["gate_proj"], mode=self.quant_mode)
        glu.up_proj = QuantizedSwitchLinear(
            self.hidden, self.moe_inter, n, bias=False,
            group_size=self.group_size, bits=pb["up_proj"], mode=self.quant_mode)
        glu.down_proj = QuantizedSwitchLinear(
            self.moe_inter, self.hidden, n, bias=False,
            group_size=self.group_size, bits=pb["down_proj"], mode=self.quant_mode)
        self._glu = glu
        self._n = n

    def release_bound(self) -> None:
        """Drop transient expert arrays after a long expert-major layer."""
        self._glu = None
        self._n = None
        self._bound_signature = None

    def forward(self, fetched: dict, n: int, x: mx.array, local: mx.array) -> mx.array:
        self._bind(fetched, n)
        if self.swiglu_limit > 0:
            # DeepSeek path below retains its special clipping semantics.
            x_exp = mx.expand_dims(x, (-2, -3))
            gate = self._glu.gate_proj(x_exp, local)
            up = self._glu.up_proj(x_exp, local)
            up = mx.clip(up, -self.swiglu_limit, self.swiglu_limit)
            gate = mx.minimum(gate, self.swiglu_limit)
            h = nn.silu(gate) * up
            return self._glu.down_proj(h, local).squeeze(-2)
        max_fused_seq = config.custom_fused_moe_max_seq()
        if (x.shape[1] <= max_fused_seq
                and _custom_fused_moe_enabled(
                    -1 if self.layer_idx is None else self.layer_idx,
                    self.proj_bits,
                )):
            return self._custom_fused_forward(x, local)
        max_seq = config.custom_qproj_max_seq()
        if (x.shape[1] <= max_seq
                and _custom_qproj_enabled(
                    -1 if self.layer_idx is None else self.layer_idx,
                    self.proj_bits["gate_proj"],
                )):
            return self._custom_gate_up_forward(x, local)
        return self._glu(x, local)

    def _bind(self, fetched: dict, n: int) -> None:
        self._ensure(n)
        signature = (
            int(n),
            *((key, id(value)) for key, value in sorted(fetched.items())),
        )
        if getattr(self, "_bound_signature", None) != signature:
            _update_qsl(self._glu.gate_proj, "gate_proj", fetched)
            _update_qsl(self._glu.up_proj, "up_proj", fetched)
            _update_qsl(self._glu.down_proj, "down_proj", fetched)
            self._bound_signature = signature

    def forward_gate_up(
        self, fetched: dict, n: int, x: mx.array, local: mx.array,
    ) -> mx.array:
        """Consume only the six gate/up arrays of a partially-ready row."""
        self._bind(fetched, n)
        x_exp = mx.expand_dims(x, (-2, -3))
        gate = self._glu.gate_proj(x_exp, local)
        up = self._glu.up_proj(x_exp, local)
        return self._glu.activation(up, gate)

    def forward_down(
        self, fetched: dict, n: int, hidden: mx.array, local: mx.array,
    ) -> mx.array:
        """Consume the down arrays after their second readiness event."""
        self._bind(fetched, n)
        return self._glu.down_proj(hidden, local).squeeze(-2)

    def _custom_fused_forward(
        self,
        x: mx.array,
        local: mx.array,
        active_mask: "mx.array | None" = None,
    ) -> mx.array:
        """完整替换 gate/up/SwiGLU/down，保持 SwitchGLU 的 [B,S,K,H] 输出契约。"""
        shape = local.shape
        k = int(shape[-1])
        x_flat = mx.broadcast_to(mx.expand_dims(x, -2), x.shape[:-1] + (k, x.shape[-1]))
        x_flat = x_flat.reshape(-1, self.hidden).astype(mx.float32)
        idx = local.reshape(-1).astype(mx.uint32)
        y = _custom_fused_moe_indexed(
            x_flat, idx,
            self._glu.gate_proj["weight"], self._glu.gate_proj["scales"], self._glu.gate_proj["biases"],
            self._glu.up_proj["weight"], self._glu.up_proj["scales"], self._glu.up_proj["biases"],
            self._glu.down_proj["weight"], self._glu.down_proj["scales"], self._glu.down_proj["biases"],
            self.hidden, self.moe_inter, self.group_size, self.proj_bits["gate_proj"],
            active_mask=active_mask,
        )
        return y.reshape(shape + (self.hidden,))

    def forward_masked(
        self,
        fetched: dict,
        n: int,
        x: mx.array,
        local: mx.array,
        active_mask: mx.array,
    ) -> mx.array:
        """Fused expert pass that performs no matmul for inactive routes."""
        self._ensure(n)
        _update_qsl(self._glu.gate_proj, "gate_proj", fetched)
        _update_qsl(self._glu.up_proj, "up_proj", fetched)
        _update_qsl(self._glu.down_proj, "down_proj", fetched)
        if not _custom_fused_moe_enabled(
            -1 if self.layer_idx is None else self.layer_idx,
            self.proj_bits,
        ):
            raise RuntimeError("masked split compute requires CUSTOM_FUSED_MOE")
        return self._custom_fused_forward(
            x, local, active_mask=active_mask,
        )

    def _custom_gate_up_forward(self, x: mx.array, local: mx.array) -> mx.array:
        """只替换 gate/up projection；down 仍走 MLX QuantizedSwitchLinear。"""
        shape = local.shape
        k = int(shape[-1])
        x_flat = mx.broadcast_to(mx.expand_dims(x, -2), x.shape[:-1] + (k, x.shape[-1]))
        x_flat = x_flat.reshape(-1, self.hidden).astype(mx.float32)
        idx = local.reshape(-1).astype(mx.uint32)
        tile = config.custom_qproj_tile()
        up = _custom_qlinear_indexed(
            x_flat, idx,
            self._glu.up_proj["weight"], self._glu.up_proj["scales"], self._glu.up_proj["biases"],
            self.moe_inter, self.hidden, self.group_size, self.proj_bits["up_proj"], tile)
        gate = _custom_qlinear_indexed(
            x_flat, idx,
            self._glu.gate_proj["weight"], self._glu.gate_proj["scales"], self._glu.gate_proj["biases"],
            self.moe_inter, self.hidden, self.group_size, self.proj_bits["gate_proj"], tile)
        up = up.reshape(shape + (1, self.moe_inter))
        gate = gate.reshape(shape + (1, self.moe_inter))
        a = self._glu.activation(up, gate)
        if "down" in _custom_qproj_targets():
            a_flat = a.squeeze(-2).reshape(-1, self.moe_inter).astype(mx.float32)
            y = _custom_qlinear_indexed(
                a_flat, idx,
                self._glu.down_proj["weight"], self._glu.down_proj["scales"], self._glu.down_proj["biases"],
                self.hidden, self.moe_inter, self.group_size, self.proj_bits["down_proj"], tile)
            return y.reshape(shape + (self.hidden,))
        y = self._glu.down_proj(a, local)
        return y.squeeze(-2)
