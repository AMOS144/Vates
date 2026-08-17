"""把 Qwen3-Next 的 12 个全注意力层就地切换到 K4/V3 旋转量化 KV。

做法(隔离、低侵入):
- 子类 `_RotatedQuantAttn` 仅重写 `__call__`:RoPE 后对 q/k 旋转(R_k)、v 旋转(R_v),
  存非对称量化 cache,走 asym_quantized_sdpa,输出做 V 逆旋转复原。保参数树不变
  (只换 `__class__`、用 object.__setattr__ 挂常量,不进 nn.Module 的参数/子模块树)。
- 覆盖 `model.make_cache`:全注意力层→AsymmetricQuantizedKVCache,线性层→ArraysCache(size=2)。
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from mlx_streaming.core.cache.quant_kv import (
    AsymmetricQuantizedKVCache,
    asym_quantized_sdpa,
    build_block_so4,
    rotate_last,
)


class _RotatedQuantAttn(Qwen3NextAttention):
    """与原 Qwen3NextAttention.__call__ 数值等价(高 bit 时),叠加旋转 + 非对称量化。"""

    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # 旋转去相关(RoPE 之后):q、k 用 R_k(分数自动抵消),v 用 R_v(输出后逆旋转)。
        Rk = self._kvq_Rk
        Rv = self._kvq_Rv
        if Rk is not None:
            queries = rotate_last(queries, Rk)
            keys = rotate_last(keys, Rk)
            values = rotate_last(values, Rv)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
            output = asym_quantized_sdpa(
                queries, keys, values, scale=self.scale, mask=mask,
                group_size=self._kvq_gs, k_bits=self._kvq_kb, v_bits=self._kvq_vb)
        else:
            # 无 cache(罕见,如纯前向探针):走稠密 SDPA,V 仍在旋转空间,下方逆旋转复原。
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask)

        if Rv is not None:
            output = rotate_last(output, Rv.T)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output * mx.sigmoid(gate))


def _inner(model):
    """取到含 .layers 的内层模块(mlx_lm Model 包了一层 .model)。"""
    return model.model if hasattr(model, "model") else model


def patch_kv_quant(model, *, group_size=64, k_bits=4, v_bits=3, rotate=True, seed=0):
    """就地把所有全注意力层切到旋转 + 非对称量化 KV,并覆盖 make_cache。返回 model。"""
    inner = _inner(model)
    layers = inner.layers

    head_dim = None
    for l in layers:
        if not l.is_linear:
            head_dim = l.self_attn.head_dim
            break
    if head_dim is None:
        return model  # 没有全注意力层,无需 patch

    Rk = build_block_so4(head_dim, seed=seed) if rotate else None
    Rv = build_block_so4(head_dim, seed=seed + 1) if rotate else None

    for l in layers:
        if l.is_linear:
            continue
        attn = l.self_attn
        # 常量挂载用 object.__setattr__,避免进 nn.Module 参数/子模块树(否则破坏权重 save/树遍历)。
        object.__setattr__(attn, "_kvq_Rk", Rk)
        object.__setattr__(attn, "_kvq_Rv", Rv)
        object.__setattr__(attn, "_kvq_gs", group_size)
        object.__setattr__(attn, "_kvq_kb", k_bits)
        object.__setattr__(attn, "_kvq_vb", v_bits)
        attn.__class__ = _RotatedQuantAttn

    def make_cache():
        return [
            AsymmetricQuantizedKVCache(group_size, k_bits, v_bits)
            if not l.is_linear else ArraysCache(size=2)
            for l in layers
        ]

    # 覆盖实例方法(plain function,nn.Module.__setattr__ 不会把函数纳入参数树)。
    model.make_cache = make_cache
    return model
