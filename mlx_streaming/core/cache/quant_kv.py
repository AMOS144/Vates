"""IsoQuant 风格 K4/V3 非对称量化 KV cache(仅用于全注意力层)。

设计要点(详见 docs/superpowers/specs/2026-06-30-isoquant-kv-quant-design.md):
- SO(4) 块对角旋转去相关:head_dim=256 切成 64 个 4D 块,每块由两个单位四元数构造
  一个 SO(4) 旋转(left·right_conj sandwich)。旋转正交,在注意力分数里自动抵消
  (q、k 同旋转 → q·kᵀ 不变);对 V 旋转后需在 SDPA 输出上做逆旋转复原。
- 非对称仿射量化:K 用 k_bits(默认 4)、V 用 v_bits(默认 3),分别用 mx.quantize 打包,
  attention 走 mx.quantized_matmul 快路径(每次调用可传不同 bits),无需自写 Metal。

打包宽度用 dim*bits//32(对 3-bit 必须如此:mlx_lm 自带 8*4//bits 公式对 3-bit 会算错)。
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map, tree_reduce
# 注意:make_mask 用的是 cache 模块里的 create_attention_mask(签名含 offset/return_array),
# 与 base 模块同名但不同签名的那个不可混用。
from mlx_lm.models.cache import create_attention_mask


# ----------------------------- SO(4) 块旋转 -----------------------------

def _quat_left(q: np.ndarray) -> np.ndarray:
    """单位四元数 q 的左乘矩阵 L(q):把 v 映射到 q*v(Hamilton 积)。"""
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w],
    ], dtype=np.float32)


def _quat_right_conj(q: np.ndarray) -> np.ndarray:
    """单位四元数 q 的"右乘其共轭"矩阵 R(conj q):把 v 映射到 v*conj(q)。

    L(qL)·R(conj qR) 即 SO(4) 的双边旋转 v -> qL * v * conj(qR),覆盖全部 4D 旋转。
    """
    w, x, y, z = q
    return np.array([
        [ w,  x,  y,  z],
        [-x,  w,  z, -y],
        [-y, -z,  w,  x],
        [-z,  y, -x,  w],
    ], dtype=np.float32)


def build_block_so4(head_dim: int, seed: int = 0, blocks_of: int = 4) -> mx.array:
    """构造 head_dim×head_dim 的块对角正交矩阵,每 blocks_of(=4)维一个 SO(4) 旋转。

    data-oblivious:仅由 seed 决定,固定可复现,不依赖数据分布。
    """
    assert head_dim % blocks_of == 0, "head_dim 必须被 blocks_of 整除"
    rng = np.random.default_rng(seed)
    M = np.zeros((head_dim, head_dim), dtype=np.float32)
    for b in range(head_dim // blocks_of):
        qL = rng.standard_normal(4).astype(np.float32)
        qL /= np.linalg.norm(qL)
        qR = rng.standard_normal(4).astype(np.float32)
        qR /= np.linalg.norm(qR)
        blk = _quat_left(qL) @ _quat_right_conj(qR)   # 4×4 ∈ SO(4)
        i = b * blocks_of
        M[i:i + blocks_of, i:i + blocks_of] = blk
    return mx.array(M)


def rotate_last(x: mx.array, R: mx.array) -> mx.array:
    """对末维做正交旋转:x[..., D] @ R[D, D](按 x 的 dtype 计算)。"""
    return x @ R.astype(x.dtype)


# --------------------- 非对称量化 KV cache(K4/V3)---------------------

def _packed_words(dim: int, bits: int) -> int:
    """量化后每行占的 uint32 个数:dim*bits//32(对 2/3/4/5/6/8 bit 均正确)。"""
    return dim * bits // 32


class AsymmetricQuantizedKVCache:
    """K、V 各用独立位宽的量化 KV cache(仿 mlx_lm.QuantizedKVCache,但 K/V 不同 bits)。

    keys/values 各存为 (wq, scales, biases) 三元组,与 mx.quantized_matmul 接口对齐。
    """

    step = 256

    def __init__(self, group_size: int = 64, k_bits: int = 4, v_bits: int = 3):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.k_bits = k_bits
        self.v_bits = v_bits

    def _init_quant(self, dim, bits, B, H, steps, dt):
        packed = _packed_words(dim, bits)
        ngroups = dim // self.group_size
        return (
            mx.zeros((B, H, steps, packed), dtype=mx.uint32),
            mx.zeros((B, H, steps, ngroups), dtype=dt),
            mx.zeros((B, H, steps, ngroups), dtype=dt),
        )

    def update_and_fetch(self, keys, values):
        B, H, n, kD = keys.shape
        vD = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + n) > self.keys[0].shape[-2]:
            new_steps = (self.step + n - 1) // self.step * self.step
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = tree_map(lambda x: x[..., :prev, :], self.keys)
                    self.values = tree_map(lambda x: x[..., :prev, :], self.values)

                def _expand(x):
                    z = mx.zeros((B, H, new_steps, x.shape[-1]), dtype=x.dtype)
                    return mx.concatenate([x, z], axis=-2)

                self.keys = tree_map(_expand, self.keys)
                self.values = tree_map(_expand, self.values)
            else:
                self.keys = self._init_quant(kD, self.k_bits, B, H, new_steps, keys.dtype)
                self.values = self._init_quant(vD, self.v_bits, B, H, new_steps, values.dtype)

        self.offset += n
        kq = mx.quantize(keys, group_size=self.group_size, bits=self.k_bits)
        vq = mx.quantize(values, group_size=self.group_size, bits=self.v_bits)
        for i in range(3):
            self.keys[i][..., prev:self.offset, :] = kq[i]
            self.values[i][..., prev:self.offset, :] = vq[i]

        return (
            tree_map(lambda x: x[..., :self.offset, :], self.keys),
            tree_map(lambda x: x[..., :self.offset, :], self.values),
        )

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


# ----------------- 非对称量化 SDPA(K 用 k_bits、V 用 v_bits)-----------------

def asym_quantized_sdpa(queries, q_keys, q_values, scale, mask,
                        group_size, k_bits, v_bits):
    """fork 自 mlx_lm.quantized_scaled_dot_product_attention,K/V 各自传 bits。

    queries: [B, n_q_heads, L, D];q_keys/q_values: 量化三元组(n_kv_heads)。
    返回 [B, n_q_heads, L, D](仍在旋转后的 V 空间,调用方负责逆旋转)。
    """
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=k_bits)
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=v_bits)

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out
