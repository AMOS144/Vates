"""quant_kv.py 单元测试。

分层原则:
- 机制正确性(strict):旋转正交/抵消、GQA/mask/逆旋转的代数链路用"高 bit(8/8)"验,
  排除量化噪声,容差严格。
- 量化质量(lenient):随机高斯是 K4/V3 的最坏情况(无结构可去相关),此处只做宽松下界,
  真正的质量验收在真实模型的端到端测试(token 一致率 + logits cosine)。
"""
import numpy as np
import mlx.core as mx

from mlx_streaming.core.cache.quant_kv import (
    AsymmetricQuantizedKVCache,
    asym_quantized_sdpa,
    build_block_so4,
    rotate_last,
)


def _cos(a, b):
    return float((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b)))


# ----------------------------- SO(4) 旋转(机制)-----------------------------

def test_block_so4_orthonormal():
    """块对角 SO(4) 构造应正交(用 numpy 验构造本身,排除 mlx matmul 精度)。"""
    R = np.array(build_block_so4(256, seed=0))
    assert np.abs(R @ R.T - np.eye(256)).max() < 1e-4


def test_block_so4_determinant_blocks_are_rotations():
    """每个 4×4 块行列式 ≈ +1(纯旋转,无反射)。"""
    R = np.array(build_block_so4(16, seed=3))
    for b in range(4):
        blk = R[b * 4:(b + 1) * 4, b * 4:(b + 1) * 4]
        assert abs(np.linalg.det(blk) - 1.0) < 1e-3


def test_rotation_cancels_in_scores():
    """q、k 同旋转后注意力分数不变:(qR)·(kR)ᵀ = q·kᵀ(用 cosine 抗 float32 matmul 噪)。"""
    mx.random.seed(1)
    q = mx.random.normal((1, 2, 5, 256))
    k = mx.random.normal((1, 2, 7, 256))
    R = build_block_so4(256, seed=0)
    s0 = q @ k.transpose(0, 1, 3, 2)
    s1 = rotate_last(q, R) @ rotate_last(k, R).transpose(0, 1, 3, 2)
    assert _cos(s0, s1) > 0.9999


def test_v_inverse_rotation_recovers():
    """对 V 旋转后再逆旋转应复原:(vR)Rᵀ = v。"""
    mx.random.seed(7)
    v = mx.random.normal((1, 2, 9, 256))
    R = build_block_so4(256, seed=5)
    back = rotate_last(rotate_last(v, R), R.T)
    assert _cos(back, v) > 0.9999
    # 逐元素相对误差(放宽到 float32 256 维 matmul 的现实精度)
    assert float(mx.abs(back - v).max()) < 2e-2


# --------------------- 非对称量化 cache ---------------------

def test_asym_cache_roundtrip_shapes_and_bits():
    c = AsymmetricQuantizedKVCache(group_size=64, k_bits=4, v_bits=3)
    k = mx.random.normal((1, 2, 10, 256))
    v = mx.random.normal((1, 2, 10, 256))
    kq, vq = c.update_and_fetch(k, v)
    assert kq[0].dtype == mx.uint32 and vq[0].dtype == mx.uint32
    # K 4-bit:256*4//32 = 32 个 uint32;V 3-bit:256*3//32 = 24
    assert kq[0].shape[-1] == 32
    assert vq[0].shape[-1] == 24
    assert c.offset == 10
    kd = mx.dequantize(*kq, group_size=64, bits=4)
    vd = mx.dequantize(*vq, group_size=64, bits=3)
    assert kd.shape == (1, 2, 10, 256)
    # 4-bit K 误差应小于 3-bit V 误差
    assert float(mx.abs(kd - k).mean()) < float(mx.abs(vd - v).mean())


def test_asym_cache_incremental_growth():
    """逐 token 追加跨过 step 边界仍正确累加 offset 且形状对齐。"""
    c = AsymmetricQuantizedKVCache(group_size=64, k_bits=4, v_bits=3)
    c.update_and_fetch(mx.random.normal((1, 2, 300, 256)),
                       mx.random.normal((1, 2, 300, 256)))
    assert c.offset == 300
    kq = vq = None
    for _ in range(5):
        kq, vq = c.update_and_fetch(mx.random.normal((1, 2, 1, 256)),
                                    mx.random.normal((1, 2, 1, 256)))
    assert c.offset == 305
    assert kq[0].shape[-2] == 305


def test_asym_cache_nbytes_smaller_than_bf16():
    """K4/V3 量化存储应显著小于 bf16(含 scales/biases 开销)。"""
    c = AsymmetricQuantizedKVCache(group_size=64, k_bits=4, v_bits=3)
    n = 256
    c.update_and_fetch(mx.random.normal((1, 2, n, 256)),
                       mx.random.normal((1, 2, n, 256)))
    bf16_bytes = 2 * (1 * 2 * n * 256) * 2  # K+V, bf16=2B
    assert c.nbytes < bf16_bytes * 0.45     # 目标 ~K4/V3 ≈ 0.22x + 元数据


# --------------------- 非对称 SDPA ---------------------

def test_asym_sdpa_mechanism_high_bits_matches_dense():
    """机制验证:8-bit 下非对称 SDPA(GQA)应与稠密参考几乎一致(排除量化噪)。"""
    mx.random.seed(2)
    nq, nkv, L = 16, 2, 40
    q = mx.random.normal((1, nq, 1, 256))
    k = mx.random.normal((1, nkv, L, 256))
    v = mx.random.normal((1, nkv, L, 256))
    c = AsymmetricQuantizedKVCache(64, 8, 8)
    kq, vq = c.update_and_fetch(k, v)
    out = asym_quantized_sdpa(q, kq, vq, scale=256 ** -0.5, mask=None,
                              group_size=64, k_bits=8, v_bits=8)
    rep = nq // nkv
    qr = q.reshape(1, nkv, rep, 1, 256)
    kr = k.reshape(1, nkv, 1, L, 256)
    vr = v.reshape(1, nkv, 1, L, 256)
    s = mx.softmax((qr * 256 ** -0.5) @ kr.transpose(0, 1, 2, 4, 3), axis=-1, precise=True)
    ref = (s @ vr).reshape(1, nq, 1, 256)
    assert _cos(out, ref) > 0.999


def test_asym_sdpa_causal_mask_shape():
    """因果 mask(字符串/bool)路径不报错且形状正确。"""
    mx.random.seed(4)
    q = mx.random.normal((1, 16, 4, 256))
    k = mx.random.normal((1, 2, 4, 256))
    v = mx.random.normal((1, 2, 4, 256))
    c = AsymmetricQuantizedKVCache(64, 4, 3)
    kq, vq = c.update_and_fetch(k, v)
    out = asym_quantized_sdpa(q, kq, vq, scale=256 ** -0.5, mask="causal",
                              group_size=64, k_bits=4, v_bits=3)
    assert out.shape == (1, 16, 4, 256)


def test_asym_sdpa_k4v3_random_lower_bound():
    """质量下界(随机数据=最坏情况):K4/V3 cos 应 > 0.9(真实质量见 e2e)。"""
    mx.random.seed(2)
    nq, nkv, L = 16, 2, 40
    q = mx.random.normal((1, nq, 1, 256))
    k = mx.random.normal((1, nkv, L, 256))
    v = mx.random.normal((1, nkv, L, 256))
    c = AsymmetricQuantizedKVCache(64, 4, 3)
    kq, vq = c.update_and_fetch(k, v)
    out = asym_quantized_sdpa(q, kq, vq, scale=256 ** -0.5, mask=None,
                              group_size=64, k_bits=4, v_bits=3)
    rep = nq // nkv
    qr = q.reshape(1, nkv, rep, 1, 256)
    kr = k.reshape(1, nkv, 1, L, 256)
    vr = v.reshape(1, nkv, 1, L, 256)
    s = mx.softmax((qr * 256 ** -0.5) @ kr.transpose(0, 1, 2, 4, 3), axis=-1, precise=True)
    ref = (s @ vr).reshape(1, nq, 1, 256)
    assert _cos(out, ref) > 0.9
