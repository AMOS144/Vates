import mlx.core as mx
import mlx.nn as nn

from mlx_streaming.core.moe.custom_kernel import _expert_major_fused_moe


def test_expert_major_gemm_matches_affine_reference():
    mx.random.seed(31)
    experts, hidden, inter = 3, 64, 64
    counts = [3, 1, 4]
    offsets = mx.array([0, 3, 4, 8], dtype=mx.uint32)
    x = mx.random.normal((sum(counts), hidden)).astype(mx.float32)
    gate = mx.random.normal((experts, inter, hidden)).astype(mx.float16)
    up = mx.random.normal((experts, inter, hidden)).astype(mx.float16)
    down = mx.random.normal((experts, hidden, inter)).astype(mx.float16)
    qg, qu, qd = (
        mx.quantize(weight, group_size=64, bits=4)
        for weight in (gate, up, down)
    )
    actual = _expert_major_fused_moe(
        x, offsets, *qg, *qu, *qd, hidden, inter, 64, 4,
        shards_per_expert=2,
    )
    dg, du, dd = (
        mx.dequantize(*weight, group_size=64, bits=4).astype(mx.float32)
        for weight in (qg, qu, qd)
    )
    parts = []
    start = 0
    for expert, count in enumerate(counts):
        xe = x[start:start + count]
        g = xe @ dg[expert].T
        u = xe @ du[expert].T
        parts.append((nn.silu(g) * u) @ dd[expert].T)
        start += count
    expected = mx.concatenate(parts, axis=0)
    mx.eval(actual, expected)
    cosine = mx.sum(actual * expected) / (
        mx.linalg.norm(actual) * mx.linalg.norm(expected)
    )
    assert float(cosine.item()) > 0.999999
    assert float(mx.max(mx.abs(actual - expected)).item()) < 2.0
