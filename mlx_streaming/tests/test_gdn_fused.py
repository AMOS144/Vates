import mlx.core as mx

from mlx_lm.models.gated_delta import gated_delta_kernel
from mlx_streaming.core.attention.gdn_fused import gated_delta_blocked


def test_blocked_gdn_matches_stock_kernel():
    mx.random.seed(41)
    B, T, Hk, Hv, D = 1, 17, 2, 4, 128
    q = mx.random.normal((B, T, Hk, D)).astype(mx.float16) * 0.05
    k = mx.random.normal((B, T, Hk, D)).astype(mx.float16) * 0.05
    v = mx.random.normal((B, T, Hv, D)).astype(mx.float16) * 0.05
    g = mx.full((B, T, Hv), 0.98, dtype=mx.float32)
    beta = mx.full((B, T, Hv), 0.4, dtype=mx.float32)
    state = mx.zeros((B, Hv, D, D), dtype=mx.float32)
    expected, expected_state = gated_delta_kernel(q, k, v, g, beta, state)
    actual, actual_state = gated_delta_blocked(
        q, k, v, g, beta, state, block_t=16,
    )
    mx.eval(expected, expected_state, actual, actual_state)
    assert bool(mx.allclose(actual, expected, rtol=2e-3, atol=2e-3))
    assert bool(mx.allclose(actual_state, expected_state, rtol=2e-3, atol=2e-3))
