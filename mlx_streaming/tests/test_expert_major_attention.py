import mlx.core as mx
from mlx_streaming.core.attention.expert_major import (
    _tiled_asymmetric_attention,
)
from mlx_streaming.core.cache.quant_kv import asym_quantized_sdpa


def test_tiled_asymmetric_attention_matches_untiled():
    """Expert-major keeps the existing IsoQuant K4/V3 algebra exactly."""
    mx.random.seed(6)
    queries = mx.random.normal((1, 4, 9, 64)).astype(mx.float16)
    keys = mx.random.normal((1, 2, 9, 64)).astype(mx.float16)
    values = mx.random.normal((1, 2, 9, 64)).astype(mx.float16)
    q_keys = mx.quantize(keys, group_size=64, bits=4)
    q_values = mx.quantize(values, group_size=64, bits=3)

    expected = asym_quantized_sdpa(
        queries * 1, q_keys, q_values, scale=64**-0.5, mask="causal",
        group_size=64, k_bits=4, v_bits=3,
    )
    actual = _tiled_asymmetric_attention(
        queries * 1, q_keys, q_values, scale=64**-0.5, mask="causal",
        group_size=64, k_bits=4, v_bits=3,
    )
    mx.eval(expected, actual)
    assert bool(mx.allclose(actual, expected, rtol=2e-3, atol=2e-3))


def test_causal_visible_key_pruning_with_prefix_is_exact():
    """Dropping keys beyond a tile's last query preserves prefix attention."""
    mx.random.seed(16)
    prefix, q_len, dim = 5, 9, 64
    queries = mx.random.normal((1, 4, q_len, dim)).astype(mx.float16)
    keys = mx.random.normal((1, 2, prefix + q_len, dim)).astype(mx.float16)
    values = mx.random.normal((1, 2, prefix + q_len, dim)).astype(mx.float16)
    q_keys = mx.quantize(keys, group_size=64, bits=4)
    q_values = mx.quantize(values, group_size=64, bits=3)
    expected = asym_quantized_sdpa(
        queries, q_keys, q_values, scale=dim**-0.5, mask="causal",
        group_size=64, k_bits=4, v_bits=3,
    )

    parts = []
    for start in range(0, q_len, 3):
        end = min(q_len, start + 3)
        visible = prefix + end
        tile_keys = tuple(value[..., :visible, :] for value in q_keys)
        tile_values = tuple(value[..., :visible, :] for value in q_values)
        positions = mx.arange(prefix + start, prefix + end)[:, None]
        tile_mask = positions >= mx.arange(visible)[None, :]
        parts.append(_tiled_asymmetric_attention(
            queries[..., start:end, :], tile_keys, tile_values,
            scale=dim**-0.5, mask=tile_mask, group_size=64,
            k_bits=4, v_bits=3,
        ))
    actual = mx.concatenate(parts, axis=-2)
    mx.eval(expected, actual)
    assert bool(mx.allclose(actual, expected, rtol=2e-3, atol=2e-3))
