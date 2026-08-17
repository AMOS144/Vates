"""The single production attention/GDN path for Expert-major prefill."""

import math

import mlx.core as mx
from mlx.utils import tree_map

from mlx_streaming.core.attention.prefill_scope import expert_major_prefill_active


# Winning real-model settings. They are constants on this clean branch so a
# stale shell environment cannot silently select a rejected implementation.
ATTENTION_TILE = 2048
ATTENTION_SCORE_BUDGET = 16_777_216
GDN_TILE = 512

_INSTALLED = False
_ORIGINAL_GATED_DELTA = None


def _causal_query_tile(start: int, length: int, cache_offset: int) -> int:
    """Largest tile whose visible causal score area stays within budget."""
    remaining = length - start
    maximum = min(remaining, ATTENTION_TILE)
    base = cache_offset + start
    bounded = (
        math.isqrt(base * base + 4 * ATTENTION_SCORE_BUDGET) - base
    ) // 2
    return max(1, min(maximum, bounded))


def _tiled_asymmetric_attention(
    queries, q_keys, q_values, *, scale, mask, group_size, k_bits, v_bits,
):
    """Memory-bounded exact-algebra K4/V3 attention."""
    B, n_q_heads, q_len, dim = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    repeats = n_q_heads // n_kv_heads
    k_len = q_keys[0].shape[-2]
    tile = min(
        q_len,
        ATTENTION_TILE,
        max(1, ATTENTION_SCORE_BUDGET // k_len),
    )
    expanded_keys = tree_map(lambda value: mx.expand_dims(value, -3), q_keys)
    expanded_values = tree_map(
        lambda value: mx.expand_dims(value, -3), q_values,
    )
    key_positions = mx.arange(k_len)[None, :]
    causal_offset = k_len - q_len
    outputs = []
    for start in range(0, q_len, tile):
        end = min(q_len, start + tile)
        q = queries[..., start:end, :] * scale
        if repeats > 1:
            q = q.reshape(B, n_kv_heads, repeats, end - start, dim)
            keys = expanded_keys
            values = expanded_values
        else:
            keys = q_keys
            values = q_values
        scores = mx.quantized_matmul(
            q, *keys, transpose=True,
            group_size=group_size, bits=k_bits,
        )
        tile_mask = mask
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"unsupported attention mask {mask!r}")
            query_positions = mx.arange(
                causal_offset + start, causal_offset + end,
            )[:, None]
            tile_mask = query_positions >= key_positions
        elif mask is not None and getattr(mask, "ndim", 0) >= 2:
            tile_mask = mask[..., start:end, :]
        if tile_mask is not None:
            if tile_mask.dtype == mx.bool_:
                scores = mx.where(
                    tile_mask, scores, mx.finfo(scores.dtype).min,
                )
            else:
                scores = scores + tile_mask
        probs = mx.softmax(scores, axis=-1, precise=True)
        out = mx.quantized_matmul(
            probs, *values, transpose=False,
            group_size=group_size, bits=v_bits,
        )
        if repeats > 1:
            out = out.reshape(B, n_q_heads, end - start, dim)
        mx.eval(out)
        leaf = mx.zeros(out.shape, dtype=out.dtype)
        leaf[:] = out
        mx.eval(leaf)
        outputs.append(mx.stop_gradient(leaf))
    return mx.concatenate(outputs, axis=-2)


def _install_rotated_quant_attention() -> None:
    """Patch only the production IsoQuant K4/V3 attention class."""
    from mlx_streaming.core.cache import kv_quant_patch

    rotated_cls = kv_quant_patch._RotatedQuantAttn
    if getattr(rotated_cls, "_expert_major_installed", False):
        return
    original = rotated_cls.__call__

    def bounded_rotated(self, x, mask=None, cache=None):
        if not expert_major_prefill_active() or cache is None:
            return original(self, x, mask=mask, cache=cache)
        from mlx_streaming.core.cache.quant_kv import rotate_last

        B, length, _ = x.shape
        cache_offset = int(cache.offset)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        keys = self.k_norm(
            keys.reshape(B, length, self.num_key_value_heads, -1),
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, length, self.num_key_value_heads, -1,
        ).transpose(0, 2, 1, 3)
        keys = self.rope(keys, offset=cache_offset)
        if self._kvq_Rk is not None:
            keys = rotate_last(keys, self._kvq_Rk)
            values = rotate_last(values, self._kvq_Rv)
        keys, values = cache.update_and_fetch(keys, values)

        key_length = int(keys[0].shape[-2])
        result = mx.zeros(x.shape, dtype=x.dtype)
        mx.eval(result)
        start = 0
        while start < length:
            tile = _causal_query_tile(start, length, cache_offset)
            end = min(length, start + tile)
            q_proj = self.q_proj(x[:, start:end, :])
            queries, gate = mx.split(
                q_proj.reshape(
                    B, end - start, self.num_attention_heads, -1,
                ),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, end - start, -1)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            queries = self.rope(queries, offset=cache_offset + start)
            if self._kvq_Rk is not None:
                queries = rotate_last(queries, self._kvq_Rk)

            visible_end = min(key_length, cache_offset + end)
            visible_keys = tree_map(
                lambda value: value[..., :visible_end, :], keys,
            )
            visible_values = tree_map(
                lambda value: value[..., :visible_end, :], values,
            )
            tile_mask = mask
            if isinstance(mask, str):
                query_positions = mx.arange(
                    cache_offset + start, cache_offset + end,
                )[:, None]
                tile_mask = query_positions >= mx.arange(visible_end)[None, :]
            elif mask is not None and getattr(mask, "ndim", 0) >= 2:
                tile_mask = mask[..., start:end, :visible_end]
            attended = _tiled_asymmetric_attention(
                queries, visible_keys, visible_values,
                scale=self.scale, mask=tile_mask,
                group_size=self._kvq_gs,
                k_bits=self._kvq_kb, v_bits=self._kvq_vb,
            )
            if self._kvq_Rv is not None:
                attended = rotate_last(attended, self._kvq_Rv.T)
            attended = attended.transpose(0, 2, 1, 3).reshape(
                B, end - start, -1,
            )
            result[:, start:end, :] = self.o_proj(
                attended * mx.sigmoid(gate),
            )
            mx.eval(result)
            result = mx.stop_gradient(result)
            mx.eval(result)
            start = end
        return result

    rotated_cls.__call__ = bounded_rotated
    rotated_cls._expert_major_installed = True


def install() -> None:
    """Install the fixed prefill attention and blocked-GDN implementation."""
    global _INSTALLED, _ORIGINAL_GATED_DELTA
    if _INSTALLED:
        _install_rotated_quant_attention()
        return
    import mlx_lm.models.qwen3_next as qwen3_next

    _ORIGINAL_GATED_DELTA = qwen3_next.Qwen3NextGatedDeltaNet.__call__

    def bounded_gated_delta(self, inputs, mask=None, cache=None):
        if not expert_major_prefill_active():
            return _ORIGINAL_GATED_DELTA(
                self, inputs, mask=mask, cache=cache,
            )
        if mask is not None:
            raise RuntimeError(
                "optimal Expert-major GDN expects the Qwen prefill mask to be None",
            )
        from mlx_streaming.core.attention.gdn_fused import fused_gdn_layer

        length = int(inputs.shape[1])
        result = mx.zeros(
            (int(inputs.shape[0]), length, int(inputs.shape[-1])),
            dtype=inputs.dtype,
        )
        mx.eval(result)
        for start in range(0, length, GDN_TILE):
            end = min(length, start + GDN_TILE)
            tile_output = fused_gdn_layer(
                self, inputs[:, start:end, :], mask=None, cache=cache,
            )
            result[:, start:end, :] = tile_output
            mx.eval(result)
            result = mx.stop_gradient(result)
            mx.eval(result)
        return result

    qwen3_next.Qwen3NextGatedDeltaNet.__call__ = bounded_gated_delta
    _install_rotated_quant_attention()
    _INSTALLED = True
