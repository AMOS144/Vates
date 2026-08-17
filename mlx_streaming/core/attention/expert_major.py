"""Memory-bounded quantized attention used only by long Expert-major prefill."""

import math

import mlx.core as mx
from mlx.utils import tree_map

from mlx_streaming import config
from mlx_streaming.core.attention.prefill_scope import expert_major_prefill_active


_INSTALLED = False
_ORIGINAL = None
_ORIGINAL_QWEN_ATTENTION = None
_ORIGINAL_GATED_DELTA = None


def _causal_query_tile(start: int, length: int, cache_offset: int) -> int:
    """Largest tile whose visible causal score area stays within budget."""
    remaining = length - start
    maximum = min(remaining, config.expert_major_attention_tile())
    base = cache_offset + start
    budget = config.expert_major_attention_score_budget()
    # tile * (base + tile) <= budget
    bounded = (math.isqrt(base * base + 4 * budget) - base) // 2
    return max(1, min(maximum, bounded))


def _tiled_asymmetric_attention(
    queries, q_keys, q_values, *, scale, mask, group_size, k_bits, v_bits,
):
    """Memory-bounded K/V-asymmetric quantized attention (K4/V3)."""
    B, n_q_heads, q_len, dim = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    repeats = n_q_heads // n_kv_heads
    k_len = q_keys[0].shape[-2]
    tile = min(
        q_len,
        config.expert_major_attention_tile(),
        max(1, config.expert_major_attention_score_budget() // k_len),
    )
    expanded_keys = tree_map(lambda value: mx.expand_dims(value, -3), q_keys)
    expanded_values = tree_map(lambda value: mx.expand_dims(value, -3), q_values)
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
        # ``stop_gradient`` alone can leave the evaluated command graph and
        # its score/probability temporaries reachable.  Copy into a fresh
        # buffer before retaining this tile for concatenation.
        leaf = mx.zeros(out.shape, dtype=out.dtype)
        leaf[:] = out
        mx.eval(leaf)
        outputs.append(mx.stop_gradient(leaf))
    return mx.concatenate(outputs, axis=-2)


def _streaming_asymmetric_attention(
    queries, q_keys, q_values, *, scale, mask, group_size, k_bits, v_bits,
    query_tile=512, key_tile=2048,
):
    """Exact block-online K4/V3 attention without a QxK materialization.

    This is the executable reference for the native fused kernel. Peak score
    storage is bounded by ``query_tile * key_tile`` regardless of context.
    """
    B, n_q_heads, q_len, dim = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    repeats = n_q_heads // n_kv_heads
    k_len = q_keys[0].shape[-2]
    query_tile = max(1, min(int(query_tile), q_len))
    key_tile = max(1, min(int(key_tile), k_len))
    key_positions = mx.arange(k_len)[None, :]
    causal_offset = k_len - q_len
    outputs = []

    for q_start in range(0, q_len, query_tile):
        q_end = min(q_len, q_start + query_tile)
        q = queries[..., q_start:q_end, :] * scale
        if repeats > 1:
            q = q.reshape(B, n_kv_heads, repeats, q_end - q_start, dim)
        row_max = mx.full((*q.shape[:-1], 1), -mx.inf, dtype=mx.float32)
        row_sum = mx.zeros(row_max.shape, dtype=mx.float32)
        accumulator = mx.zeros((*q.shape[:-1], dim), dtype=mx.float32)

        visible_end = k_len
        if isinstance(mask, str):
            if mask != "causal":
                raise ValueError(f"unsupported attention mask {mask!r}")
            visible_end = min(k_len, causal_offset + q_end)

        for k_start in range(0, visible_end, key_tile):
            k_end = min(visible_end, k_start + key_tile)
            keys = tree_map(
                lambda value: value[..., k_start:k_end, :], q_keys,
            )
            values = tree_map(
                lambda value: value[..., k_start:k_end, :], q_values,
            )
            if repeats > 1:
                keys = tree_map(lambda value: mx.expand_dims(value, -3), keys)
                values = tree_map(lambda value: mx.expand_dims(value, -3), values)
            scores = mx.quantized_matmul(
                q, *keys, transpose=True,
                group_size=group_size, bits=k_bits,
            ).astype(mx.float32)

            block_mask = mask
            if isinstance(mask, str):
                query_positions = mx.arange(
                    causal_offset + q_start, causal_offset + q_end,
                )[:, None]
                block_mask = query_positions >= key_positions[:, k_start:k_end]
            elif mask is not None and getattr(mask, "ndim", 0) >= 2:
                block_mask = mask[..., q_start:q_end, k_start:k_end]
            if block_mask is not None:
                if block_mask.dtype == mx.bool_:
                    scores = mx.where(block_mask, scores, -mx.inf)
                else:
                    scores = scores + block_mask

            block_max = mx.max(scores, axis=-1, keepdims=True)
            new_max = mx.maximum(row_max, block_max)
            old_factor = mx.exp(row_max - new_max)
            block_exp = mx.exp(scores - new_max)
            block_exp = mx.where(mx.isfinite(scores), block_exp, 0.0)
            weighted = mx.quantized_matmul(
                block_exp.astype(queries.dtype), *values, transpose=False,
                group_size=group_size, bits=v_bits,
            ).astype(mx.float32)
            accumulator = accumulator * old_factor + weighted
            row_sum = row_sum * old_factor + mx.sum(
                block_exp, axis=-1, keepdims=True,
            )
            row_max = new_max

        out = (accumulator / row_sum).astype(queries.dtype)
        if repeats > 1:
            out = out.reshape(B, n_q_heads, q_end - q_start, dim)
        mx.eval(out)
        outputs.append(mx.stop_gradient(out))
    return mx.concatenate(outputs, axis=-2)


def _fused_asymmetric_attention(
    queries, q_keys, q_values, *, scale, mask, group_size, k_bits, v_bits,
):
    """Dispatch the packed-cache Steel kernel, with a strict ABI guard."""
    if (mask != "causal" or group_size != 64 or k_bits != 4 or v_bits != 3
            or int(queries.shape[-1]) != 256):
        raise ValueError("fused attention requires causal head256 K4/V3 group64")
    from mlx_streaming import native_moe_ext
    return native_moe_ext.k4v3_fused_causal_attention(
        queries, *q_keys, *q_values, float(scale), q_block=32, k_block=8,
    )


def _install_rotated_quant_attention() -> None:
    """Patch the IsoQuant subclass, which overrides the base Qwen method."""
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
        dense_keys = dense_values = None
        configured_fused_layers = config.expert_major_fused_attention_layers()
        dense_enabled = (
            config.expert_major_dense_steel_attention()
            and (
                configured_fused_layers is None
                or int(getattr(self, "_kvq_layer_idx", -1))
                in configured_fused_layers
            )
        )
        if dense_enabled:
            dense_keys = mx.dequantize(
                *keys, group_size=self._kvq_gs, bits=self._kvq_kb,
            )
            dense_values = mx.dequantize(
                *values, group_size=self._kvq_gs, bits=self._kvq_vb,
            )
            mx.eval(dense_keys, dense_values)
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
            # Exact causal pruning: this query tile can never observe keys
            # after its final absolute position.  Slicing the quantized tuples
            # before QK avoids computing the masked upper triangle instead of
            # paying for it and discarding it afterward.  Prefix-cache keys
            # remain visible because ``cache_offset`` is included.
            visible_end = min(key_length, cache_offset + end)
            visible_keys = tree_map(
                lambda value: value[..., :visible_end, :], keys,
            )
            visible_values = tree_map(
                lambda value: value[..., :visible_end, :], values,
            )
            key_positions = mx.arange(visible_end)[None, :]
            tile_mask = mask
            if isinstance(mask, str):
                query_positions = mx.arange(
                    cache_offset + start, cache_offset + end,
                )[:, None]
                tile_mask = query_positions >= key_positions
            elif mask is not None and getattr(mask, "ndim", 0) >= 2:
                tile_mask = mask[..., start:end, :visible_end]
            fused_layers = config.expert_major_fused_attention_layers()
            use_fused = (
                config.expert_major_fused_attention()
                and isinstance(mask, str)
                and (
                    fused_layers is None
                    or int(getattr(self, "_kvq_layer_idx", -1)) in fused_layers
                )
            )
            if dense_keys is not None and isinstance(mask, str):
                from mlx_streaming import native_moe_ext
                attended = native_moe_ext.dense_fused_causal_attention(
                    queries,
                    dense_keys[..., :visible_end, :],
                    dense_values[..., :visible_end, :],
                    float(self.scale),
                )
            elif use_fused:
                attended = _fused_asymmetric_attention(
                    queries, visible_keys, visible_values,
                    scale=self.scale, mask="causal",
                    group_size=self._kvq_gs,
                    k_bits=self._kvq_kb, v_bits=self._kvq_vb,
                )
            else:
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
            tile_output = self.o_proj(attended * mx.sigmoid(gate))
            result[:, start:end, :] = tile_output
            mx.eval(result)
            result = mx.stop_gradient(result)
            mx.eval(result)
            start = end
        return result

    rotated_cls.__call__ = bounded_rotated
    rotated_cls._expert_major_installed = True


def _tiled_quantized_attention(
    queries, q_keys, q_values, *, scale, mask, group_size, bits,
):
    """Exact query tiling for the mlx-lm quantized SDPA fallback.

    mlx-lm's quantized path materialises [B,KVH,repeats,Q,K] scores.  At 256K
    that is impossible even though the KV cache itself is compact.  Tiling Q
    preserves every dot product and softmax row while bounding score storage by
    ``tile*K``.  The causal offset accounts for an existing cache prefix.
    """
    B, n_q_heads, q_len, dim = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    repeats = n_q_heads // n_kv_heads
    k_len = q_keys[0].shape[-2]
    tile = min(
        q_len,
        config.expert_major_attention_tile(),
        max(1, config.expert_major_attention_score_budget() // k_len),
    )
    expanded_keys = tree_map(lambda value: mx.expand_dims(value, -3), q_keys)
    expanded_values = tree_map(lambda value: mx.expand_dims(value, -3), q_values)
    key_positions = mx.arange(k_len)[None, :]
    causal_offset = k_len - q_len
    outputs = []

    for start in range(0, q_len, tile):
        end = min(q_len, start + tile)
        q = queries[:, :, start:end, :] * scale
        if repeats > 1:
            q = q.reshape(B, n_kv_heads, repeats, end - start, dim)
            keys = expanded_keys
            values = expanded_values
        else:
            keys = q_keys
            values = q_values
        scores = mx.quantized_matmul(
            q, *keys, transpose=True, group_size=group_size, bits=bits,
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
            group_size=group_size, bits=bits,
        )
        if repeats > 1:
            out = out.reshape(B, n_q_heads, end - start, dim)
        mx.eval(out)
        leaf = mx.zeros(out.shape, dtype=out.dtype)
        leaf[:] = out
        mx.eval(leaf)
        outputs.append(mx.stop_gradient(leaf))
    return mx.concatenate(outputs, axis=-2)


def _tiled_fast_attention(
    queries, keys, values, *, scale, mask,
):
    """Query-tiled wrapper around MLX fast SDPA for an ordinary KV cache."""
    q_len = int(queries.shape[-2])
    k_len = int(keys.shape[-2])
    tile = min(q_len, config.expert_major_attention_tile())
    key_positions = mx.arange(k_len)[None, :]
    causal_offset = k_len - q_len
    B, n_q_heads, _, dim = queries.shape
    n_kv_heads = int(keys.shape[-3])
    repeats = n_q_heads // n_kv_heads
    expanded_keys = mx.expand_dims(keys, -3) if repeats > 1 else keys
    expanded_values = mx.expand_dims(values, -3) if repeats > 1 else values
    outputs = []
    for start in range(0, q_len, tile):
        end = min(q_len, start + tile)
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
        q = queries[..., start:end, :] * scale
        if repeats > 1:
            q = q.reshape(B, n_kv_heads, repeats, end - start, dim)
        scores = mx.matmul(q, mx.swapaxes(expanded_keys, -1, -2))
        if tile_mask is not None:
            if tile_mask.dtype == mx.bool_:
                scores = mx.where(
                    tile_mask, scores, mx.finfo(scores.dtype).min,
                )
            else:
                scores = scores + tile_mask
        probs = mx.softmax(scores, axis=-1, precise=True)
        out = mx.matmul(probs, expanded_values)
        if repeats > 1:
            out = out.reshape(B, n_q_heads, end - start, dim)
        mx.eval(out)
        leaf = mx.zeros(out.shape, dtype=out.dtype)
        leaf[:] = out
        mx.eval(leaf)
        outputs.append(mx.stop_gradient(leaf))
    return mx.concatenate(outputs, axis=-2)


def install() -> None:
    """Patch Qwen3-Next once; calls outside prefill scope fall through."""
    global _INSTALLED, _ORIGINAL, _ORIGINAL_QWEN_ATTENTION, _ORIGINAL_GATED_DELTA
    if _INSTALLED:
        _install_rotated_quant_attention()
        return
    import mlx_lm.models.qwen3_next as qwen3_next

    _ORIGINAL = qwen3_next.scaled_dot_product_attention
    _ORIGINAL_QWEN_ATTENTION = qwen3_next.Qwen3NextAttention.__call__
    _ORIGINAL_GATED_DELTA = qwen3_next.Qwen3NextGatedDeltaNet.__call__

    def bounded_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        if expert_major_prefill_active() and sinks is None:
            if hasattr(cache, "bits"):
                return _tiled_quantized_attention(
                    queries, keys, values, scale=scale, mask=mask,
                    group_size=cache.group_size, bits=cache.bits,
                )
            return _tiled_fast_attention(
                queries, keys, values, scale=scale, mask=mask,
            )
        return _ORIGINAL(
            queries, keys, values, cache=cache, scale=scale,
            mask=mask, sinks=sinks,
        )

    qwen3_next.scaled_dot_product_attention = bounded_sdpa

    def bounded_qwen_attention(self, x, mask=None, cache=None):
        if not expert_major_prefill_active():
            return _ORIGINAL_QWEN_ATTENTION(self, x, mask=mask, cache=cache)

        B, length, _ = x.shape
        cache_offset = int(cache.offset) if cache is not None else 0
        # K/V are the actual persistent cache payload.  Q and gate are much
        # wider and purely temporary, so only those are tiled below.
        keys = self.k_proj(x)
        values = self.v_proj(x)
        keys = self.k_norm(
            keys.reshape(B, length, self.num_key_value_heads, -1),
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, length, self.num_key_value_heads, -1,
        ).transpose(0, 2, 1, 3)
        if cache is not None:
            keys = self.rope(keys, offset=cache_offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            keys = self.rope(keys)

        key_length = int(keys.shape[-2])
        key_positions = mx.arange(key_length)[None, :]
        tile = min(
            length,
            config.expert_major_attention_tile(),
            max(
                1,
                config.expert_major_attention_score_budget() // key_length,
            ),
        )
        result = mx.zeros(x.shape, dtype=x.dtype)
        mx.eval(result)
        for start in range(0, length, tile):
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

            tile_mask = mask
            if isinstance(mask, str):
                if mask != "causal":
                    raise ValueError(f"unsupported attention mask {mask!r}")
                query_positions = mx.arange(
                    cache_offset + start, cache_offset + end,
                )[:, None]
                tile_mask = query_positions >= key_positions
            elif mask is not None and getattr(mask, "ndim", 0) >= 2:
                tile_mask = mask[..., start:end, :]
            if hasattr(cache, "bits"):
                attended = _tiled_quantized_attention(
                    queries, keys, values, scale=self.scale, mask=tile_mask,
                    group_size=cache.group_size, bits=cache.bits,
                )
            else:
                attended = _tiled_fast_attention(
                    queries, keys, values, scale=self.scale, mask=tile_mask,
                )
            attended = attended.transpose(0, 2, 1, 3).reshape(
                B, end - start, -1,
            )
            tile_output = self.o_proj(attended * mx.sigmoid(gate))
            result[:, start:end, :] = tile_output
            mx.eval(result)
            result = mx.stop_gradient(result)
            mx.eval(result)
        return result

    qwen3_next.Qwen3NextAttention.__call__ = bounded_qwen_attention

    def bounded_gated_delta(self, inputs, mask=None, cache=None):
        length = int(inputs.shape[1])
        if not expert_major_prefill_active():
            return _ORIGINAL_GATED_DELTA(
                self, inputs, mask=mask, cache=cache,
            )
        tile = min(length, config.expert_major_gdn_tile())
        result = mx.zeros(
            (int(inputs.shape[0]), length, int(inputs.shape[-1])),
            dtype=inputs.dtype,
        )
        mx.eval(result)
        for start in range(0, length, tile):
            end = min(length, start + tile)
            tile_mask = mask
            if mask is not None and getattr(mask, "ndim", 0) >= 2:
                tile_mask = mask[..., start:end]
            if config.expert_major_fused_gdn() and tile_mask is None:
                from mlx_streaming.core.attention.gdn_fused import fused_gdn_layer
                tile_output = fused_gdn_layer(
                    self, inputs[:, start:end, :], mask=None, cache=cache,
                )
            else:
                tile_output = _ORIGINAL_GATED_DELTA(
                    self, inputs[:, start:end, :],
                    mask=tile_mask, cache=cache,
                )
            result[:, start:end, :] = tile_output
            mx.eval(result)
            result = mx.stop_gradient(result)
            mx.eval(result)
        return result

    qwen3_next.Qwen3NextGatedDeltaNet.__call__ = bounded_gated_delta
    _install_rotated_quant_attention()
    _INSTALLED = True
