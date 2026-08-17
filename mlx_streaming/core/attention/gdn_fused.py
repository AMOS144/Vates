# SPDX-License-Identifier: Apache-2.0
"""Blocked Gated-DeltaNet scan for long Expert-major prefill.

The recurrence is exact. Each threadgroup owns 32 value rows, stages a time
block of q/k/v once, and keeps its state fragment in registers.
"""

import mlx.core as mx
import mlx.nn as nn


_KERNELS = {}
_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"
_SOURCE = r"""
    constexpr int TB = BLOCK_T;
    constexpr int DB = 32;
    const int tid = thread_position_in_threadgroup.x;
    const int blk = threadgroup_position_in_grid.x;
    const int hv = threadgroup_position_in_grid.y;
    const int batch = threadgroup_position_in_grid.z;
    const int hk = hv / (Hv / Hk);
    const int dv0 = blk * DB;
    const int dv = tid / 8;
    const int seg = tid % 8;
    const int d0 = seg * 16;

    threadgroup InT ks[TB][Dk + 8];
    threadgroup InT qs[TB][Dk + 8];
    threadgroup InT vs[TB][DB + 8];
    threadgroup float gs[TB];
    threadgroup float bs[TB];
    const device InT* kb = k + ((size_t)batch * T * Hk + hk) * Dk;
    const device InT* qb = q + ((size_t)batch * T * Hk + hk) * Dk;
    const device InT* vb = v + ((size_t)batch * T * Hv + hv) * Dv + dv0;
    const size_t krow = (size_t)Hk * Dk;
    float4 state[4];
    const device float4* sin = (const device float4*)(
        state_in + (((size_t)batch * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
    for (int i = 0; i < 4; ++i) state[i] = sin[i];
    device InT* yb = y + ((size_t)batch * T * Hv + hv) * Dv + dv0;

    for (int t0 = 0; t0 < T; t0 += TB) {
      const int count = min(TB, T - t0);
      for (int p = tid; p < count * Dk; p += 256) {
        const int row = p / Dk, d = p % Dk;
        ks[row][d] = kb[(size_t)(t0 + row) * krow + d];
        qs[row][d] = qb[(size_t)(t0 + row) * krow + d];
      }
      for (int p = tid; p < count * DB; p += 256) {
        const int row = p / DB, d = p % DB;
        vs[row][d] = vb[(size_t)(t0 + row) * Hv * Dv + d];
      }
      for (int p = tid; p < count; p += 256) {
        gs[p] = g[((size_t)batch * T + t0 + p) * Hv + hv];
        bs[p] = beta[((size_t)batch * T + t0 + p) * Hv + hv];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int t = 0; t < count; ++t) {
        const threadgroup vec<InT,4>* k4 =
            (const threadgroup vec<InT,4>*)&ks[t][d0];
        const threadgroup vec<InT,4>* q4 =
            (const threadgroup vec<InT,4>*)&qs[t][d0];
        float4 kf[4];
        float4 product = 0.0f;
        for (int i = 0; i < 4; ++i) {
          kf[i] = float4(k4[i]);
          state[i] *= gs[t];
          product += state[i] * kf[i];
        }
        float mem = product.x + product.y + product.z + product.w;
        mem += simd_shuffle_down(mem, 4);
        mem += simd_shuffle_down(mem, 2);
        mem += simd_shuffle_down(mem, 1);
        mem = simd_shuffle(mem, (tid % 32) / 8 * 8);
        const float delta = (float(vs[t][dv]) - mem) * bs[t];
        float4 result = 0.0f;
        for (int i = 0; i < 4; ++i) {
          state[i] += kf[i] * delta;
          result += state[i] * float4(q4[i]);
        }
        float out = result.x + result.y + result.z + result.w;
        out += simd_shuffle_down(out, 4);
        out += simd_shuffle_down(out, 2);
        out += simd_shuffle_down(out, 1);
        if (seg == 0) yb[(size_t)(t0 + t) * Hv * Dv + dv] = InT(out);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4* sout = (device float4*)(
        state_out + (((size_t)batch * Hv + hv) * Dv + dv0 + dv) * Dk + d0);
    for (int i = 0; i < 4; ++i) sout[i] = state[i];
"""


def gated_delta_blocked(q, k, v, g, beta, state=None, block_t=32):
    B, T, Hk, Dk = (int(vv) for vv in q.shape)
    Hv, Dv = (int(vv) for vv in v.shape[-2:])
    if Dk != 128 or Dv % 32:
        raise ValueError("blocked GDN requires Dk=128 and Dv divisible by 32")
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    block_t = int(block_t)
    key = (block_t, q.dtype)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"vates_gdn_blocked_t{block_t}",
            input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"],
            source=_SOURCE.replace("BLOCK_T", str(block_t)), header=_HEADER,
        )
        _KERNELS[key] = kernel
    return kernel(
        inputs=[q, k, v, g.astype(mx.float32), beta.astype(mx.float32), state, T],
        template=[("InT", q.dtype), ("Dk", Dk), ("Dv", Dv),
                  ("Hk", Hk), ("Hv", Hv)],
        grid=(256 * (Dv // 32), Hv, B), threadgroup=(256, 1, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, mx.float32],
    )


def fused_gdn_layer(module, inputs, mask=None, cache=None):
    """Run Qwen3-Next projections/conv around the blocked fused scan."""
    from mlx_lm.models.gated_delta import compute_g
    object.__setattr__(
        module, "_expert_major_fused_gdn_calls",
        int(getattr(module, "_expert_major_fused_gdn_calls", 0)) + 1,
    )

    B, S, _ = inputs.shape
    q, k, v, z, b, a = module.fix_query_key_value_ordering(
        module.in_proj_qkvz(inputs), module.in_proj_ba(inputs),
    )
    conv_state = (
        cache[0] if cache is not None and cache[0] is not None
        else mx.zeros(
            (B, module.conv_kernel_size - 1, module.conv_dim),
            dtype=inputs.dtype,
        )
    )
    mixed = mx.concatenate(
        [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)],
        axis=-1,
    )
    if mask is not None:
        mixed = mx.where(mask[..., None], mixed, 0)
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    if cache is not None:
        keep = module.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S)
            positions = (ends[:, None] + mx.arange(keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -keep:, :])
    conv_out = nn.silu(module.conv1d(conv_input))
    q, k, v = [
        value.reshape(B, S, heads, dim)
        for value, heads, dim in zip(
            mx.split(conv_out, [module.key_dim, 2 * module.key_dim], -1),
            [module.num_k_heads, module.num_k_heads, module.num_v_heads],
            [module.head_k_dim, module.head_k_dim, module.head_v_dim],
        )
    ]
    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    beta = mx.sigmoid(b)
    g = compute_g(module.A_log, a, module.dt_bias)
    state = cache[1] if cache else None
    out, state = gated_delta_blocked(q, k, v, g, beta, state, block_t=32)
    if cache is not None:
        cache[1] = state
        cache.advance(S)
    out = module.norm(out, z)
    return module.out_proj(out.reshape(B, S, -1))
