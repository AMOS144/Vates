"""逐步状态输出版 gated-delta Metal kernel（vendored 自 mlx_lm.models.gated_delta）。

来源：mlx-lm 0.x `mlx_lm/models/gated_delta.py` 的 `_make_gated_delta_kernel`
（本机 mlx 0.31.2）。本文件**只**在上游 kernel 基础上多加一个输出 `states_out`：
在时间维 `for (t=0..T-1)` 的串行递归里，每算完一步就把当前 state 写回
`states_out[B, T, Hv, Dv, Dk]`。

为什么 bit-exact：上游 kernel 的时间维本就是线程内串行递归（不是分块并行），第 t 步
寄存器里的 `state` 就是「逐 token 走 t+1 步后的状态」，全程 fp32、同序、同融合。因此
一次 seq=T 调用产出的 `states_out[:, t]`，与逐 token（T 次 seq=1）调用链得到的 state
逐 bit 相等。这正是 MTP 投机验证后能「按接受长度直接提交、零 replay」的数值根基。

用途：MTP 验证前向里捕获每个 token 后的 ssm 递归态作为 checkpoint，替换原先走
`_gated_delta_step_ops`（与 baseline kernel 不 bit-exact）的捕获路径。

非目标：仅 Metal(GPU) 路径；CPU/ops 回退不在此实现（部署恒为 Metal）。
"""
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# 与上游一致：g = exp(-exp(A_log) * softplus(a + dt_bias))，fp32。
from mlx_lm.models.gated_delta import compute_g


def _make_gated_delta_multistate_kernel(has_mask=False, vectorized=False):
    """构造逐步状态输出版 kernel；除多写 states_out 外与上游 kernel 完全一致。"""
    if not mx.metal.is_available():
        return None
    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    # g 索引方式：标量门控 [B,T,Hv] vs 向量门控 [B,T,Hv,Dk]，与上游对齐。
    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    # 与上游唯一的差异：时间循环每步末把 state 写入 states_out（见下方标注的新增块）。
    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }} else {{
            y[dv_idx] = static_cast<InT>(0);
          }}

          // ===== 新增：每步把当前 state 写回 states_out[B, T, Hv, Dv, Dk] =====
          // masked 步 state 不变（上面 else 分支不更新 state），写回的就是上一步的状态，
          // 与逐 token 单步链语义一致。
          auto ms_ = states_out
              + ((((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk);
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            ms_[s_idx] = static_cast<StT>(state[i]);
          }}
          // ===================================================================

          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }}
    """
    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_step_multistate{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "states_out"],
        source=source,
    )


_kernel = _make_gated_delta_multistate_kernel(has_mask=False, vectorized=False)
_kernel_masked = _make_gated_delta_multistate_kernel(has_mask=True, vectorized=False)
_kernel_vec = _make_gated_delta_multistate_kernel(has_mask=False, vectorized=True)
_kernel_vec_masked = _make_gated_delta_multistate_kernel(has_mask=True, vectorized=True)


def gated_delta_multistate_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """逐步状态输出版 kernel 调用，返回 (y, final_state, states_out)。

    形状（与上游 `gated_delta_kernel` 一致，不在 Python 侧做 GQA repeat）：
      - q, k: [B, T, Hk, Dk]
      - v:    [B, T, Hv, Dv]
      - g:    [B, T, Hv]（标量）或 [B, T, Hv, Dk]（向量）
      - beta: [B, T, Hv]
      - state:[B, Hv, Dv, Dk]
    返回：
      - y:          [B, T, Hv, Dv]
      - final_state:[B, Hv, Dv, Dk]（== states_out[:, -1]）
      - states_out: [B, T, Hv, Dv, Dk]（每个 token 处理完后的递归态）
    """
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    input_type = q.dtype
    state_type = state.dtype
    if g.ndim == 4:
        kernel = _kernel_vec_masked if mask is not None else _kernel_vec
    else:
        kernel = _kernel_masked if mask is not None else _kernel

    inputs = [q, k, v, g, beta, state, T]
    if mask is not None:
        inputs.append(mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("StT", state_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type, state_type],
    )


def gated_delta_update_multistate(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """对齐上游 `gated_delta_update` 的签名，多返回逐步状态 states_out。

    内部 beta/g 计算与上游完全一致（sigmoid(b) / compute_g），保证数值路径相同。
    仅走 Metal kernel 路径（multistate 仅 GPU 实现）。
    """
    beta = mx.sigmoid(b)
    g = compute_g(A_log, a, dt_bias)
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return gated_delta_multistate_kernel(q, k, v, g, beta, state, mask)
