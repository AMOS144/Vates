"""multistate kernel 的 bit-exact 守门测试（方案 3 的 first gate）。

核心断言：multistate kernel 一次 seq=T 调用产出的 `states_out[:, t]`，与逐 token
（T 次 seq=1）调用上游 `gated_delta_kernel` 得到的递归态**逐 bit 相等**；并且
`states_out[:, -1]` 与上游一次性调用的 final state 相等。这是 MTP 直提交（零 replay）
正确性的根基——过不了则方案不成立。
"""
import mlx.core as mx
import pytest

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="multistate kernel 仅 Metal 路径"
)


def _rand_inputs(B=1, T=3, Hk=2, Hv=4, Dk=128, Dv=128, seed=0):
    """造与 Qwen3-Next 同形状量级的随机输入（Dk=Dv=128，n_per_t=4）。"""
    mx.random.seed(seed)
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    v = mx.random.normal((B, T, Hv, Dv))
    g = mx.random.uniform(low=0.5, high=1.0, shape=(B, T, Hv))   # 标量门控
    beta = mx.random.uniform(low=0.0, high=1.0, shape=(B, T, Hv))
    state = mx.random.normal((B, Hv, Dv, Dk)).astype(mx.float32)
    mx.eval(q, k, v, g, beta, state)
    return q, k, v, g, beta, state


def test_multistate_matches_stepwise_kernel_bit_exact():
    """states_out[:, t] 必须与「逐 token 调用上游 kernel」得到的 state 逐 bit 相等。"""
    from mlx_lm.models.gated_delta import gated_delta_kernel
    from mlx_streaming.core.linear_attn.gated_delta_multistate import (
        gated_delta_multistate_kernel,
    )

    B, T, Hk, Hv, Dk, Dv = 1, 4, 2, 4, 128, 128
    q, k, v, g, beta, state0 = _rand_inputs(B, T, Hk, Hv, Dk, Dv, seed=3)

    # 一次 multistate 调用
    _, final_ms, states_out = gated_delta_multistate_kernel(q, k, v, g, beta, state0)
    mx.eval(final_ms, states_out)

    # 逐 token 单步链：每步 seq=1 调上游 kernel，携带 state
    step_state = state0
    for t in range(T):
        _, step_state = gated_delta_kernel(
            q[:, t : t + 1], k[:, t : t + 1], v[:, t : t + 1],
            g[:, t : t + 1], beta[:, t : t + 1], step_state,
        )
        mx.eval(step_state)
        assert mx.array_equal(states_out[:, t], step_state), (
            f"第 {t} 步 state 与单步链不 bit-exact"
        )

    # 末步 state 应等于 multistate 的 final_state
    assert mx.array_equal(states_out[:, -1], final_ms)


def test_multistate_final_matches_upstream_oneshot():
    """final_state 应与上游一次性 seq=T 调用的 final state 逐 bit 相等。"""
    from mlx_lm.models.gated_delta import gated_delta_kernel
    from mlx_streaming.core.linear_attn.gated_delta_multistate import (
        gated_delta_multistate_kernel,
    )

    q, k, v, g, beta, state0 = _rand_inputs(T=3, seed=7)
    y_up, final_up = gated_delta_kernel(q, k, v, g, beta, state0)
    y_ms, final_ms, _ = gated_delta_multistate_kernel(q, k, v, g, beta, state0)
    mx.eval(y_up, final_up, y_ms, final_ms)
    assert mx.array_equal(final_up, final_ms)
    assert mx.array_equal(y_up, y_ms)


def test_multistate_update_wrapper_matches_upstream():
    """gated_delta_update_multistate 与上游 gated_delta_update（kernel 路径）一致。"""
    from mlx_lm.models.gated_delta import gated_delta_update
    from mlx_streaming.core.linear_attn.gated_delta_multistate import (
        gated_delta_update_multistate,
    )

    mx.random.seed(11)
    B, T, Hk, Hv, Dk, Dv = 1, 3, 2, 4, 128, 128
    q = mx.random.normal((B, T, Hk, Dk))
    k = mx.random.normal((B, T, Hk, Dk))
    v = mx.random.normal((B, T, Hv, Dv))
    a = mx.random.normal((B, T, Hv))
    bb = mx.random.normal((B, T, Hv))
    A_log = mx.random.normal((Hv,))
    dt_bias = mx.random.normal((Hv,))
    state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    out_up, st_up = gated_delta_update(
        q, k, v, a, bb, A_log, dt_bias, state, None, use_kernel=True
    )
    out_ms, st_ms, states = gated_delta_update_multistate(
        q, k, v, a, bb, A_log, dt_bias, state, None
    )
    mx.eval(out_up, st_up, out_ms, st_ms, states)
    assert mx.array_equal(out_up, out_ms)
    assert mx.array_equal(st_up, st_ms)
    assert mx.array_equal(states[:, -1], st_up)
