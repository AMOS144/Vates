import mlx.core as mx

from mlx_streaming.core.moe.custom_kernel import _custom_fused_moe_indexed


def _stack_quantized(experts: int, out_dim: int, in_dim: int, group: int, bits: int):
    weights = []
    scales = []
    biases = []
    for _ in range(experts):
        w = mx.random.normal((out_dim, in_dim)).astype(mx.float32) * 0.1
        wq, s, b = mx.quantize(w, group_size=group, bits=bits)
        weights.append(wq)
        scales.append(s)
        biases.append(b)
    return mx.stack(weights), mx.stack(scales), mx.stack(biases)


def _indexed_qlinear(x, indices, weight, scales, biases, group: int, bits: int):
    ys = []
    for pair, expert in enumerate(indices.tolist()):
        y = mx.quantized_matmul(
            x[pair:pair + 1],
            weight[int(expert)],
            scales[int(expert)],
            biases[int(expert)],
            transpose=True,
            group_size=group,
            bits=bits,
            mode="affine",
        )
        ys.append(y)
    return mx.concatenate(ys, axis=0)


def test_custom_fused_moe_indexed_matches_three_quantized_projections():
    # fused kernel 的契约：每个 pair 使用 indices[pair] 选本地专家，输出 down 后 hidden。
    mx.random.seed(0)
    experts = 3
    pairs = 5
    hidden = 64
    inter = 32
    group = 32
    bits = 6
    x = mx.random.normal((pairs, hidden)).astype(mx.float32) * 0.1
    indices = mx.array([0, 1, 2, 1, 0], dtype=mx.uint32)
    gate_w, gate_s, gate_b = _stack_quantized(experts, inter, hidden, group, bits)
    up_w, up_s, up_b = _stack_quantized(experts, inter, hidden, group, bits)
    down_w, down_s, down_b = _stack_quantized(experts, hidden, inter, group, bits)

    gate = _indexed_qlinear(x, indices, gate_w, gate_s, gate_b, group, bits)
    up = _indexed_qlinear(x, indices, up_w, up_s, up_b, group, bits)
    act = gate * mx.sigmoid(gate) * up
    ref = _indexed_qlinear(act, indices, down_w, down_s, down_b, group, bits)
    got = _custom_fused_moe_indexed(
        x,
        indices,
        gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        down_w, down_s, down_b,
        hidden,
        inter,
        group,
        bits,
    )
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-3).item()
