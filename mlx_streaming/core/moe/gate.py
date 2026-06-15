"""MoE 门控与专家选择：top-k 激活数开关 + 跨层专家预测（gate 前向 + argpartition）。"""
import mlx.core as mx

from mlx_streaming import config


def _effective_top_k(default_k: int) -> int:
    """实验开关：降低每 token 激活专家数以测速度/质量取舍。默认不改变模型。"""
    override = config.moe_topk_override()
    if not override:
        return default_k
    return max(1, min(default_k, int(override)))


def _predict_layer_experts(norm, gate, top_k: int, x: mx.array, mult: int) -> "tuple[dict[int, float], int]":
    """用 gate(norm(x)) 预测专家集，返回 ({expert_id: 最大 softmax 分数}, num_experts)。

    norm/gate 必须取自**目标层**（被预测的那层），以匹配 probe 验证的
    gate_L(post_attention_layernorm_L(h)) 配置（recall_miss≈0.95）。
    """
    gates = mx.softmax(gate(norm(x)), axis=-1, precise=True)
    num_experts = gates.shape[-1]
    k = min(num_experts, top_k * mult)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(gates, inds, axis=-1)
    mx.eval(inds, vals)
    best: "dict[int, float]" = {}
    for e, s in zip(inds.reshape(-1).tolist(), vals.reshape(-1).tolist()):
        e, s = int(e), float(s)
        if s > best.get(e, -1.0):
            best[e] = s
    return best, num_experts
