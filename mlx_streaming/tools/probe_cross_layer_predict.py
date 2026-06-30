"""评估跨层专家预测信号。

在每层真正执行前，用该层输入 hidden 近似预测该层 MoE experts：
- input_norm: gate(input_layernorm(h_in))
- post_norm:  gate(post_attention_layernorm(h_in))

然后执行真实 layer，读取实际 MoE 路由，对比 recall/precision。
这是“上一层输出 -> 下一层专家”的离线 probe，不接入推理热路径。
"""
import json
import os
from collections import defaultdict

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.qwen3_next import ModelArgs

import mlx_streaming.mtp.generate as mg
from mlx_streaming import config
from mlx_streaming.core import route_trace
from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

QN_CONFIG = config.qn_config()
MTP_OUT = config.mtp_out()
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "48"))
K = int(os.environ.get("K", "3"))
MULTS = [int(x) for x in os.environ.get("CROSS_LAYER_MULTS", "1,2,4,8").split(",")]
# 逐层 recall 只看一个预测器（默认 post_norm × PER_LAYER_MULT），用于评估"砍常驻池、靠预取"可行性。
PER_LAYER_NAME = os.environ.get("PER_LAYER_NAME", "post_norm")
PER_LAYER_MULT = int(os.environ.get("PER_LAYER_MULT", "1"))
# 预测集"绝对截断"上界列表：把本层预测专家集合截到 top-N（按门控概率），
# 模拟"预测集必须塞进 cap=N 槽常驻池"的约束，量此时 recall。
PREDICT_CAPS = [int(x) for x in os.environ.get("PREDICT_CAPS", "16,24,32,40").split(",")]

STATS = defaultdict(lambda: defaultdict(float))
LAYER_STATS = defaultdict(lambda: defaultdict(float))  # layer_idx -> {hit, actual, n, recall_sum}
# cap -> {hit, actual, pred, n, recall_sum}（聚合）
CAP_STATS = defaultdict(lambda: defaultdict(float))
# cap -> layer_idx -> {hit, actual, n, recall_sum}（逐层）
CAP_LAYER = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))


def _predict(blk: FileStreamingMoeBlock, h: mx.array, mult: int) -> set[int]:
    gates = mx.softmax(blk.gate(h), axis=-1, precise=True)
    k = min(gates.shape[-1], blk.top_k * mult)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(inds)
    return {int(e) for e in inds.reshape(-1).tolist()}


def _predict_capped(blk: FileStreamingMoeBlock, h: mx.array, cap: int) -> set[int]:
    """预测本层专家并把集合"绝对截断"到 top-cap 个。

    模拟运行时约束：预取集必须塞进 cap 个常驻槽。
    跨多个 token 位置取每个专家的最大门控概率作为排序分，保留 top-cap。
    """
    gates = mx.softmax(blk.gate(h), axis=-1, precise=True)  # [b, s, E]
    # 每个专家在所有 token 位置上的最大门控概率（覆盖"任一位置强烈想要"的专家）
    score = gates.reshape(-1, gates.shape[-1]).max(axis=0)  # [E]
    k = min(int(score.shape[-1]), cap)
    inds = mx.argpartition(score, kth=-k)[-k:]
    mx.eval(inds)
    return {int(e) for e in inds.tolist()}


def _score(name: str, mult: int, pred: set[int], actual: set[int]) -> None:
    key = f"{name}_x{mult}"
    hit = len(pred & actual)
    STATS[key]["layers"] += 1
    STATS[key]["hit"] += hit
    STATS[key]["pred"] += len(pred)
    STATS[key]["actual"] += len(actual)
    STATS[key]["recall_sum"] += hit / max(1, len(actual))
    STATS[key]["precision_sum"] += hit / max(1, len(pred))
    STATS[key]["missed_layers"] += int(hit < len(actual))


def instrumented_forward_with_hidden(model, ids, cache):
    """镜像 mtp_generate.forward_with_hidden，但在每层前后采集预测/真实路由。"""
    inner = model.model
    h = inner.embed_tokens(ids)
    layers = inner.layers
    has_full = any(not l.is_linear for l in layers)
    has_linear = any(l.is_linear for l in layers)
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx]) if has_full else None
    ssm_mask = create_ssm_mask(h, cache[ssm_idx]) if has_linear else None
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        mlp = getattr(layer, "mlp", None)
        preds = {}
        capped = {}  # cap -> 截断后的预测集
        if isinstance(mlp, FileStreamingMoeBlock):
            # 真实 routing 用的是 post_attention_layernorm(h_after_attn)，这里在 attention 前近似。
            h_input_norm = layer.input_layernorm(h)
            h_post_norm = layer.post_attention_layernorm(h)
            for mult in MULTS:
                preds[("input_norm", mult)] = _predict(mlp, h_input_norm, mult)
                preds[("post_norm", mult)] = _predict(mlp, h_post_norm, mult)
            # 绝对截断预测集（用 post_norm，更准），模拟"预取集 ≤ cap 槽"
            for cap in PREDICT_CAPS:
                capped[cap] = _predict_capped(mlp, h_post_norm, cap)

        route_trace.enable()
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        events = route_trace.events()
        route_trace.disable()
        if events and preds:
            actual = {int(e) for e in events[-1]["experts"]}
            for (name, mult), pred in preds.items():
                _score(name, mult, pred, actual)
            # 绝对截断预测集的 recall（聚合 + 逐层）
            li_cap = int(mlp.layer_idx)
            for cap, cpred in capped.items():
                chit = len(cpred & actual)
                CAP_STATS[cap]["n"] += 1
                CAP_STATS[cap]["hit"] += chit
                CAP_STATS[cap]["actual"] += len(actual)
                CAP_STATS[cap]["pred"] += len(cpred)
                CAP_STATS[cap]["recall_sum"] += chit / max(1, len(actual))
                CAP_LAYER[cap][li_cap]["hit"] += chit
                CAP_LAYER[cap][li_cap]["actual"] += len(actual)
                CAP_LAYER[cap][li_cap]["n"] += 1
                CAP_LAYER[cap][li_cap]["recall_sum"] += chit / max(1, len(actual))
            # 逐层累计选定预测器的 recall（mlp.layer_idx 为绝对层号）
            sel = preds.get((PER_LAYER_NAME, PER_LAYER_MULT))
            if sel is not None:
                li = int(mlp.layer_idx)
                hit = len(sel & actual)
                LAYER_STATS[li]["hit"] += hit
                LAYER_STATS[li]["actual"] += len(actual)
                LAYER_STATS[li]["pred"] += len(sel)
                LAYER_STATS[li]["n"] += 1
                LAYER_STATS[li]["recall_sum"] += hit / max(1, len(actual))
    H = inner.norm(h)
    return model.lm_head(H), H


def main():
    os.environ["ROUTE_TRACE"] = "1"
    os.environ.setdefault("MTP_VERIFY_MODE", "batch")
    os.environ.setdefault("MTP_ARRAY_COMMIT", "1")
    model, tok, store = build_streaming_model()
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    old_forward = mg.forward_with_hidden
    mg.forward_with_hidden = instrumented_forward_with_hidden
    try:
        ids, mtp_stats = mtp_generate(
            model, drafter, tok, mx.array([tok.encode(PROMPT)]),
            MAXTOK, K=K, ids_mode=True, profile=False)
    finally:
        mg.forward_with_hidden = old_forward

    rows = []
    for key, s in sorted(STATS.items()):
        layers = max(1, int(s["layers"]))
        rows.append({
            "predictor": key,
            "layers": layers,
            "recall": round(s["recall_sum"] / layers, 4),
            "precision": round(s["precision_sum"] / layers, 4),
            "global_recall": round(s["hit"] / max(1, s["actual"]), 4),
            "global_precision": round(s["hit"] / max(1, s["pred"]), 4),
            "avg_pred_experts": round(s["pred"] / layers, 2),
            "avg_actual_experts": round(s["actual"] / layers, 2),
            "missed_layers": int(s["missed_layers"]),
        })
    per_layer = []
    for li in sorted(LAYER_STATS):
        s = LAYER_STATS[li]
        n = max(1, int(s["n"]))
        per_layer.append({
            "layer": li,
            "recall": round(s["recall_sum"] / n, 4),          # 逐 token 平均后再按层
            "global_recall": round(s["hit"] / max(1, s["actual"]), 4),
            "avg_actual": round(s["actual"] / n, 2),
            "avg_pred": round(s["pred"] / n, 2),
            "n": n,
        })
    recalls = [r["recall"] for r in per_layer]
    # 绝对截断预测集的 recall 曲线（核心产出）：cap=N 时 recall 还剩多少
    cap_curve = []
    for cap in sorted(CAP_STATS):
        s = CAP_STATS[cap]
        n = max(1, int(s["n"]))
        per_layer_recalls = [
            CAP_LAYER[cap][li]["recall_sum"] / max(1, CAP_LAYER[cap][li]["n"])
            for li in CAP_LAYER[cap]
        ]
        cap_curve.append({
            "cap": cap,
            "recall_mean": round(s["recall_sum"] / n, 4),       # 逐拍平均
            "global_recall": round(s["hit"] / max(1, s["actual"]), 4),
            "avg_pred": round(s["pred"] / n, 2),
            "avg_actual": round(s["actual"] / n, 2),
            "per_layer_recall_min": round(min(per_layer_recalls), 4) if per_layer_recalls else None,
            "per_layer_recall_mean": round(sum(per_layer_recalls) / max(1, len(per_layer_recalls)), 4) if per_layer_recalls else None,
        })
    print(json.dumps({
        "K": K,
        "tokens": len(ids),
        "steps": mtp_stats["steps"],
        "avg_accept_len": mtp_stats["avg_accept_len"],
        "per_layer_predictor": f"{PER_LAYER_NAME}_x{PER_LAYER_MULT}",
        "per_layer_recall_min": round(min(recalls), 4) if recalls else None,
        "per_layer_recall_mean": round(sum(recalls) / max(1, len(recalls)), 4) if recalls else None,
        "per_layer_recall_max": round(max(recalls), 4) if recalls else None,
        "predict_cap_curve": cap_curve,
        "per_layer": per_layer,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
