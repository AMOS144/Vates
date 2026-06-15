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
from mlx_streaming.core import route_trace
from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "48"))
K = int(os.environ.get("K", "3"))
MULTS = [int(x) for x in os.environ.get("CROSS_LAYER_MULTS", "1,2,4,8").split(",")]

STATS = defaultdict(lambda: defaultdict(float))


def _predict(blk: FileStreamingMoeBlock, h: mx.array, mult: int) -> set[int]:
    gates = mx.softmax(blk.gate(h), axis=-1, precise=True)
    k = min(gates.shape[-1], blk.top_k * mult)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(inds)
    return {int(e) for e in inds.reshape(-1).tolist()}


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
        if isinstance(mlp, FileStreamingMoeBlock):
            # 真实 routing 用的是 post_attention_layernorm(h_after_attn)，这里在 attention 前近似。
            h_input_norm = layer.input_layernorm(h)
            h_post_norm = layer.post_attention_layernorm(h)
            for mult in MULTS:
                preds[("input_norm", mult)] = _predict(mlp, h_input_norm, mult)
                preds[("post_norm", mult)] = _predict(mlp, h_post_norm, mult)

        route_trace.enable()
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        events = route_trace.events()
        route_trace.disable()
        if events and preds:
            actual = {int(e) for e in events[-1]["experts"]}
            for (name, mult), pred in preds.items():
                _score(name, mult, pred, actual)
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
    print(json.dumps({
        "K": K,
        "tokens": len(ids),
        "steps": mtp_stats["steps"],
        "avg_accept_len": mtp_stats["avg_accept_len"],
        "store_misses": store.misses,
        "store_hits": store.hits,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
