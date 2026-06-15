"""测「同 token、上层 hidden 预测下一层专家」对 miss 子集的 recall（命门 recall_miss）。

与跨-token 历史信号不同：这里用的是**当前 token** 第 L-AHEAD 层的真实 hidden（携带本 token
新颖内容），经第 L 层自己的 gate/post_norm 预测第 L 层专家。
- AHEAD=0：用第 L 层 attention 前的 hidden（窗口仅 ~70µs）。
- AHEAD=1：用第 L-1 层输入 hidden（窗口 ≈ 整层 ≈ 1.1ms，够盖住 340µs 物化）。

命门：recall_miss 高 → 该信号能预测真正要预取的 miss，且 AHEAD≥1 窗口够 → 值得落运行时。
中池由 EXPERT_SLOTS 控制。
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
MAXTOK = int(os.environ.get("MAXTOK", "64"))
K = int(os.environ.get("K", "2"))
AHEADS = [int(x) for x in os.environ.get("AHEADS", "0,1,2").split(",")]
MULTS = [int(x) for x in os.environ.get("MULTS", "1,2,4").split(",")]

STATS = defaultdict(lambda: defaultdict(float))


def _predict(mlp, norm, h, mult) -> set:
    gates = mx.softmax(mlp.gate(norm(h)), axis=-1, precise=True)
    k = min(gates.shape[-1], mlp.top_k * mult)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(inds)
    return {int(e) for e in inds.reshape(-1).tolist()}


def _score(name, mult, pred, actual, miss) -> None:
    s = STATS[f"{name}_x{mult}"]
    s["n"] += 1
    s["hit_full"] += len(pred & actual)
    s["tot_full"] += len(actual)
    s["hit_miss"] += len(pred & miss)
    s["tot_miss"] += len(miss)
    s["pred"] += len(pred)


def instrumented_forward_with_hidden(model, ids, cache):
    inner = model.model
    h = inner.embed_tokens(ids)
    layers = inner.layers
    has_full = any(not l.is_linear for l in layers)
    has_linear = any(l.is_linear for l in layers)
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx]) if has_full else None
    ssm_mask = create_ssm_mask(h, cache[ssm_idx]) if has_linear else None
    snaps = []  # 每层进入前（attention 前）的 hidden 快照
    for i, (layer, c) in enumerate(zip(layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        snaps.append(h)
        route_trace.enable()
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        events = route_trace.events()
        route_trace.disable()
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FileStreamingMoeBlock) and events:
            actual = {int(e) for e in events[-1]["experts"]}
            miss = {int(e) for e in events[-1].get("miss", [])}
            for ahead in AHEADS:
                if i - ahead < 0:
                    continue
                src_h = snaps[i - ahead]
                for mult in MULTS:
                    pred = _predict(mlp, layer.post_attention_layernorm, src_h, mult)
                    _score(f"ahead{ahead}", mult, pred, actual, miss)
    H = inner.norm(h)
    return model.lm_head(H), H


def main():
    os.environ["ROUTE_TRACE"] = "1"
    os.environ.setdefault("RESIDENT_POOL", "1")
    os.environ.setdefault("MTP_VERIFY_MODE", "batch")
    os.environ.setdefault("MTP_ARRAY_COMMIT", "1")
    model, tok, store = build_streaming_model()
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    old = mg.forward_with_hidden
    mg.forward_with_hidden = instrumented_forward_with_hidden
    try:
        ids, mtp_stats = mtp_generate(
            model, drafter, tok, mx.array([tok.encode(PROMPT)]),
            MAXTOK, K=K, ids_mode=True, profile=False)
    finally:
        mg.forward_with_hidden = old

    rows = []
    for key, s in sorted(STATS.items()):
        n = max(1, int(s["n"]))
        rows.append({
            "predictor": key,
            "n": int(s["n"]),
            "recall_full": round(s["hit_full"] / max(1, s["tot_full"]), 4),
            "recall_miss": round(s["hit_miss"] / max(1, s["tot_miss"]), 4),
            "tot_miss": int(s["tot_miss"]),
            "avg_pred": round(s["pred"] / n, 2),
        })
    print(json.dumps({
        "expert_slots": os.environ.get("EXPERT_SLOTS"),
        "K": K,
        "tokens": len(ids),
        "hit_rate": round(store.hits / max(1, store.hits + store.misses), 3),
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
