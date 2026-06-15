"""统计指定 MoE 层在 MTP verify 中每次调用的 unique expert 数分布。"""
import json
import os
from collections import Counter, defaultdict

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

import mlx_streaming.mtp.generate as mg
from mlx_streaming.core import route_trace
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "3"))
LAYERS = {int(x) for x in os.environ.get("LAYERS", "43,47").split(",")}


COUNTS: dict[int, list[int]] = defaultdict(list)


def traced_forward_with_hidden(model, ids, cache):
    route_trace.enable()
    logits, H = mg._ORIG_FORWARD_WITH_HIDDEN(model, ids, cache)
    mx.eval(logits, H)
    for rec in route_trace.events():
        layer = int(rec["layer"])
        if layer in LAYERS:
            COUNTS[layer].append(len(set(int(e) for e in rec["experts"])))
    route_trace.disable()
    return logits, H


def _summary(vals: list[int]) -> dict:
    vals = sorted(vals)
    if not vals:
        return {"n": 0}
    def pct(p):
        return vals[min(len(vals) - 1, int((len(vals) - 1) * p))]
    return {
        "n": len(vals),
        "min": vals[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": vals[-1],
        "hist": dict(sorted(Counter(vals).items())),
    }


def main():
    os.environ["ROUTE_TRACE"] = "1"
    model, tok, _store = build_streaming_model()
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    mg._ORIG_FORWARD_WITH_HIDDEN = mg.forward_with_hidden
    mg.forward_with_hidden = traced_forward_with_hidden
    try:
        ids, stats = mtp_generate(
            model, drafter, tok, mx.array([tok.encode(PROMPT)]),
            MAXTOK, K=K, ids_mode=True, profile=False)
    finally:
        mg.forward_with_hidden = mg._ORIG_FORWARD_WITH_HIDDEN

    print(json.dumps({
        "K": K,
        "tokens": len(ids),
        "steps": stats["steps"],
        "avg_accept_len": stats["avg_accept_len"],
        "layers": {str(k): _summary(v) for k, v in sorted(COUNTS.items())},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
