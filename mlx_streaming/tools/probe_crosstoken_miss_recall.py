"""Milestone 1：测「上一/前几次同层路由」对当前 token 各层 miss 的 recall（命门 recall_miss）。

中池由 EXPERT_SLOTS 控制（跑 64 / 96 两档）。high→大窗口预取能藏住中池 miss；
low→miss ⊥ 历史信号（墙坐实），转 Plan B。
history_n>1 的并集天然涵盖 MTP 草稿/验证多次 occurrence。
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core import route_trace
from mlx_streaming.tools.crosstoken_recall import crosstoken_recall
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "64"))
K = int(os.environ.get("K", "2"))
HISTORY_NS = [int(x) for x in os.environ.get("HISTORY_NS", "1,2,3").split(",")]


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

    route_trace.enable()
    ids, mtp_stats = mtp_generate(
        model, drafter, tok, mx.array([tok.encode(PROMPT)]),
        MAXTOK, K=K, ids_mode=True, profile=False)
    events = route_trace.events()
    route_trace.disable()

    rows = [crosstoken_recall(events, history_n=n) for n in HISTORY_NS]
    print(json.dumps({
        "expert_slots": os.environ.get("EXPERT_SLOTS"),
        "K": K,
        "tokens": len(ids),
        "n_events": len(events),
        "store_hits": store.hits,
        "store_misses": store.misses,
        "hit_rate": round(store.hits / max(1, store.hits + store.misses), 3),
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
