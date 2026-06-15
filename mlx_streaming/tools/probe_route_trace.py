"""采集 MTP 运行中的主模型 MoE 路由 trace。

输出 JSONL: {"layer": 0, "experts": [..]}。每行是一层 MoE 调用的唯一专家集合。
用于 `simulate_eviction.py` 离线比较 LRU / window / Belady 替换策略。
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

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
TRACE_OUT = os.environ.get("ROUTE_TRACE_OUT", "/tmp/qwen_route_trace.jsonl")


def main():
    os.environ["ROUTE_TRACE"] = "1"
    model, tok, store = build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    route_trace.enable()
    ids, stats = mtp_generate(model, drafter, tok, mx.array([tok.encode(PROMPT)]),
                              MAXTOK, K=K, ids_mode=True, profile=False)
    n_events = route_trace.dump_jsonl(TRACE_OUT)
    route_trace.disable()
    print(json.dumps({
        "trace_out": TRACE_OUT,
        "events": n_events,
        "tokens": len(ids),
        "steps": stats["steps"],
        "avg_accept_len": stats["avg_accept_len"],
        "store_misses": store.misses,
        "store_hits": store.hits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
