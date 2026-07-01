"""最小树(位置1 top-2)A/B:tree off vs on,实测 accept_len / hit_rate / union / exact_match 净收益。

模型只加载一次;TREE_TOP2 每 token 实时读 env,故进程内切换。ref=tree-off 的输出(MTP 自投机
贪婪精确),tree-on 必须逐 token 相等(正确性护栏)。union 需 UNION_PROF=1(在 import 前设),
其探针本身会扰动 hit_rate,故建议跑两遍:UNION_PROF=0 取干净 hit_rate,UNION_PROF=1 取 union。
"""
import os
os.environ.setdefault("MISS_ATTRIB", "1")
os.environ.setdefault("PREDICT_RECALL_PROF", "1")

import json
import time

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core.mem import reset_peak
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.core.profiling import (
    MISS_ATTRIB, PREDICT_RECALL_PROF, UNION_PROF, tprof_reset)
from mlx_streaming import config as _cfg

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
K = int(os.environ.get("K", "3"))


def _reset():
    for k in MISS_ATTRIB:
        MISS_ATTRIB[k] = 0
    for k in PREDICT_RECALL_PROF:
        PREDICT_RECALL_PROF[k] = 0
    UNION_PROF.clear()
    tprof_reset()


def main():
    reset_peak()
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)
    enc = mx.array([tok.encode(PROMPT)])

    os.environ["TREE_TOP2"] = "0"
    print(f"[warmup] tree=off width={_cfg.cross_layer_predict_width()} "
          f"budget={_cfg.stream_blob_bg_budget()} slots={store.capacity} K={K}", flush=True)
    ref, _ = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K, ids_mode=True, profile=True)

    rows = []
    for tree in (0, 1):
        os.environ["TREE_TOP2"] = str(tree)
        store.reset_stats()
        _reset()
        t0 = time.perf_counter()
        ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                                  ids_mode=True, profile=True)
        dr = max(1, MISS_ATTRIB["dec_routed"])
        u = {s: round(v[0] / max(1, v[1]), 2) for s, v in sorted(UNION_PROF.items())}
        rows.append({
            "tree_top2": bool(tree),
            "avg_accept_len": stats["avg_accept_len"],
            "accept_hist": stats["accept_hist"],
            "tree_rescues": stats["tree_rescues"],
            "recall": round(PREDICT_RECALL_PROF["hit"] / max(1, PREDICT_RECALL_PROF["routed"]), 4),
            "dec_hit_rate": round(MISS_ATTRIB["dec_resident_hit"] / dr, 4),
            "miss_A_timing": MISS_ATTRIB["dec_miss_A_timing"],
            "disk_loads": store.misses,
            "union_by_seq": u,
            "spec_tok_per_s": round(stats["tokens"] / stats["wall_s"], 2),
            "exact_match": ids == ref,
            "wall_s": round(time.perf_counter() - t0, 2),
        })
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    print("=== SUMMARY (tree off vs on) ===")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
