"""K sweep：增大 MTP 的 K 以拉长每层 verify 计算时间，观察 width=32 的 hit_rate 能否升到 recall 上限。

机制：K 越大 → verify 每次前向处理 K+1 个 token → 每层 GPU 计算时间越长 → 后台预取的完成窗口越宽
→ miss_A_timing（pread 没算完就要用）越少 → hit_rate 逼近 recall（width=32 的 recall≈0.9183）。

固定 width=32 / budget=32（=width，排除 budget 截断混淆）/ EXPERT_SLOTS=32。同时量 union 专家数：
K 越大，K+1 个 token 的路由并集越大，可能超过 32 槽物理上限 → 触发 miss_A_evicted（另一个天花板）。

口径同前：recall=真实路由∩预测集/真实路由；hit_rate=真实路由∩acquire前驻留/真实路由（decode 桶）。
exact_match：MTP 自投机精确验证，任意 K 的贪婪输出应逐 token 相等（正确性护栏）。
"""
import os
# UNION_ON 在 profiling 导入时定格，故 union/recall/miss 探针须在导入 mlx_streaming 前打开。
os.environ.setdefault("UNION_PROF", "1")
os.environ.setdefault("MISS_ATTRIB", "1")
os.environ.setdefault("PREDICT_RECALL_PROF", "1")
os.environ.setdefault("CROSS_LAYER_PREDICT_WIDTH", "32")
os.environ.setdefault("STREAM_BLOB_BG_BUDGET", "32")

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
MAXTOK = int(os.environ.get("MAXTOK", "64"))
K_SWEEP = [int(x) for x in os.environ.get("K_SWEEP", "3,4,5,6,8,10").split(",")]


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
    print(f"[warmup] K={K_SWEEP[0]} width={_cfg.cross_layer_predict_width()} "
          f"budget={_cfg.stream_blob_bg_budget()} slots={store.capacity} ...", flush=True)
    ref, _ = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K_SWEEP[0],
                          ids_mode=True, profile=True)

    rows = []
    for K in K_SWEEP:
        store.reset_stats()
        _reset()
        t0 = time.perf_counter()
        ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                                  ids_mode=True, profile=True)
        dr = max(1, MISS_ATTRIB["dec_routed"])
        # union：K 个 draft 的路由并集分桶记于 seq。取 seq∈{K, K+1} 两个可能的 verify 桶。
        u_by_seq = {s: round(v[0] / max(1, v[1]), 2) for s, v in sorted(UNION_PROF.items())}
        row = {
            "K": K,
            "recall": round(PREDICT_RECALL_PROF["hit"] / max(1, PREDICT_RECALL_PROF["routed"]), 4),
            "dec_hit_rate": round(MISS_ATTRIB["dec_resident_hit"] / dr, 4),
            "miss_A_timing": MISS_ATTRIB["dec_miss_A_timing"],
            "miss_A_evicted": MISS_ATTRIB["dec_miss_A_evicted"],
            "miss_B_unpred": MISS_ATTRIB["dec_miss_B"],
            "union_by_seq": u_by_seq,
            "avg_accept_len": stats.get("avg_accept_len"),
            "t_verify_s": round(stats.get("t_verify_s", 0.0), 3),
            "disk_loads": store.misses,
            "spec_tok_per_s": round(stats["tokens"] / stats["wall_s"], 2),
            "exact_match": ids == ref,
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print("=== SUMMARY (目标 hit_rate → recall=0.9183) ===")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
