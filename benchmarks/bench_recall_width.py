"""recall vs hit_rate 宽度 sweep：定位"~90% recall 对应多宽",并拆清 recall→hit_rate 的差距来源。

模型只加载一次；每个点改 CROSS_LAYER_PREDICT_WIDTH / STREAM_BLOB_BG_BUDGET（config 每 token 实时读
env）。同时开 PREDICT_RECALL_PROF（量预测集覆盖真实路由的 recall）+ MISS_ATTRIB（量真正进池的
hit_rate 及 miss 拆分）。K=3 / EXPERT_SLOTS=32，无计算注入（COMPUTE_INFLATE_ITERS=0）。

口径区别：
- recall     = 真实路由 ∩ 预测集 / 真实路由（预测器质量上限，随 width 增大）
- hit_rate   = 真实路由 ∩ acquire 前驻留 / 真实路由（预测对 + 预取送达 + 未被 budget 截断/驱逐）
recall−hit_rate 的差 = 预测对了却没送到（budget 截断 + 时序）。

SWEEP 形如 "w,b;w,b;..."：默认先固定 budget=16 扫 width，再给几个 width=budget 匹配点。
"""
import json
import os
import time

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core.mem import reset_peak
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.core.profiling import MISS_ATTRIB, PREDICT_RECALL_PROF, tprof_reset
from mlx_streaming import config as _cfg

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "64"))
K = int(os.environ.get("K", "3"))
# (width, budget) 点集：先固定 budget=16 扫 width，再匹配 budget=width。
SWEEP = os.environ.get(
    "SWEEP", "16,16;24,16;32,16;48,16;64,16;32,32;48,48;64,64")


def _reset():
    for k in MISS_ATTRIB:
        MISS_ATTRIB[k] = 0
    for k in PREDICT_RECALL_PROF:
        PREDICT_RECALL_PROF[k] = 0
    tprof_reset()


def _run_once(model, drafter, tok):
    ids, stats = mtp_generate(
        model, drafter, tok, mx.array([tok.encode(PROMPT)]),
        MAXTOK, K=K, ids_mode=True, profile=True)
    return ids, stats, round(stats["tokens"] / stats["wall_s"], 2)


def main():
    os.environ["PREDICT_RECALL_PROF"] = "1"
    os.environ["MISS_ATTRIB"] = "1"
    reset_peak()
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    points = [tuple(int(x) for x in p.split(",")) for p in SWEEP.split(";") if p.strip()]

    # warmup（用第一个点的配置）：编译 kernel、热池/预取，并取基线 ids 作 exact_match 护栏。
    w0, b0 = points[0]
    os.environ["CROSS_LAYER_PREDICT_WIDTH"] = str(w0)
    os.environ["STREAM_BLOB_BG_BUDGET"] = str(b0)
    print(f"[warmup] width={w0} budget={b0} ...", flush=True)
    base_ids, _, _ = _run_once(model, drafter, tok)

    rows = []
    for (w, b) in points:
        os.environ["CROSS_LAYER_PREDICT_WIDTH"] = str(w)
        os.environ["STREAM_BLOB_BG_BUDGET"] = str(b)
        store.reset_stats()
        _reset()
        t0 = time.perf_counter()
        ids, stats, tps = _run_once(model, drafter, tok)
        dr = max(1, MISS_ATTRIB["dec_routed"])
        recall = round(PREDICT_RECALL_PROF["hit"] / max(1, PREDICT_RECALL_PROF["routed"]), 4)
        hit = round(MISS_ATTRIB["dec_resident_hit"] / dr, 4)
        row = {
            "width": w, "budget": b,
            "recall": recall,
            "dec_hit_rate": hit,
            "recall_minus_hit": round(recall - hit, 4),
            "miss_A_timing": MISS_ATTRIB["dec_miss_A_timing"],
            "miss_B_unpred": MISS_ATTRIB["dec_miss_B"],
            "disk_loads": store.misses,
            "spec_tok_per_s": tps,
            "exact_match": ids == base_ids,
            "wall_s": round(time.perf_counter() - t0, 2),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    print("=== SUMMARY ===")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
