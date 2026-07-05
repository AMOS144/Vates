"""最小树 top-2(TREE_TOP2)稳态多 prompt A/B:tree-off vs tree-on 的净 tok/s 与 lossless 裁决。

模型只加载一次。lossless 口径(噪声地板对照):
  ZEROCOPY_DUAL_SOURCE=1 后端有良性 run-to-run token 漂移(字节校验 0 BAD,漂移来自 MoE 浮点
  非结合性在近平局处翻转 argmax),严格逐 token 相等对 baseline 自己都不成立。故:
    - ref  = 非投机贪婪 baseline(每 prompt 跑一次,作真参考)
    - 噪声地板 control_mm = tree-off 各重复轮相对 ref 的最大失配数
    - on_mm = tree-on 各重复轮相对 ref 的最大失配数
    - lossless_ok(逐 prompt) = on_mm <= control_mm(最小树不引入超出后端噪声的失配)
tok/s 每配置每 prompt 重复 REPEAT 次取中位数,抵抗抖动;跑前 warmup 一遍热 kernel/池。
生产配置(env,与 run_mtp_spec 一致):
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 .venv/bin/python benchmarks/bench_tree.py
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core.mem import reset_peak
from mlx_streaming.mtp.bench_verdict import median, verdict_from_delta
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import (
    forward_with_hidden, prefill_chunked, mtp_generate)
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

K = int(os.environ.get("K", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "96"))
REPEAT = int(os.environ.get("REPEAT", "3"))
MARGIN = float(os.environ.get("MARGIN", "0.05"))

# 固定 6 条多样中文 prompt(取自 router_pred_prompts.txt:概念/代码/英文/叙事/JSON 等风格)。
PROMPTS = [
    "用三句话解释什么是混合专家模型。",
    "写一段 Python 代码，演示如何用 LRU 缓存函数结果。",
    "为什么模型量化会影响困惑度和生成质量？",
    "用英文写一段关于 speculative decoding 的技术摘要。",
    "请写一个短故事，主题是工程师在午夜调试模型推理性能。",
    "请给出一个使用 Python 解析 JSONL 文件并统计字段频率的例子。",
]


def _baseline_greedy(model, enc, n):
    """非投机贪婪 baseline(逐 token argmax),作为 lossless 真参考 ref。"""
    cache = model.make_cache()
    logits, _ = prefill_chunked(model, enc, cache)
    out = []
    for _ in range(n):
        nxt = int(mx.argmax(logits[:, -1, :]))
        out.append(nxt)
        if len(out) >= n:
            break
        logits, _ = forward_with_hidden(model, mx.array([[nxt]]), cache)
    return out


def _run_once(model, drafter, tok, enc, store):
    store.reset_stats()
    ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                              ids_mode=True, profile=True)
    tps = round(stats["tokens"] / stats["wall_s"], 2)
    return ids, stats, tps


def _mismatch(ids, ref):
    return sum(1 for a, b in zip(ids, ref) if a != b)


def _bench_prompt(model, drafter, tok, prompt, store):
    enc = mx.array([tok.encode(prompt)])
    ref = _baseline_greedy(model, enc, MAXTOK)          # 非投机贪婪真参考

    os.environ["TREE_TOP2"] = "0"
    off_tps, off_mm = [], []
    for _r in range(REPEAT):
        ids, _stats, tps = _run_once(model, drafter, tok, enc, store)
        off_tps.append(tps)
        off_mm.append(_mismatch(ids, ref))

    os.environ["TREE_TOP2"] = "1"
    on_tps, on_mm, on_rescues = [], [], 0
    for _r in range(REPEAT):
        ids, stats, tps = _run_once(model, drafter, tok, enc, store)
        on_tps.append(tps)
        on_mm.append(_mismatch(ids, ref))
        on_rescues = stats["tree_rescues"]

    off_med, on_med = median(off_tps), median(on_tps)
    control_mm = max(off_mm)                             # 后端噪声地板
    on_mm_max = max(on_mm)
    lossless_ok = on_mm_max <= control_mm               # 不引入超出噪声的失配
    delta = (on_med - off_med) / max(off_med, 1e-6)
    return {
        "prompt": prompt[:16],
        "off_tps_med": off_med,
        "on_tps_med": on_med,
        "off_tps_runs": off_tps,
        "on_tps_runs": on_tps,
        "delta_pct": round(delta * 100, 2),
        "tree_rescues": on_rescues,
        "control_mm": control_mm,                       # tree-off vs ref 最大失配(噪声地板)
        "on_mm": on_mm_max,                             # tree-on vs ref 最大失配
        "lossless_ok": lossless_ok,
    }


def main():
    reset_peak()
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    # warmup:两条路径各跑一遍第一个 prompt,热 Metal kernel + 专家常驻池 + 预取。
    print(f"[warmup] slots={store.capacity} K={K} maxtok={MAXTOK} repeat={REPEAT}", flush=True)
    _enc = mx.array([tok.encode(PROMPTS[0])])
    for _t in ("0", "1"):
        os.environ["TREE_TOP2"] = _t
        _run_once(model, drafter, tok, _enc, store)

    rows = []
    for p in PROMPTS:
        row = _bench_prompt(model, drafter, tok, p, store)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    deltas = [r["delta_pct"] / 100 for r in rows]
    agg_delta = median(deltas)
    lossless_all = all(r["lossless_ok"] for r in rows)
    verdict = verdict_from_delta(agg_delta, lossless_all, margin=MARGIN)
    summary = {
        "n_prompts": len(rows),
        "repeat": REPEAT,
        "maxtok": MAXTOK,
        "margin_pct": round(MARGIN * 100, 1),
        "median_delta_pct": round(agg_delta * 100, 2),
        "lossless_all": lossless_all,
        "max_control_mm": max(r["control_mm"] for r in rows),
        "max_on_mm": max(r["on_mm"] for r in rows),
        "verdict": verdict,
    }
    print("=== SUMMARY (minimal-tree top-2 off vs on) ===")
    print(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
