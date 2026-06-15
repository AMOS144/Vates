"""端到端 A/C × cap∈{32,64,96}：同层(AHEAD=0)预测预取 vs plain。

核心问题（用数据回答，不靠假设）：
1. 窗口够不够：ready_on_time / (ready_on_time + not_ready) —— attention/GDN 窗口
   能否藏住缺失专家的后台物化。
2. 净收益：C 的 tok/s 是否 > A（同 cap），命中是否提升，内存是否仍是该 cap 水平。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERT_DIR = os.path.join(ROOT, "models", "qwen3_next_experts_2bit_g128")
BLOB_DIR = "/tmp/cb_2bit_blob"

COMMON = {
    "MODEL": "/tmp/qwen3_next_80b_4bit",
    "EXPERT_DIR": EXPERT_DIR,
    "RESIDENT_POOL": "1",
    "MTP_VERIFY_MODE": "batch",
    "MTP_ARRAY_COMMIT": "1",
    "K": "2",
    "MAXTOK": "64",
    "NATIVE_MOE": "0",
    "STREAM_BLOB": "0",
}

CAPS = [32, 64, 96]
FIELDS = ["spec_tok_per_s", "baseline_tok_per_s", "spec_hit_rate",
          "mlx_peak_gb", "rss_gb", "exact_match", "n_mismatch", "bg_stats"]


def _variants_for_cap(cap: int):
    slots = str(cap)
    return [
        (f"A_plain_{cap}", {"EXPERT_SLOTS": slots, "STREAM_BLOB_LOADER": "1",
                            "STREAM_BLOB_BG": "0", "BLOB_DIR": BLOB_DIR,
                            "CROSS_LAYER_PREFETCH": "0"}),
        # C：同层预取 —— AHEAD=0、预算 2×top_k、只取「预测∩非常驻」
        (f"C_samelayer_{cap}", {"EXPERT_SLOTS": slots, "STREAM_BLOB_BG": "1",
                                "BLOB_DIR": BLOB_DIR, "STREAM_BLOB_WINDOW": "2",
                                "CROSS_LAYER_PREFETCH": "1",
                                "CROSS_LAYER_PREFETCH_AHEAD": "0",
                                "CROSS_LAYER_PREFETCH_MULT": "2",
                                "STREAM_BLOB_BG_BUDGET": "4"}),
    ]


def _run(name, overrides):
    env = os.environ.copy()
    env.update(COMMON)
    env.update(overrides)
    out = subprocess.check_output(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"], env=env, text=True)
    rec = json.loads(out)
    row = {"variant": name}
    row.update({k: rec.get(k) for k in FIELDS})
    bg = rec.get("bg_stats") or {}
    rot = bg.get("ready_on_time", 0)
    nr = bg.get("not_ready", 0)
    row["ready_rate"] = round(rot / max(rot + nr, 1), 3) if (rot + nr) else None
    return row


def main():
    rows = []
    for cap in CAPS:
        variants = _variants_for_cap(cap)
        a = _run(*variants[0])
        c = _run(*variants[1])
        base_tps = a.get("spec_tok_per_s") or 1e-9
        for r in (a, c):
            r["tok_s_vs_A"] = round((r.get("spec_tok_per_s") or 0) / base_tps, 3)
        rows.extend([a, c])
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
