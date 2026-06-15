"""端到端 A/B：常驻 vs 全流式 blob（STREAM_BLOB=1）。

对比 spec_tok_per_s / mlx_peak_gb / rss_gb / exact_match，验证"低内存流式"是否达成
（目标：exact_match=true、峰值内存大降、tok/s ≥ 常驻 80%）。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EXPERT_DIR = os.path.join(ROOT, "models", "qwen3_next_experts_2bit_g128")

VARIANTS = [
    ("resident", {
        "STREAM_BLOB": "0",
        "CROSS_LAYER_PREFETCH": "0",
    }),
    ("stream_blob", {
        "STREAM_BLOB": "1",
        "BLOB_DIR": "/tmp/cb_2bit_blob",
        "STREAM_BLOB_WORKERS": "8",
        "STREAM_BLOB_NOCACHE": "1",
        "STREAM_BLOB_WINDOW": "3",
        "CROSS_LAYER_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH_AHEAD": "1",
        "CROSS_LAYER_PREFETCH_MULT": "2",
    }),
]
FIELDS = ["spec_tok_per_s", "baseline_tok_per_s", "speedup", "avg_accept_len",
          "exact_match", "n_mismatch", "mlx_peak_gb", "rss_gb"]


def _run(name, overrides):
    env = os.environ.copy()
    env.setdefault("MODEL", "/tmp/qwen3_next_80b_4bit")
    env.setdefault("EXPERT_DIR", DEFAULT_EXPERT_DIR)
    env.setdefault("EXPERT_SLOTS", "256")
    env.setdefault("RESIDENT_POOL", "1")
    env.setdefault("MTP_VERIFY_MODE", "batch")
    env.setdefault("MTP_ARRAY_COMMIT", "1")
    env.setdefault("K", "2")
    env.setdefault("MAXTOK", "64")
    env.setdefault("NATIVE_MOE", "0")
    env.update(overrides)
    out = subprocess.check_output(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"], env=env, text=True)
    rec = json.loads(out)
    row = {"variant": name}
    row.update({k: rec.get(k) for k in FIELDS})
    return row


def main():
    rows = [_run(n, ov) for n, ov in VARIANTS]
    res = rows[0]
    for r in rows:
        r["tok_s_vs_resident"] = round((r.get("spec_tok_per_s") or 0) / max(res.get("spec_tok_per_s") or 1e-9, 1e-9), 3)
        r["peak_gb_vs_resident"] = round((r.get("mlx_peak_gb") or 0) / max(res.get("mlx_peak_gb") or 1e-9, 1e-9), 3)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
