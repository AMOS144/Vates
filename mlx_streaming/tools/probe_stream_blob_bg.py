"""端到端 A/B/C：常驻大池 vs 小池+blob loader vs 小池+后台预填(STREAM_BLOB_BG)。

看 C 能否用后台预填把小池命中率/吞吐拉近大池，同时保持小池内存。
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

VARIANTS = [
    ("A_resident_256", {"EXPERT_SLOTS": "256", "STREAM_BLOB_LOADER": "0", "STREAM_BLOB_BG": "0",
                        "CROSS_LAYER_PREFETCH": "0"}),
    ("B_blob_loader_32", {"EXPERT_SLOTS": "32", "STREAM_BLOB_LOADER": "1", "STREAM_BLOB_BG": "0",
                          "BLOB_DIR": BLOB_DIR, "CROSS_LAYER_PREFETCH": "0"}),
    ("C_bg_fill_32", {"EXPERT_SLOTS": "32", "STREAM_BLOB_BG": "1", "BLOB_DIR": BLOB_DIR,
                      "STREAM_BLOB_WINDOW": "3", "CROSS_LAYER_PREFETCH": "1",
                      "CROSS_LAYER_PREFETCH_AHEAD": "1", "CROSS_LAYER_PREFETCH_MULT": "2"}),
]
FIELDS = ["spec_tok_per_s", "baseline_tok_per_s", "speedup", "spec_hit_rate",
          "mlx_peak_gb", "rss_gb", "exact_match", "n_mismatch"]


def _run(name, overrides):
    env = os.environ.copy()
    env.update(COMMON)
    env.update(overrides)
    out = subprocess.check_output([sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"], env=env, text=True)
    rec = json.loads(out)
    row = {"variant": name}
    row.update({k: rec.get(k) for k in FIELDS})
    return row


def main():
    rows = [_run(n, ov) for n, ov in VARIANTS]
    a = rows[0]
    for r in rows:
        r["tok_s_vs_A"] = round((r.get("spec_tok_per_s") or 0) / max(a.get("spec_tok_per_s") or 1e-9, 1e-9), 3)
        r["peak_vs_A"] = round((r.get("mlx_peak_gb") or 0) / max(a.get("mlx_peak_gb") or 1e-9, 1e-9), 3)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
