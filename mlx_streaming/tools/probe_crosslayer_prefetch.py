"""端到端 A(plain 中池) vs C(同 token 跨层预取) × cap∈{64,96} × AHEAD∈{1,2}。

判定：C 的 tok/s 是否 > A（同 cap）、hit 是否提升、ready_on_time 率是否高（窗口够）、
内存仍是中池水平、exact_match 与 A 一致。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERT_DIR = os.environ.get("EXPERT_DIR", os.path.join(ROOT, "models", "qwen3_next_experts_2bit_g128"))
BLOB_DIR = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")

COMMON = {
    "MODEL": "/tmp/qwen3_next_80b_4bit",
    "EXPERT_DIR": EXPERT_DIR,
    "RESIDENT_POOL": "1",
    "MTP_VERIFY_MODE": "batch",
    "MTP_ARRAY_COMMIT": "1",
    "K": "2",
    "MAXTOK": os.environ.get("MAXTOK", "64"),
    "NATIVE_MOE": "0",
    "STREAM_BLOB": "0",
}
CAPS = [int(x) for x in os.environ.get("CAPS", "64,96").split(",")]
AHEADS = [int(x) for x in os.environ.get("AHEADS", "1,2").split(",")]
FIELDS = ["spec_tok_per_s", "spec_hit_rate", "mlx_peak_gb", "rss_gb",
          "exact_match", "n_mismatch", "bg_stats", "window_prof"]


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
    rot, nr = bg.get("ready_on_time", 0), bg.get("not_ready", 0)
    row["ready_rate"] = round(rot / max(rot + nr, 1), 3) if (rot + nr) else None
    return row


def main():
    rows = []
    for cap in CAPS:
        a = _run(f"A_plain_{cap}", {
            "EXPERT_SLOTS": str(cap), "STREAM_BLOB_LOADER": "1",
            "STREAM_BLOB_BG": "0", "BLOB_DIR": BLOB_DIR, "CROSS_LAYER_PREFETCH": "0"})
        rows.append(a)
        base = a.get("spec_tok_per_s") or 1e-9
        for ahead in AHEADS:
            c = _run(f"C_ahead{ahead}_{cap}", {
                "EXPERT_SLOTS": str(cap), "STREAM_BLOB_BG": "1", "BLOB_DIR": BLOB_DIR,
                "STREAM_BLOB_WINDOW": str(ahead + 1), "CROSS_LAYER_PREFETCH": "1",
                "CROSS_LAYER_PREFETCH_AHEAD": str(ahead), "CROSS_LAYER_PREFETCH_MULT": "2",
                "STREAM_BLOB_BG_BUDGET": "32"})
            c["tok_s_vs_A"] = round((c.get("spec_tok_per_s") or 0) / base, 3)
            rows.append(c)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
