"""对比 baseline 与 native MLX MoE runtime 的实机参数矩阵。"""
import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_EXPERT_DIR = os.path.join(
    ROOT, "models", "qwen3_next_experts_bnd12_l43_l47_6_g128")


VARIANTS = [
    ("baseline", {
        "NATIVE_MOE": "0",
        "CROSS_LAYER_PREFETCH": "0",
    }),
    ("native_uncached", {
        "NATIVE_MOE": "1",
        "NATIVE_MOE_MLX_OP": "1",
        "NATIVE_MOE_SLOT_POOL": "0",
        "NATIVE_MOE_STAGE_CACHE": "0",
        "NATIVE_MOE_STAGE_BUNDLE_CACHE": "0",
        "NATIVE_MOE_STAGE_PREFETCH": "0",
        "CROSS_LAYER_PREFETCH": "0",
    }),
    ("native_slot_pool", {
        "NATIVE_MOE": "1",
        "NATIVE_MOE_MLX_OP": "1",
        "NATIVE_MOE_SLOT_POOL": "1",
        "NATIVE_MOE_SLOT_CAP": "96",
        "NATIVE_MOE_STAGE_PREFETCH": "0",
        "CROSS_LAYER_PREFETCH": "0",
    }),
    ("native_cache", {
        "NATIVE_MOE": "1",
        "NATIVE_MOE_MLX_OP": "1",
        "NATIVE_MOE_SLOT_POOL": "0",
        "NATIVE_MOE_STAGE_CACHE": "1",
        "NATIVE_MOE_STAGE_BUNDLE_CACHE": "1",
        "NATIVE_MOE_STAGE_PREFETCH": "0",
        "CROSS_LAYER_PREFETCH": "0",
    }),
    ("native_cache_prefetch4", {
        "NATIVE_MOE": "1",
        "NATIVE_MOE_MLX_OP": "1",
        "NATIVE_MOE_SLOT_POOL": "0",
        "NATIVE_MOE_STAGE_CACHE": "1",
        "NATIVE_MOE_STAGE_BUNDLE_CACHE": "1",
        "NATIVE_MOE_STAGE_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH_MULT": "2",
        "STAGE_PREFETCH_PER_LAYER_BUDGET": "4",
        "STAGE_PREFETCH_GLOBAL_BUDGET": "64",
    }),
    ("native_cache_prefetch8", {
        "NATIVE_MOE": "1",
        "NATIVE_MOE_MLX_OP": "1",
        "NATIVE_MOE_SLOT_POOL": "0",
        "NATIVE_MOE_STAGE_CACHE": "1",
        "NATIVE_MOE_STAGE_BUNDLE_CACHE": "1",
        "NATIVE_MOE_STAGE_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH_MULT": "2",
        "STAGE_PREFETCH_PER_LAYER_BUDGET": "8",
        "STAGE_PREFETCH_GLOBAL_BUDGET": "96",
    }),
]


def _run(name: str, overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.setdefault("MODEL", "/tmp/qwen3_next_80b_4bit")
    env.setdefault("EXPERT_DIR", DEFAULT_EXPERT_DIR)
    env.setdefault("COMPUTE_BUFFER_DIR", "/tmp/qwen_compute_buffers")
    env.setdefault("EXPERT_SLOTS", "96")
    env.setdefault("MAXTOK", "32")
    env.setdefault("K", "3")
    env.setdefault("MTP_VERIFY_MODE", "batch")
    env.setdefault("MTP_ARRAY_COMMIT", "1")
    env.update(overrides)
    out = subprocess.check_output(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"],
        env=env,
        text=True,
    )
    rec = json.loads(out)
    rec["variant"] = name
    rec["env"] = {
        k: env[k]
        for k in (
            "MODEL", "EXPERT_DIR", "COMPUTE_BUFFER_DIR", "EXPERT_SLOTS",
            "MAXTOK", "K", "NATIVE_MOE", "NATIVE_MOE_MLX_OP",
            "NATIVE_MOE_SLOT_POOL", "NATIVE_MOE_SLOT_CAP",
            "NATIVE_MOE_STAGE_CACHE", "NATIVE_MOE_STAGE_BUNDLE_CACHE",
            "NATIVE_MOE_STAGE_PREFETCH", "CROSS_LAYER_PREFETCH",
            "STAGE_PREFETCH_PER_LAYER_BUDGET", "STAGE_PREFETCH_GLOBAL_BUDGET",
        )
        if k in env
    }
    return rec


def main():
    subprocess.run(["make", "-C", "native/ext", "native_moe_ext"], check=True)
    rows = [_run(name, overrides) for name, overrides in VARIANTS]
    baseline = rows[0]
    summary = []
    for rec in rows:
        summary.append({
            "variant": rec["variant"],
            "exact_match": rec.get("exact_match"),
            "spec_tok_per_s": rec.get("spec_tok_per_s"),
            "delta_tok_s": round(rec.get("spec_tok_per_s", 0) - baseline.get("spec_tok_per_s", 0), 4),
            "speedup_vs_greedy": rec.get("speedup"),
            "t_verify_s": rec.get("t_verify_s"),
            "t_sync_s": rec.get("t_sync_s"),
            "disk_load_ratio": rec.get("disk_load_ratio"),
            "spec_hit_rate": rec.get("spec_hit_rate"),
            "mlx_peak_gb": rec.get("mlx_peak_gb"),
            "rss_gb": rec.get("rss_gb"),
            "native_stage_cache": rec.get("native_stage_cache"),
        })
    print(json.dumps({
        "summary": summary,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
