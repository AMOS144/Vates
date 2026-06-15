"""量化"常驻一层 + 下一层靠异步预取"能把 demand miss 压到多低。

对照组（其余参数一致，native 关闭，纯 streaming + resident pool）：
  off       : 不开 cross-layer 预取（基准 miss）
  ahead_b10 : AHEAD=1，每层预取预算=top_k(10)，ASYNC_PREFETCH=1
  ahead_b20 : AHEAD=1，每层预取预算=20（覆盖误预测冗余）

关注指标：spec_disk_loads / disk_load_ratio / spec_hit_rate / spec_prefetch_hits。
"""
import json
import os
import subprocess
import sys


COMMON = {
    "MODEL": "/tmp/qwen3_next_80b_4bit",
    "EXPERT_DIR": "/tmp/qwen3_next_experts",
    "EXPERT_SLOTS": "256",
    "RESIDENT_POOL": "1",
    "NATIVE_MOE": "0",
    "MTP_VERIFY_MODE": "batch",
    "MTP_ARRAY_COMMIT": "1",
    "K": "2",
    "MAXTOK": "48",
}

VARIANTS = [
    ("off", {
        "CROSS_LAYER_PREFETCH": "0",
    }),
    # 直接进 resident pool（受每层池容量限制，不走 2048 buffer，开句柄少）
    ("ahead_b10_pool", {
        "CROSS_LAYER_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH_AHEAD": "1",
        "CROSS_LAYER_PREFETCH_MULT": "2",
        "ASYNC_PREFETCH": "0",
        "STAGE_PREFETCH_PER_LAYER_BUDGET": "10",
        "STAGE_PREFETCH_GLOBAL_BUDGET": "480",
    }),
    # async buffer 路径，但把 buffer 压小，验证是否是惰性句柄累积导致崩溃
    ("ahead_b10_async", {
        "CROSS_LAYER_PREFETCH": "1",
        "CROSS_LAYER_PREFETCH_AHEAD": "1",
        "CROSS_LAYER_PREFETCH_MULT": "2",
        "ASYNC_PREFETCH": "1",
        "PREFETCH_BUFFER_EXPERTS": "128",
        "STAGE_PREFETCH_PER_LAYER_BUDGET": "10",
        "STAGE_PREFETCH_GLOBAL_BUDGET": "480",
    }),
]

FIELDS = [
    "spec_tok_per_s", "baseline_tok_per_s", "speedup",
    "spec_disk_loads", "baseline_disk_loads", "disk_load_ratio",
    "spec_hit_rate", "spec_prefetch_loads", "spec_prefetch_hits",
    "mlx_peak_gb", "exact_match",
]


def _run(name: str, overrides: dict) -> dict:
    env = os.environ.copy()
    env.update(COMMON)
    env.update(overrides)
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"],
        env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return {"variant": name, "error": " | ".join(tail)}
    rec = json.loads(proc.stdout)
    row = {"variant": name}
    row.update({k: rec.get(k) for k in FIELDS})
    return row


def main():
    rows = [_run(name, ov) for name, ov in VARIANTS]
    off = rows[0]
    base_miss = off.get("spec_disk_loads") or 1
    for r in rows:
        if r.get("error"):
            continue
        miss = r.get("spec_disk_loads") or 0
        r["miss_vs_off"] = round(miss / max(base_miss, 1), 3)
    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
