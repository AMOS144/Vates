"""汇总纯 Metal staging 下 gate/up/down 三投影耗时。

这是 fused MoE 前的低成本 de-risk：复用 `native/bench/metal_staging_bench`，
估算 mmap->MTLBuffer->custom qlinear 的三投影链路下界。
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "native" / "bench" / "metal_staging_bench"
BUF_DIR = Path(os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/qwen_compute_buffers"))
LAYER = int(os.environ.get("LAYER", "43"))
ACTIVE = int(os.environ.get("ACTIVE", "16"))
BITS = int(os.environ.get("BITS", "6"))
TILE = int(os.environ.get("TILE", "4"))
REPEAT = int(os.environ.get("REPEAT", "50"))


def _run(proj: str, in_dim: int, out_dim: int, async_mode: bool) -> dict:
    cmd = [
        str(BENCH),
        "--weight", str(BUF_DIR / f"layer{LAYER:02d}.{proj}.weight.bin"),
        "--scales", str(BUF_DIR / f"layer{LAYER:02d}.{proj}.scales.bin"),
        "--biases", str(BUF_DIR / f"layer{LAYER:02d}.{proj}.biases.bin"),
        "--active", str(ACTIVE),
        "--in", str(in_dim),
        "--out", str(out_dim),
        "--group", "128",
        "--bits", str(BITS),
        "--tile", str(TILE),
        "--repeat", str(REPEAT),
    ]
    if async_mode:
        cmd.append("--async")
    rec = json.loads(subprocess.check_output(cmd, text=True))
    rec["proj"] = proj
    return rec


def main():
    subprocess.run(["make", "metal_staging_bench"], cwd=ROOT / "native" / "bench", check=True)
    rows = []
    for async_mode in (False, True):
        parts = [
            _run("gate_proj", 2048, 512, async_mode),
            _run("up_proj", 2048, 512, async_mode),
            _run("down_proj", 512, 2048, async_mode),
        ]
        rows.append({
            "async": async_mode,
            "active": ACTIVE,
            "bits": BITS,
            "tile": TILE,
            "gate_ms": parts[0]["metal_ms"],
            "up_ms": parts[1]["metal_ms"],
            "down_ms": parts[2]["metal_ms"],
            "sum_ms_no_activation": round(sum(p["metal_ms"] for p in parts), 4),
            "parts": parts,
        })
    print(json.dumps({
        "layer": LAYER,
        "repeat": REPEAT,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
