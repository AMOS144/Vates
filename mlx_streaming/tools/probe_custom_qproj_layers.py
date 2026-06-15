"""只针对高价值 6-bit 层的 projection custom qlinear de-risk。

当前最佳混合精度目录里 layer43/47 是 6-bit。本 probe 用这些层的真实投影维度：
- gate/up: hidden -> moe_intermediate = 2048 -> 512
- down: moe_intermediate -> hidden = 512 -> 2048

它调用 C++ `qlinear_bench`，比较 custom Metal qlinear 与 MLX C++ quantized_matmul loop。
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "native" / "bench"
BENCH = Path(os.environ.get("QLINEAR_BENCH", str(BENCH_DIR / "qlinear_bench")))
BITS = int(os.environ.get("BITS", "6"))
GROUP = int(os.environ.get("GROUP", "128"))
EXPERTS = [int(x) for x in os.environ.get("EXPERTS_LIST", "1,2,4,8").split(",")]
REPEAT = int(os.environ.get("REPEAT", "30"))


def _run_case(name: str, in_dim: int, out_dim: int, experts: int) -> dict:
    out = subprocess.check_output([
        str(BENCH),
        "--in", str(in_dim),
        "--out", str(out_dim),
        "--group", str(GROUP),
        "--bits", str(BITS),
        "--experts", str(experts),
        "--repeat", str(REPEAT),
    ], text=True)
    rec = json.loads(out)
    rec["projection"] = name
    rec["pass_70pct"] = rec["custom_vs_mlx"] >= 0.70
    rec["pass_error"] = rec["max_abs"] < 1e-4
    return rec


def main():
    subprocess.run(["make", "qlinear_bench"], cwd=BENCH_DIR, check=True)
    rows = []
    for experts in EXPERTS:
        rows.append(_run_case("gate_or_up", 2048, 512, experts))
        rows.append(_run_case("down", 512, 2048, experts))
    print(json.dumps({
        "bits": BITS,
        "group": GROUP,
        "repeat": REPEAT,
        "rows": rows,
        "all_pass_70pct": all(r["pass_70pct"] for r in rows),
        "all_pass_error": all(r["pass_error"] for r in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
