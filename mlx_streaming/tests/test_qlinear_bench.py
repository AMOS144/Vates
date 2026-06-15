import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "native" / "bench"
BENCH = BENCH_DIR / "qlinear_bench"


def test_qlinear_bench_runs_6bit_smoke():
    subprocess.run(["make", "qlinear_bench"], cwd=BENCH_DIR, check=True)
    out = subprocess.check_output([
        str(BENCH),
        "--in", "128",
        "--out", "64",
        "--group", "64",
        "--bits", "6",
        "--experts", "2",
        "--repeat", "3",
    ], text=True)
    rec = json.loads(out)
    assert rec["bits"] == 6
    assert rec["experts"] == 2
    assert rec["max_abs"] < 1e-4
    assert rec["custom_vs_mlx"] > 0.5
