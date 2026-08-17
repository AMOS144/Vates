import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "native" / "bench"
RUNNER = BENCH_DIR / "metal" / "native_moe_runner"


def test_native_moe_runner_synthetic_smoke():
    subprocess.run([
        "make", f"PYTHON={sys.executable}", "metal/native_moe_runner",
    ], cwd=BENCH_DIR, check=True)
    out = subprocess.check_output([
        str(RUNNER),
        "--synthetic", "1",
        "--steps", "3",
        "--active", "4",
        "--hidden", "64",
        "--inter", "32",
        "--group", "32",
        "--bits", "6",
        "--repeat", "2",
    ], text=True)
    rec = json.loads(out)
    for key in ("stage_ms", "kernel_ms", "total_ms", "async", "checksum"):
        assert key in rec
    assert rec["synthetic"] is True
    assert rec["steps"] == 3
    assert rec["active"] == 4
    assert rec["stage_ms"] >= 0
    assert rec["kernel_ms"] >= 0
    assert rec["total_ms"] >= 0
    assert rec["checksum_ok"] is True
