"""运行 native MoE runner，汇总 sync/async staging + fused kernel 指标。"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "native" / "bench"
RUNNER = BENCH_DIR / "native_moe_runner"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _run_case(active: int, synthetic: bool) -> dict:
    cmd = [
        str(RUNNER),
        "--dir", os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/qwen_compute_buffers"),
        "--layer", os.environ.get("LAYER", "43"),
        "--active", str(active),
        "--steps", os.environ.get("STEPS", "16"),
        "--all-experts", os.environ.get("ALL_EXPERTS", "512"),
        "--hidden", os.environ.get("HIDDEN", "2048"),
        "--inter", os.environ.get("INTER", "512"),
        "--group", os.environ.get("GROUP", "128"),
        "--bits", os.environ.get("BITS", "6"),
        "--repeat", os.environ.get("REPEAT", "10"),
        "--synthetic", "1" if synthetic else "0",
    ]
    experts = os.environ.get("EXPERTS", "")
    trace = os.environ.get("TRACE", "")
    if experts:
        cmd.extend(["--experts", experts])
    if trace:
        cmd.extend(["--trace", trace])
    return json.loads(subprocess.check_output(cmd, text=True))


def main():
    subprocess.run(["make", "native_moe_runner"], cwd=BENCH_DIR, check=True)
    synthetic = os.environ.get("SYNTHETIC", "0") == "1"
    actives = [
        int(x)
        for x in os.environ.get("ACTIVES", str(_env_int("ACTIVE", 16))).split(",")
        if x.strip()
    ]
    rows = [_run_case(active, synthetic) for active in actives]
    print(json.dumps({
        "runner": str(RUNNER),
        "synthetic": synthetic,
        "rows": rows,
        "all_checksum_ok": all(bool(r.get("checksum_ok")) for r in rows),
        "all_async_faster": all(r["async_ms"] < r["sync_ms"] for r in rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
