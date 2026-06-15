"""对比 per-expert mx.load 与自定义 pack range read 的微基准。"""
import json
import os
import random
import subprocess
import time

import mlx.core as mx

from mlx_streaming.prep.pack_expert_ranges import pack_layer

EXPERT_DIR = os.environ.get(
    "EXPERT_DIR",
    "/Users/amos/project/flash-moe/mlx-streaming-moe/models/qwen3_next_experts_bnd12_l43_l47_6_g128",
)
PACK_DIR = os.environ.get("EXPERT_PACK_DIR", os.path.join(EXPERT_DIR, "layer_packs"))
LAYER = int(os.environ.get("LAYER", "43"))
SIZES = [int(x) for x in os.environ.get("SIZES", "1,2,4,8,16").split(",")]
REPEAT = int(os.environ.get("REPEAT", "8"))
SEED = int(os.environ.get("SEED", "0"))
LOADER = os.environ.get(
    "PACK_LOADER",
    "/Users/amos/project/flash-moe/mlx-streaming-moe/native/bench/pack_loader",
)
CHECKSUM = os.environ.get("PACK_CHECKSUM", "0") == "1"


def _ensure_pack(num_experts: int):
    pack = os.path.join(PACK_DIR, f"layer{LAYER:02d}.pack")
    idx = os.path.join(PACK_DIR, f"layer{LAYER:02d}.idx")
    if not os.path.exists(pack) or not os.path.exists(idx):
        pack_layer(EXPERT_DIR, PACK_DIR, LAYER, num_experts)
    return pack, idx


def _small_load(experts: list[int], repeat: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeat):
        for e in experts:
            path = os.path.join(EXPERT_DIR, f"layer{LAYER:02d}_expert{e:03d}.safetensors")
            rec = mx.load(path)
            mx.eval(list(rec.values()))
    return (time.perf_counter() - t0) * 1000


def _pack_load(pack: str, idx: str, experts: list[int], repeat: int) -> dict:
    cmd = [
        LOADER, "bench",
        "--pack", pack,
        "--index", idx,
        "--experts", ",".join(str(e) for e in experts),
        "--repeat", str(repeat),
    ]
    if not CHECKSUM:
        cmd.append("--no-checksum")
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def main():
    with open(os.path.join(EXPERT_DIR, "_split_meta.json")) as f:
        meta = json.load(f)
    num_experts = int(meta["dims"]["num_experts"])
    pack, idx = _ensure_pack(num_experts)
    rng = random.Random(SEED)
    rows = []
    for n in SIZES:
        experts = rng.sample(range(num_experts), n)
        small_avg = _small_load(experts, REPEAT) / REPEAT
        pack_rec = _pack_load(pack, idx, experts, REPEAT)
        pack_avg = float(pack_rec["elapsed_ms"]) / REPEAT
        rows.append({
            "n_experts": n,
            "small_files_ms": round(small_avg, 4),
            "pack_read_ms": round(pack_avg, 4),
            "speedup": round(small_avg / max(pack_avg, 1e-9), 3),
        })
    print(json.dumps({
        "expert_dir": EXPERT_DIR,
        "layer": LAYER,
        "repeat": REPEAT,
        "pack": pack,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
