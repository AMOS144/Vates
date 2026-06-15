"""冷读对照：老格式散读 vs blob 串行 vs blob 并行。调用方需先 purge 页缓存。

各模式用不同层，确保单次 purge 后首读都冷：
  old_scatter  : 老 compute buffer，9 文件 × K 散读（基线，~0.85 GB/s）
  blob_serial  : 每专家 1 次 pread，串行
  blob_parallel: 每专家 1 次 pread，线程池并行（提高 SSD 队列深度）
"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512
SRC = os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/cb_2bit_g128")
BLOB_DIR = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")
K = int(os.environ.get("K_EXPERTS", "10"))
WORKERS = int(os.environ.get("READ_WORKERS", "8"))
PROJS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER))


def _stride():
    total = 0
    for _, out_dim, in_dim in PROJS:
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        total += out_dim * words * 4 + 2 * (out_dim * groups * 2)
    return total


def old_scatter(layer: int, ids):
    total = 0
    t0 = time.perf_counter()
    for proj, out_dim, in_dim in PROJS:
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        base = os.path.join(SRC, f"layer{layer:02d}.{proj}")
        wb = out_dim * words * 4
        sb = out_dim * groups * 2
        for tensor, nb in (("weight", wb), ("scales", sb), ("biases", sb)):
            fd = os.open(f"{base}.{tensor}.bin", os.O_RDONLY)
            try:
                for e in ids:
                    total += len(os.pread(fd, nb, e * nb))
            finally:
                os.close(fd)
    return total, time.perf_counter() - t0


def blob_serial(layer: int, ids, stride: int):
    path = os.path.join(BLOB_DIR, f"layer{layer:02d}.blob")
    fd = os.open(path, os.O_RDONLY)
    total = 0
    t0 = time.perf_counter()
    try:
        for e in ids:
            total += len(os.pread(fd, stride, e * stride))
    finally:
        os.close(fd)
    return total, time.perf_counter() - t0


def blob_parallel(layer: int, ids, stride: int, workers: int):
    path = os.path.join(BLOB_DIR, f"layer{layer:02d}.blob")
    fd = os.open(path, os.O_RDONLY)
    t0 = time.perf_counter()
    try:
        def rd(e):
            return len(os.pread(fd, stride, e * stride))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            total = sum(ex.map(rd, ids))
    finally:
        os.close(fd)
    return total, time.perf_counter() - t0


def _rep(mode, layer, total, dt, extra=None):
    r = {"mode": mode, "layer": layer, "MB": round(total / 1e6, 2),
         "ms": round(dt * 1e3, 2), "GBps": round(total / dt / 1e9, 3)}
    if extra:
        r.update(extra)
    return r


def main():
    stride = _stride()
    rng = random.Random(0)
    ids = sorted(rng.sample(range(NUM_EXPERTS), K))
    rows = [
        _rep("old_scatter", 5, *old_scatter(5, ids)),
        _rep("blob_serial", 15, *blob_serial(15, ids, stride)),
        _rep("blob_parallel", 25, *blob_parallel(25, ids, stride, WORKERS), {"workers": WORKERS}),
    ]
    print(json.dumps({"K": K, "stride_KB": round(stride / 1024, 1), "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
