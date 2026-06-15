"""测真冷 SSD 读带宽（调用方需在运行前 purge 页缓存）。

两种模式，各用不同层避免互相污染缓存：
  seq   : 顺序读整个 layerXX.gate_proj.weight.bin（峰值顺序带宽）
  slice : 走真实 native 路径，读 K 个专家的 gate/up/down 切片（实际访问模式）
"""
import os
import time

import numpy as np

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512
COMPUTE_DIR = os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/cb_2bit_g128")
SEQ_LAYER = int(os.environ.get("SEQ_LAYER", "10"))
SLICE_LAYER = int(os.environ.get("SLICE_LAYER", "30"))
K = int(os.environ.get("K_EXPERTS", "10"))


def seq_read(layer: int) -> dict:
    path = os.path.join(COMPUTE_DIR, f"layer{layer:02d}.gate_proj.weight.bin")
    size = os.path.getsize(path)
    t0 = time.perf_counter()
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(4 * 1024 * 1024)
            if not b:
                break
            n += len(b)
    dt = time.perf_counter() - t0
    return {"mode": "seq", "layer": layer, "MB": round(n / 1e6, 1),
            "ms": round(dt * 1e3, 2), "GBps": round(n / dt / 1e9, 2)}


def slice_read(layer: int, k: int) -> dict:
    total = 0
    t0 = time.perf_counter()
    for proj, out_dim, in_dim in (
        ("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER)):
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        base = os.path.join(COMPUTE_DIR, f"layer{layer:02d}.{proj}")
        w = np.memmap(base + ".weight.bin", dtype=np.uint32, mode="r", shape=(NUM_EXPERTS, out_dim, words))
        s = np.memmap(base + ".scales.bin", dtype=np.uint16, mode="r", shape=(NUM_EXPERTS, out_dim, groups))
        b = np.memmap(base + ".biases.bin", dtype=np.uint16, mode="r", shape=(NUM_EXPERTS, out_dim, groups))
        for e in range(k):
            total += np.asarray(w[e]).nbytes + np.asarray(s[e]).nbytes + np.asarray(b[e]).nbytes
        del w, s, b
    dt = time.perf_counter() - t0
    return {"mode": "slice", "layer": layer, "K": k, "MB": round(total / 1e6, 2),
            "ms": round(dt * 1e3, 2), "GBps": round(total / dt / 1e9, 2)}


def main():
    import json
    # seq 与 slice 用不同层，确保各自首读都是冷的
    print(json.dumps([seq_read(SEQ_LAYER), slice_read(SLICE_LAYER, K)], ensure_ascii=False))


if __name__ == "__main__":
    main()
