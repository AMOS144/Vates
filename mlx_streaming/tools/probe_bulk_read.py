"""冷读对照：当前 mmap 切片 vs bulk pread（连续/散列），看读法能否把带宽拉到接近顺序峰值。

三种模式各用不同层，确保单次 purge 后每个首读都是冷的：
  mmap        : np.memmap + np.asarray（当前 native 路径）
  bulk_contig : 专家 0..K-1（文件内连续），每个张量一次大 pread
  bulk_scatter: 随机 K 个专家 id，每个专家切片一次 pread（真实 decode 访问模式）
"""
import json
import os
import random
import time

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512
COMPUTE_DIR = os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/cb_2bit_g128")
K = int(os.environ.get("K_EXPERTS", "10"))

# 每投影的 (out_dim, in_dim)
PROJS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER))


def _slice_bytes(out_dim: int, in_dim: int):
    w = out_dim * (in_dim * BITS // 32) * 4      # uint32 weight
    sb = out_dim * (in_dim // GROUP) * 2          # uint16 scales / biases（各一份）
    return w, sb


def mmap_read(layer: int, ids):
    import numpy as np
    total = 0
    t0 = time.perf_counter()
    for proj, out_dim, in_dim in PROJS:
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        base = os.path.join(COMPUTE_DIR, f"layer{layer:02d}.{proj}")
        w = np.memmap(base + ".weight.bin", dtype=np.uint32, mode="r", shape=(NUM_EXPERTS, out_dim, words))
        s = np.memmap(base + ".scales.bin", dtype=np.uint16, mode="r", shape=(NUM_EXPERTS, out_dim, groups))
        b = np.memmap(base + ".biases.bin", dtype=np.uint16, mode="r", shape=(NUM_EXPERTS, out_dim, groups))
        for e in ids:
            total += np.asarray(w[e]).nbytes + np.asarray(s[e]).nbytes + np.asarray(b[e]).nbytes
        del w, s, b
    return total, time.perf_counter() - t0


def bulk_read(layer: int, ids, contiguous: bool):
    total = 0
    t0 = time.perf_counter()
    for proj, out_dim, in_dim in PROJS:
        wb, sbb = _slice_bytes(out_dim, in_dim)
        for tensor, slot in (("weight", wb), ("scales", sbb), ("biases", sbb)):
            path = os.path.join(COMPUTE_DIR, f"layer{layer:02d}.{proj}.{tensor}.bin")
            fd = os.open(path, os.O_RDONLY)
            try:
                if contiguous:
                    # ids 连续 → 一次大 pread
                    off = min(ids) * slot
                    data = os.pread(fd, slot * len(ids), off)
                    total += len(data)
                else:
                    for e in ids:
                        data = os.pread(fd, slot, e * slot)
                        total += len(data)
            finally:
                os.close(fd)
    return total, time.perf_counter() - t0


def _report(mode, layer, total, dt):
    return {"mode": mode, "layer": layer, "MB": round(total / 1e6, 2),
            "ms": round(dt * 1e3, 2), "GBps": round(total / dt / 1e9, 3)}


def main():
    contig_ids = list(range(K))
    rng = random.Random(0)
    scatter_ids = sorted(rng.sample(range(NUM_EXPERTS), K))
    rows = [
        _report("mmap", 5, *mmap_read(5, scatter_ids)),
        _report("bulk_contig", 15, *bulk_read(15, contig_ids, True)),
        _report("bulk_scatter", 25, *bulk_read(25, scatter_ids, False)),
    ]
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
