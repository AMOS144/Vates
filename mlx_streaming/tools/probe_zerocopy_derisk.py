"""M0 de-risk：blob 字节 → mx.array(quantized) → MLX quantized_matmul。

验证三件事：
  1. 正确性：blob 路径 MoE 输出与参考(safetensors 直接 load)逐专家一致。
  2. 不崩：走正常 mx.array + quantized_matmul（不碰自定义 primitive / 裸 MTLBuffer）。
  3. 重叠：后台线程读下一层专家，和主线程当前层计算能否重叠（wall ≈ max 而非 sum）。
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512
BLOB_DIR = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")
EXPERT_DIR = os.environ.get(
    "EXPERT_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128"))
K = int(os.environ.get("K_EXPERTS", "10"))
PROJS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER))


def _seg_table():
    """blob 内每段 (proj, tensor, dtype, shape, nbytes)，按打包顺序。"""
    segs = []
    for proj, out_dim, in_dim in PROJS:
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        segs.append((proj, "weight", np.uint32, (out_dim, words), out_dim * words * 4))
        segs.append((proj, "scales", np.uint16, (out_dim, groups), out_dim * groups * 2))
        segs.append((proj, "biases", np.uint16, (out_dim, groups), out_dim * groups * 2))
    return segs, sum(s[4] for s in segs)


SEGS, STRIDE = _seg_table()


def load_expert_from_blob(fd: int, e: int) -> dict:
    """pread 一个专家 blob，零拷贝 frombuffer + mx.array，scales/biases 位重解释为 bf16。"""
    raw = os.pread(fd, STRIDE, e * STRIDE)
    out = {}
    off = 0
    for proj, tensor, dt, shape, nb in SEGS:
        view = np.frombuffer(raw, dtype=dt, count=nb // np.dtype(dt).itemsize, offset=off).reshape(shape)
        arr = mx.array(view)
        if tensor in ("scales", "biases"):
            arr = arr.view(mx.bfloat16)
        out[f"{proj}.{tensor}"] = arr
        off += nb
    return out


def moe(x, w):
    g = mx.quantized_matmul(x, w["gate_proj.weight"], w["gate_proj.scales"], w["gate_proj.biases"],
                            transpose=True, group_size=GROUP, bits=BITS)
    u = mx.quantized_matmul(x, w["up_proj.weight"], w["up_proj.scales"], w["up_proj.biases"],
                            transpose=True, group_size=GROUP, bits=BITS)
    a = g * mx.sigmoid(g) * u
    return mx.quantized_matmul(a, w["down_proj.weight"], w["down_proj.scales"], w["down_proj.biases"],
                               transpose=True, group_size=GROUP, bits=BITS)


def main():
    import json
    layer = 15
    ids = list(range(K))
    x = (mx.random.normal((1, HIDDEN)) * 0.1).astype(mx.float32)
    mx.eval(x)

    # ---- 正确性：blob 路径 vs safetensors 参考 ----
    fd = os.open(os.path.join(BLOB_DIR, f"layer{layer:02d}.blob"), os.O_RDONLY)
    max_rel = 0.0
    try:
        for e in ids:
            wb = load_expert_from_blob(fd, e)
            wr = mx.load(os.path.join(EXPERT_DIR, f"layer{layer:02d}_expert{e:03d}.safetensors"))
            yb, yr = moe(x, wb), moe(x, wr)
            mx.eval(yb, yr)
            d = float(mx.max(mx.abs(yb - yr)) / (mx.max(mx.abs(yr)) + 1e-9))
            max_rel = max(max_rel, d)
    finally:
        os.close(fd)

    # ---- 重叠：后台读下一层 K 专家 vs 主线程算当前层 ----
    fd2 = os.open(os.path.join(BLOB_DIR, "layer25.blob"), os.O_RDONLY)
    cur = [load_expert_from_blob(fd2, e) for e in ids]  # 预载“当前层”
    mx.eval([v for w in cur for v in w.values()])

    def read_next():
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(lambda e: load_expert_from_blob(fd2, e), ids))

    # 只算
    t0 = time.perf_counter()
    for _ in range(10):
        mx.eval([moe(x, w) for w in cur])
    compute_s = (time.perf_counter() - t0) / 10
    # 只读
    t0 = time.perf_counter()
    for _ in range(10):
        read_next()
    read_s = (time.perf_counter() - t0) / 10
    # 重叠：后台读 + 主线程算
    t0 = time.perf_counter()
    for _ in range(10):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(read_next)
            mx.eval([moe(x, w) for w in cur])
            fut.result()
    overlap_s = (time.perf_counter() - t0) / 10
    os.close(fd2)

    print(json.dumps({
        "K": K,
        "correctness_max_rel_diff": round(max_rel, 6),
        "correct": max_rel < 1e-3,
        "compute_ms": round(compute_s * 1e3, 3),
        "read_ms": round(read_s * 1e3, 3),
        "overlap_ms": round(overlap_s * 1e3, 3),
        "sum_ms": round((compute_s + read_s) * 1e3, 3),
        "overlap_saves_ms": round((compute_s + read_s - overlap_s) * 1e3, 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
