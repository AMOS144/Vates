"""零拷贝物化 spike：os.preadv 把 blob 专家字节直接散读进 MLX buffer（免 numpy→mx 拷贝）。

验证：
  1. 正确性：零拷贝路径 MoE 输出 vs safetensors 参考，0 误差。
  2. 提速：物化时间 从 ~1.3ms（frombuffer+mx.array）降到 ~pread 级。
纯 Python（np.array(mx, copy=False) 视图可写 + os.preadv 散读）。
"""
import os
import time

import mlx.core as mx
import numpy as np

HIDDEN, INTER, GROUP, BITS, NE = 2048, 512, 128, 2, 512
BLOB_DIR = "/tmp/cb_2bit_blob"
EXPERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")
K = int(os.environ.get("K_EXPERTS", "10"))
LAYER = 15

# blob 段：(proj, tensor, np_dtype, mx_dtype, shape)
PROJS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER))
SEGS = []
for _proj, _out, _in in PROJS:
    _w, _g = _in * BITS // 32, _in // GROUP
    SEGS.append((_proj, "weight", np.uint32, (_out, _w)))
    SEGS.append((_proj, "scales", np.uint16, (_out, _g)))
    SEGS.append((_proj, "biases", np.uint16, (_out, _g)))
STRIDE = sum(int(np.prod(s[3])) * np.dtype(s[2]).itemsize for s in SEGS)


def alloc_expert_buffers():
    """预分配一个专家的 9 个 MLX buffer（uint32/uint16），返回 (arrays, numpy 视图)。"""
    arrs, views = [], []
    for _proj, _tensor, dt, shape in SEGS:
        mxdt = mx.uint32 if dt == np.uint32 else mx.uint16
        a = mx.zeros(shape, dtype=mxdt)
        mx.eval(a)
        arrs.append(a)
        views.append(memoryview(np.array(a, copy=False)).cast("B"))
    return arrs, views


def zerocopy_load(fd, expert, arrs, views):
    """os.preadv 一次散读整个专家 blob 进 9 个 MLX buffer；scales/biases 位重解释 bf16。"""
    os.preadv(fd, views, expert * STRIDE)
    out = {}
    for (proj, tensor, _, _), a in zip(SEGS, arrs):
        out[f"{proj}.{tensor}"] = a.view(mx.bfloat16) if tensor in ("scales", "biases") else a
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
    x = (mx.random.normal((1, HIDDEN)) * 0.1).astype(mx.float32)
    mx.eval(x)
    fd = os.open(os.path.join(BLOB_DIR, f"layer{LAYER:02d}.blob"), os.O_RDONLY)

    # ---- 正确性 ----
    max_rel = 0.0
    for e in [3, 7, 100]:
        arrs, views = alloc_expert_buffers()
        wz = zerocopy_load(fd, e, arrs, views)
        wr = mx.load(os.path.join(EXPERT_DIR, f"layer{LAYER:02d}_expert{e:03d}.safetensors"))
        yz, yr = moe(x, wz), moe(x, wr)
        mx.eval(yz, yr)
        max_rel = max(max_rel, float(mx.max(mx.abs(yz - yr)) / (mx.max(mx.abs(yr)) + 1e-9)))

    # ---- 提速：零拷贝 vs 旧 frombuffer+mx.array（都 warm，K 个专家）----
    ids = list(range(K))
    pool = [alloc_expert_buffers() for _ in ids]   # 复用 buffer
    for _ in range(5):
        for (arrs, views), e in zip(pool, ids):
            zerocopy_load(fd, e, arrs, views)
        mx.eval([a for arrs, _ in pool for a in arrs])
    t = time.perf_counter()
    for _ in range(50):
        for (arrs, views), e in zip(pool, ids):
            zerocopy_load(fd, e, arrs, views)
        mx.eval([a for arrs, _ in pool for a in arrs])
    zerocopy_ms = (time.perf_counter() - t) / 50 * 1e3

    def old_materialize(raw):
        out, off = {}, 0
        for proj, tensor, dt, shape in SEGS:
            nb = int(np.prod(shape)) * np.dtype(dt).itemsize
            v = np.frombuffer(raw, dtype=dt, count=nb // np.dtype(dt).itemsize, offset=off).reshape(shape)
            a = mx.array(v)
            out[f"{proj}.{tensor}"] = a.view(mx.bfloat16) if tensor in ("scales", "biases") else a
            off += nb
        return out
    t = time.perf_counter()
    for _ in range(50):
        outs = [old_materialize(os.pread(fd, STRIDE, e * STRIDE)) for e in ids]
        mx.eval([a for o in outs for a in o.values()])
    old_ms = (time.perf_counter() - t) / 50 * 1e3
    os.close(fd)

    print(json.dumps({
        "K": K,
        "correct_max_rel_diff": round(max_rel, 6),
        "correct": max_rel < 1e-4,
        "zerocopy_materialize_ms": round(zerocopy_ms, 3),
        "old_materialize_ms": round(old_ms, 3),
        "speedup": round(old_ms / max(zerocopy_ms, 1e-9), 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
