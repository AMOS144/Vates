"""验证 BlobLoadPrimitive：图内 pread 进 MLX 自有 buffer（无拷贝、无 per-expert 同步）。

对比 lazy frombuffer+mx.array：正确性 + 批量物化耗时（都 1 次 eval）。
"""
import os
import time

import mlx.core as mx
import numpy as np

import mlx_streaming.native_moe_ext as ext

HIDDEN, INTER, GROUP, BITS, NE = 2048, 512, 128, 2, 512
BLOB = "/tmp/cb_2bit_blob"
EXPERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")
LAYER, K = 15, int(os.environ.get("K_EXPERTS", "10"))

PROJS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER))
SEGS = []
for _proj, _out, _in in PROJS:
    _w, _g = _in * BITS // 32, _in // GROUP
    SEGS.append((_proj, "weight", np.uint32, (_out, _w), _out * _w * 4))
    SEGS.append((_proj, "scales", np.uint16, (_out, _g), _out * _g * 2))
    SEGS.append((_proj, "biases", np.uint16, (_out, _g), _out * _g * 2))
STRIDE = sum(s[4] for s in SEGS)


def view_expert(row_u8):
    """把 (STRIDE,) uint8 切成 9 个 typed 数组（全是 view，零拷贝）。"""
    out, off = {}, 0
    for proj, tensor, dt, shape, nb in SEGS:
        seg = row_u8[off:off + nb]
        if dt == np.uint32:
            a = seg.view(mx.uint32).reshape(shape)
        else:
            a = seg.view(mx.uint16).reshape(shape).view(mx.bfloat16)
        out[f"{proj}.{tensor}"] = a
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
    path = os.path.join(BLOB, f"layer{LAYER:02d}.blob")
    x = (mx.random.normal((1, HIDDEN)) * 0.1).astype(mx.float32)
    mx.eval(x)

    # 正确性：blob_load 视图 vs safetensors
    ids = mx.array([3, 7, 100], dtype=mx.uint32)
    raw = ext.blob_load(path, ids, STRIDE)        # (3, STRIDE) uint8
    mx.eval(raw)
    max_rel = 0.0
    for i, e in enumerate([3, 7, 100]):
        w = view_expert(raw[i])
        wr = mx.load(os.path.join(EXPERT_DIR, f"layer{LAYER:02d}_expert{e:03d}.safetensors"))
        y, yr = moe(x, w), moe(x, wr)
        mx.eval(y, yr)
        max_rel = max(max_rel, float(mx.max(mx.abs(y - yr)) / (mx.max(mx.abs(yr)) + 1e-9)))

    fd = os.open(path, os.O_RDONLY)
    ids_k = mx.array(list(range(K)), dtype=mx.uint32)
    # warm
    for _ in range(3):
        mx.eval(ext.blob_load(path, ids_k, STRIDE))

    # (a) blob_load 图内
    t = time.perf_counter()
    for _ in range(50):
        mx.eval(ext.blob_load(path, ids_k, STRIDE))
    blob_ms = (time.perf_counter() - t) / 50 * 1e3

    # (b) lazy frombuffer+mx.array（批量 1 次 eval）
    def lazy_load():
        arrs = []
        for e in range(K):
            raw_e = os.pread(fd, STRIDE, e * STRIDE)
            off = 0
            for proj, tensor, dt, shape, nb in SEGS:
                v = np.frombuffer(raw_e, dtype=dt, count=nb // np.dtype(dt).itemsize, offset=off).reshape(shape)
                arrs.append(mx.array(v))
                off += nb
        return arrs
    t = time.perf_counter()
    for _ in range(50):
        mx.eval(lazy_load())
    lazy_ms = (time.perf_counter() - t) / 50 * 1e3
    os.close(fd)

    print(json.dumps({
        "K": K,
        "correct_max_rel_diff": round(max_rel, 6),
        "correct": max_rel < 1e-4,
        "blob_load_ms": round(blob_ms, 3),
        "lazy_frombuffer_ms": round(lazy_ms, 3),
        "speedup": round(lazy_ms / max(blob_ms, 1e-9), 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
