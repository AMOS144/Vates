"""⑤ 决断微基准：取同样 8 个专家，比较两种读法的耗时。

A) per-expert：8 次 mx.load(小文件)（当前方案）
B) layerfile-partial：对每层单文件解析 safetensors 头，只 pread 选中 8 个专家的
   字节区间，再用 numpy/mlx 构造数组（1 次 open，按需读字节，无读放大）。

若 B 明显快于 A → ⑤(partial) 值得实现；否则瓶颈在读字节而非 open/解析，⑤ 无益。
"""
import os
import sys
import json
import glob
import time
import struct

import numpy as np
import mlx.core as mx

PER_EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts")
LAYER_DIR = os.environ.get("LAYER_DIR", "/tmp/mlx_qwen3_layerfiles")
LAYER = int(os.environ.get("PROBE_LAYER", "0"))
LAYER_PATH = os.path.join(LAYER_DIR, f"layer{LAYER:02d}.safetensors")
ITERS = int(os.environ.get("ITERS", "200"))

_ST_DTYPE = {"F16": np.float16, "F32": np.float32, "BF16": np.float16,
             "U32": np.uint32, "I32": np.int32, "U8": np.uint8}


def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    return hdr, 8 + n  # 头后是数据区起点


def partial_load_expert(path, hdr, data_start, e):
    """从每层单文件只读第 e 个专家在每个堆叠张量里的字节区间。"""
    out = {}
    with open(path, "rb") as f:
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            shape = meta["shape"]            # (E, ...)
            E = shape[0]
            per = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            dt = _ST_DTYPE[meta["dtype"]]
            itemsize = np.dtype(dt).itemsize
            stride = per * itemsize          # 单个专家字节数
            s0, _ = meta["data_offsets"]
            off = data_start + s0 + e * stride
            f.seek(off)
            buf = f.read(stride)
            arr = np.frombuffer(buf, dtype=dt).reshape(shape[1:])
            out[name] = mx.array(arr)
    return out


def bench_per_expert(expert_ids):
    files = [os.path.join(PER_EXPERT_DIR, f"layer{LAYER:02d}_expert{e:03d}.safetensors")
             for e in expert_ids]
    t0 = time.perf_counter()
    for _ in range(ITERS):
        picked = []
        for fp in files:
            w = mx.load(fp)
            mx.eval(w)
            picked.append(w)
    return time.perf_counter() - t0


def bench_partial(expert_ids):
    hdr, data_start = _read_header(LAYER_PATH)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        picked = []
        for e in expert_ids:
            w = partial_load_expert(LAYER_PATH, hdr, data_start, e)
            mx.eval(w)
            picked.append(w)
    return time.perf_counter() - t0


def main():
    if not os.path.exists(LAYER_PATH):
        print("先用 probe_layerfile build 生成每层单文件")
        sys.exit(1)
    expert_ids = [3, 17, 42, 60, 75, 88, 100, 120]
    # 正确性校验：partial 与 per-expert 应一致
    hdr, ds = _read_header(LAYER_PATH)
    a = partial_load_expert(LAYER_PATH, hdr, ds, expert_ids[0])
    b = mx.load(os.path.join(PER_EXPERT_DIR, f"layer{LAYER:02d}_expert{expert_ids[0]:03d}.safetensors"))
    ok = all(mx.array_equal(a[k], b[k]).item() for k in a.keys())
    t_pe = bench_per_expert(expert_ids)
    t_pa = bench_partial(expert_ids)
    print(json.dumps({
        "iters": ITERS, "experts_per_iter": len(expert_ids),
        "partial_correct": bool(ok),
        "per_expert_s": round(t_pe, 3),
        "partial_pread_s": round(t_pa, 3),
        "speedup_x": round(t_pe / t_pa, 2) if t_pa else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
