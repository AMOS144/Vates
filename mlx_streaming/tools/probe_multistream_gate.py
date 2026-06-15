"""Gating 测试:后台线程在独立 stream 上物化+eval，能否与主线程计算并发重叠且不崩。

这是"预取队列 + 主队列"架构的成败前提。
"""
import os
import threading
import time

import mlx.core as mx
import numpy as np

HIDDEN, INTER, GROUP, BITS = 2048, 512, 128, 2
BLOB = "/tmp/cb_2bit_blob"
STRIDE = (INTER * (HIDDEN * BITS // 32) * 4 + 2 * INTER * (HIDDEN // GROUP) * 2) * 2 \
    + HIDDEN * (INTER * BITS // 32) * 4 + 2 * HIDDEN * (INTER // GROUP) * 2
K = 10
ITERS = 30


def main():
    import json
    s2 = mx.new_stream(mx.default_device())
    fd = os.open(os.path.join(BLOB, "layer15.blob"), os.O_RDONLY)

    # 主线程计算负载：模拟一层 matmul
    W = mx.random.normal((HIDDEN, HIDDEN)).astype(mx.float32)
    x = mx.random.normal((1, HIDDEN)).astype(mx.float32)
    mx.eval(W, x)

    def compute_once():
        y = x
        for _ in range(8):
            y = mx.tanh(y @ W)
        return y

    # 后台预取负载：在 s2 上物化 K 个专家（uint32 weight 段近似）
    def prefetch_once(stream):
        with mx.stream(stream):
            arrs = []
            for e in range(K):
                raw = os.pread(fd, STRIDE, e * STRIDE)
                v = np.frombuffer(raw, dtype=np.uint32, count=STRIDE // 4)
                arrs.append(mx.array(v))
            mx.eval(arrs)

    # 预热
    mx.eval(compute_once())
    prefetch_once(s2)

    # 只算
    t = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(compute_once())
    compute_ms = (time.perf_counter() - t) / ITERS * 1e3

    # 只预取(主线程, s2)
    t = time.perf_counter()
    for _ in range(ITERS):
        prefetch_once(s2)
    prefetch_ms = (time.perf_counter() - t) / ITERS * 1e3

    # 并发:后台线程预取(s2) + 主线程计算(默认 stream)
    crashed = [None]

    def bg():
        try:
            for _ in range(ITERS):
                prefetch_once(s2)
        except Exception as e:
            crashed[0] = repr(e)

    t = time.perf_counter()
    th = threading.Thread(target=bg)
    th.start()
    for _ in range(ITERS):
        mx.eval(compute_once())
    th.join()
    concurrent_ms = (time.perf_counter() - t) / ITERS * 1e3
    os.close(fd)

    print(json.dumps({
        "crashed": crashed[0],
        "compute_ms": round(compute_ms, 3),
        "prefetch_ms": round(prefetch_ms, 3),
        "concurrent_ms": round(concurrent_ms, 3),
        "sum_ms": round(compute_ms + prefetch_ms, 3),
        "overlap_factor": round((compute_ms + prefetch_ms) / max(concurrent_ms, 1e-9), 2),
        "verdict": "重叠有效" if concurrent_ms < 0.85 * (compute_ms + prefetch_ms) else "几乎不重叠",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
