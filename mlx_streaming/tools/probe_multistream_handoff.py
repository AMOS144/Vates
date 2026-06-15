"""Gating 测试 C:后台 s2 物化「私有」专家 array(与主线程计算重叠)→ 交接 → 主线程拷进池。

验证:正确性 + 重叠 + 不崩。这是可行架构(避开 Test B 的共享写问题)。
"""
import os
import threading
import time

import mlx.core as mx
import numpy as np

HIDDEN, INTER, GROUP, BITS = 2048, 512, 128, 2
BLOB = "/tmp/cb_2bit_blob"
GU_W = INTER * (HIDDEN * BITS // 32)          # uint32 元素数
K, CAP, N = 10, 32, 30


def main():
    import json
    s2 = mx.new_stream(mx.default_device())
    fd = os.open(os.path.join(BLOB, "layer15.blob"), os.O_RDONLY)
    stride = (INTER * (HIDDEN * BITS // 32) * 4 + 2 * INTER * (HIDDEN // GROUP) * 2) * 2 \
        + HIDDEN * (INTER * BITS // 32) * 4 + 2 * HIDDEN * (INTER // GROUP) * 2

    # 主线程拥有的池(gate.weight 段示意)
    pool = mx.zeros((CAP, GU_W), dtype=mx.uint32)
    mx.eval(pool)

    handoff = {}
    errors = []
    done = threading.Event()

    def bg_materialize(experts):
        try:
            with mx.stream(s2):
                for slot, e in enumerate(experts):
                    raw = os.pread(fd, stride, e * stride)
                    v = np.frombuffer(raw, dtype=np.uint32, count=GU_W)  # 取 gate.weight 段
                    a = mx.array(v)
                    handoff[slot] = a
                mx.eval(list(handoff.values()))   # 在 s2 上物化(与主线程计算重叠)
            done.set()
        except Exception as ex:
            errors.append(repr(ex))
            done.set()

    # 主线程计算负载
    W = mx.random.normal((HIDDEN, HIDDEN)).astype(mx.float32)
    x = mx.random.normal((1, HIDDEN)).astype(mx.float32)
    mx.eval(W, x)

    experts = list(range(K))
    t0 = time.perf_counter()
    th = threading.Thread(target=bg_materialize, args=(experts,))
    th.start()
    # 主线程"算当前层"
    y = x
    for _ in range(N):
        y = mx.tanh(y @ W)
    mx.eval(y)
    done.wait()
    th.join()
    overlap_wall = time.perf_counter() - t0

    # 主线程把交接的 array 拷进池槽(默认 stream，便宜的 scatter)
    for slot, a in handoff.items():
        pool[slot] = a
    mx.eval(pool)

    # 校验:池槽 == 直接读该专家 gate.weight
    ok = True
    for slot, e in enumerate(experts):
        raw = os.pread(fd, stride, e * stride)
        ref = np.frombuffer(raw, dtype=np.uint32, count=GU_W)
        got = np.array(pool[slot], copy=False)
        if not np.array_equal(got, ref):
            ok = False
            break
    os.close(fd)

    print(json.dumps({
        "errors": errors[:3],
        "all_correct": ok,
        "handoff_count": len(handoff),
        "overlap_wall_ms": round(overlap_wall * 1e3, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
