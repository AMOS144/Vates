"""Gating 测试 B:后台线程(独立 stream)写共享池槽，主线程读，是否正确且无竞争。

模拟"后台预取队列把 L+1 专家物化进 NativeComputeSlotPool 槽位"。
"""
import threading
import time

import mlx.core as mx
import numpy as np

CAP, D = 32, 4096
N = 50


def main():
    import json
    s2 = mx.new_stream(mx.default_device())
    pool = mx.zeros((CAP, D), dtype=mx.uint32)
    mx.eval(pool)

    # 期望值：每个 slot 填 slot 编号的常数
    expected = {}
    errors = []

    def bg_fill(slots):
        try:
            with mx.stream(s2):
                p = pool
                for slot in slots:
                    val = np.full((D,), slot + 1, dtype=np.uint32)
                    p[slot] = mx.array(val)
                    expected[slot] = slot + 1
                mx.eval(p)
        except Exception as e:
            errors.append(repr(e))

    # 主线程并发跑点计算
    W = mx.random.normal((512, 512)).astype(mx.float32)
    mx.eval(W)

    th = threading.Thread(target=bg_fill, args=(list(range(CAP)),))
    th.start()
    y = mx.random.normal((1, 512)).astype(mx.float32)
    for _ in range(N):
        y = mx.tanh(y @ W)
    mx.eval(y)
    th.join()

    # join 后主线程读池，校验
    mx.eval(pool)
    ok = True
    for slot, val in expected.items():
        got = int(pool[slot, 0].item())
        if got != val:
            ok = False
            errors.append(f"slot {slot}: got {got} want {val}")

    print(json.dumps({
        "crashed_or_errors": errors[:5],
        "all_slots_correct": ok,
        "filled_slots": len(expected),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
