"""de-risk：测「把单个专家写进连续 pool 的某个槽位」三种写法的成本。

判定：单次写应 ≈ 单专家字节量级、且 N 次写后内存不线性膨胀。
谁最便宜就用谁作为 ResidentExpertPool._write_slot。
"""
import time
import mlx.core as mx

CAP, O, I, GROUP, BITS = 96, 2048, 768, 64, 2
# 2-bit 打包：weight 是 uint32，列数 = I*BITS/32
W_COLS = I * BITS // 32


def _new_pool():
    return {
        "weight": mx.zeros((CAP, O, W_COLS), dtype=mx.uint32),
        "scales": mx.zeros((CAP, O, I // GROUP), dtype=mx.float16),
        "biases": mx.zeros((CAP, O, I // GROUP), dtype=mx.float16),
    }


def _new_expert():
    return {
        "weight": mx.random.randint(0, 2**31, (O, W_COLS)).astype(mx.uint32),
        "scales": mx.random.normal((O, I // GROUP)).astype(mx.float16),
        "biases": mx.random.normal((O, I // GROUP)).astype(mx.float16),
    }


def _pool_for_cap(cap):
    return {
        "weight": mx.zeros((cap, O, W_COLS), dtype=mx.uint32),
        "scales": mx.zeros((cap, O, I // GROUP), dtype=mx.float16),
        "biases": mx.zeros((cap, O, I // GROUP), dtype=mx.float16),
    }


def bench(name, write_fn, cap=CAP, n=200):
    pool, ex = _pool_for_cap(cap), _new_expert()
    mx.eval(list(pool.values()) + list(ex.values()))
    t0 = time.perf_counter()
    for s in range(n):
        write_fn(pool, s % cap, ex)
        mx.eval(list(pool.values()))
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"{name:>16} (cap={cap:>4}): {dt:.3f} ms/write")


def w_inplace(pool, slot, ex):
    for k, v in ex.items():
        pool[k][slot] = v


def w_slice_update(pool, slot, ex):
    for k, v in ex.items():
        pool[k] = mx.slice_update(pool[k], v[None], mx.array([slot, 0, 0]), axes=(0, 1, 2))


def main():
    print(f"单专家字节量 weight={O*W_COLS*4/1e6:.2f}MB  整池(cap=96)≈{96*O*W_COLS*4/1e6:.1f}MB")
    # 判别：若写入耗时随 cap 线性增长 => 整池拷贝(坏)；若恒定 => 真单槽写(好)
    for cap in (32, 96, 384):
        bench("inplace", w_inplace, cap=cap)
    for cap in (32, 96, 384):
        bench("slice_update", w_slice_update, cap=cap)


if __name__ == "__main__":
    main()
