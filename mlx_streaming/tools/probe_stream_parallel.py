"""干净测量两条 MLX stream 的真实并行度（不靠忙循环夸大争用）。

分别测：
  T_main : 主线程 main_work 单独
  T_bg   : 后台单独物化 K 批专家（同样的活，无主线程）
  T_both : 主线程 main_work 的同时，后台物化恰好 K 批后停
判定：
  T_both ≈ max(T_main, T_bg)  → 真并行（无争用）
  T_both ≈ T_main + T_bg      → 完全串行
  介于之间                     → 部分重叠，overlap=(T_main+T_bg-T_both)/min(...)
K 取成与真实每层预取相当（每个 main 层物化一批），而不是无限忙循环。
"""
import os
import threading
import time

import mlx.core as mx

from mlx_streaming.core.cache.blob_loader import BlobExpertSource

BLOB = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")
BITS = int(os.environ.get("BITS", "2"))
LAYERS = int(os.environ.get("PROBE_LAYERS", "48"))
ITERS = int(os.environ.get("PROBE_ITERS", "20"))
MM = int(os.environ.get("PROBE_MM", "6"))
DIM = int(os.environ.get("PROBE_DIM", "2048"))
N_EXPERT = int(os.environ.get("PROBE_N", "8"))
K_BATCHES = int(os.environ.get("PROBE_K_BATCHES", str(LAYERS * ITERS)))  # 真实量：每层一批
MODE = os.environ.get("MAT_MODE", "py")  # py | nat


def main_work(w):
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for _ in range(LAYERS):
            x = mx.random.normal((1, DIM))
            for k in range(MM):
                x = x @ w[k]
            _ = x[0, :4].tolist()
    mx.eval(x)
    return time.perf_counter() - t0


def bg_materialize(n_batches):
    src = BlobExpertSource(BLOB, 2048, 512, 128, BITS, num_experts=512)
    s2 = mx.new_stream(mx.default_device())
    e0 = 0
    done = 0
    for _ in range(n_batches):
        ids = [(e0 + j) % 512 for j in range(N_EXPERT)]
        e0 = (e0 + N_EXPERT) % 512
        with mx.stream(s2):
            if MODE == "nat":
                ex = src.load_experts_native(15, ids, view_bf16=False)
            else:
                ex = src.load_experts(15, ids, view_bf16=False)
            mx.eval([v for d in ex.values() for v in d.values()])
        done += 1
    src.close()
    return done


def main():
    w = [mx.random.normal((DIM, DIM)) for _ in range(MM)]
    mx.eval(w)
    main_work(w)              # 预热
    bg_materialize(20)        # 预热

    t = time.perf_counter(); main_work(w); T_main = time.perf_counter() - t
    t = time.perf_counter(); n = bg_materialize(K_BATCHES); T_bg = time.perf_counter() - t

    # 并发：主线程跑，同时后台物化 K 批
    done = {"n": 0}
    def runner():
        done["n"] = bg_materialize(K_BATCHES)
    th = threading.Thread(target=runner); 
    t = time.perf_counter()
    th.start()
    main_work(w)
    th.join()
    T_both = time.perf_counter() - t

    serial = T_main + T_bg
    ideal = max(T_main, T_bg)
    overlap = (serial - T_both) / max(serial - ideal, 1e-9)  # 1.0=全并行,0=全串行
    print(f"mode={MODE} K_batches={K_BATCHES} ({n} bg mat/批{N_EXPERT})")
    print(f"  T_main      = {T_main:.3f}s")
    print(f"  T_bg        = {T_bg:.3f}s")
    print(f"  T_both      = {T_both:.3f}s")
    print(f"  serial(sum) = {serial:.3f}s   ideal(max) = {ideal:.3f}s")
    print(f"  overlap     = {overlap*100:.0f}%   (100%=纯并行, 0%=纯串行)")
    print(f"  main 增量    = {(T_both - T_main):.3f}s  (+{(T_both/T_main-1)*100:.0f}%)")


if __name__ == "__main__":
    main()
