"""隔离测试:后台物化是否因 GIL 拖慢主线程,以及 native(C++ blob_load)能否消除。

主线程:模拟 verify 前向的 per-layer 模式 —— 每"层"做若干 quantized_matmul + 一次
.tolist()(host 同步,需 GIL,模拟路由/promote 编排)。测主线程跑完固定工作的墙钟。
三种条件对比:
  base   : 无后台
  bg_py  : 后台线程用 Python 物化(np.frombuffer+mx.array,占 GIL)
  bg_nat : 后台线程用 native blob_load(C++ pread+拷,绕 GIL)
若 bg_py 明显拖慢 base 而 bg_nat 接近 base → GIL 是主因且 native 修复它。
"""
import os
import threading
import time

import mlx.core as mx

from mlx_streaming.core.cache.blob_loader import BlobExpertSource

BLOB = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")
BITS = int(os.environ.get("BITS", "2"))
LAYERS = int(os.environ.get("PROBE_LAYERS", "48"))
ITERS = int(os.environ.get("PROBE_ITERS", "20"))      # 模拟 token 数
MM = int(os.environ.get("PROBE_MM", "6"))             # 每层 matmul 个数
DIM = int(os.environ.get("PROBE_DIM", "2048"))
N_EXPERT = int(os.environ.get("PROBE_N", "8"))        # 每层后台物化专家数


def main_work():
    """跑一遍"前向":LAYERS 层 × ITERS token,每层若干 matmul + 一次 host 同步。"""
    w = [mx.random.normal((DIM, DIM)) for _ in range(MM)]
    mx.eval(w)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for _ in range(LAYERS):
            x = mx.random.normal((1, DIM))
            for k in range(MM):
                x = x @ w[k]
            # 模拟路由/promote 的 host 同步 + Python 编排(需 GIL)
            _ = x[0, :4].tolist()
    mx.eval(x)
    return time.perf_counter() - t0


def run(mode: str) -> float:
    stop = threading.Event()
    src = BlobExpertSource(BLOB, 2048, 512, 128, BITS, num_experts=512)
    cnt = {"n": 0}

    def bg():
        s2 = mx.new_stream(mx.default_device())  # 在 bg 线程内部建 stream（跨线程注册问题）
        e0 = 0
        while not stop.is_set():
            ids = [(e0 + j) % 512 for j in range(N_EXPERT)]
            e0 = (e0 + N_EXPERT) % 512
            layer = 15
            with mx.stream(s2):
                if mode == "bg_nat":
                    ex = src.load_experts_native(layer, ids, view_bf16=False)
                else:
                    ex = src.load_experts(layer, ids, view_bf16=False)
                mx.eval([v for d in ex.values() for v in d.values()])
            cnt["n"] += 1

    t = None
    if mode != "base":
        t = threading.Thread(target=bg, daemon=True)
        t.start()
        time.sleep(0.05)  # 让后台先转起来
    dt = main_work()
    stop.set()
    if t:
        t.join(timeout=2)
    src.close()
    return dt, cnt["n"]


def main():
    # 预热
    main_work()
    res = {}
    for mode in ("base", "bg_py", "bg_nat", "base", "bg_py", "bg_nat"):
        dt, n = run(mode)
        res.setdefault(mode, []).append((round(dt, 3), n))
    for mode in ("base", "bg_py", "bg_nat"):
        runs = res[mode]
        best = min(r[0] for r in runs)
        print(f"{mode:>8}: main_wall={runs}  best={best}s")


if __name__ == "__main__":
    main()
