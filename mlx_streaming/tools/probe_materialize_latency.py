"""量化「物化链路」耗时：把 N 个专家从 blob 变成可用 mx.array 要多久。

拆成两个口径：
- read_raw_ms：仅并行 pread 读字节（不物化）。
- load_eval_ms：load_experts（np.frombuffer→mx.array）+ mx.eval 同步（= demand 真实物化）。
- submit_ready_ms：经后台 BackgroundExpertPrefetcher 的 submit→ready 端到端延迟
  （含线程调度 + s2 stream 物化 + eval），这才是同层预取真正要塞进 attention 窗口的链路。

与 run_mtp_spec 的 window_prof（attention/GDN 窗口）对比即可判定「窗口够不够」。
"""
import os
import statistics
import time

import mlx.core as mx

from mlx_streaming.core.cache.blob_loader import BlobExpertSource
from mlx_streaming.core.prefetch.bg_prefetch import BackgroundExpertPrefetcher

BLOB = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")
BITS = int(os.environ.get("BITS", "2"))
LAYER = int(os.environ.get("PROBE_LAYER", "15"))
N = int(os.environ.get("PROBE_N", "4"))          # 同层缺失专家数（budget≈2×top_k）
TRIALS = int(os.environ.get("PROBE_TRIALS", "30"))
NOCACHE = os.environ.get("PROBE_NOCACHE", "0") == "1"


def _pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    src = BlobExpertSource(BLOB, 2048, 512, 128, BITS, num_experts=512,
                           workers=int(os.environ.get("STREAM_BLOB_WORKERS", "8")),
                           nocache=NOCACHE)
    base = LAYER * 1000
    ids_pool = [[(base + t * N + j) % 512 for j in range(N)] for t in range(TRIALS)]

    read_ms, load_ms = [], []
    for ids in ids_pool:
        ids = [int(i) for i in ids]
        t = time.perf_counter()
        src.read_raw(LAYER, ids)
        read_ms.append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        ex = src.load_experts(LAYER, ids)
        mx.eval([v for d in ex.values() for v in d.values()])
        load_ms.append((time.perf_counter() - t) * 1000)

    # 后台 submit→ready 端到端延迟
    pf = BackgroundExpertPrefetcher(src, window=4)
    submit_ms = []
    for ids in ids_pool:
        ids = [int(i) for i in ids]
        # take_ready_layer 每轮清空 ready，故复用真实 LAYER 即可（每轮 ids 不同）
        t = time.perf_counter()
        pf.submit(LAYER, ids)
        while pf.ready_count(LAYER) < len(ids):
            if time.perf_counter() - t > 2:
                break
            time.sleep(0.0002)
        submit_ms.append((time.perf_counter() - t) * 1000)
        pf.take_ready_layer(LAYER)
    pf.close()
    src.close()

    def row(name, xs):
        print(f"{name:>16}: mean={statistics.mean(xs):7.3f}ms  "
              f"p50={_pctl(xs, 0.5):7.3f}  p90={_pctl(xs, 0.9):7.3f}  "
              f"min={min(xs):7.3f}  max={max(xs):7.3f}")

    print(f"blob={BLOB} layer={LAYER} N={N} trials={TRIALS} nocache={NOCACHE}")
    row("read_raw", read_ms)
    row("load+eval", load_ms)
    row("submit→ready", submit_ms)


if __name__ == "__main__":
    main()
