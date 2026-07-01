"""隔离实验:验证"Mac 上后台预取(pread)是否抢占 GPU 计算(统一内存)带宽"。

命题(待验真伪):batch=1 decode 时,后台线程把专家字节 pread 进统一内存的写入带宽,
会挤占 GPU 做专家 matmul 所需的那份内存带宽 → GPU 计算变慢。

方法:一个"内存带宽受限"的 GPU 量化 matmul 循环(常驻权重 ~1GB,每轮全量读 → 纯带宽瓶颈),
分别在三种背景负载下测【每轮 GPU 耗时】:
  A. baseline    : GPU 单独跑
  B. +ssd_pread  : 同时 8 线程 F_NOCACHE pread 真实 qwen blob(复刻真实预取)
  C. +mem_copy   : 同时若干线程纯内存 memcpy(无盘,只吃统一内存带宽)
  A2. baseline   : 再测一次基线,排除漂移/热节流

判读:
  - B 比 A 慢 → 后台预取确实拖慢 GPU(命题为真)
  - C 也比 A 慢 → 根因是"统一内存带宽争用"(不是 SSD/IO 本身)
  - 仅 B 慢、C 不慢 → 是 SSD/DMA 特有的争用,而非纯内存带宽
  - 都不慢 → 命题为假

用法:.venv/bin/python benchmarks/bench_bw_contention.py
可调:ITERS REPEAT SSD_WORKERS MEM_WORKERS BLOB_DIR
"""
import os
import time
import random
import threading
import statistics
from functools import partial

import numpy as np
import mlx.core as mx

ITERS = int(os.environ.get("ITERS", "150"))
REPEAT = int(os.environ.get("REPEAT", "5"))
SSD_WORKERS = int(os.environ.get("SSD_WORKERS", "8"))     # 复刻 stream_blob_workers 默认 8
MEM_WORKERS = int(os.environ.get("MEM_WORKERS", "4"))
BLOB_DIR = os.environ.get(
    "BLOB_DIR", "models/qwen3_next_experts_4bit_g64/blobs")
N_LAYERS = 48
N_EXPERTS = 512
GIB = 1024 ** 3

# ---- 常驻量化权重:做成 ~1GB,保证 GPU 循环是内存带宽瓶颈(值无所谓,读的是字节)----
K = 4096                      # in_dim
N = 1 << 19                   # out_dim = 524288 行
GROUP = 64
BITS = 4
_words = K * BITS // 32       # 每行打包 uint32 个数 = 512
_ngroups = K // GROUP         # 64

w = mx.zeros((N, _words), dtype=mx.uint32)
scales = mx.ones((N, _ngroups), dtype=mx.float16)
biases = mx.zeros((N, _ngroups), dtype=mx.float16)
mx.eval(w, scales, biases)
_W_BYTES = w.nbytes + scales.nbytes + biases.nbytes
print(f"常驻权重:{_W_BYTES/GIB:.3f} GiB(每轮 GPU 全量读一遍 → 带宽瓶颈)")


def gpu_loop(iters: int) -> float:
    """跑 iters 轮量化 matmul(每轮读全量 W),返回本次 eval 的墙钟秒。"""
    x = mx.ones((1, K), dtype=mx.float16)
    s = mx.zeros((1,), dtype=mx.float32)
    for i in range(iters):
        xi = x + (i * 1e-6)                       # 每轮微扰,防被优化/复用
        y = mx.quantized_matmul(xi, w, scales, biases,
                                transpose=True, group_size=GROUP, bits=BITS)
        s = s + y.sum().astype(mx.float32)
    t0 = time.perf_counter()
    mx.eval(s)                                     # 阻塞直到 GPU 算完(期间释放 GIL,后台线程可跑)
    return time.perf_counter() - t0


# ---- 背景负载 B:真实 SSD pread(F_NOCACHE)----
_F_NOCACHE = 48
STRIDE = os.path.getsize(os.path.join(BLOB_DIR, "layer00.blob")) // N_EXPERTS


def ssd_reader(stop: threading.Event, out: list, idx: int):
    import fcntl
    fds = []
    for L in range(N_LAYERS):
        fd = os.open(os.path.join(BLOB_DIR, f"layer{L:02d}.blob"), os.O_RDONLY)
        try:
            fcntl.fcntl(fd, _F_NOCACHE, 1)
        except OSError:
            pass
        fds.append(fd)
    rng = random.Random(idx * 7919 + 1)
    while not stop.is_set():
        L = rng.randrange(N_LAYERS)
        e = rng.randrange(N_EXPERTS)
        os.pread(fds[L], STRIDE, e * STRIDE)       # 释放 GIL 的系统调用 → 真并发
        out[idx] += STRIDE                         # 实时累加(各线程各自下标,免锁)
    for fd in fds:
        os.close(fd)


def ssd_reader_throttled(stop: threading.Event, out: list, idx: int, bps_per_thread: float):
    """节流版:pread 后按目标每线程速率 bps_per_thread 睡眠,逼近真实预取的稳态盘读速率。"""
    import fcntl
    fds = []
    for L in range(N_LAYERS):
        fd = os.open(os.path.join(BLOB_DIR, f"layer{L:02d}.blob"), os.O_RDONLY)
        try:
            fcntl.fcntl(fd, _F_NOCACHE, 1)
        except OSError:
            pass
        fds.append(fd)
    rng = random.Random(idx * 7919 + 3)
    t0 = time.perf_counter()
    done = 0
    while not stop.is_set():
        L = rng.randrange(N_LAYERS)
        e = rng.randrange(N_EXPERTS)
        os.pread(fds[L], STRIDE, e * STRIDE)
        done += STRIDE
        out[idx] += STRIDE
        expected = done / bps_per_thread            # 到目前为止"应该"花的秒数
        actual = time.perf_counter() - t0
        if actual < expected:
            time.sleep(expected - actual)           # 睡到目标速率,压住带宽
    for fd in fds:
        os.close(fd)


# ---- 背景负载 C:纯内存拷贝(无盘,只吃统一内存带宽)----
def mem_hog(stop: threading.Event, out: list, idx: int):
    a = np.ones(256 * 1024 * 1024, dtype=np.uint8)   # 256MB
    b = np.empty_like(a)
    while not stop.is_set():
        np.copyto(b, a)                              # C memcpy,释放 GIL → 真并发
        out[idx] += a.nbytes                         # 实时累加(读+写各一遍,实际流量 x2)


def measure(label: str, bg_fn=None, workers: int = 0):
    stop = threading.Event()
    counters = [0] * workers
    threads = []
    if bg_fn is not None:
        for i in range(workers):
            t = threading.Thread(target=bg_fn, args=(stop, counters, i), daemon=True)
            t.start()
            threads.append(t)
        time.sleep(0.5)                              # 让后台先跑起来、进入稳态
    runs = []
    bg_gbps = 0.0
    for r in range(REPEAT):
        if bg_fn is not None:
            c0 = sum(counters)
            t0 = time.perf_counter()
        runs.append(gpu_loop(ITERS))
        if bg_fn is not None:                        # 只统计"与本轮 GPU 重叠"那段的背景速率
            bg_gbps += (sum(counters) - c0) / (time.perf_counter() - t0) / GIB
    if bg_fn is not None:
        bg_gbps /= REPEAT
        stop.set()
        for t in threads:
            t.join()
    med = statistics.median(runs)
    ms_iter = med / ITERS * 1000
    print(f"{label:16s} 每轮 {ms_iter:7.3f} ms  ({REPEAT}次: {[round(r*1000,1) for r in runs]})"
          + (f"  背景 {bg_gbps:5.2f} GB/s" if bg_fn else ""))
    return ms_iter, bg_gbps


def main():
    print(f"ITERS={ITERS} REPEAT={REPEAT} SSD_WORKERS={SSD_WORKERS} "
          f"MEM_WORKERS={MEM_WORKERS} STRIDE={STRIDE/1024/1024:.3f}MiB\n")
    gpu_loop(20)                                     # warmup
    # 配对交错:基线与背景档交替,每个背景档用【相邻两次基线的均值】做比较 → 抵消慢速热漂移。
    conds = [
        ("B' ssd@1.0GB/s", partial(ssd_reader_throttled, bps_per_thread=1.0 * GIB / 2), 2),
        ("B' ssd@1.5GB/s", partial(ssd_reader_throttled, bps_per_thread=1.5 * GIB / 2), 2),
        ("B' ssd@2.0GB/s", partial(ssd_reader_throttled, bps_per_thread=2.0 * GIB / 2), 2),
        ("B  ssd 饱和",     ssd_reader, SSD_WORKERS),
        ("C  纯内存拷贝",   mem_hog, MEM_WORKERS),
    ]
    seq = []                                          # [(kind, label, ms, gbps)] kind: 'base'|'cond'
    b0, _ = measure("A baseline")
    seq.append(("base", "A", b0, 0.0))
    for label, fn, wk in conds:
        m, g = measure(label, fn, wk)
        seq.append(("cond", label, m, g))
        b, _ = measure("A baseline")                  # 每个档后补一次基线
        seq.append(("base", "A", b, 0.0))

    print(f"\n==== 配对结论(每档 vs 相邻基线均值,已抵消漂移)====")
    for i, (kind, label, ms, g) in enumerate(seq):
        if kind != "cond":
            continue
        pre = seq[i - 1][2]                           # 前一次基线
        post = seq[i + 1][2]                          # 后一次基线
        local = (pre + post) / 2
        print(f"  {label:16s} {(ms/local-1)*100:+6.1f}%   "
              f"(局部基线 {local:.2f}ms, 背景 {g:.2f} GB/s)")
    bases = [ms for kind, _, ms, _ in seq if kind == "base"]
    print(f"\n  基线漂移范围:{min(bases):.2f}~{max(bases):.2f} ms "
          f"(极差 {(max(bases)/min(bases)-1)*100:.1f}%);配对法已消掉它")


if __name__ == "__main__":
    main()
