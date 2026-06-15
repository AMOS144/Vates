"""de-risk 探针:SpQR-style 推理路径在 MLX 上的可行性(临时,GO/NO-GO 用)。

SpQR 推理 = 稠密低 bit matmul + 稀疏 FP16 离群修正。本 spike 用合成数据(不需 FP16 模型)
测三条路的单次 matmul 墙钟:
  A. 纯 2-bit 稠密(mx.quantized_matmul)          —— 我们现有的快路径
  B. 2-bit 稠密 + 结构化离群通道 FP16 修正(C 个输入通道走小 FP16 matmul)
  C. 纯 FP16 稠密(参考上界)
判据:B/A 开销要小(修正便宜),且 A 明显 < C(2-bit 真比 FP16 快),SpQR 才有运行时意义。
注:结构化"离群通道"是 GPU 友好变体;非结构化逐权重离群只会更慢。

环境变量:OUT/IN(矩阵维度,默认 512/2048=gate-up)/ T(token 数,默认 1=decode)/
          OUTLIER_FRAC(离群通道比例,默认 0.01)/ REPS
"""
import os
import time

import mlx.core as mx

OUT = int(os.environ.get("OUT", "512"))
IN = int(os.environ.get("IN", "2048"))
T = int(os.environ.get("T", "1"))
FRAC = float(os.environ.get("OUTLIER_FRAC", "0.01"))
REPS = int(os.environ.get("REPS", "200"))
GROUP = int(os.environ.get("GROUP", "128"))
BITS = int(os.environ.get("BITS", "2"))


def _time(fn):
    fn()                                  # warm/编译
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(REPS):
        mx.eval(fn())
    return (time.perf_counter() - t0) / REPS * 1e6   # us/次


def main():
    mx.random.seed(0)
    x = mx.random.normal((T, IN)).astype(mx.float16)
    W = mx.random.normal((OUT, IN)).astype(mx.float16)        # 参考 FP16 权重
    wq, scales, biases = mx.quantize(W, group_size=GROUP, bits=BITS)

    n_out_ch = max(1, int(IN * FRAC))                          # 离群输入通道数
    cols = mx.array(sorted(range(0, IN, max(1, IN // n_out_ch)))[:n_out_ch])
    W_out = W[:, cols]                                         # 这些通道的 FP16 权重 (OUT, C)

    def dense_2bit():
        return mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                   group_size=GROUP, bits=BITS)

    def dense_2bit_plus_outlier():
        y = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                group_size=GROUP, bits=BITS)
        corr = x[:, cols] @ W_out.T                           # 小 FP16 修正 (T, C)x(C, OUT)
        return y + corr

    def dense_fp16():
        return x @ W.T

    a = _time(dense_2bit)
    b = _time(dense_2bit_plus_outlier)
    c = _time(dense_fp16)

    print(f"shape OUT={OUT} IN={IN} T={T} bits={BITS} g{GROUP} 离群通道={n_out_ch}({FRAC*100:.0f}%)")
    print(f"A 纯2-bit稠密:         {a:8.2f} us")
    print(f"B 2-bit+离群FP16修正:  {b:8.2f} us   (B/A = {b/a:.2f}x 开销)")
    print(f"C 纯FP16稠密(参考):    {c:8.2f} us   (A/C = {a/c:.2f}x,2-bit 相对 FP16)")
    print(f"判据: B/A 越接近 1 越好(修正便宜);A/C < 1 说明 2-bit 真比 FP16 快")


if __name__ == "__main__":
    main()
