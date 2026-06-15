"""钉死"流式读能否藏进计算"这堵墙：测每层字节量、mmap 读带宽、单层 MoE 计算耗时。

得出 读/算 比值：<1 则"掩盖读取 + 只驻留 2 层"成立；>1 则带宽墙挡住。

环境变量：
  EXPERT_DIR        2bit per-expert safetensors 目录
  COMPUTE_BUFFER_DIR  打包后的 compute buffer 目录
  LAYER (默认 20)、K_EXPERTS (默认 10，单 token top_k)、ITERS (默认 50)
"""
import os
import time

import mlx.core as mx
import numpy as np

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERT_DIR = os.environ.get("EXPERT_DIR", os.path.join(ROOT, "models", "qwen3_next_experts_2bit_g128"))
COMPUTE_DIR = os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/cb_2bit_g128")
LAYER = int(os.environ.get("LAYER", "20"))
K = int(os.environ.get("K_EXPERTS", "10"))
ITERS = int(os.environ.get("ITERS", "50"))


def _bytes_per_expert() -> int:
    gu_w = INTER * (HIDDEN * BITS // 32) * 4          # uint32
    gu_sb = INTER * (HIDDEN // GROUP) * 2 * 2         # scales+biases, uint16
    dn_w = HIDDEN * (INTER * BITS // 32) * 4
    dn_sb = HIDDEN * (INTER // GROUP) * 2 * 2
    return (gu_w + gu_sb) * 2 + (dn_w + dn_sb)        # gate+up + down


def _read_layer_via_memmap(layer: int, k: int) -> int:
    """走真实代码路径：np.memmap 大 .bin → np.asarray 拷贝（与 native staging 一致）。返回搬运字节。"""
    total = 0
    for proj, out_dim, in_dim in (
        ("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER)):
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        base = os.path.join(COMPUTE_DIR, f"layer{layer:02d}.{proj}")
        w = np.memmap(base + ".weight.bin", dtype=np.uint32, mode="r", shape=(NUM_EXPERTS, out_dim, words))
        s = np.memmap(base + ".scales.bin", dtype=np.uint16, mode="r", shape=(NUM_EXPERTS, out_dim, groups))
        b = np.memmap(base + ".biases.bin", dtype=np.uint16, mode="r", shape=(NUM_EXPERTS, out_dim, groups))
        for e in range(k):
            total += np.asarray(w[e]).nbytes + np.asarray(s[e]).nbytes + np.asarray(b[e]).nbytes
        del w, s, b
    return total


def _try_purge() -> bool:
    """尝试 sudo -n purge 清页缓存（非交互；没权限就跳过）。"""
    return os.system("sudo -n purge >/dev/null 2>&1") == 0


def _load_k_experts(layer: int, k: int):
    out = []
    for e in range(k):
        w = mx.load(os.path.join(EXPERT_DIR, f"layer{layer:02d}_expert{e:03d}.safetensors"))
        out.append(w)
    mx.eval([v for d in out for v in d.values()])
    return out


def _moe_compute(experts, x):
    acc = None
    for w in experts:
        g = mx.quantized_matmul(x, w["gate_proj.weight"], w["gate_proj.scales"], w["gate_proj.biases"],
                                transpose=True, group_size=GROUP, bits=BITS)
        u = mx.quantized_matmul(x, w["up_proj.weight"], w["up_proj.scales"], w["up_proj.biases"],
                                transpose=True, group_size=GROUP, bits=BITS)
        a = g * mx.sigmoid(g) * u
        o = mx.quantized_matmul(a, w["down_proj.weight"], w["down_proj.scales"], w["down_proj.biases"],
                                transpose=True, group_size=GROUP, bits=BITS)
        acc = o if acc is None else acc + o
    return acc


def main():
    bpe = _bytes_per_expert()
    layer_bytes = bpe * K

    # ---- READ: cold (purge then time) ----
    purged = _try_purge()
    t0 = time.perf_counter()
    read_bytes = _read_layer_via_memmap(LAYER, K)
    cold_s = time.perf_counter() - t0

    # ---- READ: warm (页缓存命中，重复取稳态) ----
    best_warm = 1e9
    for _ in range(10):
        t0 = time.perf_counter()
        _read_layer_via_memmap(LAYER, K)
        best_warm = min(best_warm, time.perf_counter() - t0)

    # ---- COMPUTE: 单层 MoE（K 个专家，batch=1） ----
    experts = _load_k_experts(LAYER, K)
    x = (mx.random.normal((1, HIDDEN)) * 0.1).astype(mx.float32)
    mx.eval(x)
    for _ in range(5):
        mx.eval(_moe_compute(experts, x))
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(_moe_compute(experts, x))
    compute_s = (time.perf_counter() - t0) / ITERS

    cold_gbps = read_bytes / cold_s / 1e9
    warm_gbps = read_bytes / best_warm / 1e9
    print({
        "layer": LAYER, "K_experts": K,
        "bytes_per_expert_KB": round(bpe / 1024, 1),
        "layer_bytes_MB": round(layer_bytes / 1e6, 2),
        "purged_cold": purged,
        "read_cold_ms": round(cold_s * 1e3, 2),
        "read_cold_GBps": round(cold_gbps, 2),
        "read_warm_ms": round(best_warm * 1e3, 3),
        "read_warm_GBps": round(warm_gbps, 2),
        "compute_ms_per_layer": round(compute_s * 1e3, 3),
        "ratio_cold_read_over_compute": round(cold_s / compute_s, 2),
        "ratio_warm_read_over_compute": round(best_warm / compute_s, 2),
        "verdict": "读>算(藏不住)" if cold_s > compute_s else "读<算(可藏)",
    })


if __name__ == "__main__":
    main()
