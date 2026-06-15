"""【证伪用，可丢弃】模拟「旋转 + 2-bit 非均匀(Lloyd-Max-Gaussian)码本」的质量。

思路：把非均匀码本的*重构结果*存进 8-bit affine 容器（8-bit 近无损，只是忠实搬运
Lloyd-Max-2bit 的重构值），从而复用现有旋转 runtime（EXPERT_ROT=1, EXPERT_BITS=8）
跑困惑度——无需自写非均匀 dequant 内核。

用法：derisk_lloydmax.py SRC_DIR DST_DIR
"""
import os
import sys
import json
import math
import time

import mlx.core as mx

PROJ = [("gate_proj", 2048), ("up_proj", 2048), ("down_proj", 768)]
# Lloyd-Max N(0,1) 2-bit(4 level) 最优质心（大样本 Lloyd 迭代求得）
CENT = mx.array([-1.508, -0.4505, 0.4556, 1.5137])


def lloydmax_roundtrip(W: mx.array, group_size: int) -> mx.array:
    """旋转后(近高斯)权重按 Lloyd-Max-Gaussian 4-level 码本逐组量化再重构。"""
    O, I = W.shape
    Wg = W.reshape(O, I // group_size, group_size).astype(mx.float32)
    mu = Wg.mean(axis=-1, keepdims=True)
    sd = Wg.std(axis=-1, keepdims=True) + 1e-8
    z = (Wg - mu) / sd                                  # 标准化到 ~N(0,1)
    dist = mx.abs(z[..., None] - CENT.reshape(1, 1, 1, -1))
    idx = mx.argmin(dist, axis=-1)
    zq = CENT[idx]                                      # 最近质心
    return (zq * sd + mu).reshape(O, I)


def gen(src_dir: str, dst_dir: str) -> dict:
    with open(os.path.join(src_dir, "_split_meta.json")) as f:
        src_meta = json.load(f)
    sb = src_meta["dims"]["bits"]
    sg = src_meta["dims"]["group_size"]
    in_dims = {n: d for n, d in PROJ}
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".safetensors"))
    t = time.perf_counter()
    for i, fn in enumerate(files):
        src = mx.load(os.path.join(src_dir, fn))
        out = {}
        for name, n_in in PROJ:
            if f"{name}.weight" not in src:
                continue
            W = mx.dequantize(src[f"{name}.weight"], src[f"{name}.scales"],
                              src[f"{name}.biases"], group_size=sg, bits=sb)
            Wr = mx.hadamard_transform(W, scale=1.0 / math.sqrt(n_in))   # 旋转
            Wlm = lloydmax_roundtrip(Wr, sg)                            # 非均匀 2-bit 重构
            wq, s, b = mx.quantize(Wlm, group_size=sg, bits=8)          # 存 8-bit 容器
            out[f"{name}.weight"] = wq
            out[f"{name}.scales"] = s
            out[f"{name}.biases"] = b
        mx.eval(out)
        mx.save_safetensors(os.path.join(dst_dir, fn), out)
        if (i + 1) % 512 == 0:
            print(f"  {i+1}/{len(files)} ({round(time.perf_counter()-t,1)}s)", flush=True)
    dm = dict(src_meta)
    dm["dims"] = dict(src_meta["dims"], bits=8, group_size=sg)
    dm["rotated"] = True
    dm["note"] = "lloydmax-2bit reconstruction stored in 8bit affine container (de-risk)"
    with open(os.path.join(dst_dir, "_split_meta.json"), "w") as f:
        json.dump(dm, f, ensure_ascii=False, indent=2)
    return {"files": len(files), "elapsed_s": round(time.perf_counter() - t, 1)}


if __name__ == "__main__":
    print(json.dumps(gen(sys.argv[1], sys.argv[2]), ensure_ascii=False, indent=2))
