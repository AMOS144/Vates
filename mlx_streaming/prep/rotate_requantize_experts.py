"""把已拆分的 4-bit per-expert 专家，沿 input 维做 Hadamard 旋转后重量化到 2-bit。

原理（QuIP#-lite）：量化分组沿 input 维，旋转把每组能量打散成近高斯、压低组内动态
范围 → 低比特误差大降。W' = hadamard(W, 1/√n_in)（逐行、作用在最后一维=input 维）。
运行时输入也做同样旋转，gather_qmm 数学等价 W·x（见 streaming_moe.RotatedSubGLU）。
"""
import os
import sys
import json
import math
import time

import mlx.core as mx

PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def rotate_requantize_file(src_path: str, dst_path: str, src_bits: int, src_group: int,
                           dst_bits: int, dst_group: int, in_dims: dict) -> None:
    """单个专家文件：每个 proj 反量化 → 沿 input 维 Hadamard 旋转 → 重量化。"""
    src = mx.load(src_path)
    out = {}
    for name in PROJ_NAMES:
        wq = src.get(f"{name}.weight")
        if wq is None:
            continue
        scales = src[f"{name}.scales"]
        biases = src[f"{name}.biases"]
        W = mx.dequantize(wq, scales, biases, group_size=src_group, bits=src_bits)
        n_in = in_dims[name]
        # Hadamard 作用在最后一维（input 维），逐行旋转
        Wp = mx.hadamard_transform(W, scale=1.0 / math.sqrt(n_in))
        nwq, ns, nb = mx.quantize(Wp, group_size=dst_group, bits=dst_bits)
        out[f"{name}.weight"] = nwq
        out[f"{name}.scales"] = ns
        out[f"{name}.biases"] = nb
    mx.eval(out)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mx.save_safetensors(dst_path, out)


def rotate_requantize_dir(src_dir: str, dst_dir: str, dst_bits: int, dst_group: int) -> dict:
    """把 src_dir（4-bit 拆分专家）全部旋转重量化到 dst_dir（2-bit），并写 meta。"""
    with open(os.path.join(src_dir, "_split_meta.json")) as f:
        src_meta = json.load(f)
    src_bits = src_meta["dims"]["bits"]
    src_group = src_meta["dims"]["group_size"]
    hidden = src_meta["dims"]["hidden"]
    moe_inter = src_meta["dims"]["moe_intermediate"]
    in_dims = {"gate_proj": hidden, "up_proj": hidden, "down_proj": moe_inter}
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".safetensors"))
    t = time.perf_counter()
    for i, fn in enumerate(files):
        rotate_requantize_file(os.path.join(src_dir, fn), os.path.join(dst_dir, fn),
                               src_bits, src_group, dst_bits, dst_group, in_dims)
        if (i + 1) % 512 == 0:
            print(f"  {i+1}/{len(files)} ({round(time.perf_counter()-t,1)}s)", flush=True)

    dst_meta = dict(src_meta)
    dst_meta["out_dir"] = dst_dir
    dst_meta["dims"] = dict(src_meta["dims"], bits=dst_bits, group_size=dst_group)
    dst_meta["rotated"] = True
    dst_meta["rotation"] = {"type": "hadamard", "scale": "1/sqrt(n_in)", "in_dims": in_dims}
    dst_meta["rotated_from"] = {"dir": src_dir, "bits": src_bits, "group_size": src_group}
    with open(os.path.join(dst_dir, "_split_meta.json"), "w") as f:
        json.dump(dst_meta, f, ensure_ascii=False, indent=2)
    return {"files": len(files), "dst_bits": dst_bits, "dst_group": dst_group,
            "elapsed_s": round(time.perf_counter() - t, 1)}


if __name__ == "__main__":
    # 用法：rotate_requantize_experts.py SRC_DIR DST_DIR [DST_BITS=2] [DST_GROUP=64]
    src = sys.argv[1]
    dst = sys.argv[2]
    bits = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    group = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    info = rotate_requantize_dir(src, dst, bits, group)
    print(json.dumps(info, ensure_ascii=False, indent=2))
