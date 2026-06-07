"""把已拆分的 per-expert 量化文件重量化到更低 bit（4-bit → 3/2-bit）。

做法：逐专家、逐 proj 把源量化权重反量化回 float，再用 mx.quantize 量到目标
bit/group，写出同名文件到目标目录。全程一次只物化一个专家，低内存。

注意（严谨性）：源权重本身已是 4-bit，这里是「4bit→反量化→2/3bit」的二次量化，
质量是真实「从原始 bf16 量化」的下界；测大小/I/O/速度完全有效。
"""
import os
import sys
import json
import time

import mlx.core as mx

PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def _norm_bits(dst_bits):
    """把 dst_bits 规整成 {proj: bits}：int → 三 proj 同 bit；dict → 原样补全。"""
    if isinstance(dst_bits, dict):
        return {n: int(dst_bits[n]) for n in PROJ_NAMES if n in dst_bits}
    return {n: int(dst_bits) for n in PROJ_NAMES}


def requantize_file(src_path: str, dst_path: str, src_bits: int, src_group: int,
                    dst_bits, dst_group: int) -> None:
    """重量化单个 per-expert 文件：每个 proj 反量化后按目标 bit 重新量化。

    dst_bits 可为 int（三 proj 统一 bit）或 {proj: bits}（混合精度，逐 proj 不同 bit）。
    """
    bits_map = _norm_bits(dst_bits)
    src = mx.load(src_path)
    out = {}
    for name in PROJ_NAMES:
        wq = src.get(f"{name}.weight")
        if wq is None:
            continue
        scales = src[f"{name}.scales"]
        biases = src[f"{name}.biases"]
        W = mx.dequantize(wq, scales, biases, group_size=src_group, bits=src_bits)
        nwq, ns, nb = mx.quantize(W, group_size=dst_group, bits=bits_map[name])
        out[f"{name}.weight"] = nwq
        out[f"{name}.scales"] = ns
        out[f"{name}.biases"] = nb
    mx.eval(out)   # 只物化这一个专家
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    mx.save_safetensors(dst_path, out)


def requantize_dir(src_dir: str, dst_dir: str, dst_bits, dst_group: int) -> dict:
    """把 src_dir 里所有 per-expert 文件重量化到 dst_dir。源 bit/group 取自源 meta。

    dst_bits 可为 int（统一）或 {proj: bits}（混合精度）。混合时 meta.dims 记录
    proj_bits，并把 bits 设为各 proj 的最大值（仅作名义/兼容旧读取用）。
    """
    with open(os.path.join(src_dir, "_split_meta.json")) as f:
        src_meta = json.load(f)
    src_bits = src_meta["dims"]["bits"]
    src_group = src_meta["dims"]["group_size"]
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".safetensors"))
    t = time.perf_counter()
    for i, fn in enumerate(files):
        requantize_file(os.path.join(src_dir, fn), os.path.join(dst_dir, fn),
                        src_bits, src_group, dst_bits, dst_group)
        if (i + 1) % 512 == 0:
            print(f"  {i+1}/{len(files)} ({round(time.perf_counter()-t,1)}s)", flush=True)

    dst_meta = dict(src_meta)
    dst_meta["out_dir"] = dst_dir
    if isinstance(dst_bits, dict):
        proj_bits = _norm_bits(dst_bits)
        dst_meta["dims"] = dict(src_meta["dims"], group_size=dst_group,
                                bits=max(proj_bits.values()), proj_bits=proj_bits)
    else:
        dst_meta["dims"] = dict(src_meta["dims"], bits=int(dst_bits), group_size=dst_group)
    dst_meta["requantized_from"] = {"dir": src_dir, "bits": src_bits, "group_size": src_group}
    with open(os.path.join(dst_dir, "_split_meta.json"), "w") as f:
        json.dump(dst_meta, f, ensure_ascii=False, indent=2)
    return {"files": len(files), "dst_bits": dst_bits, "dst_group": dst_group,
            "elapsed_s": round(time.perf_counter() - t, 1)}


def _layer_idx_from_name(fn: str) -> int:
    """从 layer{LL}_expert{EEE}.safetensors 解析绝对层号。"""
    return int(fn[len("layer"):fn.index("_expert")])


def boundary_scheme(moe_layers, bnd: int, bnd_bits, mid_bits) -> dict:
    """构造逐层 bit 映射：MoE 层序的首 bnd + 尾 bnd 用 bnd_bits，其余用 mid_bits。

    返回 {绝对层号: bits}（bits 可为 int 或 {proj:bits}）。moe_layers 为绝对层号有序列表。
    依据 §4.12 实测：本模型首尾 MoE 层对 2-bit 几乎免疫，故首尾压更狠、中间留高 bit。
    """
    n = len(moe_layers)
    bnd_ord = set(range(bnd)) | set(range(n - bnd, n))
    return {abs_idx: (bnd_bits if o in bnd_ord else mid_bits)
            for o, abs_idx in enumerate(moe_layers)}


def requantize_dir_layered(src_dir: str, dst_dir: str, layer_bits: dict, dst_group: int) -> dict:
    """逐层重量化：layer_bits={绝对层号: bits(int 或 {proj:bits})}，按文件名层号取对应 bit。

    meta.dims 记录 per_layer_proj_bits（每个 MoE 层规整后的 {proj:bits}），供 runtime 逐层建 QSL。
    """
    with open(os.path.join(src_dir, "_split_meta.json")) as f:
        src_meta = json.load(f)
    src_bits = src_meta["dims"]["bits"]
    src_group = src_meta["dims"]["group_size"]
    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(fn for fn in os.listdir(src_dir) if fn.endswith(".safetensors"))
    t = time.perf_counter()
    for i, fn in enumerate(files):
        li = _layer_idx_from_name(fn)
        requantize_file(os.path.join(src_dir, fn), os.path.join(dst_dir, fn),
                        src_bits, src_group, layer_bits[li], dst_group)
        if (i + 1) % 512 == 0:
            print(f"  {i+1}/{len(files)} ({round(time.perf_counter()-t,1)}s)", flush=True)

    per_layer_proj_bits = {str(li): _norm_bits(b) for li, b in layer_bits.items()}
    all_bits = [v for pb in per_layer_proj_bits.values() for v in pb.values()]
    dst_meta = dict(src_meta)
    dst_meta["out_dir"] = dst_dir
    dst_meta["dims"] = dict(src_meta["dims"], group_size=dst_group,
                            bits=max(all_bits), per_layer_proj_bits=per_layer_proj_bits)
    dst_meta["requantized_from"] = {"dir": src_dir, "bits": src_bits, "group_size": src_group}
    with open(os.path.join(dst_dir, "_split_meta.json"), "w") as f:
        json.dump(dst_meta, f, ensure_ascii=False, indent=2)
    return {"files": len(files), "layers": len(layer_bits), "dst_group": dst_group,
            "elapsed_s": round(time.perf_counter() - t, 1)}


def _parse_bits_arg(arg: str):
    """解析 CLI bit 参数：'3' → 3；'gate=2,up=3,down=3' → {proj_proj: bits}。"""
    if "=" not in arg:
        return int(arg)
    alias = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    out = {}
    for pair in arg.split(","):
        k, v = pair.split("=")
        k = k.strip()
        out[alias.get(k, k)] = int(v)
    return out


def _main_layered(argv):
    """用法：requantize_experts.py --layered SRC DST BND MID_BITS BND_BITS [GROUP]
    例：--layered /src /dst 4 gate=2,up=3,down=3 gate=2,up=2,down=2 64
        首/尾各 4 个 MoE 层用 BND_BITS，其余用 MID_BITS。
    """
    src, dst = argv[0], argv[1]
    bnd = int(argv[2])
    mid_bits = _parse_bits_arg(argv[3])
    bnd_bits = _parse_bits_arg(argv[4])
    group = int(argv[5]) if len(argv) > 5 else 64
    with open(os.path.join(src, "_split_meta.json")) as f:
        moe_layers = json.load(f)["moe_layers"]
    layer_bits = boundary_scheme(moe_layers, bnd, bnd_bits, mid_bits)
    info = requantize_dir_layered(src, dst, layer_bits, group)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 统一/混合：requantize_experts.py SRC DST DST_BITS [GROUP]
    #   DST_BITS 统一 "3" 或 混合 "gate=2,up=3,down=3"
    # 逐层：requantize_experts.py --layered SRC DST BND MID_BITS BND_BITS [GROUP]
    if sys.argv[1] == "--layered":
        _main_layered(sys.argv[2:])
    else:
        src = sys.argv[1]
        dst = sys.argv[2]
        bits = _parse_bits_arg(sys.argv[3])
        group = int(sys.argv[4]) if len(sys.argv) > 4 else 64
        info = requantize_dir(src, dst, bits, group)
        print(json.dumps(info, ensure_ascii=False, indent=2))
