"""把 per-expert safetensors 的原始 payload 打包成 per-layer range pack。

这个格式只用于 de-risk：避免运行时解析大量小 safetensors header，先验证
“按 expert byte range 读取”是否比多个 `mx.load` 更快。
"""
import argparse
import json
import os
import struct
import time
from pathlib import Path

ALIGN = 64


def _align(n: int, align: int = ALIGN) -> int:
    rem = n % align
    return n if rem == 0 else n + (align - rem)


def read_safetensors_payloads(path: str) -> list[dict]:
    """读取 safetensors header 和每个 tensor 的原始 payload bytes。"""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out = []
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            payload = f.read(end - start)
            out.append({
                "key": key,
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "payload": payload,
            })
        return out


def pack_layer(src_dir: str, out_dir: str, layer: int, num_experts: int) -> dict:
    """打包单层所有专家，返回 index dict。"""
    os.makedirs(out_dir, exist_ok=True)
    pack_path = os.path.join(out_dir, f"layer{layer:02d}.pack")
    index_path = os.path.join(out_dir, f"layer{layer:02d}.index.json")
    tsv_path = os.path.join(out_dir, f"layer{layer:02d}.idx")
    tensors = []
    experts = []
    with open(pack_path, "wb") as pack:
        for expert in range(num_experts):
            src = os.path.join(src_dir, f"layer{layer:02d}_expert{expert:03d}.safetensors")
            expert_start = _align(pack.tell())
            if expert_start > pack.tell():
                pack.write(b"\0" * (expert_start - pack.tell()))
            for rec in read_safetensors_payloads(src):
                offset = _align(pack.tell())
                if offset > pack.tell():
                    pack.write(b"\0" * (offset - pack.tell()))
                payload = rec["payload"]
                pack.write(payload)
                tensors.append({
                    "layer": layer,
                    "expert_id": expert,
                    "key": rec["key"],
                    "dtype": rec["dtype"],
                    "shape": rec["shape"],
                    "offset": offset,
                    "nbytes": len(payload),
                })
            expert_end = pack.tell()
            experts.append({
                "layer": layer,
                "expert_id": expert,
                "offset": expert_start,
                "nbytes": expert_end - expert_start,
            })
    index = {
        "format": "mlx_streaming_expert_pack_v1",
        "alignment": ALIGN,
        "layer": layer,
        "num_experts": num_experts,
        "pack": os.path.basename(pack_path),
        "experts": experts,
        "tensors": tensors,
    }
    with open(index_path, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    with open(tsv_path, "w") as f:
        f.write("kind\tlayer\texpert_id\tkey\tdtype\tshape\toffset\tnbytes\n")
        for rec in experts:
            f.write(
                f"EXPERT\t{rec['layer']}\t{rec['expert_id']}\t*\t*\t*\t"
                f"{rec['offset']}\t{rec['nbytes']}\n"
            )
        for rec in tensors:
            shape = ",".join(str(x) for x in rec["shape"])
            f.write(
                f"TENSOR\t{rec['layer']}\t{rec['expert_id']}\t{rec['key']}\t"
                f"{rec['dtype']}\t{shape}\t{rec['offset']}\t{rec['nbytes']}\n"
            )
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts"))
    ap.add_argument("--out", default=os.environ.get("EXPERT_PACK_DIR"))
    ap.add_argument("--layers", default=os.environ.get("LAYERS", ""))
    args = ap.parse_args()
    src = args.src
    out = args.out or os.path.join(src, "layer_packs")
    with open(os.path.join(src, "_split_meta.json")) as f:
        meta = json.load(f)
    all_layers = [int(x) for x in meta["moe_layers"]]
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else all_layers
    num_experts = int(meta["dims"]["num_experts"])
    t0 = time.perf_counter()
    for i, layer in enumerate(layers):
        pack_layer(src, out, layer, num_experts)
        print(f"  {i + 1}/{len(layers)} layer{layer:02d} ({round(time.perf_counter() - t0, 1)}s)", flush=True)
    with open(os.path.join(out, "_pack_meta.json"), "w") as f:
        json.dump({
            "src": src,
            "out": out,
            "layers": layers,
            "num_experts": num_experts,
            "format": "mlx_streaming_expert_pack_v1",
        }, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": out, "layers": len(layers)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
