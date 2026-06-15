"""把单层单 projection 打包成连续 compute buffers。

输出三个大 buffer:
- layer43.gate_proj.weight.bin
- layer43.gate_proj.scales.bin
- layer43.gate_proj.biases.bin

这个格式用于验证“整层少数大 buffer + custom qlinear kernel 直接按 expert_id 读取”。
"""
import argparse
import json
import os

from mlx_streaming.prep.pack_expert_ranges import read_safetensors_payloads


def pack_projection(src_dir: str, out_dir: str, layer: int, proj: str, num_experts: int) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    tensors = ["weight", "scales", "biases"]
    handles = {
        name: open(os.path.join(out_dir, f"layer{layer:02d}.{proj}.{name}.bin"), "wb")
        for name in tensors
    }
    meta = {
        "format": "mlx_streaming_compute_buffer_v1",
        "src_dir": src_dir,
        "layer": layer,
        "proj": proj,
        "num_experts": num_experts,
        "tensors": {},
    }
    try:
        for expert in range(num_experts):
            path = os.path.join(src_dir, f"layer{layer:02d}_expert{expert:03d}.safetensors")
            payloads = {rec["key"]: rec for rec in read_safetensors_payloads(path)}
            for name in tensors:
                key = f"{proj}.{name}"
                rec = payloads[key]
                f = handles[name]
                offset = f.tell()
                f.write(rec["payload"])
                entry = meta["tensors"].setdefault(name, {
                    "dtype": rec["dtype"],
                    "shape_per_expert": rec["shape"],
                    "nbytes_per_expert": len(rec["payload"]),
                    "file": f"layer{layer:02d}.{proj}.{name}.bin",
                    "offsets": [],
                })
                entry["offsets"].append(offset)
    finally:
        for f in handles.values():
            f.close()
    meta_path = os.path.join(out_dir, f"layer{layer:02d}.{proj}.index.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("EXPERT_DIR", ""))
    ap.add_argument("--out", default=os.environ.get("COMPUTE_BUFFER_DIR", ""))
    ap.add_argument("--layer", type=int, default=int(os.environ.get("LAYER", "43")))
    ap.add_argument("--proj", default=os.environ.get("PROJ", "gate_proj"))
    args = ap.parse_args()
    if not args.src:
        raise SystemExit("--src / EXPERT_DIR required")
    out_dir = args.out or os.path.join(args.src, "compute_buffers")
    with open(os.path.join(args.src, "_split_meta.json")) as f:
        model_meta = json.load(f)
    num_experts = int(model_meta["dims"]["num_experts"])
    meta = pack_projection(args.src, out_dir, args.layer, args.proj, num_experts)
    print(json.dumps({
        "out_dir": out_dir,
        "layer": args.layer,
        "proj": args.proj,
        "num_experts": num_experts,
        "files": {k: v["file"] for k, v in meta["tensors"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
