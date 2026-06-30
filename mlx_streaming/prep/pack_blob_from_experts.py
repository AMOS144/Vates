"""直接把 per-expert safetensors 打包成「每专家一个连续 blob」(按层一个文件)。

字节布局由 prep/blob_layout.py 统一描述（v1 affine / v2 mxfp4），与 blob_loader 完全一致：
按 _split_meta.json 的 blob_format 选段表，每专家依次写各 proj 的 [weight, scales(, biases)]。
- weight: uint32；v1 的 scales/biases 为 bf16 按 uint16 原始 2 字节；v2 mxfp4 的 scales 为 uint8。

环境变量：EXPERT_DIR(源 per-expert) / BLOB_DIR(输出) / BITS / GROUP / LAYERS(逗号或 all)
"""
import json
import os

import mlx.core as mx
import numpy as np

from mlx_streaming.prep.blob_layout import layout_for, BLOB_V1_AFFINE, BLOB_V2_MXFP4

EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts_8bit_g128")
OUT = os.environ.get("BLOB_DIR", "/tmp/cb_8bit_blob")
BITS = int(os.environ.get("BITS", "8"))
GROUP = int(os.environ.get("GROUP", "128"))


def _meta():
    m = json.load(open(os.path.join(EXPERT_DIR, "_split_meta.json")))
    d = m["dims"]
    fmt = m.get("blob_format", BLOB_V1_AFFINE)
    return (int(d["num_experts"]), int(d["hidden"]), int(d["moe_intermediate"]),
            fmt, d.get("quant_mode", "affine"))


def _raw_bytes(arr) -> bytes:
    """uint32 → 4B/元素；uint8 → 1B/元素；其余(affine 的 bf16 scales/biases) → uint16 2B/元素。"""
    if arr.dtype == mx.uint32:
        return np.array(arr, copy=False).tobytes()
    if arr.dtype == mx.uint8:
        return np.array(arr, copy=False).tobytes()
    return np.array(arr.view(mx.uint16), copy=False).tobytes()


def pack_layer(layer, num_experts, segs, stride, fmt, quant_mode) -> None:
    out_path = os.path.join(OUT, f"layer{layer:02d}.blob")
    with open(out_path, "wb") as f:
        for e in range(num_experts):
            w = mx.load(os.path.join(EXPERT_DIR, f"layer{layer:02d}_expert{e:03d}.safetensors"))
            for proj, tensor, dt, shape, nb in segs:
                b = _raw_bytes(w[f"{proj}.{tensor}"])
                assert len(b) == nb, f"L{layer} e{e} {proj}.{tensor}: {len(b)}!={nb}"
                f.write(b)
    assert os.path.getsize(out_path) == stride * num_experts
    index = {"format": fmt, "quant_mode": quant_mode, "layer": layer,
             "num_experts": num_experts, "stride": stride,
             "page_aligned": stride % 16384 == 0,
             "segments": [{"proj": p, "tensor": t, "nbytes": n} for p, t, _, _, n in segs]}
    with open(os.path.join(OUT, f"layer{layer:02d}.blob.index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def _resolve_layers(spec: str, num_experts: int) -> list:
    spec = spec.strip()
    if spec.lower() == "all":
        layers = set()
        for name in os.listdir(EXPERT_DIR):
            if name.startswith("layer") and name.endswith("_expert000.safetensors"):
                layers.add(int(name[5:7]))
        return sorted(layers)
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    os.makedirs(OUT, exist_ok=True)
    num_experts, hidden, inter, fmt, quant_mode = _meta()
    segs, stride = layout_for(fmt, hidden, inter, BITS, GROUP)
    layers = _resolve_layers(os.environ.get("LAYERS", "all"), num_experts)
    for i, L in enumerate(layers):
        pack_layer(L, num_experts, segs, stride, fmt, quant_mode)
        print(f"  layer {L} packed ({i+1}/{len(layers)})", flush=True)
    summary = {"format": fmt, "quant_mode": quant_mode, "stride": stride,
               "page_aligned": stride % 16384 == 0, "num_experts": num_experts,
               "layers": layers, "bits": BITS, "group_size": GROUP}
    with open(os.path.join(OUT, "blob_index.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": OUT, "n_layers": len(layers), "stride_bytes": stride,
                      "page_aligned": stride % 16384 == 0, "format": fmt}, ensure_ascii=False))


if __name__ == "__main__":
    main()
