"""直接把 per-expert safetensors 打包成「每专家一个连续 blob」(按层一个文件)。

绕过 compute-buffer 中间体（省一份磁盘），字节布局与 repack_expert_blobs.py / blob_loader 完全一致：
每专家段顺序 = 对每个 proj(gate,up,down)依次写 [weight, scales, biases] 原始字节。
- weight: uint32（out_dim * (in_dim*bits//32) 个），scales/biases: bfloat16 按 uint16 原始 2 字节。

环境变量：EXPERT_DIR(源 per-expert) / BLOB_DIR(输出) / BITS / GROUP / LAYERS(逗号或 all)
"""
import json
import os

import mlx.core as mx
import numpy as np

EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts_8bit_g128")
OUT = os.environ.get("BLOB_DIR", "/tmp/cb_8bit_blob")
BITS = int(os.environ.get("BITS", "8"))
GROUP = int(os.environ.get("GROUP", "128"))
PROJS = (("gate_proj", 512, 2048), ("up_proj", 512, 2048), ("down_proj", 2048, 512))


def _meta():
    m = json.load(open(os.path.join(EXPERT_DIR, "_split_meta.json")))
    d = m["dims"]
    return int(d["num_experts"]), int(d["hidden"]), int(d["moe_intermediate"])


def _layout(hidden, inter):
    projs = (("gate_proj", inter, hidden), ("up_proj", inter, hidden), ("down_proj", hidden, inter))
    segs = []
    for proj, out_dim, in_dim in projs:
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        segs.append((proj, "weight", out_dim * words * 4))
        segs.append((proj, "scales", out_dim * groups * 2))
        segs.append((proj, "biases", out_dim * groups * 2))
    return segs, sum(s[2] for s in segs)


def _raw_bytes(arr) -> bytes:
    """uint32 直接取字节；bfloat16/其它按 uint16 重解释取 2 字节/元素。"""
    if arr.dtype == mx.uint32:
        return np.array(arr, copy=False).tobytes()
    return np.array(arr.view(mx.uint16), copy=False).tobytes()


def pack_layer(layer: int, num_experts: int, segs, stride) -> None:
    out_path = os.path.join(OUT, f"layer{layer:02d}.blob")
    with open(out_path, "wb") as f:
        for e in range(num_experts):
            path = os.path.join(EXPERT_DIR, f"layer{layer:02d}_expert{e:03d}.safetensors")
            w = mx.load(path)
            for proj, tensor, nb in segs:
                b = _raw_bytes(w[f"{proj}.{tensor}"])
                assert len(b) == nb, f"L{layer} e{e} {proj}.{tensor}: {len(b)}!={nb}"
                f.write(b)
    assert os.path.getsize(out_path) == stride * num_experts
    index = {
        "format": "expert_blob_v1", "layer": layer, "num_experts": num_experts,
        "stride": stride, "page_aligned": stride % 16384 == 0,
        "segments": [{"proj": p, "tensor": t, "nbytes": n} for p, t, n in segs],
    }
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
    num_experts, hidden, inter = _meta()
    segs, stride = _layout(hidden, inter)
    layers = _resolve_layers(os.environ.get("LAYERS", "all"), num_experts)
    for i, L in enumerate(layers):
        pack_layer(L, num_experts, segs, stride)
        print(f"  layer {L} packed ({i+1}/{len(layers)})", flush=True)
    summary = {"format": "expert_blob_v1", "stride": stride,
               "page_aligned": stride % 16384 == 0, "num_experts": num_experts,
               "layers": layers, "bits": BITS, "group_size": GROUP}
    with open(os.path.join(OUT, "blob_index.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": OUT, "n_layers": len(layers), "stride_bytes": stride,
                      "page_aligned": stride % 16384 == 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
