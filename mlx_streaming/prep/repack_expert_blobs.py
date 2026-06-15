"""把 compute buffer 重打包成"每专家一个连续 blob"格式（按层一个文件）。

blob 内顺序：gate[w,s,b] + up[w,s,b] + down[w,s,b]，正好 864KB（= 54×16KB 页对齐）。
读一个专家 = 1 次 pread(stride, e*stride)，而不是当前的 9 次散读。

环境变量：COMPUTE_BUFFER_DIR(源)、BLOB_DIR(输出)、LAYERS(逗号分隔，默认 15,25)
"""
import json
import os

import numpy as np

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512
SRC = os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/cb_2bit_g128")
OUT = os.environ.get("BLOB_DIR", "/tmp/cb_2bit_blob")
PROJS = (("gate_proj", INTER, HIDDEN), ("up_proj", INTER, HIDDEN), ("down_proj", HIDDEN, INTER))


def _layout():
    """返回 [(proj, tensor, nbytes_per_expert), ...] 和单专家总字节。"""
    segs = []
    for proj, out_dim, in_dim in PROJS:
        words = in_dim * BITS // 32
        groups = in_dim // GROUP
        segs.append((proj, "weight", out_dim * words * 4))
        segs.append((proj, "scales", out_dim * groups * 2))
        segs.append((proj, "biases", out_dim * groups * 2))
    return segs, sum(s[2] for s in segs)


def repack_layer(layer: int) -> dict:
    os.makedirs(OUT, exist_ok=True)
    segs, stride = _layout()
    # 源 .bin 以 uint8 原始字节映射，按专家字节偏移切片
    raw = {}
    for proj, _, _ in PROJS:
        base = os.path.join(SRC, f"layer{layer:02d}.{proj}")
        for tensor in ("weight", "scales", "biases"):
            raw[(proj, tensor)] = np.memmap(f"{base}.{tensor}.bin", dtype=np.uint8, mode="r")
    out_path = os.path.join(OUT, f"layer{layer:02d}.blob")
    with open(out_path, "wb") as f:
        for e in range(NUM_EXPERTS):
            for proj, tensor, nb in segs:
                buf = raw[(proj, tensor)]
                f.write(buf[e * nb:(e + 1) * nb].tobytes())
    assert os.path.getsize(out_path) == stride * NUM_EXPERTS
    index = {
        "format": "expert_blob_v1",
        "layer": layer,
        "num_experts": NUM_EXPERTS,
        "stride": stride,
        "page_aligned": stride % 16384 == 0,
        "segments": [{"proj": p, "tensor": t, "nbytes": n} for p, t, n in segs],
    }
    with open(os.path.join(OUT, f"layer{layer:02d}.blob.index.json"), "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return index


def _resolve_layers(spec: str) -> list[int]:
    """LAYERS=all → 读源 compute buffer 现有的全部层；否则按逗号解析。"""
    spec = spec.strip()
    if spec.lower() == "all":
        layers = []
        for name in os.listdir(SRC):
            if name.endswith(".gate_proj.weight.bin") and name.startswith("layer"):
                layers.append(int(name[len("layer"):len("layer") + 2]))
        return sorted(set(layers))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    layers = _resolve_layers(os.environ.get("LAYERS", "15,25"))
    _, stride = _layout()
    for L in layers:
        repack_layer(L)
    # 汇总 index，供 loader 校验/发现
    summary = {"format": "expert_blob_v1", "stride": stride,
               "page_aligned": stride % 16384 == 0, "num_experts": NUM_EXPERTS,
               "layers": layers}
    with open(os.path.join(OUT, "blob_index.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"out": OUT, "n_layers": len(layers), "stride_bytes": stride,
                      "page_aligned": stride % 16384 == 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
