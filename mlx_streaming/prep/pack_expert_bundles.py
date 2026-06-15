"""把 per-expert safetensors 打包成 per-layer bundle。

输入目录:
  layer00_expert000.safetensors
  layer00_expert001.safetensors
  ...

输出目录:
  layer_bundles/layer00.safetensors

bundle 内 key:
  expert000.gate_proj.weight
  expert000.gate_proj.scales
  ...

运行时设置 EXPERT_BUNDLE=1 即可优先读取 bundle。
"""
import json
import os
import time

import mlx.core as mx

SRC_DIR = os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts")
OUT_DIR = os.environ.get("EXPERT_BUNDLE_DIR", os.path.join(SRC_DIR, "layer_bundles"))


def _pack_layer(src_dir: str, out_dir: str, layer: int, num_experts: int) -> str:
    out = {}
    for e in range(num_experts):
        path = os.path.join(src_dir, f"layer{layer:02d}_expert{e:03d}.safetensors")
        rec = mx.load(path)
        prefix = f"expert{e:03d}."
        for k, v in rec.items():
            out[prefix + k] = v
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"layer{layer:02d}.safetensors")
    mx.save_safetensors(out_path, out)
    return out_path


def main():
    t0 = time.perf_counter()
    with open(os.path.join(SRC_DIR, "_split_meta.json")) as f:
        meta = json.load(f)
    layers = [int(x) for x in meta["moe_layers"]]
    num_experts = int(meta["dims"]["num_experts"])
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, layer in enumerate(layers):
        out_path = _pack_layer(SRC_DIR, OUT_DIR, layer, num_experts)
        print(f"  {i + 1}/{len(layers)} layer{layer:02d} -> {out_path} "
              f"({round(time.perf_counter() - t0, 1)}s)", flush=True)
    bundle_meta = {
        "src_dir": SRC_DIR,
        "out_dir": OUT_DIR,
        "layers": layers,
        "num_experts": num_experts,
        "format": "expert{EEE}.{tensor_key}",
    }
    with open(os.path.join(OUT_DIR, "_bundle_meta.json"), "w") as f:
        json.dump(bundle_meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "out_dir": OUT_DIR,
        "layers": len(layers),
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
