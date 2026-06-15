"""路线 B 离线工具：把模型里堆叠的 switch_mlp 专家权重按专家拆成 per-expert 小文件。

拆分后每个文件 layer{L:02d}_expert{E:03d}.safetensors 含扁平 dict：
  gate_proj.weight / gate_proj.scales / gate_proj.biases
  up_proj.weight   / up_proj.scales   / up_proj.biases
  down_proj.weight / down_proj.scales / down_proj.biases
（非量化模型则只有 .weight，可能还有 .bias）

一次拆分、逐专家物化，全程低内存。
"""
import os
import sys
import json

import mlx.core as mx

PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def split_switch_glu(switch_glu, out_dir: str, layer: int) -> int:
    """把一个 SwitchGLU 的三组 SwitchLinear 沿专家维拆成 per-expert 文件。返回专家数。"""
    os.makedirs(out_dir, exist_ok=True)
    E = switch_glu.gate_proj.num_experts
    for e in range(E):
        d = {}
        for proj_name in PROJ_NAMES:
            proj = getattr(switch_glu, proj_name)
            for pname, p in proj.parameters().items():
                if isinstance(p, mx.array) and p.ndim >= 1 and p.shape[0] == E:
                    d[f"{proj_name}.{pname}"] = p[e]
        mx.eval(d)   # 只物化这一个专家
        path = os.path.join(out_dir, f"layer{layer:02d}_expert{e:03d}.safetensors")
        mx.save_safetensors(path, d)
    return E


def split_model(model_path: str, out_dir: str) -> dict:
    """加载模型（lazy）并把所有 MoE 层的专家拆到 out_dir。返回 {dims/统计}。"""
    from mlx_lm import load
    model, _ = load(model_path, lazy=True)
    os.makedirs(out_dir, exist_ok=True)
    moe_layers = []
    dims = None
    for l, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            sm = mlp.switch_mlp
            split_switch_glu(sm, out_dir, l)
            moe_layers.append(l)
            if dims is None:
                gp = sm.gate_proj
                dims = {
                    "hidden": gp.input_dims,
                    "moe_intermediate": gp.output_dims,
                    "num_experts": gp.num_experts,
                    "group_size": getattr(gp, "group_size", None),
                    "bits": getattr(gp, "bits", None),
                }
    meta = {"out_dir": out_dir, "moe_layers": moe_layers, "dims": dims}
    with open(os.path.join(out_dir, "_split_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    mp = sys.argv[1]
    od = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mlx_qwen3_experts"
    m = split_model(mp, od)
    print(json.dumps(m, ensure_ascii=False, indent=2))
