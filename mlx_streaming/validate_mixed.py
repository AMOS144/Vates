"""在流式路径上对比各专家量化方案的困惑度（含混合精度）。

各目录从自己的 _split_meta.json 读 bits/group/proj_bits，故混合精度目录自动走逐 proj bit。
用较长文本求平均 NLL → 困惑度，信号比短句更稳。

环境变量：MODEL、SLOTS、DIRS（逗号分隔的 tag:dir，留空用默认集合）。
"""
import os
import json

import mlx.core as mx
from mlx_lm import load

from mlx_streaming.expert_store import FileExpertStore
from mlx_streaming.streaming_moe import patch_model_filebacked

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
SLOTS = int(os.environ.get("SLOTS", "128"))

DEFAULT_DIRS = {
    "4bit":  "/tmp/mlx_qwen3_experts",
    "2bit":  "/tmp/mlx_qwen3_experts_2bit",
    "mixA_g2u3d2": "/tmp/mlx_qwen3_experts_mixA",
    "mixB_g2u3d3": "/tmp/mlx_qwen3_experts_mixB",
    "mixL_bnd2bit": "/tmp/mlx_qwen3_experts_mixL",
    "3bit":  "/tmp/mlx_qwen3_experts_3bit",
}

TEXT = os.environ.get("TEXT", (
    "混合专家模型通过路由器为每个 token 选择少数专家参与计算，从而在巨大参数量下保持较低的"
    "激活计算成本。它的关键在于稀疏激活：虽然总参数量很大，但每个 token 只用到其中一小部分。"
    "在 Transformer 中，前馈层被替换为多个并行的专家网络，一个门控网络根据输入决定把 token "
    "分发给哪些专家，再把它们的输出按权重汇总。这样既扩大了模型容量，又不会让单次前向的计算量"
    "线性增长。训练时为避免负载不均，通常会加入辅助的均衡损失，鼓励各专家被均匀使用。"
    "推理时由于每个 token 只激活极少数专家，配合专家权重的按需加载，可以显著降低常驻内存占用。"
))


def _first_moe_dims(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            return {"hidden": gp.input_dims, "moe_inter": gp.output_dims}
    raise RuntimeError("无 MoE 层")


def _ppl(model, ids):
    x = ids[None, :-1]
    tgt = ids[1:]
    logits = model(x)[0]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    return float(mx.exp(mx.mean(nll)))


def _build(expert_dir):
    model, tok = load(MODEL, lazy=True)
    dims = _first_moe_dims(model)
    with open(os.path.join(expert_dir, "_split_meta.json")) as f:
        ed = json.load(f)["dims"]
    bits = ed.get("bits", 4)
    group = ed.get("group_size", 64)
    proj_bits = ed.get("proj_bits")
    layer_proj_bits = None
    if "per_layer_proj_bits" in ed:
        layer_proj_bits = {int(k): v for k, v in ed["per_layer_proj_bits"].items()}
    store = FileExpertStore(expert_dir, capacity=SLOTS)
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           group, bits, proj_bits=proj_bits, layer_proj_bits=layer_proj_bits)
    return model, tok


def main():
    _, tok = load(MODEL, lazy=True)
    ids = mx.array(tok.encode(TEXT))

    out = {}
    for tag, d in DEFAULT_DIRS.items():
        if not os.path.exists(os.path.join(d, "_split_meta.json")):
            continue
        model, _ = _build(d)
        out[tag] = round(_ppl(model, ids), 3)
        print(f"{tag:14s} ppl={out[tag]}", flush=True)
        del model

    print(json.dumps({"slots": SLOTS, "n_tokens": int(ids.size), "ppl": out},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
