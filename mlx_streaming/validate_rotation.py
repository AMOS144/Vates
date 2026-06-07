"""验证旋转 2-bit 是否把质量拉近 4-bit：
1) 整段文本上对比 4-bit / plain-2bit / rotated-2bit 的困惑度（token NLL）。
2) rot_recovers = ppl(plain) - ppl(rot)，>0 表示旋转救回质量。
环境变量：MODEL、DIR_4BIT、DIR_2BIT、DIR_2BIT_ROT、SLOTS、TEXT。
"""
import os
import json

import mlx.core as mx
from mlx_lm import load

from mlx_streaming.expert_store import FileExpertStore
from mlx_streaming.streaming_moe import patch_model_filebacked

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
DIR_4BIT = os.environ.get("DIR_4BIT", "/tmp/mlx_qwen3_experts")
DIR_2BIT = os.environ.get("DIR_2BIT", "/tmp/mlx_qwen3_experts_2bit")
DIR_2BIT_ROT = os.environ.get("DIR_2BIT_ROT", "/tmp/mlx_qwen3_experts_2bit_rot")
SLOTS = int(os.environ.get("SLOTS", "96"))
TEXT = os.environ.get("TEXT", "混合专家模型通过路由器为每个 token 选择少数专家参与计算，"
                                "从而在巨大参数量下保持较低的激活计算成本。"
                                "它的关键在于稀疏激活：虽然总参数量很大，"
                                "但每个 token 只用到其中一小部分。")


def _first_moe_dims(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            return {"hidden": gp.input_dims, "moe_inter": gp.output_dims}
    raise RuntimeError("无 MoE 层")x


def _ppl(model, ids):
    # teacher-forcing：用前缀预测下一 token，算平均 NLL → 困惑度
    x = ids[None, :-1]
    tgt = ids[1:]
    logits = model(x)[0]                       # (L-1, vocab)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    return float(mx.exp(mx.mean(nll)))


def _build(expert_dir, bits, rotated):
    model, tok = load(MODEL, lazy=True)
    dims = _first_moe_dims(model)
    store = FileExpertStore(expert_dir, capacity=SLOTS)
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           64, bits, rotated=rotated)
    return model, tok


def main():
    _, tok = load(MODEL, lazy=True)
    ids = mx.array(tok.encode(TEXT))

    out = {}
    for tag, (d, bits, rot) in {
        "4bit": (DIR_4BIT, 4, False),
        "2bit_plain": (DIR_2BIT, 2, False),
        "2bit_rot": (DIR_2BIT_ROT, 2, True),
    }.items():
        model, _ = _build(d, bits, rot)
        out[tag] = round(_ppl(model, ids), 3)
        del model

    print(json.dumps({"slots": SLOTS, "ppl": out,
                      "rot_recovers": round(out["2bit_plain"] - out["2bit_rot"], 3)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
