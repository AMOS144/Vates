"""teacher-forcing 实测 Qwen3-Next MTP 草稿接受率(spike,无生成循环)。

指标:
  mtp_vs_text_acc   : MTP 对 t_{i+2} 的 argmax 命中真实文本 token 的比例
  mtp_vs_greedy_acc : 先用主模型贪心生成参考序列 g,再以 g 为输入测命中 g_{i+2} 的比例
                      (真正的自投机接受率代理,决策主依据)
环境变量:MODEL / MTP_OUT(MTP 权重) / QN_CONFIG / PROMPT / MAXTOK / HIDDEN_VARIANT
"""
import json
import os

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.expert_store import FileExpertStore
from mlx_streaming.qwen3_next_mtp import load_mtp
from mlx_streaming.streaming_moe import patch_model_filebacked

MODEL = os.environ.get("MODEL", "/tmp/qwen3_next_80b_4bit")
EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts")
EXPERT_SLOTS = int(os.environ.get("EXPERT_SLOTS", "96"))
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
# pre_final_norm(默认)| post_final_norm(排错时切换)
HIDDEN_VARIANT = os.environ.get("HIDDEN_VARIANT", "pre_final_norm")


def _build_streaming_model():
    """用文件后端流式 patch 加载主模型(32GB 机器装不下 41GB 非流式)。"""
    model, tok = load(MODEL, lazy=True)
    # 取首个 MoE 维度
    dims = None
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            dims = {"hidden": gp.input_dims, "moe_inter": gp.output_dims,
                    "group_size": getattr(gp, "group_size", 64),
                    "bits": getattr(gp, "bits", 4)}
            break
    bits, group, proj_bits, layer_proj_bits = dims["bits"], dims["group_size"], None, None
    meta_path = os.path.join(EXPERT_DIR, "_split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ed = json.load(f).get("dims", {})
        bits = ed.get("bits", bits)
        group = ed.get("group_size", group)
        proj_bits = ed.get("proj_bits")
        if "per_layer_proj_bits" in ed:
            layer_proj_bits = {int(k): v for k, v in ed["per_layer_proj_bits"].items()}
    # 可选：每层池预算 profile(probe_pool_footprint 产出)，按各层真实工作集分配，
    # 无损省内存(命中率/输出/吞吐不变，仅不再为低占用层预留满 capacity)。
    layer_caps = None
    profile_path = os.environ.get("EXPERT_POOL_PROFILE", "")
    if profile_path and os.path.exists(profile_path):
        with open(profile_path) as f:
            prof = json.load(f)
        layer_caps = {int(k): int(v) for k, v in prof.get("layer_caps", {}).items()}
    store = FileExpertStore(EXPERT_DIR, capacity=EXPERT_SLOTS, layer_caps=layer_caps)
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           group, bits, proj_bits=proj_bits,
                           layer_proj_bits=layer_proj_bits)
    return model, tok, store


def capture_prenorm_hidden(model, input_ids: mx.array) -> mx.array:
    """跑主模型层循环但跳过最后的 model.norm,返回 last-layer hidden(norm 前)。

    HIDDEN_VARIANT=post_final_norm 时返回 norm 之后(用于消歧排错)。
    """
    inner = model.model
    h = inner.embed_tokens(input_ids)
    layers = inner.layers
    if not layers:
        return h
    cache = model.make_cache()
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx])
    ssm_mask = create_ssm_mask(h, cache[ssm_idx])
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
    if HIDDEN_VARIANT == "post_final_norm":
        h = inner.norm(h)
    return h


def _greedy(model, input_ids: mx.array, n: int) -> mx.array:
    cache = model.make_cache()
    cur = input_ids
    out = []
    for _ in range(n):
        logits = model(cur, cache=cache)
        nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        out.append(nxt)
        cur = nxt
        mx.eval(nxt)
    return mx.concatenate([input_ids] + out, axis=1)


def _acceptance(model, mtp, ids: mx.array) -> float:
    """teacher forcing:对 ids[0..L-1],MTP 预测 t_{i+2},与 ids[i+2] 比。"""
    hidden = capture_prenorm_hidden(model, ids)          # (1, L, H)
    next_ids = ids[:, 1:]                                  # 位置 i 喂 ids[i+1]
    hid = hidden[:, :-1, :]                                # 对齐 (1, L-1, H)
    logits = mtp(hid, next_ids, model.lm_head)             # (1, L-1, vocab)
    pred = mx.argmax(logits, axis=-1)                      # 预测 t_{i+2}
    target = ids[:, 2:]                                    # (1, L-2)
    pred = pred[:, : target.shape[1]]
    match = (pred == target).astype(mx.float32)
    return float(match.mean())


def main():
    model, tok, _store = _build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT)
    # MTP 不含 embed_tokens,共享主模型的 embedding(与 vLLM/sglang 一致)
    mtp.embed_tokens = model.model.embed_tokens

    prompt_ids = mx.array([tok.encode(PROMPT)])
    greedy_ids = _greedy(model, prompt_ids, MAXTOK)
    mx.eval(greedy_ids)

    greedy_acc = _acceptance(model, mtp, greedy_ids)
    nat = mx.array([tok.encode(
        PROMPT
        + "混合专家模型通过门控网络为每个 token 选择少量专家参与计算,"
        + "从而在巨大参数量下保持较低的激活成本。"
    )])
    text_acc = _acceptance(model, mtp, nat)

    print(json.dumps({
        "mtp_vs_greedy_acc": round(greedy_acc, 4),
        "mtp_vs_text_acc": round(text_acc, 4),
        "n_greedy_positions": int(greedy_ids.shape[1] - 2),
        "hidden_variant": HIDDEN_VARIANT,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
