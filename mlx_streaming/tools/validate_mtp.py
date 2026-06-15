"""teacher-forcing 实测 Qwen3-Next MTP 草稿接受率(spike,无生成循环)。

指标:
  mtp_vs_text_acc   : MTP 对 t_{i+2} 的 argmax 命中真实文本 token 的比例
  mtp_vs_greedy_acc : 先用主模型贪心生成参考序列 g,再以 g 为输入测命中 g_{i+2} 的比例
                      (真正的自投机接受率代理,决策主依据)
环境变量:MODEL / MTP_OUT(MTP 权重) / QN_CONFIG / PROMPT / MAXTOK / HIDDEN_VARIANT
         (建模相关 MODEL/EXPERT_DIR/EXPERT_SLOTS/HIDDEN_VARIANT 见 model_builder)
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.model_builder import build_streaming_model, capture_prenorm_hidden, greedy
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))


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
    model, tok, _store = build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT)
    # MTP 不含 embed_tokens,共享主模型的 embedding(与 vLLM/sglang 一致)
    mtp.embed_tokens = model.model.embed_tokens

    prompt_ids = mx.array([tok.encode(PROMPT)])
    greedy_ids = greedy(model, prompt_ids, MAXTOK)
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
        "hidden_variant": os.environ.get("HIDDEN_VARIANT", "pre_final_norm"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
