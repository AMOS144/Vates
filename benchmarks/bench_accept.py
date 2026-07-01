"""实测 K=3 的 MTP 接受分布(matched 直方图),不做几何分布假设。

emitted/step = min(matched+1, K)，本实现 verify 只喂 K-1 个草稿,故每步上限 = K。
逐位置接受率 α_i = P(matched≥i | matched≥i-1)：第 i 个草稿在前 i-1 个都命中的条件下被接受的概率。
"""
import os
import json

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core.mem import reset_peak
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
K = int(os.environ.get("K", "3"))


def main():
    reset_peak()
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp_quant = os.environ.get("MTP_QUANT", "1") == "1"     # 0=fp16 全精度 MTP
    mtp_bits = int(os.environ.get("MTP_BITS", "4"))          # MTP 线性层量化位宽(4/8)
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=mtp_quant, bits=mtp_bits)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    enc = mx.array([tok.encode(PROMPT)])
    ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                              ids_mode=True, profile=True)
    hist = stats["accept_hist"]                 # [恰好命中 0,1,...,K 个]
    steps = sum(hist)
    # 生存概率 P(matched≥i) 与逐位置条件接受率 α_i
    ge = [sum(hist[i:]) / max(1, steps) for i in range(K + 1)]   # ge[i]=P(matched≥i)
    alpha = [round(ge[i] / ge[i - 1], 4) if ge[i - 1] > 0 else 0.0 for i in range(1, K + 1)]
    out = {
        "K": K, "steps": steps, "tokens": len(ids),
        "mtp_quant": mtp_quant, "mtp_bits": mtp_bits if mtp_quant else "fp16",
        "avg_accept_len": stats["avg_accept_len"],
        "accept_hist": hist,
        "P(matched=j)": [round(h / max(1, steps), 4) for h in hist],
        "P(matched>=i)": [round(g, 4) for g in ge],
        "alpha_per_pos(1..K)": alpha,
        "max_per_step": K,
        "exact_match_note": "verify 只喂 K-1 草稿,emit 上限=K",
    }
    tp = stats.get("topk_probe")
    if tp:
        n = tp["n"]
        c1, c2, c3 = tp["cover_top1"], tp["cover_top2"], tp["cover_top3"]
        pos = []
        for i in range(K):
            ni = max(1, n[i])
            rej = ni - c1[i]                       # top1 未命中(=被拒)的步数
            pos.append({
                "position": i + 1,
                "n": n[i],
                "top1_hit_rate": round(c1[i] / ni, 4),
                "top2_cover": round(c2[i] / ni, 4),
                "top3_cover": round(c3[i] / ni, 4),
                # 被拒时的救回率:真实 token 在 top2/top3(但不在 top1)/ 被拒步数
                "rescue_top2_given_rejected": round((c2[i] - c1[i]) / max(1, rej), 4),
                "rescue_top3_given_rejected": round((c3[i] - c1[i]) / max(1, rej), 4),
            })
        out["topk_probe_by_position"] = pos
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
