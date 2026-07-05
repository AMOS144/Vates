"""逐位置 top-1/2/3 覆盖率探针:量 MTP 每个草稿位置的"主模型真值落在 top-k"比例。

用途:界定按需深化救回的每位置天花板——
  gap(pos_i) = cover_top2[i] - cover_top1[i] = 该位置 top-1 被拒但 top-2 命中的比例(可救回上界)。
走 plain 单链草稿路径(TREE_TOP2=0)才会采集 draft_cands,故本探针强制关树。
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

os.environ["TREE_TOP2"] = "0"          # 关树 → 走 plain 单链,采集每位置 top-k 候选
K = int(os.environ.get("K", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "96"))

PROMPTS = [
    "用三句话解释什么是混合专家模型。",
    "写一段 Python 代码，演示如何用 LRU 缓存函数结果。",
    "为什么模型量化会影响困惑度和生成质量？",
    "用英文写一段关于 speculative decoding 的技术摘要。",
    "请写一个短故事，主题是工程师在午夜调试模型推理性能。",
    "请给出一个使用 Python 解析 JSONL 文件并统计字段频率的例子。",
]


def main():
    os.environ["ACCEPT_TOPK"] = "3"
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    agg_n = [0] * K
    agg_c1 = [0] * K
    agg_c2 = [0] * K
    agg_c3 = [0] * K
    for p in PROMPTS:
        enc = mx.array([tok.encode(p)])
        _ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                                   ids_mode=True, profile=True)
        pr = stats.get("topk_probe")
        if not pr:
            continue
        for i in range(K):
            agg_n[i] += pr["n"][i]
            agg_c1[i] += pr["cover_top1"][i]
            agg_c2[i] += pr["cover_top2"][i]
            agg_c3[i] += pr["cover_top3"][i]

    print("\n=== 逐位置 top-1/2/3 覆盖率(6 prompt 汇总) ===")
    print(f"{'pos':>3} {'n':>6} {'top1':>7} {'top2':>7} {'top3':>7} "
          f"{'gap21(可救)':>12} {'gap32':>8}")
    for i in range(K):
        n = max(agg_n[i], 1)
        t1, t2, t3 = agg_c1[i] / n, agg_c2[i] / n, agg_c3[i] / n
        print(f"{i:>3} {agg_n[i]:>6} {t1:>7.3f} {t2:>7.3f} {t3:>7.3f} "
              f"{t2 - t1:>12.3f} {t3 - t2:>8.3f}")


if __name__ == "__main__":
    main()
