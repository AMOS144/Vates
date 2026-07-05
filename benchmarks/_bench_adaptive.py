"""置信度门控动态深度消融:fixed K=3 vs 多组自适应(tau × depth_max)。

信号:avg_accept_len(近确定性)看接受长度变化;tok/s 多轮中位数看净收益;lossless 用逐轮 vs
fixed 首轮的最大失配(应为 0,自适应只提交模型 argmax token → bit-lossless)。
配置(与生产一致):
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 .venv/bin/python benchmarks/_bench_adaptive.py
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mtp.bench_verdict import median
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

K = int(os.environ.get("K", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "96"))
REPEAT = int(os.environ.get("REPEAT", "3"))

PROMPTS = [
    "用三句话解释什么是混合专家模型。",
    "写一段 Python 代码，演示如何用 LRU 缓存函数结果。",
    "为什么模型量化会影响困惑度和生成质量？",
    "用英文写一段关于 speculative decoding 的技术摘要。",
    "请写一个短故事，主题是工程师在午夜调试模型推理性能。",
    "请给出一个使用 Python 解析 JSONL 文件并统计字段频率的例子。",
]

# (名称, ADAPTIVE, TAU, DEPTH_MAX)
CONFIGS = [
    ("fixed-K3",       "0", "0.3", "3"),
    ("adap t.3 d4",    "1", "0.3", "4"),
    ("adap t.5 d4",    "1", "0.5", "4"),
    ("adap t.3 d3",    "1", "0.3", "3"),   # 纯向下收缩(不扩展到 4),隔离 K=4 贡献
]


def _set(cfg):
    _, ad, tau, dmax = cfg
    os.environ["MTP_ADAPTIVE_DEPTH"] = ad
    os.environ["MTP_CONF_TAU"] = tau
    os.environ["MTP_DEPTH_MAX"] = dmax
    os.environ["TREE_TOP2"] = "0"


def _run(model, drafter, tok, enc, store):
    store.reset_stats()
    ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                              ids_mode=True, profile=True)
    return ids, stats, round(stats["tokens"] / stats["wall_s"], 2)


def _mm(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    _enc = mx.array([tok.encode(PROMPTS[0])])
    for c in CONFIGS:
        _set(c)
        _run(model, drafter, tok, _enc, store)

    rows = []
    for p in PROMPTS:
        enc = mx.array([tok.encode(p)])
        ref = None
        rec = {"prompt": p[:14]}
        for c in CONFIGS:
            name = c[0]
            _set(c)
            tps_runs, acc, mm = [], None, 0
            for _r in range(REPEAT):
                ids, stats, tps = _run(model, drafter, tok, enc, store)
                tps_runs.append(tps)
                acc = stats["avg_accept_len"]
                if name == "fixed-K3" and ref is None:
                    ref = ids
                elif ref is not None:
                    mm = max(mm, _mm(ids, ref))
            rec[name] = {"tps": median(tps_runs), "acc": acc, "mm": mm}
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    print("\n=== 动态深度消融汇总(6 prompt 平均)===")
    print(f"{'config':>12} {'acc_len':>8} {'tps_med_avg':>12} {'max_mm':>7} {'vs_fixed_tps':>13}")
    base_tps = sum(r["fixed-K3"]["tps"] for r in rows) / len(rows)
    for name in [c[0] for c in CONFIGS]:
        acc = sum(r[name]["acc"] for r in rows) / len(rows)
        tps = sum(r[name]["tps"] for r in rows) / len(rows)
        mm = max(r[name]["mm"] for r in rows)
        d = (tps / base_tps - 1) * 100
        print(f"{name:>12} {acc:>8.3f} {tps:>12.2f} {mm:>7} {d:>+12.2f}%")


if __name__ == "__main__":
    main()
