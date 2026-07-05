"""pos1 救回增量三向消融:off / pos0-only / pos0+pos1。

关键信号用 avg_accept_len(近确定性,不受热漂移影响)判断 pos1 是否真的抬升接受率;
tok/s 取多轮中位数辅证净收益。lossless 用逐轮 vs off 首轮的最大失配数(应为 0)。
配置(与生产一致):
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 .venv/bin/python benchmarks/_bench_p1_ablation.py
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
MAXTOK = int(os.environ.get("MAXTOK", "128"))
REPEAT = int(os.environ.get("REPEAT", "4"))

PROMPTS = [
    "用三句话解释什么是混合专家模型。",
    "写一段 Python 代码，演示如何用 LRU 缓存函数结果。",
    "为什么模型量化会影响困惑度和生成质量？",
    "用英文写一段关于 speculative decoding 的技术摘要。",
    "请写一个短故事，主题是工程师在午夜调试模型推理性能。",
    "请给出一个使用 Python 解析 JSONL 文件并统计字段频率的例子。",
]

# (名称, TREE_TOP2, TREE_TOP2_P1)
CONFIGS = [
    ("off",       "0", "0"),
    ("pos0",      "1", "0"),
    ("pos0+pos1", "1", "1"),
]


def _set(cfg):
    _, t2, p1 = cfg
    os.environ["TREE_TOP2"] = t2
    os.environ["TREE_TOP2_P1"] = p1


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

    # warmup 三配置各一遍
    _enc = mx.array([tok.encode(PROMPTS[0])])
    for c in CONFIGS:
        _set(c)
        _run(model, drafter, tok, _enc, store)

    rows = []
    for p in PROMPTS:
        enc = mx.array([tok.encode(p)])
        ref = None
        rec = {"prompt": p[:16]}
        for c in CONFIGS:
            name = c[0]
            _set(c)
            tps_runs, acc, resc, resc1 = [], None, 0, 0
            mm = 0
            for _r in range(REPEAT):
                ids, stats, tps = _run(model, drafter, tok, enc, store)
                tps_runs.append(tps)
                acc = stats["avg_accept_len"]
                resc = stats["tree_rescues"]
                resc1 = stats["tree_rescues_p1"]
                if name == "off" and ref is None:
                    ref = ids
                elif ref is not None:
                    mm = max(mm, _mm(ids, ref))
            rec[name] = {"tps": median(tps_runs), "acc": acc,
                         "resc": resc, "resc1": resc1, "mm": mm}
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    # 汇总:accept-len 与 tps 的相对增量
    def _avg(key, sub):
        return sum(r[key][sub] for r in rows) / len(rows)

    print("\n=== 三向消融汇总(6 prompt 平均)===")
    print(f"{'config':>10} {'acc_len':>8} {'tps_med_avg':>12} {'resc':>6} {'resc1':>6} {'max_mm':>7}")
    for name in ("off", "pos0", "pos0+pos1"):
        acc = _avg(name, "acc")
        tps = sum(r[name]["tps"] for r in rows) / len(rows)
        resc = sum(r[name]["resc"] for r in rows)
        resc1 = sum(r[name]["resc1"] for r in rows)
        mm = max(r[name]["mm"] for r in rows)
        print(f"{name:>10} {acc:>8.3f} {tps:>12.2f} {resc:>6} {resc1:>6} {mm:>7}")

    a_off = _avg("off", "acc")
    a_p0 = _avg("pos0", "acc")
    a_p01 = _avg("pos0+pos1", "acc")
    print(f"\naccept_len: pos0 vs off = {(a_p0/a_off-1)*100:+.2f}% | "
          f"pos0+pos1 vs pos0 = {(a_p01/a_p0-1)*100:+.2f}% (← pos1 增量) | "
          f"pos0+pos1 vs off = {(a_p01/a_off-1)*100:+.2f}%")


if __name__ == "__main__":
    main()
