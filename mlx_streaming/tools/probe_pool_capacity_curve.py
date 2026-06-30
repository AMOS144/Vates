"""每层工作集 + LRU 命中率-容量曲线。

采集一段 MTP 生成里每层「每次 acquire 的唯一专家序列」，按层模拟 ResidentExpertPool 的
LRU 行为（当前调用的专家永不被驱逐），扫描容量 C 给出命中率，并算出「保住 X% 命中所需的最小 C」。
用于把「全层统一 96 槽」改成逐层自适应容量。
"""
import json
import os
from collections import OrderedDict, defaultdict

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

import mlx_streaming.mtp.generate as mg
from mlx_streaming.core import route_trace
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

from mlx_streaming import config as _cfg
QN_CONFIG = _cfg.qn_config()
MTP_OUT = _cfg.mtp_out()
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "2"))
CAPS = [int(x) for x in os.environ.get("CAPS", "16,24,32,48,64,96").split(",")]
THRESHOLDS = [float(x) for x in os.environ.get("THRESHOLDS", "0.95,0.99").split(",")]
# 跳过每层前 SKIP_FIRST 次调用（prefill 把整段 prompt 一次路由，唯一专家暴多，污染 decode 工作集）。
SKIP_FIRST = int(os.environ.get("SKIP_FIRST", "1"))

SEQ: "dict[int, list[list[int]]]" = defaultdict(list)  # layer -> [每次 acquire 的唯一专家列表]


def traced_forward_with_hidden(model, ids, cache):
    route_trace.enable()
    logits, H = mg._ORIG_FORWARD_WITH_HIDDEN(model, ids, cache)
    mx.eval(logits, H)
    for rec in route_trace.events():
        uniq = list(dict.fromkeys(int(e) for e in rec["experts"]))
        SEQ[int(rec["layer"])].append(uniq)
    route_trace.disable()
    return logits, H


def lru_hit_rate(seq: "list[list[int]]", cap: int) -> float:
    """模拟 ResidentExpertPool：当前调用专家受保护不被驱逐；命中=已驻留。返回专家粒度命中率。"""
    lru: "OrderedDict[int, None]" = OrderedDict()
    hits = misses = 0
    for uniq in seq:
        cur = set(uniq)
        for e in uniq:
            if e in lru:
                hits += 1
                lru.move_to_end(e)
            else:
                misses += 1
                lru[e] = None
                while len(lru) > cap:
                    victim = next((k for k in lru if k not in cur), None)
                    if victim is None:
                        break          # 全是当前专家 → 装不下，本调用超容量（cap 太小）
                    del lru[victim]
    tot = hits + misses
    return hits / tot if tot else 1.0


def lfu_hit_rate(seq: "list[list[int]]", cap: int) -> float:
    """LFU 驱逐：满了驱逐历史使用次数最少的（当前调用专家受保护）。"""
    from collections import defaultdict
    resident: set = set()
    freq: "dict[int,int]" = defaultdict(int)
    hits = misses = 0
    for uniq in seq:
        cur = set(uniq)
        for e in uniq:
            freq[e] += 1
            if e in resident:
                hits += 1
            else:
                misses += 1
                resident.add(e)
                while len(resident) > cap:
                    victim = min((x for x in resident if x not in cur), key=lambda x: freq[x], default=None)
                    if victim is None:
                        break
                    resident.discard(victim)
    tot = hits + misses
    return hits / tot if tot else 1.0


def belady_hit_rate(seq: "list[list[int]]", cap: int) -> float:
    """Belady 最优（上帝视角）：满了就驱逐"下一次使用最晚/永不再用"的专家。任何驱逐策略的命中上界。"""
    n = len(seq)
    occ: "dict[int, list[int]]" = {}      # 专家 -> 出现的调用号（升序）
    for t, uniq in enumerate(seq):
        for e in set(uniq):
            occ.setdefault(e, []).append(t)
    import bisect
    INF = n + 1
    def next_use(e: int, t: int) -> int:
        lst = occ.get(e, [])
        i = bisect.bisect_right(lst, t)
        return lst[i] if i < len(lst) else INF
    resident: set = set()
    hits = misses = 0
    for t, uniq in enumerate(seq):
        cur = set(uniq)
        for e in uniq:
            if e in resident:
                hits += 1
            else:
                misses += 1
                resident.add(e)
                while len(resident) > cap:
                    victim = max((x for x in resident if x not in cur),
                                 key=lambda x: next_use(x, t), default=None)
                    if victim is None:
                        break
                    resident.discard(victim)
    tot = hits + misses
    return hits / tot if tot else 1.0


def main():
    os.environ["ROUTE_TRACE"] = "1"
    model, tok, _store = build_streaming_model()
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    mg._ORIG_FORWARD_WITH_HIDDEN = mg.forward_with_hidden
    mg.forward_with_hidden = traced_forward_with_hidden
    try:
        ids, stats = mtp_generate(model, drafter, tok, mx.array([tok.encode(PROMPT)]),
                                  MAXTOK, K=K, ids_mode=True, profile=False)
    finally:
        mg.forward_with_hidden = mg._ORIG_FORWARD_WITH_HIDDEN

    per_layer = []
    for li in sorted(SEQ):
        seq = SEQ[li][SKIP_FIRST:]          # 丢掉 prefill 调用，只看 decode/verify 稳态
        if not seq:
            continue
        working_set = len(set(e for s in seq for e in s))
        max_call = max((len(s) for s in seq), default=0)
        curve = {str(c): round(lru_hit_rate(seq, c), 4) for c in CAPS}
        min_cap = {}
        for th in THRESHOLDS:
            c = next((c for c in range(max_call, max(CAPS) + 1) if lru_hit_rate(seq, c) >= th), None)
            min_cap[f"ge_{th}"] = c
        per_layer.append({"layer": li, "working_set": working_set, "max_call_uniq": max_call,
                          "hit_at_cap": curve, "min_cap": min_cap})

    # 聚合：各容量下的「全层均值命中」+ 「保 99% 命中所需 cap 的分布」
    agg_hit = {str(c): round(sum(lru_hit_rate(SEQ[li][SKIP_FIRST:], c) for li in SEQ) / max(1, len(SEQ)), 4) for c in CAPS}
    # 策略对照（全层均值命中）：LRU vs LFU vs Belady(最优上界)，看 LRU 还有多少头部空间。
    policy_cmp = {}
    for c in [int(x) for x in os.environ.get("POLICY_CAPS", "24,32,48").split(",")]:
        nl = max(1, len(SEQ))
        policy_cmp[str(c)] = {
            "LRU": round(sum(lru_hit_rate(SEQ[li][SKIP_FIRST:], c) for li in SEQ) / nl, 4),
            "LFU": round(sum(lfu_hit_rate(SEQ[li][SKIP_FIRST:], c) for li in SEQ) / nl, 4),
            "Belady": round(sum(belady_hit_rate(SEQ[li][SKIP_FIRST:], c) for li in SEQ) / nl, 4),
        }
    th99 = sorted([p["min_cap"].get("ge_0.99") for p in per_layer if p["min_cap"].get("ge_0.99")])
    dump = os.environ.get("DUMP_SEQ", "")
    if dump:
        json.dump({str(li): SEQ[li][SKIP_FIRST:] for li in sorted(SEQ)}, open(dump, "w"))
    print(json.dumps({
        "K": K, "tokens": len(ids), "steps": stats["steps"], "n_layers": len(SEQ),
        "agg_mean_hit_at_cap": agg_hit,
        "policy_cmp": policy_cmp,
        "min_cap_ge_0.99_min/median/max": [th99[0], th99[len(th99)//2], th99[-1]] if th99 else None,
        "per_layer": per_layer,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
