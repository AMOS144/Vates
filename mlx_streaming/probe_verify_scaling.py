"""探针:真实条件下验证前向是否随 L 摊薄(快照复用版,避免反复 prefill)。

真实 verify 的条件:完整 KV 上下文已在 cache 里 + 专家 warm。
流程:
  1) 贪心生成一段序列,prefill 除最后 maxL 个外的全部进 master cache;
  2) 快照 master cache(_snapshot,廉价);
  3) warm:把待测窗口在副本上跑两遍,让专家驻留 + PersistentSubGLU 各 n 建好;
  4) 计时:restore→对最后 L 个 token 做一次前向(带完整 cache),报 ms 与 ms/token。

ms_ratio(L) 远小于 L → 前向摊薄(投机有红利,瓶颈是实现);
ms_ratio(L) ≈ L → 不摊薄(投机本质净亏)。
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.mtp_generate import forward_with_hidden, _snapshot, _restore
from mlx_streaming.validate_mtp import _build_streaming_model, _greedy

MAXTOK = int(os.environ.get("MAXTOK", "64"))
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
LS = [int(x) for x in os.environ.get("LS", "1,2,3,5").split(",")]
REPEAT = int(os.environ.get("REPEAT", "4"))


def main():
    model, tok, store = _build_streaming_model()
    prompt_ids = mx.array([tok.encode(PROMPT)])
    g = _greedy(model, prompt_ids, MAXTOK)
    mx.eval(g)
    seq = g[0]
    T = int(seq.shape[0])
    maxL = max(LS)
    prefill_len = T - maxL

    # 1) prefill 一次进 master cache
    mc = model.make_cache()
    pre = seq[None, :prefill_len]
    out, _ = forward_with_hidden(model, pre, mc)
    mx.eval(out)
    snap = _snapshot(mc)

    # 2) warm:对各 L 窗口跑两遍,确保窗口涉及的专家全部驻留
    for _w in range(2):
        for L in LS:
            _restore(mc, snap)
            w = seq[None, prefill_len:prefill_len + L]
            o, _ = forward_with_hidden(model, w, mc)
            mx.eval(o)
    _restore(mc, snap)

    # 3) 计时(warm 后):每次 restore → 跑窗口前向
    results = {}
    for L in LS:
        ts = []
        window = seq[None, prefill_len:prefill_len + L]
        for _r in range(REPEAT):
            _restore(mc, snap)
            t0 = time.perf_counter()
            o, _ = forward_with_hidden(model, window, mc)
            mx.eval(o)
            ts.append(time.perf_counter() - t0)
        ms = min(ts) * 1000.0
        results[L] = {"ms": round(ms, 1), "ms_per_token": round(ms / L, 1)}

    base = results[LS[0]]["ms"]
    print(json.dumps({
        "context_len": prefill_len,
        "warm_hit_rate": round(store.hits / max(1, store.hits + store.misses), 4),
        "per_L": results,
        "ms_ratio_vs_L1": {L: round(results[L]["ms"] / base, 3) for L in LS},
        "note": "ms_ratio(L) << L → 摊薄(投机有戏);≈ L → 不摊薄(本质净亏)",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
