"""双源双缓冲(ZEROCOPY_DUAL_SOURCE)端到端验证 + A/B。

单进程只跑一种配置(dual on/off 在 build 时决定池构造,无法进程内切换),用贪婪解码打印
token ids、命中率、读盘、内存。跑两次(off 作 ref、on 作对照)后 diff token ids 判 exact_match。

用法:
  # 基线(dual off)
  ZEROCOPY_DUAL_SOURCE=0 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 \
    .venv/bin/python -m benchmarks.bench_dual_source > /tmp/ds_off.json
  # 双源(dual on),侧区第二级 = POOL_SPEC_SLOTS
  ZEROCOPY_DUAL_SOURCE=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 POOL_SPEC_SLOTS=32 \
    .venv/bin/python -m benchmarks.bench_dual_source > /tmp/ds_on.json
  # 比对
  .venv/bin/python -m benchmarks.bench_dual_source --diff /tmp/ds_off.json /tmp/ds_on.json
"""
import json
import os
import sys
import time

import mlx.core as mx

from mlx_streaming.core.mem import snapshot, reset_peak
from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
WARMUP_TOK = int(os.environ.get("WARMUP_TOK", str(MAXTOK)))


def _greedy(model, tok, prompt, n):
    cache = model.make_cache()
    ids = mx.array([tok.encode(prompt)])
    logits, _ = prefill_chunked(model, ids, cache)
    out = []
    for _ in range(n):
        nxt = int(mx.argmax(logits[:, -1, :]))
        out.append(nxt)
        if len(out) >= n:
            break
        logits, _ = forward_with_hidden(model, mx.array([[nxt]]), cache)
    return out


def _diff(off_path, on_path):
    off = json.load(open(off_path))
    on = json.load(open(on_path))
    a, b = off["ids"], on["ids"]
    n_mm = sum(1 for x, y in zip(a, b) if x != y)
    fpos = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1)
    print(json.dumps({
        "exact_match": a == b,
        "n_mismatch": n_mm,
        "first_mm_pos": fpos,
        "off": {k: off[k] for k in ("dual", "expert_slots", "spec_slots",
                                    "hit_rate", "disk_loads", "active_gb", "peak_gb", "tok_per_s")},
        "on": {k: on[k] for k in ("dual", "expert_slots", "spec_slots",
                                  "hit_rate", "disk_loads", "active_gb", "peak_gb", "tok_per_s")},
    }, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--diff":
        _diff(sys.argv[2], sys.argv[3])
        return

    reset_peak()
    model, tok, store = build_streaming_model()

    if WARMUP_TOK > 0:                       # 装满常驻池 + 编译 kernel,保证正式测量即稳态
        _greedy(model, tok, PROMPT, WARMUP_TOK)

    reset_peak()
    store.reset_stats()
    t0 = time.perf_counter()
    ids = _greedy(model, tok, PROMPT, MAXTOK)
    tps = round(MAXTOK / (time.perf_counter() - t0), 2)
    snap = snapshot()

    rp = store._resident
    out = {
        "dual": _cfg.zerocopy_dual_source(),
        "expert_slots": store.capacity,
        "spec_slots": getattr(rp, "spec_slots", 0),
        "spec_gens": getattr(rp, "spec_gens", 1),
        "tok_per_s": tps,
        "hit_rate": round(store.hit_rate(), 4),
        "disk_loads": store.misses,
        "gpu_fastpath": rp.gpu_fastpath,
        "gpu_fallback": rp.gpu_fallback,
        "active_gb": round(snap.mlx_active_bytes / 1e9, 2),
        "peak_gb": round(snap.mlx_peak_bytes / 1e9, 2),
        "ids": ids,
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
