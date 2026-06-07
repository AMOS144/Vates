"""隔离探针:只测主模型前向耗时随序列长度 L 的标度,判断多 token 是否摊薄。

不含投机/草稿/快照/重放。对每个 L:fresh cache,喂 L 个真实 token,计时一次前向。
若 per-token 耗时随 L 下降 → 批量摊薄成立(投机有红利);若近似不变 → 不摊薄。
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.mem import reset_peak
from mlx_streaming.mtp_generate import forward_with_hidden
from mlx_streaming.validate_mtp import _build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。混合专家模型通过门控网络")
LS = [int(x) for x in os.environ.get("LS", "1,2,4,8,16").split(",")]


def main():
    reset_peak()
    model, tok, store = _build_streaming_model()
    ids_all = tok.encode(PROMPT)

    # 先 warmup 让专家 LRU 热(避免冷启动污染)
    c = model.make_cache()
    forward_with_hidden(model, mx.array([ids_all[:8]]), c)
    mx.eval(model.make_cache())

    rows = []
    for L in LS:
        seq = ids_all[:L] if L <= len(ids_all) else (ids_all * (L // len(ids_all) + 1))[:L]
        ids = mx.array([seq])
        cache = model.make_cache()                 # fresh,等价 prefill L 个 token
        store.reset_stats()
        # 计时:跑两次取第二次(避开该 L 首次的对象重建/编译开销)
        forward_with_hidden(model, ids, cache)
        mx.eval(model.make_cache())
        cache = model.make_cache()
        store.reset_stats()
        t0 = time.perf_counter()
        logits, H = forward_with_hidden(model, ids, cache)
        mx.eval(logits, H)
        dt = time.perf_counter() - t0
        rows.append({
            "L": L,
            "total_ms": round(dt * 1000, 1),
            "per_token_ms": round(dt * 1000 / L, 1),
            "disk_loads": store.misses,
            "loads_per_token": round(store.misses / L, 1),
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    # 摊薄判据:per_token_ms 是否随 L 下降
    base = rows[0]["per_token_ms"]
    print("\n--- 标度小结 (per_token_ms 相对 L=1) ---")
    for r in rows:
        print(f"L={r['L']:>2}  per_token={r['per_token_ms']:>7.1f}ms  "
              f"× L=1 = {r['per_token_ms'] / base:.2f}")


if __name__ == "__main__":
    main()
