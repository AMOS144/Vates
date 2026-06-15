"""de-risk 探针③:量化「miss 层读盘」在一个 token 解码里的真实占比(临时,出结论后删)。

已证瓶颈是 miss 层读盘。本探针包裹 `store._load_one` 计时,跑真实 decode,直接给出:
  - 单专家冷加载 / 热重载耗时(mx.load+eval);
  - 计时窗口内总读盘秒数 / 总 decode 秒数 = 读盘占比;
  - 每 token 平均 miss 数与读盘耗时。
据此判断 mmap / 批量读 / 预取重叠值不值得、上限多少。

环境变量:MODEL / EXPERT_DIR / EXPERT_SLOTS / RESIDENT_POOL=1 / PROMPT / WARM / STEPS
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.model_builder import build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型，并举一个实际应用的例子。")
WARM = int(os.environ.get("WARM", "16"))
STEPS = int(os.environ.get("STEPS", "48"))


def main():
    assert os.environ.get("RESIDENT_POOL", "1") == "1"
    model, tok, store = build_streaming_model()
    prompt_ids = mx.array([tok.encode(PROMPT)])

    # 包裹 _load_one 计时(读盘总耗时/次数)
    orig_load = store._load_one
    stats = {"load_s": 0.0, "n": 0}

    def timed_load(layer, e):
        t = time.perf_counter()
        w = orig_load(layer, e)
        stats["load_s"] += time.perf_counter() - t
        stats["n"] += 1
        return w
    store._load_one = timed_load
    store._resident.loader = timed_load

    # 单专家冷/热加载基准(挑一个未必驻留的专家)
    cold_e = 511
    t = time.perf_counter(); store._load_one(0, cold_e); cold_ms = (time.perf_counter() - t) * 1e3
    t = time.perf_counter(); store._load_one(0, cold_e); warm_ms = (time.perf_counter() - t) * 1e3

    # prefill + warm
    cache = model.make_cache()
    cur = mx.argmax(model(prompt_ids, cache=cache)[:, -1:, :], axis=-1)
    mx.eval(cur)
    for _ in range(WARM):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)

    # 计时窗口:重置读盘/命中统计,只统计这 STEPS 步
    stats["load_s"] = 0.0; stats["n"] = 0
    store.reset_stats()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)
    decode_s = time.perf_counter() - t0

    out = {
        "expert_slots": int(os.environ.get("EXPERT_SLOTS", "256")),
        "steps": STEPS,
        "tok_per_s": round(STEPS / decode_s, 2),
        "decode_s": round(decode_s, 3),
        "disk_load_s": round(stats["load_s"], 3),
        "disk_frac_of_decode": round(stats["load_s"] / decode_s, 4),
        "n_loads_window": stats["n"],
        "loads_per_token": round(stats["n"] / STEPS, 2),
        "avg_load_ms": round(stats["load_s"] / max(1, stats["n"]) * 1e3, 3),
        "single_cold_load_ms": round(cold_ms, 3),
        "single_warm_load_ms": round(warm_ms, 3),
        "steady_hit_rate": round(store.hit_rate(), 4),
        "steady_misses": store.misses,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
