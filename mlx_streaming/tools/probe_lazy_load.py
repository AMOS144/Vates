"""de-risk 探针⑤:验证「_load_one 去掉 mx.eval、让读盘惰性并入异步图」能否逼近去读盘上界(临时)。

消融拆解显示:读盘占 61% decode 时间,但其中大头是 `_load_one` 里 `mx.eval(w)` 每次强制同步、
打碎 MLX 异步流水线(42 次/token)。本探针对比三种 loader,均跑真实 decode(精确性靠 token 末
统一 eval 保证):
  base   : 现状(mx.load + mx.eval(w),每专家强制同步)
  lazy   : 只 mx.load,不 eval(读盘并入异步图,与计算重叠)
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


def _decode_window(model, prompt_ids, warm, steps):
    cache = model.make_cache()
    cur = mx.argmax(model(prompt_ids, cache=cache)[:, -1:, :], axis=-1)
    mx.eval(cur)
    for _ in range(warm):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)
    t0 = time.perf_counter()
    last = cur
    for _ in range(steps):
        last = mx.argmax(model(last, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(last)
    return steps / (time.perf_counter() - t0), last


def main():
    assert os.environ.get("RESIDENT_POOL", "1") == "1"
    os.environ.setdefault("GPU_REMAP", "0")          # 走 host 路径,隔离 loader 变量
    model, tok, store = build_streaming_model()
    prompt_ids = mx.array([tok.encode(PROMPT)])

    path = store.path
    base_load = lambda layer, e: _eval_load(path(layer, e))   # noqa: E731
    lazy_load = lambda layer, e: mx.load(path(layer, e))      # noqa: E731

    def _eval_load(p):
        w = mx.load(p)
        mx.eval(w)
        return w

    def window(loader, label):
        store._load_one = loader
        store._resident.loader = loader
        store.reset_stats()
        tps, last = _decode_window(model, prompt_ids, WARM, STEPS)
        return {"label": label, "tps": round(tps, 2),
                "hit_rate": round(store.hit_rate(), 4), "_t": last}

    # 充分预热(真读盘灌满池),消除冷启动污染
    store._load_one = base_load
    store._resident.loader = base_load
    _decode_window(model, prompt_ids, WARM * 2, 0)

    # 三窗口对照:base→lazy→base,看 lazy 收益是否真实(而非顺序/预热假象)
    w1 = window(base_load, "base_eval")
    w2 = window(lazy_load, "lazy_noeval")
    w3 = window(base_load, "base_eval_2")

    out = {
        "windows": [{k: v for k, v in w.items() if k != "_t"} for w in (w1, w2, w3)],
        "lazy_over_base1": round(w2["tps"] / w1["tps"], 3),
        "lazy_over_base2": round(w2["tps"] / w3["tps"], 3),
        "same_token_base_vs_lazy": bool(mx.array_equal(w1["_t"], w2["_t"]).item()),
        "warm": WARM, "steps": STEPS,
        "expert_slots": int(os.environ.get("EXPERT_SLOTS", "256")),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
