"""路线 A：lazy（+ 若支持则 mmap）加载，可选设常驻/缓存上限，量内存与速度。

环境探针结论（mlx-lm 0.31.3）：load 支持 lazy，但不支持 use_mmap。
本脚本用 _filtered_load_kwargs 只传该版本真正支持的参数，缺失的自动剔除。
"""
import os
import time
import json
import inspect

import mlx.core as mx
from mlx_lm import load, generate

from mlx_streaming.mem import snapshot, reset_peak, clear_cache

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
WIRED_GB = os.environ.get("WIRED_GB")      # 例如 "6"；空=不设
CACHE_GB = os.environ.get("CACHE_GB")      # 例如 "1"；空=不设


def _maybe(fn_name, val_bytes):
    fn = getattr(mx, fn_name, None) or getattr(getattr(mx, "metal", object()), fn_name, None)
    if fn and val_bytes is not None:
        prev = fn(int(val_bytes))
        print(f"{fn_name}({int(val_bytes)}) 旧值={prev}")


def _filtered_load_kwargs():
    sig = inspect.signature(load)
    want = {"lazy": True, "use_mmap": True}
    return {k: v for k, v in want.items() if k in sig.parameters}


def main():
    if WIRED_GB:
        _maybe("set_wired_limit", float(WIRED_GB) * 1e9)
    if CACHE_GB:
        _maybe("set_cache_limit", float(CACHE_GB) * 1e9)

    reset_peak()
    kw = _filtered_load_kwargs()
    print("load kwargs:", kw)
    t0 = time.perf_counter()
    model, tok = load(MODEL, **kw)       # 不强制 eval 全部参数！
    t1 = time.perf_counter()
    after_load = snapshot()              # 关键：此时 RSS 应远低于基线（专家还没换入）

    text = generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK, verbose=False)
    t2 = time.perf_counter()
    clear_cache()
    after_gen = snapshot()

    out = {
        "mode": "mmap_lazy", "model": MODEL,
        "wired_gb": WIRED_GB, "cache_gb": CACHE_GB,
        "load_s": round(t1 - t0, 2), "gen_s": round(t2 - t1, 2),
        "tok_per_s": round(MAXTOK / (t2 - t1), 2),
        "rss_gb_after_load": round(after_load.rss_bytes / 1e9, 2),
        "rss_gb_after_gen": round(after_gen.rss_bytes / 1e9, 2),
        "mlx_active_gb_after_gen": round(after_gen.mlx_active_bytes / 1e9, 2),
        "mlx_peak_gb": round(after_gen.mlx_peak_bytes / 1e9, 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
