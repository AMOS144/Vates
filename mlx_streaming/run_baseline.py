"""全量驻留基线：正常加载 + 生成，记录 RSS/峰值/decode 速度。"""
import os
import time
import json

import mlx.core as mx
from mlx_lm import load, generate

from mlx_streaming.mem import snapshot, reset_peak

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))


def main():
    reset_peak()
    t0 = time.perf_counter()
    model, tok = load(MODEL)            # 默认全量加载
    mx.eval(model.parameters())         # 强制全部驻留，作为对照上界
    load_done = snapshot()
    t1 = time.perf_counter()

    text = generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK, verbose=False)
    t2 = time.perf_counter()
    after = snapshot()

    out = {
        "mode": "baseline_resident",
        "model": MODEL,
        "load_s": round(t1 - t0, 2),
        "gen_s": round(t2 - t1, 2),
        "tok_per_s": round(MAXTOK / (t2 - t1), 2),
        "rss_gb_after_load": round(load_done.rss_bytes / 1e9, 2),
        "rss_gb_after_gen": round(after.rss_bytes / 1e9, 2),
        "mlx_peak_gb": round(after.mlx_peak_bytes / 1e9, 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
