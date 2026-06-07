"""定位流式 MoE 单 token 前向的 ~230ms 固定开销构成。

需 STREAM_PROF=1。warm 后跑 N 步单 token 解码,打印各段累计耗时占比:
  route   : 路由(gate/softmax/argpartition)
  pyremap : Python uniq/remap/local 构造(含 .tolist 同步)
  fetch   : store.fetch 专家加载
  matmul  : 专家 SwitchGLU 计算(含 QSL update)
  combine : 加权合并 + 共享专家
"""
import os
import time

import mlx.core as mx

from mlx_streaming import streaming_moe as sm
from mlx_streaming.mtp_generate import forward_with_hidden
from mlx_streaming.validate_mtp import _build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
N = int(os.environ.get("N", "16"))


def main():
    model, tok, store = _build_streaming_model()
    ids = tok.encode(PROMPT)
    cache = model.make_cache()
    # prefill 预热(让专家驻留 + 触发编译)
    forward_with_hidden(model, mx.array([ids]), cache)
    cur = mx.array([[ids[-1]]])
    for _ in range(4):                       # 再 warm 几步单 token
        logits, _ = forward_with_hidden(model, cur, cache)
        cur = mx.array([[int(mx.argmax(logits[:, -1, :]))]])
        mx.eval(cur)

    sm.prof_reset()
    t0 = time.perf_counter()
    for _ in range(N):
        logits, _ = forward_with_hidden(model, cur, cache)
        cur = mx.array([[int(mx.argmax(logits[:, -1, :]))]])
        mx.eval(cur)
    wall = time.perf_counter() - t0

    p = sm.PROF
    seg_sum = p["route"] + p["pyremap"] + p["fetch"] + p["matmul"] + p["combine"]
    n_layers = p["n_calls"] / N
    print(f"steps={N}  wall={wall:.3f}s  per_token={wall/N*1000:.1f}ms  "
          f"moe_calls/step={n_layers:.0f}")
    print(f"MoE 段累计(占 wall):segsum={seg_sum:.3f}s ({seg_sum/wall*100:.0f}%)")
    for seg in ("route", "pyremap", "fetch", "matmul", "combine"):
        print(f"  {seg:>8}: {p[seg]*1000/N:>7.1f} ms/step  ({p[seg]/seg_sum*100:>4.0f}% of MoE)")


if __name__ == "__main__":
    main()
