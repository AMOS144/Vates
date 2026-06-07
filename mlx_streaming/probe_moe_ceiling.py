"""de-risk：fused MoE Metal kernel 的速度天花板。

把专家计算(gather_qmm + dequant + SwiGLU + down + _update_qsl，即 fused kernel 要攻的
全部)短路成 0，保留路由与专家加载(I/O，kernel 不改这块)。测「专家计算免费」时的
单 token 解码 tok/s —— 这是 fused kernel 的绝对上界(它只能让专家计算更快、变不成 0)。

判读：
  tps_stub 远 > 30 → 专家计算是大头，fused kernel 有冲 30 的空间
  tps_stub 也到不了 30 → 瓶颈在别处(attention/deltanet/dispatch)，kernel 白搭
"""
import os
import time

import mlx.core as mx

from mlx_streaming import streaming_moe as sm
from mlx_streaming.mtp_generate import forward_with_hidden
from mlx_streaming.validate_mtp import _build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
N = int(os.environ.get("N", "32"))
WARM = int(os.environ.get("WARM", "8"))


def _decode_tps(model, ids0, cache, n):
    cur = ids0
    t0 = time.perf_counter()
    for _ in range(n):
        logits, _ = forward_with_hidden(model, cur, cache)
        cur = mx.array([[int(mx.argmax(logits[:, -1, :]))]])
        mx.eval(cur)
    return n / (time.perf_counter() - t0)


def main():
    model, tok, store = _build_streaming_model()
    ids = tok.encode(PROMPT)
    cache = model.make_cache()
    forward_with_hidden(model, mx.array([ids]), cache)
    cur = mx.array([[ids[-1]]])
    for _ in range(WARM):
        logits, _ = forward_with_hidden(model, cur, cache)
        cur = mx.array([[int(mx.argmax(logits[:, -1, :]))]])
        mx.eval(cur)

    tps_full = _decode_tps(model, cur, cache, N)

    # 短路所有 MoE 块的专家计算：保留 routing/acquire，_sub.forward → 0
    blocks = [l.mlp for l in model.model.layers
              if isinstance(l.mlp, sm.FileStreamingMoeBlock)]
    originals = []
    for b in blocks:
        sub = b._sub
        originals.append((sub, sub.forward))

        def stub(fetched, n, x, local, _orig=None):
            return mx.zeros((x.shape[0], x.shape[1], local.shape[-1], x.shape[-1]),
                            dtype=x.dtype)
        sub.forward = stub

    tps_stub = _decode_tps(model, cur, cache, N)

    for sub, orig in originals:
        sub.forward = orig

    import json
    print(json.dumps({
        "moe_layers": len(blocks),
        "tps_full": round(tps_full, 2),
        "tps_expert_free": round(tps_stub, 2),
        "expert_compute_fraction": round(1 - tps_full / tps_stub, 3),
        "note": "tps_expert_free 是 fused kernel 的绝对上界(专家计算变0)；到不了30则kernel白搭",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
