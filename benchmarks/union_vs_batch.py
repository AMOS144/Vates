"""专家并集 vs 前向 token 数:量"调大 batch 能不能摊薄专家流式带宽"的天花板。

512 专家、top-k=10。N 个 token 一次前向,每 MoE 层路由并集 union(N)。摊薄比 = N*10/union(N)
= 每个被加载专家平均服务多少 token;experts_per_token = union(N)/N = 每 token 需加载多少专家。
只有 union 随 N 明显亚线性(即 tokens 开始共享专家)时,调大 batch 才省带宽。
"""
import os
os.environ["UNION_PROF"] = "1"           # 必须在 import mlx_streaming 前

import mlx.core as mx
from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked
from mlx_streaming.mtp.kv_cache import _snapshot, _restore
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.core.profiling import UNION_PROF, union_reset

PROMPT = "请写一段关于人工智能发展历史的详细介绍,尽量长一些。"


def main():
    model, tok, store = build_streaming_model()
    cache = model.make_cache()
    logits, _ = prefill_chunked(model, mx.array([tok.encode(PROMPT)]), cache)

    # 贪婪解码 128 个真实 token,作为不同长度前向的输入(真实路由,非随机)。
    x = int(mx.argmax(logits[:, -1, :]))
    toks = [x]
    for _ in range(160):
        l, _ = forward_with_hidden(model, mx.array([[toks[-1]]]), cache); mx.eval(l)
        toks.append(int(mx.argmax(l[:, -1, :])))

    snap = _snapshot(cache)                # 稳态 cache(prompt+160)后作为共享前缀
    print(f"{'N':>4} {'union/layer':>12} {'experts/token':>14} {'tokens/expert':>14}")
    for N in [1, 2, 4, 6, 8, 16, 32, 64, 128]:
        _restore(cache, snap)
        union_reset()          # 同时清 UNION_PROF 与 UNION_SAMPLES，避免跨迭代陈旧样本/无界增长
        seq = mx.array([toks[:N]])
        l, _ = forward_with_hidden(model, seq, cache); mx.eval(l)
        # UNION_PROF[N] = [sum_union_over_layers, num_layers]
        v = UNION_PROF.get(N)
        if v is None:                      # 兜底:取唯一桶
            v = list(UNION_PROF.values())[0]
        union = v[0] / max(1, v[1])
        print(f"{N:>4} {union:>12.1f} {union / N:>14.2f} {N * 10 / union:>14.2f}")


if __name__ == "__main__":
    main()
