"""探针:验证 MoE 专家 union(唯一专家数)是否随验证窗口 L 近似线性增长。

若 union(L) ≈ L × union(1),则多 token 验证前向在专家计算上不摊薄,
这是 80B 流式上投机解码净亏的根因(verify(K) ≈ K × verify(1))。

复用 validate_mtp._build_streaming_model;monkeypatch streaming_moe._unique_and_local
统计每次前向里每层的 union 专家数(对真实连续 token,逐 L 测量)。
"""
import json
import os

import mlx.core as mx

import mlx_streaming.streaming_moe as sm
from mlx_streaming.validate_mtp import _build_streaming_model, _greedy

MAXTOK = int(os.environ.get("MAXTOK", "48"))
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
LS = [int(x) for x in os.environ.get("LS", "1,2,3,5").split(",")]

# 累计器:每次前向把每层 union 大小(=fetch 的专家 id 列表长度)append 进来
_acc = {"sizes": []}


def main():
    model, tok, store = _build_streaming_model()

    # 包 store.fetch:第二个参数就是该层本次前向的 union 专家 id 列表
    _orig_fetch = store.fetch

    def _patched_fetch(layer, uniq_ids):
        try:
            _acc["sizes"].append(len(uniq_ids))
        except TypeError:
            pass
        return _orig_fetch(layer, uniq_ids)

    store.fetch = _patched_fetch

    prompt_ids = mx.array([tok.encode(PROMPT)])
    # 先贪心生成一段真实 token,作为验证窗口的来源
    g = _greedy(model, prompt_ids, MAXTOK)
    mx.eval(g)
    seq = g[0]  # (T,)
    T = int(seq.shape[0])

    # 预热:把序列尾部跑一遍,稳定专家缓存(避免首次加载影响 union 计数本身)
    _ = model(seq[None, -8:])
    mx.eval(_)

    results = {}
    for L in LS:
        window = seq[None, T - L:]  # 取最后 L 个真实 token
        _acc["sizes"] = []
        out = model(window)
        mx.eval(out)
        sizes = _acc["sizes"]
        # 取每层一次:解码窗口里 48 层各 record 一次(若有 MTP 等额外层会多,按 48 截断不必要,直接平均)
        n_layers = len(sizes)
        avg = sum(sizes) / max(1, n_layers)
        results[L] = {
            "n_moe_calls": n_layers,
            "avg_union_per_layer": round(avg, 3),
            "total_union": sum(sizes),
        }

    base = results.get(1, {}).get("avg_union_per_layer", None)
    print(json.dumps({
        "per_L": results,
        "union_ratio_vs_L1": {
            L: round(results[L]["avg_union_per_layer"] / base, 3)
            for L in LS if base
        },
        "note": "若 ratio(L) ≈ L,则专家计算随 L 线性增长,验证前向不摊薄",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
