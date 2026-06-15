import mlx.core as mx

from mlx_streaming.core.moe.gate import _predict_layer_experts


def test_predict_layer_experts_returns_topk_with_scores():
    # norm/gate 都用恒等：gates=softmax(x)，top_k=1、mult=2 → k=2 → 取最大两个下标 {1,3}
    ident = lambda t: t
    x = mx.array([[[1.0, 5.0, 2.0, 9.0, 3.0]]])
    best, num_experts = _predict_layer_experts(ident, ident, top_k=1, x=x, mult=2)
    assert set(best.keys()) == {1, 3}
    assert num_experts == 5
    # 分数为 softmax 概率，下标 3（logit 9）应高于下标 1（logit 5）
    assert best[3] > best[1]
