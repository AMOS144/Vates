"""分块 prefill 数值等价性测试(迷你 Qwen3-Next,不需 80B 权重)。

验证 prefill_chunked(chunk=2/3)与整段 prefill:
- 末位 logits cosine ≥ 0.99
- 首 token argmax 完全一致
- cache.offset 推进正确(= prompt 长度)
全注意力层完全等价;线性层 chunk 边界有末位浮点差异,故用 cosine + argmax 判据。
"""
import mlx.core as mx
from mlx_lm.models.qwen3_next import Model, ModelArgs

from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked


def _tiny_model():
    args = ModelArgs(
        model_type="qwen3_next",
        hidden_size=128,
        num_hidden_layers=4,            # 含线性层(idx0,1,2)与全注意力层(idx3, interval=4)
        intermediate_size=256,
        num_attention_heads=4,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=0,
        num_experts_per_tok=1,
        decoder_sparse_step=1000,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[0, 1, 2, 3],
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        max_position_embeddings=4096,
        head_dim=64,
        full_attention_interval=4,
    )
    mx.random.seed(0)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _cos(a, b):
    return float((a * b).sum() / (mx.linalg.norm(a) * mx.linalg.norm(b)))


def test_chunked_prefill_matches_full():
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3, 11, 4, 6, 8, 0, 13, 2, 5]])

    full_cache = model.make_cache()
    full_logits, full_H = forward_with_hidden(model, ids, full_cache)

    for chunk in (2, 3, 5):
        c = model.make_cache()
        logits, H = prefill_chunked(model, ids, c, chunk=chunk)
        # offset 推进到整段长度
        fa = next(x for x, l in zip(c, model.layers) if not l.is_linear)
        assert fa.offset == ids.shape[1]
        # 末位 logits 与整段高度一致 + 首 token 完全一致
        assert _cos(logits[:, -1, :], full_logits[:, -1, :]) > 0.99
        assert int(mx.argmax(logits[:, -1, :])) == int(mx.argmax(full_logits[:, -1, :]))


def test_chunk_zero_is_full_prefill():
    """chunk<=0 退化为整段 prefill(逐位一致)。"""
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3, 8]])
    c1 = model.make_cache()
    l1, _ = forward_with_hidden(model, ids, c1)
    c2 = model.make_cache()
    l2, _ = prefill_chunked(model, ids, c2, chunk=0)
    assert mx.allclose(l1, l2, atol=1e-5)


def test_short_prompt_single_chunk():
    """prompt 不超过一块时直接整段,行为不变。"""
    model = _tiny_model()
    ids = mx.array([[1, 5]])
    c = model.make_cache()
    logits, H = prefill_chunked(model, ids, c, chunk=4)
    assert logits.shape[1] == 2
    fa = next(x for x, l in zip(c, model.layers) if not l.is_linear)
    assert fa.offset == 2
