"""分块 prefill 数值等价性测试(迷你 Qwen3-Next,不需 80B 权重)。

验证 Expert-major prefill_chunked(chunk=2/3)与整段参考前向:
- 末位 logits cosine ≥ 0.99
- 首 token argmax 完全一致
- cache.offset 推进正确(= prompt 长度)
全注意力层完全等价;线性层 chunk 边界有末位浮点差异,故用 cosine + argmax 判据。
"""
import mlx.core as mx
from mlx_lm.models.qwen3_next import Model, ModelArgs

from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked


def test_production_chunk_stays_below_metal_scatter_32k_boundary():
    from mlx_streaming import config

    assert config.prefill_chunk() == 32767


def _tiny_model():
    args = ModelArgs(
        model_type="qwen3_next",
        hidden_size=128,
        num_hidden_layers=4,            # 含线性层(idx0,1,2)与全注意力层(idx3, interval=4)
        intermediate_size=256,
        num_attention_heads=4,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        # Match the production blocked-GDN ABI; this clean branch deliberately
        # does not carry the generic-shape fallback from the experiment branch.
        linear_key_head_dim=128,
        linear_value_head_dim=128,
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
    model.set_dtype(mx.bfloat16)
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
    """chunk<=0 整段执行，但只返回消费方需要的最后一位。"""
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3, 8]])
    c1 = model.make_cache()
    l1, _ = forward_with_hidden(model, ids, c1)
    c2 = model.make_cache()
    l2, _ = prefill_chunked(model, ids, c2, chunk=0)
    assert l2.shape == (1, 1, 64)
    assert _cos(l1[:, -1, :], l2[:, -1, :]) > 0.99
    assert int(mx.argmax(l1[:, -1, :])) == int(mx.argmax(l2[:, -1, :]))


def test_short_prompt_single_chunk():
    """短 prompt 也使用 Expert-major，并只投影最后一位。"""
    model = _tiny_model()
    ids = mx.array([[1, 5]])
    c = model.make_cache()
    logits, H = prefill_chunked(model, ids, c, chunk=4)
    assert logits.shape == (1, 1, 64)
    fa = next(x for x, l in zip(c, model.layers) if not l.is_linear)
    assert fa.offset == 2


def test_expert_major_prefill_projects_only_last_token(monkeypatch):
    """Prefill never materialises unused prompt-wide vocab logits."""
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3, 8]])
    cache = model.make_cache()
    logits, hidden = prefill_chunked(model, ids, cache, chunk=0)
    assert logits.shape == (1, 1, 64)
    assert hidden.shape == (1, 1, 128)
    fa = next(x for x, l in zip(cache, model.layers) if not l.is_linear)
    assert fa.offset == ids.shape[1]


def test_expert_major_last_token_matches_full_attention(monkeypatch):
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3, 8]])
    reference_cache = model.make_cache()
    reference, _ = forward_with_hidden(model, ids, reference_cache)
    mx.eval(reference)

    major_cache = model.make_cache()
    actual, _ = prefill_chunked(model, ids, major_cache, chunk=0)
    mx.eval(actual)
    # Gated-delta recurrence is regrouped at tile boundaries, matching the
    # project's established chunked-prefill numerical contract.
    assert _cos(actual[:, -1, :], reference[:, -1, :]) > 0.99
    assert int(mx.argmax(actual[:, -1, :])) == int(mx.argmax(reference[:, -1, :]))


def test_prefill_scope_does_not_leak_into_decode():
    """Public prefill is Expert-major; a later direct verify remains token-major."""
    model = _tiny_model()
    prefill_ids = mx.array([[1, 5]])
    prefill_cache = model.make_cache()
    prefill_logits, _ = prefill_chunked(
        model, prefill_ids, prefill_cache, chunk=8,
    )
    assert prefill_logits.shape == (1, 1, 64)

    verify_ids = mx.array([[1, 5, 9]])
    verify_cache = model.make_cache()
    verify_logits, _ = forward_with_hidden(model, verify_ids, verify_cache)
    assert verify_logits.shape == (1, 3, 64)
