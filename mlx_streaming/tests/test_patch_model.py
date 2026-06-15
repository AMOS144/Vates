"""用本地构造的微型 Qwen3-MoE 模型（无需下载）验证 patch_model 端到端可用。"""
import mlx.core as mx
from mlx_lm.models.qwen3_moe import Model, ModelArgs

from mlx_streaming.core.prefetch.patch import patch_model
from mlx_streaming.core.moe.block import StreamingMoeBlock


def _tiny_model():
    args = ModelArgs(
        model_type="qwen3_moe",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=1,      # 每层都是 MoE
        mlp_only_layers=[],
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=1000000.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
        norm_topk_prob=True,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model, args


def test_patch_replaces_moe_blocks():
    model, args = _tiny_model()
    n = patch_model(model)
    assert n == args.num_hidden_layers
    for layer in model.layers:
        assert isinstance(layer.mlp, StreamingMoeBlock)


def test_patched_forward_matches_original():
    # 流式块与原块数值等价 => patch 前后 logits 应一致
    model, args = _tiny_model()
    inputs = mx.array([[1, 2, 3, 4, 5]])

    ref = model(inputs)
    mx.eval(ref)

    patch_model(model)
    got = model(inputs)
    mx.eval(got)

    assert got.shape == ref.shape
    assert mx.allclose(ref, got, atol=1e-4).item()
