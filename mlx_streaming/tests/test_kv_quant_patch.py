"""patch_kv_quant 在迷你 Qwen3-Next 上的机制测试(不需要 80B 权重)。

验证:
1. make_cache 覆盖后,全注意力层用 AsymmetricQuantizedKVCache、线性层用 ArraysCache。
2. patch(8/8 bit、无旋转)与未 patch 的 bf16 基线 logits 高度一致(证明 __call__ 代数等价,
   仅多了 8-bit 量化噪声)。
3. patch(K4/V3 + 旋转)能正常前向、输出有限。
"""
import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_next import Model, ModelArgs

from mlx_streaming.core.cache.kv_quant_patch import patch_kv_quant
from mlx_streaming.core.cache.quant_kv import AsymmetricQuantizedKVCache


def _tiny_model():
    args = ModelArgs(
        model_type="qwen3_next",
        hidden_size=128,
        num_hidden_layers=2,            # idx0 线性, idx1 全注意力(interval=2)
        intermediate_size=256,
        num_attention_heads=4,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        num_experts=0,                  # 全用 dense MLP,免造 MoE
        num_experts_per_tok=1,
        decoder_sparse_step=1000,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[0, 1],
        moe_intermediate_size=128,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        max_position_embeddings=4096,
        head_dim=64,
        full_attention_interval=2,
    )
    mx.random.seed(0)
    model = Model(args)
    mx.eval(model.parameters())
    return model


def test_make_cache_types_after_patch():
    model = _tiny_model()
    patch_kv_quant(model, group_size=64, k_bits=4, v_bits=3, rotate=True, seed=0)
    cache = model.make_cache()
    assert isinstance(cache[0], ArraysCache)                 # 线性层
    assert isinstance(cache[1], AsymmetricQuantizedKVCache)  # 全注意力层
    assert cache[1].k_bits == 4 and cache[1].v_bits == 3


def _greedy_logits(model, ids, n=4):
    cache = model.make_cache()
    out = model(ids, cache=cache)
    last = [out[:, -1, :]]
    cur = mx.argmax(out[:, -1, :], axis=-1)[:, None]
    for _ in range(n):
        out = model(cur, cache=cache)
        last.append(out[:, -1, :])
        cur = mx.argmax(out[:, -1, :], axis=-1)[:, None]
    return mx.concatenate(last, axis=0)


def test_patch_8bit_norotate_matches_bf16_baseline():
    """8/8 bit、无旋转:逐步 logits 与 bf16 基线 cosine ≥ 0.99。"""
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3]])
    base = _greedy_logits(model, ids)
    patch_kv_quant(model, group_size=32, k_bits=8, v_bits=8, rotate=False, seed=0)
    quant = _greedy_logits(model, ids)
    cos = float((base * quant).sum() / (mx.linalg.norm(base) * mx.linalg.norm(quant)))
    assert cos > 0.99


def test_patch_k4v3_rotated_runs_finite():
    """K4/V3 + 旋转:前向能跑、输出有限(质量阈值由真实模型 e2e 把关)。"""
    model = _tiny_model()
    ids = mx.array([[1, 5, 9, 2, 7, 3]])
    patch_kv_quant(model, group_size=32, k_bits=4, v_bits=3, rotate=True, seed=0)
    logits = _greedy_logits(model, ids)
    mx.eval(logits)
    assert bool(mx.all(mx.isfinite(logits)))
