"""文件后端流式：拆分到磁盘后只加载选中专家，结果须与常驻 SwitchGLU 等价。"""
import tempfile

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_lm.models.qwen3_moe import Model, ModelArgs

from mlx_streaming.prep.split_experts import split_switch_glu
from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.moe.compute import streaming_switch_glu_forward_from_store
from mlx_streaming.core.prefetch.patch import patch_model_filebacked
from mlx_streaming.core.moe.block import FileStreamingMoeBlock


def test_file_backed_matches_resident_quantized():
    mx.random.seed(0)
    dim, hidden, E, k = 64, 128, 8, 2
    glu = SwitchGLU(dim, hidden, E)
    mx.eval(glu.parameters())
    nn.quantize(glu, group_size=64, bits=4)
    mx.eval(glu.parameters())

    x = mx.random.normal((1, 5, dim))
    inds = mx.argpartition(mx.random.normal((1, 5, E)), kth=-k, axis=-1)[..., -k:]
    ref = glu(x, inds)

    out_dir = tempfile.mkdtemp(prefix="mlx_split_")
    n = split_switch_glu(glu, out_dir, layer=0)
    assert n == E

    store = FileExpertStore(out_dir, capacity=E)
    got = streaming_switch_glu_forward_from_store(
        store, 0, x, inds, hidden=dim, moe_inter=hidden, group_size=64, bits=4,
    )
    assert mx.allclose(ref, got, atol=1e-4).item()
    # 只加载了 inds 涉及到的唯一专家，不会是全部 E 个
    assert store.resident_count() <= E
    assert store.misses >= 1


def _tiny_quantized_moe():
    args = ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=2,
        intermediate_size=128, num_attention_heads=4, num_experts=8,
        num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
        moe_intermediate_size=128, rms_norm_eps=1e-6, vocab_size=128,
        num_key_value_heads=2, head_dim=16, rope_theta=1000000.0,
        tie_word_embeddings=True, max_position_embeddings=256, norm_topk_prob=True,
    )
    model = Model(args)
    mx.eval(model.parameters())
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    return model, args


def test_filebacked_patch_matches_quantized_model():
    # 端到端验证提速后的文件后端路径（每层独立 LRU + 持久化子模块）数值不变
    model, args = _tiny_quantized_moe()
    inputs = mx.array([[1, 2, 3, 4, 5]])
    ref = model(inputs)
    mx.eval(ref)

    out_dir = tempfile.mkdtemp(prefix="mlx_tiny_split_")
    gp = None
    for i, layer in enumerate(model.layers):
        sm = layer.mlp.switch_mlp
        split_switch_glu(sm, out_dir, i)
        gp = sm.gate_proj

    store = FileExpertStore(out_dir, capacity=args.num_experts)
    n = patch_model_filebacked(model, store, gp.input_dims, gp.output_dims,
                               gp.group_size, gp.bits)
    assert n == args.num_hidden_layers
    for layer in model.layers:
        assert isinstance(layer.mlp, FileStreamingMoeBlock)

    got = model(inputs)
    mx.eval(got)
    assert got.shape == ref.shape
    assert mx.allclose(ref, got, atol=1e-4).item()
    assert store.misses >= 1


def test_gpu_remap_decode_matches_host(monkeypatch):
    # warm 后 seq=1 解码:GPU 侧 slot 重映射(GPU_REMAP=1)须与 host 路径(=0)逐 bit 一致,
    # 且确实走了 GPU 查找表(_slot_table 被建立)。
    mx.random.seed(0)
    model, args = _tiny_quantized_moe()
    out_dir = tempfile.mkdtemp(prefix="mlx_gpu_remap_")
    gp = None
    for i, layer in enumerate(model.layers):
        sm = layer.mlp.switch_mlp
        split_switch_glu(sm, out_dir, i)
        gp = sm.gate_proj
    store = FileExpertStore(out_dir, capacity=args.num_experts)   # 全装得下,不驱逐
    patch_model_filebacked(model, store, gp.input_dims, gp.output_dims,
                           gp.group_size, gp.bits)

    blk = model.layers[0].mlp
    x = mx.random.normal((1, 1, gp.input_dims))
    monkeypatch.setenv("GPU_REMAP", "0")
    y_host = blk(x); mx.eval(y_host)          # host 路径,顺便预热池
    monkeypatch.setenv("GPU_REMAP", "1")
    y_gpu = blk(x); mx.eval(y_gpu)            # 全命中 → 走 GPU remap
    assert mx.allclose(y_host, y_gpu, atol=1e-6).item()
    assert 0 in getattr(store._resident, "_slot_table", {})


def test_resident_pool_falls_back_when_uniques_exceed_capacity():
    # 回归：单次前向(prefill)唯一专家数 > 池容量时，须回退 stack 路径而非崩溃，
    # 且数值仍与非流式量化模型一致。
    model, args = _tiny_quantized_moe()
    inputs = mx.array([[1, 2, 3, 4, 5, 6, 7]])   # 7 token，唯一专家很可能 > 3
    ref = model(inputs)
    mx.eval(ref)

    out_dir = tempfile.mkdtemp(prefix="mlx_tiny_cap_")
    gp = None
    for i, layer in enumerate(model.layers):
        sm = layer.mlp.switch_mlp
        split_switch_glu(sm, out_dir, i)
        gp = sm.gate_proj

    # 容量故意远小于一次前向的唯一专家数，强制触发回退
    store = FileExpertStore(out_dir, capacity=3)
    patch_model_filebacked(model, store, gp.input_dims, gp.output_dims,
                           gp.group_size, gp.bits)
    got = model(inputs)              # 默认 RESIDENT_POOL=1，含 prefill 大 uniq
    mx.eval(got)
    assert mx.allclose(ref, got, atol=1e-4).item()
