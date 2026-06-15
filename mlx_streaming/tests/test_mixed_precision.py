"""混合精度（逐 proj 不同 bit）专家相关单测。"""
import os
import json
import tempfile

import mlx.core as mx


def _make_src_experts(d, n_experts=4):
    """造 n_experts 个源 4-bit 专家文件，返回 (dir, dims)。"""
    dims = {"hidden": 2048, "moe_intermediate": 768,
            "num_experts": n_experts, "group_size": 64, "bits": 4}
    os.makedirs(d, exist_ok=True)
    shapes = {"gate_proj": (768, 2048), "up_proj": (768, 2048), "down_proj": (2048, 768)}
    for e in range(n_experts):
        rec = {}
        for name, (out, inp) in shapes.items():
            W = mx.random.normal((out, inp)) * 0.1
            wq, s, b = mx.quantize(W, group_size=64, bits=4)
            rec[f"{name}.weight"] = wq
            rec[f"{name}.scales"] = s
            rec[f"{name}.biases"] = b
        mx.save_safetensors(os.path.join(d, f"layer00_expert{e:03d}.safetensors"), rec)
    with open(os.path.join(d, "_split_meta.json"), "w") as f:
        json.dump({"dims": dims, "moe_layers": [0]}, f)
    return d, dims


def test_requantize_mixed_writes_per_proj_bits_and_meta():
    from mlx_streaming.prep.requantize_experts import requantize_dir
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        dst = os.path.join(root, "dst")
        _make_src_experts(src, n_experts=2)
        proj_bits = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
        requantize_dir(src, dst, proj_bits, dst_group=64)

        with open(os.path.join(dst, "_split_meta.json")) as f:
            meta = json.load(f)
        assert meta["dims"]["proj_bits"] == proj_bits

        out = mx.load(os.path.join(dst, "layer00_expert000.safetensors"))
        # gate(2-bit) 打包后元素数应少于 up(3-bit)：同形状下低 bit 占用更少 uint32
        assert out["gate_proj.weight"].size < out["up_proj.weight"].size


def test_persistent_subglu_honors_proj_bits():
    from mlx_streaming.core.moe.compute import PersistentSubGLU
    proj_bits = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
    sub = PersistentSubGLU(2048, 768, group_size=64, bits=2, proj_bits=proj_bits)
    sub._ensure(4)
    assert sub._glu.gate_proj.bits == 2
    assert sub._glu.up_proj.bits == 3
    assert sub._glu.down_proj.bits == 3


def test_mixed_forward_close_to_true_weights():
    # 混合精度前向应在容差内贴近「真实 4-bit 反量化权重」的前向（验证 bit 接线无误）。
    # 注：此处用随机高斯权重，2-bit 量化的相对误差本就较大（真实权重有结构、误差小得多，
    # 端到端困惑度才好）；本测仅用于抓「bit 接线错」这类会导致崩溃/量级错误的问题，
    # 故阈值放宽到 0.8，并固定随机种子保证可复现。
    from mlx_streaming.prep.requantize_experts import requantize_dir
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    from mlx_streaming.core.moe.compute import PersistentSubGLU
    mx.random.seed(0)
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        dst = os.path.join(root, "dst")
        _make_src_experts(src, n_experts=4)
        proj_bits = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
        requantize_dir(src, dst, proj_bits, dst_group=64)

        hidden, moe_inter = 2048, 768
        x = mx.random.normal((1, 1, hidden)) * 0.5
        local = mx.array([[0, 1, 2, 3]])

        # 参照：源 4-bit
        store_ref = FileExpertStore(src, capacity=8)
        ref = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=4)
        y_ref = ref.forward(store_ref.fetch(0, [0, 1, 2, 3]), 4, x, local)

        # 被测：混合精度
        store_mix = FileExpertStore(dst, capacity=8)
        mix = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=2, proj_bits=proj_bits)
        y_mix = mix.forward(store_mix.fetch(0, [0, 1, 2, 3]), 4, x, local)

        mx.eval(y_ref, y_mix)
        rel = float(mx.mean(mx.abs(y_ref - y_mix)) / (mx.mean(mx.abs(y_ref)) + 1e-9))
        assert rel < 0.8, f"混合精度前向偏差异常 rel={rel}"


def test_boundary_scheme_maps_layers():
    from mlx_streaming.prep.requantize_experts import boundary_scheme
    moe_layers = list(range(10))           # 假设 10 个 MoE 层
    mixB = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
    p2 = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
    sch = boundary_scheme(moe_layers, bnd=2, bnd_bits=p2, mid_bits=mixB)
    # 首 2 + 尾 2 应为 2bit，中间为 mixB
    assert sch[0] == p2 and sch[1] == p2
    assert sch[8] == p2 and sch[9] == p2
    assert sch[2] == mixB and sch[5] == mixB


def test_requantize_layered_per_layer_bits_and_meta():
    from mlx_streaming.prep.requantize_experts import requantize_dir_layered
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        dst = os.path.join(root, "dst")
        # 造 3 层 ×2 专家，层号 0/1/2（文件名 layerLL_expertEEE）
        dims = {"hidden": 2048, "moe_intermediate": 768, "num_experts": 2,
                "group_size": 64, "bits": 4}
        os.makedirs(src, exist_ok=True)
        shapes = {"gate_proj": (768, 2048), "up_proj": (768, 2048), "down_proj": (2048, 768)}
        for L in range(3):
            for e in range(2):
                rec = {}
                for name, (o, i) in shapes.items():
                    W = mx.random.normal((o, i)) * 0.1
                    wq, s, b = mx.quantize(W, group_size=64, bits=4)
                    rec[f"{name}.weight"] = wq
                    rec[f"{name}.scales"] = s
                    rec[f"{name}.biases"] = b
                mx.save_safetensors(os.path.join(src, f"layer{L:02d}_expert{e:03d}.safetensors"), rec)
        with open(os.path.join(src, "_split_meta.json"), "w") as f:
            json.dump({"dims": dims, "moe_layers": [0, 1, 2]}, f)

        # 层 0/2 → 2bit，层 1 → mixB(g2u3d3)
        p2 = {"gate_proj": 2, "up_proj": 2, "down_proj": 2}
        mixB = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
        layer_bits = {0: p2, 1: mixB, 2: p2}
        requantize_dir_layered(src, dst, layer_bits, dst_group=64)

        with open(os.path.join(dst, "_split_meta.json")) as f:
            meta = json.load(f)
        plpb = meta["dims"]["per_layer_proj_bits"]
        assert plpb["1"]["up_proj"] == 3 and plpb["0"]["up_proj"] == 2
        # 层1 的 up(3bit) 文件应比层0 的 up(2bit) 大
        f1 = mx.load(os.path.join(dst, "layer01_expert000.safetensors"))
        f0 = mx.load(os.path.join(dst, "layer00_expert000.safetensors"))
        assert f1["up_proj.weight"].size > f0["up_proj.weight"].size


def test_patch_filebacked_layer_proj_bits():
    from mlx_streaming.core.prefetch.patch import patch_model_filebacked
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    with tempfile.TemporaryDirectory() as root:
        _make_src_experts(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)

        class _Gate:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 4))

        class _MLP:
            gate = _Gate()
            top_k = 2
            norm_topk_prob = True
            switch_mlp = object()

        class _Layer:
            mlp = _MLP()

        class _Model:
            layers = [_Layer()]

        model = _Model()
        # 给该层显式指定 mixB，验证逐层映射生效
        mixB = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
        n = patch_model_filebacked(model, store, 2048, 768, 64, 2,
                                   layer_proj_bits={0: mixB})
        assert n == 1
        model.layers[0].mlp._sub._ensure(4)
        assert model.layers[0].mlp._sub._glu.up_proj.bits == 3


def test_shared_expert_added_to_output():
    # Qwen3-Next：FileStreamingMoeBlock 须把常驻共享专家 sigmoid(gate)*shared(x) 叠加到路由输出。
    import mlx.nn as nn
    from mlx_streaming.core.moe.block import FileStreamingMoeBlock
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    mx.random.seed(0)
    with tempfile.TemporaryDirectory() as root:
        _make_src_experts(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)

        class _Gate:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 4))

        shared = nn.Linear(2048, 2048, bias=False)   # 充当共享专家（可调用、常驻）
        sg = nn.Linear(2048, 1, bias=False)
        x = mx.random.normal((2, 2048)) * 0.1

        blk_no = FileStreamingMoeBlock(
            gate=_Gate(), top_k=2, norm_topk_prob=True, store=store, layer_idx=0,
            hidden=2048, moe_inter=768, group_size=64, bits=4)
        blk_sh = FileStreamingMoeBlock(
            gate=_Gate(), top_k=2, norm_topk_prob=True, store=store, layer_idx=0,
            hidden=2048, moe_inter=768, group_size=64, bits=4,
            shared_expert=shared, shared_expert_gate=sg)

        y_no = blk_no(x)
        y_sh = blk_sh(x)
        expect = y_no + mx.sigmoid(sg(x)) * shared(x)
        mx.eval(y_no, y_sh, expect)
        assert float(mx.mean(mx.abs(y_sh - expect))) < 1e-4


def test_patch_filebacked_captures_shared_expert():
    import mlx.nn as nn
    from mlx_streaming.core.prefetch.patch import patch_model_filebacked
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    with tempfile.TemporaryDirectory() as root:
        _make_src_experts(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)

        class _Gate:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 4))

        class _MLP:
            gate = _Gate()
            top_k = 2
            norm_topk_prob = True
            switch_mlp = object()
            shared_expert = nn.Linear(2048, 2048, bias=False)
            shared_expert_gate = nn.Linear(2048, 1, bias=False)

        class _Layer:
            mlp = _MLP()

        class _Model:
            layers = [_Layer()]

        model = _Model()
        n = patch_model_filebacked(model, store, 2048, 768, 64, 4)
        assert n == 1
        assert model.layers[0].mlp.shared_expert is not None
        assert model.layers[0].mlp.shared_expert_gate is not None


def test_patch_filebacked_passes_proj_bits():
    from mlx_streaming.core.moe.block import FileStreamingMoeBlock
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    with tempfile.TemporaryDirectory() as root:
        _make_src_experts(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)

        class _Gate:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 4))

        proj_bits = {"gate_proj": 2, "up_proj": 3, "down_proj": 3}
        blk = FileStreamingMoeBlock(
            gate=_Gate(), top_k=2, norm_topk_prob=True, store=store, layer_idx=0,
            hidden=2048, moe_inter=768, group_size=64, bits=2, proj_bits=proj_bits)
        blk._sub._ensure(4)
        assert blk._sub._glu.gate_proj.bits == 2
        assert blk._sub._glu.down_proj.bits == 3


def test_custom_qproj_gate_up_matches_default_path(monkeypatch):
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    from mlx_streaming.core.moe.compute import PersistentSubGLU
    mx.random.seed(1)
    with tempfile.TemporaryDirectory() as root:
        _make_src_experts(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)
        fetched = store.fetch(0, [0, 1, 2, 3])
        hidden, moe_inter = 2048, 768
        x = mx.random.normal((1, 1, hidden)) * 0.1
        local = mx.array([[[0, 1, 2, 3]]])
        ref = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=4, layer_idx=0)
        y_ref = ref.forward(fetched, 4, x, local)

        monkeypatch.setenv("CUSTOM_QPROJ", "1")
        monkeypatch.setenv("CUSTOM_QPROJ_LAYERS", "0")
        monkeypatch.setenv("CUSTOM_QPROJ_BITS", "4")
        monkeypatch.setenv("CUSTOM_QPROJ_TILE", "4")
        got = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=4, layer_idx=0)
        y_got = got.forward(fetched, 4, x, local)
        mx.eval(y_ref, y_got)
        rel = float(mx.mean(mx.abs(y_ref - y_got)) / (mx.mean(mx.abs(y_ref)) + 1e-9))
        assert rel < 1e-5


def test_custom_qproj_all_projections_matches_default_path(monkeypatch):
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    from mlx_streaming.core.moe.compute import PersistentSubGLU
    mx.random.seed(2)
    with tempfile.TemporaryDirectory() as root:
        _make_src_experts(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)
        fetched = store.fetch(0, [0, 1, 2, 3])
        hidden, moe_inter = 2048, 768
        x = mx.random.normal((1, 1, hidden)) * 0.1
        local = mx.array([[[0, 1, 2, 3]]])
        ref = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=4, layer_idx=0)
        y_ref = ref.forward(fetched, 4, x, local)

        monkeypatch.setenv("CUSTOM_QPROJ", "1")
        monkeypatch.setenv("CUSTOM_QPROJ_LAYERS", "0")
        monkeypatch.setenv("CUSTOM_QPROJ_BITS", "4")
        monkeypatch.setenv("CUSTOM_QPROJ_TILE", "4")
        monkeypatch.setenv("CUSTOM_QPROJ_TARGETS", "gate,up,down")
        got = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=4, layer_idx=0)
        y_got = got.forward(fetched, 4, x, local)
        mx.eval(y_ref, y_got)
        rel = float(mx.mean(mx.abs(y_ref - y_got)) / (mx.mean(mx.abs(y_ref)) + 1e-9))
        assert rel < 1e-5
