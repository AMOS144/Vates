"""Hadamard 旋转 2-bit 专家相关单测。"""
import os
import json
import math
import tempfile

import mlx.core as mx


def _rot(a, n):
    return mx.hadamard_transform(a, scale=1.0 / math.sqrt(n))


def test_hadamard_roundtrip_2048_and_768():
    for n in (2048, 768):
        x = mx.random.normal((4, n))
        back = _rot(_rot(x, n), n)
        assert float(mx.max(mx.abs(back - x))) < 1e-5


def test_weight_prerotation_equivalence():
    # W' = hadamard(W) 沿 input 维；W'·(Hx) 应等价 W·x
    for n in (2048, 768):
        x = mx.random.normal((4, n))
        W = mx.random.normal((5, n))
        Wp = _rot(W, n)          # 逐行旋转（最后一维=input）
        xr = _rot(x, n)
        lhs = Wp @ xr.T
        rhs = W @ x.T
        rel = float(mx.max(mx.abs(lhs - rhs)) / (mx.max(mx.abs(rhs)) + 1e-6))
        assert rel < 1e-2


def test_rotate_requantize_file_preserves_shape_and_keys():
    from mlx_streaming.rotate_requantize_experts import rotate_requantize_file
    # 造一个「源 4-bit 单专家」文件：gate/up (768,2048)，down (2048,768)
    src = {}
    for name, (out, inp) in {
        "gate_proj": (768, 2048),
        "up_proj": (768, 2048),
        "down_proj": (2048, 768),
    }.items():
        W = mx.random.normal((out, inp)) * 0.1
        wq, s, b = mx.quantize(W, group_size=64, bits=4)
        src[f"{name}.weight"] = wq
        src[f"{name}.scales"] = s
        src[f"{name}.biases"] = b
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "src.safetensors")
        dp = os.path.join(d, "dst.safetensors")
        mx.save_safetensors(sp, src)
        in_dims = {"gate_proj": 2048, "up_proj": 2048, "down_proj": 768}
        rotate_requantize_file(sp, dp, src_bits=4, src_group=64,
                               dst_bits=2, dst_group=64, in_dims=in_dims)
        out = mx.load(dp)
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert f"{name}.weight" in out
            assert f"{name}.scales" in out
            assert f"{name}.biases" in out
        # 2-bit 文件应明显小于 4-bit 源
        assert os.path.getsize(dp) < os.path.getsize(sp)


def _make_expert_files(d, n_experts=4, bits=8):
    """造 n_experts 个源专家文件，返回 (dir, dims)。

    默认 8-bit：用于「管线无 bias」断言时把量化噪声压到可忽略，
    从而隔离「旋转-反旋转逻辑是否正确」与「低比特量化损失」两件事。
    """
    dims = {"hidden": 2048, "moe_intermediate": 768,
            "num_experts": n_experts, "group_size": 64, "bits": bits}
    os.makedirs(d, exist_ok=True)
    shapes = {"gate_proj": (768, 2048), "up_proj": (768, 2048), "down_proj": (2048, 768)}
    for e in range(n_experts):
        rec = {}
        for name, (out, inp) in shapes.items():
            W = mx.random.normal((out, inp)) * 0.1
            wq, s, b = mx.quantize(W, group_size=64, bits=bits)
            rec[f"{name}.weight"] = wq
            rec[f"{name}.scales"] = s
            rec[f"{name}.biases"] = b
        mx.save_safetensors(os.path.join(d, f"layer00_expert{e:03d}.safetensors"), rec)
    with open(os.path.join(d, "_split_meta.json"), "w") as f:
        json.dump({"dims": dims}, f)
    return d, dims


def test_rotated_forward_matches_plain_switchglu():
    # 管线无 bias 断言：在 8-bit（量化噪声可忽略）下，旋转前向应与普通前向数值一致。
    # 4-bit 下 plain 与 rot 各自有独立量化网格噪声，不能直接对比（那是 Task 5 在真实
    # 权重上做的质量对比，随机高斯权重旋转无增益）。
    from mlx_streaming.rotate_requantize_experts import rotate_requantize_dir
    from mlx_streaming.expert_store import FileExpertStore
    from mlx_streaming.streaming_moe import RotatedSubGLU, PersistentSubGLU
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src")
        rot = os.path.join(root, "rot")
        _make_expert_files(src, n_experts=4, bits=8)
        # 旋转 + 重量化回 8-bit（隔离「旋转管线」与「低比特量化损失」）
        rotate_requantize_dir(src, rot, dst_bits=8, dst_group=64)

        hidden, moe_inter = 2048, 768
        x = mx.random.normal((1, 1, hidden)) * 0.5
        local = mx.array([[0, 1, 2, 3]])  # 4 个专家各一次

        # 参照：普通（非旋转）8-bit 前向
        store_p = FileExpertStore(src, capacity=8)
        plain = PersistentSubGLU(hidden, moe_inter, group_size=64, bits=8)
        fetched_p = store_p.fetch(0, [0, 1, 2, 3])
        y_plain = plain.forward(fetched_p, 4, x, local)

        # 被测：旋转 8-bit 前向
        store_r = FileExpertStore(rot, capacity=8)
        rotated = RotatedSubGLU(hidden, moe_inter, group_size=64, bits=8)
        fetched_r = store_r.fetch(0, [0, 1, 2, 3])
        y_rot = rotated.forward(fetched_r, 4, x, local)

        mx.eval(y_plain, y_rot)
        mae = float(mx.mean(mx.abs(y_plain - y_rot)))
        rel = mae / (float(mx.mean(mx.abs(y_plain))) + 1e-9)
        assert rel < 0.02, f"旋转管线偏差过大 rel={rel} mae={mae}"


def test_patch_filebacked_uses_rotated_sub():
    from mlx_streaming.streaming_moe import FileStreamingMoeBlock, RotatedSubGLU
    from mlx_streaming.expert_store import FileExpertStore
    with tempfile.TemporaryDirectory() as root:
        _make_expert_files(root, n_experts=4)
        store = FileExpertStore(root, capacity=8)

        class _Gate:
            def __call__(self, x):
                return mx.zeros((x.shape[0], 4))

        blk = FileStreamingMoeBlock(
            gate=_Gate(), top_k=2, norm_topk_prob=True, store=store, layer_idx=0,
            hidden=2048, moe_inter=768, group_size=64, bits=8, rotated=True)
        assert isinstance(blk._sub, RotatedSubGLU)
