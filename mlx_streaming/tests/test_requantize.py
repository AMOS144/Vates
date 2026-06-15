"""重量化工具单测：把 per-expert 量化文件从源 bit 重量化到目标 bit。

校验点：
- 同 bit 重量化应近似往返（dequant→quant→dequant 的二次误差很小）。
- 低 bit（2-bit）输出文件的 packed weight 维度更小（≈ 字节减半）。
- 输出文件结构完整（三组 proj 各含 weight/scales/biases），加载后数值有限。
"""
import os

import mlx.core as mx

from mlx_streaming.prep.requantize_experts import requantize_file

PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]


def _make_src_expert(path: str, hidden: int, inter: int, group_size: int, bits: int):
    """造一个 4-bit 量化的 per-expert 文件（gate/up: inter×hidden, down: hidden×inter）。"""
    d = {}
    shapes = {"gate_proj": (inter, hidden), "up_proj": (inter, hidden),
              "down_proj": (hidden, inter)}
    for name in PROJ_NAMES:
        O, I = shapes[name]
        W = mx.random.normal((O, I))
        wq, s, b = mx.quantize(W, group_size=group_size, bits=bits)
        d[f"{name}.weight"] = wq
        d[f"{name}.scales"] = s
        d[f"{name}.biases"] = b
    mx.eval(d)
    mx.save_safetensors(path, d)


def test_requantize_lower_bit_shrinks_and_loads(tmp_path):
    mx.random.seed(0)
    hidden, inter, gs = 128, 64, 64
    src = str(tmp_path / "layer00_expert000.safetensors")
    dst = str(tmp_path / "out" / "layer00_expert000.safetensors")
    _make_src_expert(src, hidden, inter, gs, bits=4)

    src_w = mx.load(src)["gate_proj.weight"]
    requantize_file(src, dst, src_bits=4, src_group=gs, dst_bits=2, dst_group=gs)

    out = mx.load(dst)
    # 结构完整
    for name in PROJ_NAMES:
        for suf in ("weight", "scales", "biases"):
            assert f"{name}.{suf}" in out
    # 2-bit packed weight 列数应为 4-bit 的一半
    assert out["gate_proj.weight"].shape[1] * 2 == src_w.shape[1]
    # 加载后可反量化、数值有限
    Wd = mx.dequantize(out["gate_proj.weight"], out["gate_proj.scales"],
                       out["gate_proj.biases"], group_size=gs, bits=2)
    mx.eval(Wd)
    assert bool(mx.all(mx.isfinite(Wd)).item())


def test_requantize_same_bit_roundtrips(tmp_path):
    mx.random.seed(1)
    hidden, inter, gs = 128, 64, 64
    src = str(tmp_path / "layer00_expert000.safetensors")
    dst = str(tmp_path / "out" / "layer00_expert000.safetensors")
    _make_src_expert(src, hidden, inter, gs, bits=4)

    s = mx.load(src)
    W_src = mx.dequantize(s["gate_proj.weight"], s["gate_proj.scales"],
                          s["gate_proj.biases"], group_size=gs, bits=4)
    requantize_file(src, dst, src_bits=4, src_group=gs, dst_bits=4, dst_group=gs)
    o = mx.load(dst)
    W_dst = mx.dequantize(o["gate_proj.weight"], o["gate_proj.scales"],
                          o["gate_proj.biases"], group_size=gs, bits=4)
    mx.eval(W_src, W_dst)
    # 同 bit 重量化近似幂等：二次量化误差应远小于 4-bit 自身的量化误差(~0.078)
    mae = float(mx.mean(mx.abs(W_dst - W_src)))
    assert mae < 0.02
