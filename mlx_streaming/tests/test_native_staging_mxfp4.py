import mlx.core as mx
from mlx_streaming.core.prefetch.native_staging import NativeStagingManager


class _FakeSrc:
    """伪 BlobExpertSource：只需 _segs / stride，供 _slice 单测。"""
    def __init__(self, segs, stride):
        self._segs = segs
        self.stride = stride
        self.dir = "/tmp"


def test_slice_mxfp4_keeps_uint8_scales_and_uint32_weight():
    # mxfp4 段表：weight uint32，scales uint8（无 biases、不 view bf16）。
    segs = [
        ("gate_proj", "weight", "uint32", (2, 2), 16),   # 2*2*4=16 字节
        ("gate_proj", "scales", "uint8", (2, 2), 4),     # 2*2*1=4 字节
    ]
    stride = 20
    mgr = NativeStagingManager(_FakeSrc(segs, stride), budget=1)
    row = mx.zeros((stride,), dtype=mx.uint8)
    out = mgr._slice(row)
    assert out["gate_proj.weight"].dtype == mx.uint32
    assert out["gate_proj.weight"].shape == (2, 2)
    assert out["gate_proj.scales"].dtype == mx.uint8     # 关键：mxfp4 scales 保持 uint8
    assert out["gate_proj.scales"].shape == (2, 2)


def test_slice_affine_still_views_bf16():
    # v1 affine：scales/biases uint16 → view bfloat16（不得回归）。
    segs = [
        ("gate_proj", "weight", "uint32", (2, 2), 16),
        ("gate_proj", "scales", "uint16", (2, 2), 8),    # 2*2*2=8 字节
    ]
    stride = 24
    mgr = NativeStagingManager(_FakeSrc(segs, stride), budget=1)
    row = mx.zeros((stride,), dtype=mx.uint8)
    out = mgr._slice(row)
    assert out["gate_proj.weight"].dtype == mx.uint32
    assert out["gate_proj.scales"].dtype == mx.bfloat16  # affine 仍 view bf16
