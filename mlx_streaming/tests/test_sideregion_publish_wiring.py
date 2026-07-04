import mlx.core as mx
import pytest


class _FakeSrc:
    _segs = [
        ("gate_proj", "weight", "uint32", (2, 2), 16),
        ("gate_proj", "scales", "uint16", (2, 2), 8),
    ]


class _FakeStg:
    def __init__(self):
        self.src = _FakeSrc()
    def sideregion_publish(self, layer, gen, seg_nbytes):
        assert seg_nbytes == [16, 8]
        rows = mx.array([5, 6], dtype=mx.int32)
        w = mx.full((2, 16), 0, dtype=mx.uint8)
        s = mx.zeros((2, 8), dtype=mx.uint8)
        return rows, [w, s]


def test_staging_side_publish_splits_segments():
    from mlx_streaming.core.prefetch.native_staging import _StagingSide
    side = _StagingSide(_FakeStg(), gen=0)
    rows, out = side.publish(0)
    assert int(rows.shape[0]) == 2
    assert set(out.keys()) == {"gate_proj.weight", "gate_proj.scales"}
    assert out["gate_proj.weight"].shape == (2, 2, 2)
    assert out["gate_proj.weight"].dtype == mx.uint32
    assert out["gate_proj.scales"].shape == (2, 2, 2)
    assert out["gate_proj.scales"].dtype == mx.bfloat16


def test_staging_side_publish_empty():
    class _EmptyStg(_FakeStg):
        def sideregion_publish(self, layer, gen, seg_nbytes):
            return mx.array([], dtype=mx.int32), [mx.zeros((0, 16), dtype=mx.uint8),
                                                  mx.zeros((0, 8), dtype=mx.uint8)]
    from mlx_streaming.core.prefetch.native_staging import _StagingSide
    side = _StagingSide(_EmptyStg(), gen=0)
    rows, out = side.publish(0)
    assert int(rows.shape[0]) == 0
    assert out == {}
