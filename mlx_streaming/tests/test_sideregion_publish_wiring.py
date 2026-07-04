import mlx.core as mx


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


def test_acquire_gpu_dual_publishes_side_rows(monkeypatch):
    import mlx.core as mx
    from mlx_streaming.core.cache.resident_pool import ResidentExpertPool

    rp = ResidentExpertPool(loader=lambda l, e: {"gate_proj.weight": mx.zeros((2, 2), mx.uint32)},
                            capacity=2, spec_slots=2, spec_gens=1)
    layer, num_experts = 0, 8
    rp._pools[layer] = {"gate_proj.weight": mx.zeros((4, 2, 2), dtype=mx.uint32)}
    mx.eval(list(rp._pools[layer].values()))
    rp._alloc[layer] = 4
    rp._slot_table[layer] = mx.array([-1] * num_experts, dtype=mx.int32)

    class _Side:
        def kv(self, l):
            return mx.array([5, 6], dtype=mx.uint32), mx.array([2, 3], dtype=mx.int32)
        def publish(self, l):
            rows = mx.array([2, 3], dtype=mx.int32)
            payload = mx.stack([mx.full((2, 2), 55, dtype=mx.uint32),
                                mx.full((2, 2), 66, dtype=mx.uint32)], axis=0)
            return rows, {"gate_proj.weight": payload}

    inds = mx.array([[5, 6]], dtype=mx.int32)
    pool, local = rp.acquire_gpu_dual(layer, inds, num_experts, _Side())
    mx.eval(pool["gate_proj.weight"])
    assert bool(mx.all(pool["gate_proj.weight"][2] == 55))
    assert bool(mx.all(pool["gate_proj.weight"][3] == 66))
