import mlx.core as mx
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _fake(seed):
    mx.random.seed(seed)
    w = mx.random.normal((32, 64))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}


def test_spec_gens_does_not_add_physical_side_rows():
    # spec_slots/spec_gens 只控制统一池内的投机准入，不再扩展物理侧池。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3, spec_gens=2)
    p.acquire(0, [10])
    assert p.allocated_slots(0) == 4


def test_spec_gens_default_one_uses_same_unified_pool():
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    assert p.allocated_slots(0) == 4
