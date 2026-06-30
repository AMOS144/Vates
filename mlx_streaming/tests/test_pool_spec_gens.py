import mlx.core as mx
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _fake(seed):
    mx.random.seed(seed)
    w = mx.random.normal((32, 64))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}


def test_spec_gens_doubles_side_rows():
    # spec_gens=2：物理行 = cap + 2*spec；真实区仍只 cap 行入 free（侧区不被 LRU）
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3, spec_gens=2)
    p.acquire(0, [10])
    assert p.allocated_slots(0) == 4 + 2 * 3


def test_spec_gens_default_one_preserves_single_buffer():
    # 默认 spec_gens=1：退回现状 cap+spec（保 test_resident_sideregion 不破）
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    assert p.allocated_slots(0) == 4 + 3
