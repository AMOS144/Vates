import mlx.core as mx
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _fake(seed):
    mx.random.seed(seed)
    w = mx.random.normal((32, 64))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}


def test_prealloc_no_grow():
    # spec_slots>0 → 首次 miss 即预分配 cap+spec 行，且不随工作集增长
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    assert p.allocated_slots(0) == 4 + 3            # 预分配满
    p.acquire(0, [11, 12, 13])                      # 填到 cap，仍不超分配
    assert p.allocated_slots(0) == 4 + 3


def test_acquire_gpu_overlays_sideregion():
    # 池里 [10]；侧区映射 {20: 行5}（伪造 contents）；inds=[10,20] 应全命中、local 指对行
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    slot10 = p._slot_of[0][10]

    class _Side:
        def contents(self, layer): return {20: 5}   # 物理侧区行 5（∈[4,7)）

    inds = mx.array([[[10, 20]]], dtype=mx.uint32)
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side())
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert loc == [slot10, 5]                        # 10→真实槽，20→侧区行5


def test_acquire_gpu_dual_fallback_true_miss():
    # 真 miss（既不在池也不在侧区）走回退：读盘落真实区 [0,cap)，gpu_fallback 计数 +1
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])

    class _Side:
        def contents(self, layer): return {20: 5}   # 20 在侧区

    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)   # 30 = 真 miss
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side())
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert p.gpu_fallback == 1
    assert loc[1] == 5                                   # 侧区命中仍指行5
    assert 0 <= loc[2] < 4                               # 30 落在真实区 [0,cap)
    assert loc[2] != 5                                   # 不会落进侧区行


def test_real_region_eviction_never_touches_sideregion():
    # 真实区填到 cap 并超额驱逐，侧区物理行 [cap,cap+spec) 永不被分配/驱逐
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    for e in [10, 11, 12, 13, 14, 15]:                   # 6 个 > cap=4 → 触发 LRU 驱逐
        p.acquire(0, [e])
    slots = set(p._slot_of[0].values())
    assert all(0 <= s < 4 for s in slots)                # 所有占用槽都在真实区
    assert p.allocated_slots(0) == 4 + 3                 # 预分配未变
