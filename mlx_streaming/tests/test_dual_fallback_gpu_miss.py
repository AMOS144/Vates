import mlx.core as mx
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _kv(d):
    return (mx.array(list(d.keys()), dtype=mx.uint32),
            mx.array(list(d.values()), dtype=mx.int32))


def _fake(seed):
    mx.random.seed(seed)
    w = mx.random.normal((32, 64))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}


class _Side:
    def __init__(self, d): self._d = d
    def kv(self, layer): return _kv(self._d)


def test_fallback_true_miss_maps_correctly():
    # 池里 [10]；侧区 {20:5}；inds=[10,20,30]，30 真 miss。
    # 回退后 local 逐元素：10→真实槽、20→侧区行5、30→真实区 [0,cap)。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    slot10 = p._slot_of[0][10]
    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side({20: 5}))
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert p.gpu_fallback == 1
    assert loc[0] == slot10
    assert loc[1] == 5
    assert 0 <= loc[2] < 4 and loc[2] != 5
    assert 30 in p._slot_of[0]          # 真 miss 已落真实区


def test_fallback_hit_count_matches_positions():
    # hit 计数按位置口径：inds.size - n_miss_positions。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    h0 = p.hits
    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)   # 2 命中(10,20) + 1 miss(30)
    p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side({20: 5}))
    assert p.hits - h0 == 2


def test_fallback_multi_miss_dedup():
    # 同一 miss 专家在多个位置出现：只读盘一次、落一个槽，两处 local 指同槽。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    inds = mx.array([[[30, 10, 30]]], dtype=mx.uint32)   # 30 出现两次
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side({}))
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert loc[0] == loc[2]                              # 两处 30 指同槽
    assert list(p._slot_of[0].keys()).count(30) == 1     # 只落一个
