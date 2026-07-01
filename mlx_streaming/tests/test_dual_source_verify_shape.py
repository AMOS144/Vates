import mlx.core as mx
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _fake(seed):
    mx.random.seed(seed)
    w = mx.random.normal((32, 64))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}


class _Side:
    def kv(self, layer):
        # 模拟 C++ sideregion_kv：(keys uint32, vals int32) device 数组。
        return (mx.array([20], dtype=mx.uint32), mx.array([5], dtype=mx.int32))


def test_acquire_gpu_dual_verify_shape_overlay():
    # verify 形状 inds=(1, K+1=3, k=2)：池里 [10]，侧区 {20:行5}，全命中且 local 指对行。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    slot10 = p._slot_of[0][10]
    inds = mx.array([[[10, 20], [20, 10], [10, 20]]], dtype=mx.uint32)  # (1,3,2)
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side())
    assert local.shape == inds.shape                       # 形状保持
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert set(loc) == {slot10, 5}                          # 仅这两行
    assert loc == [slot10, 5, 5, slot10, slot10, 5]         # 逐位对
