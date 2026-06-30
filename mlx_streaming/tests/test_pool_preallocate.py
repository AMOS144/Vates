import mlx.core as mx

from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def test_preallocate_full_cap_typed_pool():
    rp = ResidentExpertPool(capacity=5, loader=lambda l, e: {})
    sample = {"gate_proj.weight": mx.zeros((3, 4), dtype=mx.uint32),
              "gate_proj.scales": mx.zeros((3, 2), dtype=mx.bfloat16)}
    rp.preallocate(0, sample, 5)
    assert rp._pools[0]["gate_proj.weight"].shape == (5, 3, 4)
    assert rp._pools[0]["gate_proj.weight"].dtype == mx.uint32
    assert rp._pools[0]["gate_proj.scales"].dtype == mx.bfloat16
    assert rp._alloc[0] == 5
    # 幂等：二次调用不重建（保持指针稳定）
    before = rp._pools[0]["gate_proj.weight"]
    rp.preallocate(0, sample, 5)
    assert rp._pools[0]["gate_proj.weight"] is before
