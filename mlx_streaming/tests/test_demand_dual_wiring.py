"""demand_dual 接线单测：demand_dual 是双源 decode 唯一权威（无 opt-out）。

启用判据 = spec_slots>0 且 native 编译了 demand_dual；非 spec 或 native 缺失则 _native_demand=False。
dual 模式下若 native 缺失，VirtualPool.acquire 明确报错（不再有 Python 退路）。
"""
import pytest
import mlx.core as mx

from mlx_streaming import config
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool
from mlx_streaming.core.cache.virtual_pool import VirtualPool

try:
    import mlx_streaming.native_moe_ext as N
    _HAS = hasattr(N, "demand_dual")
except Exception:
    _HAS = False


def _rp(spec=8):
    return ResidentExpertPool(8, loader=lambda l, e: {}, spec_slots=spec)


def test_spec_zero_disables_native_demand():
    # 非 spec 模式（spec_slots=0）→ 不启用 demand_dual。
    rp = _rp(spec=0)
    assert rp._native_demand is False


@pytest.mark.skipif(not _HAS, reason="native 未编译")
def test_spec_mode_enables_native_demand():
    # spec 模式 + native 已编译 → 自动启用（无需任何开关）。
    rp = _rp(spec=8)
    assert rp._native_demand is True


@pytest.mark.skipif(not _HAS, reason="native 未编译")
def test_layer0_native_capacity_is_the_single_owned_main_pool(monkeypatch):
    """Direct prefetch no longer appends or advertises hidden side rows."""
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "1")
    monkeypatch.setenv("LAYER0_SLOTS", "128")
    N.real_reset()
    rp = ResidentExpertPool(
        32,
        loader=lambda _layer, _expert: {
            "weight": mx.zeros((3,), dtype=mx.uint32),
        },
        spec_slots=32,
        spec_gens=1,
    )
    rp._bootstrap_dual_pool(0)
    assert rp.native_real_cap_for(0) == 32
    assert int(rp._pools[0]["weight"].shape[0]) == 32


class _RPNoNative:
    """native 缺失的 rp mock：_native_demand=False → dual 模式下 acquire 应报错（无 Python 退路）。"""
    spec_gens = 2
    spec_slots = 16
    _native_demand = False
    _pinned = {}

    def cap_for(self, layer):
        return 32


class _Stg:
    pass


def test_vpool_raises_when_native_missing():
    rp = _RPNoNative()
    vp = VirtualPool(rp, _Stg(), spec_slots=16)
    vp.begin_forward(0)
    with pytest.raises(RuntimeError):
        vp.acquire(0, "INDS", 128, seq_len=1, layer_cap=32)


@pytest.mark.skipif(not _HAS, reason="native 未编译")
def test_resident_experts_reads_cpp_when_native():
    # _native_demand=True 时 resident_experts/_count 反映 C++ g_real 内容（预取过滤一致性）。
    N.real_reset()
    rp = _rp(spec=8)
    assert rp._native_demand is True
    layer = 3
    N.real_init(layer, 4)
    N.real_debug_place(layer, [10, 11, 12], 4, True, 0)     # 让 C++ 装入 10,11,12
    assert rp.resident_experts(layer) == {10, 11, 12}
    assert rp.resident_count(layer) == 3
    flat = N.real_region_contents(layer)
    assert {flat[i] for i in range(0, len(flat), 2)} == {10, 11, 12}


def test_native_real_pin_is_visible_and_not_evicted():
    if not _HAS:
        pytest.skip("native 未编译")
    if not hasattr(N, "real_pin"):
        pytest.skip("native extension 尚未编译 real_pin")
    N.real_reset()
    assert list(N.real_pin(0, [1, 2], 4)) == [0, 1]
    assert list(N.real_pinned_contents(0)) == [1, 2]
    # 剩余两个槽反复驱逐，pin 必须始终保留在 g_real 权威账本中。
    N.real_debug_place(0, [3, 4], 4, True, 0)
    N.real_debug_place(0, [5], 4, True, 0)
    assert {1, 2} <= {
        int(value) for value in N.real_region_contents(0)[::2]
    }


def test_resident_pool_native_pin_writes_owned_pool_bytes():
    if not _HAS or not hasattr(N, "real_pin"):
        pytest.skip("native extension 尚未编译 real_pin")
    N.real_reset()

    def loader(_layer, expert):
        return {"weight": mx.full((3,), int(expert) + 17, dtype=mx.uint32)}

    rp = ResidentExpertPool(4, loader=loader, spec_slots=2)
    rp.pin(0, [2, 5])
    contents = list(N.real_region_contents(0))
    slots = {
        int(contents[index]): int(contents[index + 1])
        for index in range(0, len(contents), 2)
    }
    assert set(N.real_pinned_contents(0)) == {2, 5}
    assert rp._pools[0]["weight"][slots[2]].tolist() == [19, 19, 19]
    assert rp._pools[0]["weight"][slots[5]].tolist() == [22, 22, 22]
