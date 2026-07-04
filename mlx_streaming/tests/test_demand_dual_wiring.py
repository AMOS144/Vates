"""Phase 2 方案B Python 接线单测：flag 门控 / native 缺失回退 / resident_experts 一致性。"""
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


def _rp(spec=8, native=None):
    return ResidentExpertPool(8, loader=lambda l, e: {}, spec_slots=spec)


def test_flag_off_keeps_python_authority(monkeypatch):
    # 显式关(NATIVE_DEMAND_DUAL=0)：_native_demand=False → 回退 Python 权威路径。
    monkeypatch.setattr(config, "native_demand_dual", lambda: False)
    rp = _rp(spec=8)
    assert rp._native_demand is False


def test_flag_on_needs_spec_mode(monkeypatch):
    # 开开关但 spec_slots=0（非 dual 侧区模式）→ 不启用。
    monkeypatch.setattr(config, "native_demand_dual", lambda: True)
    rp = _rp(spec=0)
    assert rp._native_demand is False


@pytest.mark.skipif(not _HAS, reason="native 未编译")
def test_flag_on_with_native_enables(monkeypatch):
    monkeypatch.setattr(config, "native_demand_dual", lambda: True)
    rp = _rp(spec=8)
    assert rp._native_demand is True


class _RPPy:
    """无 native 的 rp mock：_native_demand=False → VirtualPool 走 acquire_gpu_dual。"""
    spec_gens = 2
    spec_slots = 16
    _native_demand = False
    _pinned = {}

    def __init__(self):
        self.calls = []

    def cap_for(self, layer):
        return 32

    def acquire_gpu_dual(self, layer, inds, num_experts, side):
        self.calls.append(("py", layer, side._gen))
        return ("POOL_PY", "LOCAL_PY")


class _Stg:
    pass


def test_vpool_falls_back_when_native_off():
    rp = _RPPy()
    vp = VirtualPool(rp, _Stg(), spec_slots=16)
    vp.begin_forward(0)
    pool, local, n = vp.acquire(0, "INDS", 128, seq_len=1, layer_cap=32)
    assert pool == "POOL_PY" and local == "LOCAL_PY"       # 走了 Python 权威路径
    assert rp.calls and rp.calls[0][0] == "py"


@pytest.mark.skipif(not _HAS, reason="native 未编译")
def test_resident_experts_reads_cpp_when_native(monkeypatch):
    # _native_demand=True 时 resident_experts/_count 反映 C++ g_real 内容（预取过滤一致性）。
    monkeypatch.setattr(config, "native_demand_dual", lambda: True)
    N.real_reset()
    rp = _rp(spec=8)
    assert rp._native_demand is True
    layer = 3
    N.real_init(layer, 4)
    N.real_debug_place(layer, [10, 11, 12], 4, True, 0)     # 让 C++ 装入 10,11,12
    assert rp.resident_experts(layer) == {10, 11, 12}
    assert rp.resident_count(layer) == 3
    # 与 C++ 直接读一致
    flat = N.real_region_contents(layer)
    assert {flat[i] for i in range(0, len(flat), 2)} == {10, 11, 12}
