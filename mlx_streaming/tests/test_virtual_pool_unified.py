"""Phase 1（API 收口）单测：VirtualPool.acquire / acquire_host 统一取用入口等价性。

验证 VirtualPool 对外呈现「所有专家都在」的视角：dual / 非 dual GPU-remap / host / fetch
四条路径都收口进 VirtualPool，返回统一三元组 (pool_arrays, local, n_experts)，
与 block.py 原逐分支逐元素等价。沿用 test_resident_sideregion.py 的轻量 mock 风格。
"""
import mlx.core as mx

from mlx_streaming.core.cache.virtual_pool import VirtualPool


class _RPDual:
    """dual 路径 mock：记录调用、返回哨兵。"""
    spec_gens = 2
    spec_slots = 16

    def __init__(self):
        self.calls = []

    def cap_for(self, layer):
        return 32

    def acquire_gpu_dual(self, layer, inds, num_experts, side):
        self.calls.append((layer, num_experts, side._gen))
        return ("POOL_DUAL", "LOCAL_DUAL")


class _Stg:
    def sideregion_kv(self, layer, gen):
        return (mx.array([], dtype=mx.uint32), mx.array([], dtype=mx.int32))


class _RPRemap:
    """非 dual GPU-remap 路径 mock。"""
    spec_gens = 1
    spec_slots = 0

    def cap_for(self, layer):
        return 32

    def acquire_gpu(self, layer, inds, num_experts):
        return ("POOL_REMAP", "LOCAL_REMAP")


class _RPHost:
    """host/fetch 路径 mock（cap=4）。"""
    def __init__(self):
        self.calls = []

    def cap_for(self, layer):
        return 4

    def acquire(self, layer, flat):
        self.calls.append(("acquire", tuple(flat)))
        return ("POOL_HOST", [0, 1, 0])

    def fetch(self, layer, uniq_sorted):
        self.calls.append(("fetch", tuple(uniq_sorted)))
        return "POOL_FETCH"


def test_acquire_dual_returns_n_experts():
    # dual：走 acquire_gpu_dual，n_experts = layer_cap + spec_gens*spec_slots
    rp = _RPDual()
    vp = VirtualPool(rp, _Stg(), spec_slots=16)
    vp.begin_forward(0)
    pool, local, n_exp = vp.acquire(0, "INDS", 128, seq_len=4, layer_cap=32)
    assert pool == "POOL_DUAL" and local == "LOCAL_DUAL"
    assert n_exp == 32 + 2 * 16
    assert rp.calls == [(0, 128, vp.read_gen())]      # 用读代


def test_acquire_nondual_gpu_remap():
    # 无 staging / spec=0：走 acquire_gpu，n_experts = layer_cap
    vp = VirtualPool(_RPRemap(), staging=None, spec_slots=0)
    vp.begin_forward(0)
    pool, local, n_exp = vp.acquire(0, "INDS", 128, seq_len=1, layer_cap=32)
    assert pool == "POOL_REMAP" and local == "LOCAL_REMAP"
    assert n_exp == 32


def test_acquire_host_under_cap_uses_acquire():
    # host：uniq(2) <= cap(4) → acquire，n_experts = layer_cap
    rp = _RPHost()
    vp = VirtualPool(rp, staging=None, spec_slots=0)
    pool, local, n_exp = vp.acquire_host(0, [10, 11, 10], (1, 3), mx.uint32, layer_cap=4)
    assert pool == "POOL_HOST" and n_exp == 4
    assert [int(v) for v in local.reshape(-1).tolist()] == [0, 1, 0]
    assert rp.calls == [("acquire", (10, 11, 10))]


def test_acquire_host_over_cap_uses_fetch():
    # host：uniq(5) > cap(4) → fetch，local 为 remap 到 [0,5) 的连续索引
    rp = _RPHost()
    vp = VirtualPool(rp, staging=None, spec_slots=0)
    flat = [10, 11, 12, 13, 14]
    pool, local, n_exp = vp.acquire_host(0, flat, (1, 5), mx.uint32, layer_cap=4)
    assert pool == "POOL_FETCH" and n_exp == 5
    assert sorted(set(int(v) for v in local.reshape(-1).tolist())) == [0, 1, 2, 3, 4]
    assert rp.calls == [("fetch", (10, 11, 12, 13, 14))]
