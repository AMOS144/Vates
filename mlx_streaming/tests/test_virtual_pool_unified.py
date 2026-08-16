"""Phase 1（API 收口）单测：VirtualPool.acquire / acquire_host 统一取用入口等价性。

验证 VirtualPool 对外呈现「所有专家都在」的视角：dual / 非 dual GPU-remap / host / fetch
四条路径都收口进 VirtualPool，返回统一三元组 (pool_arrays, local, n_experts)，
与 block.py 原逐分支逐元素等价。用轻量 mock 风格。
"""
import mlx.core as mx

from mlx_streaming.core.cache.virtual_pool import VirtualPool


class _RPDual:
    """dual 路径 mock：demand_dual 唯一权威（_native_demand=True）。"""
    spec_gens = 2
    spec_slots = 16
    _native_demand = True

    def __init__(self):
        self.calls = []

    def cap_for(self, layer):
        return 32


class _Stg:
    def __init__(self):
        self.late_promoter = None

    def register_late_promoter(self, promoter):
        self.late_promoter = promoter

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


def test_acquire_dual_returns_n_experts(monkeypatch):
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "0")
    # unified：旧侧区容量已并入 layer_cap，staging 不增加可寻址行。
    rp = _RPDual()
    vp = VirtualPool(rp, _Stg(), spec_slots=16)
    vp.begin_forward(0)
    # stub _acquire_native（避开 native 依赖），记录派发入参
    monkeypatch.setattr(vp, "_acquire_native",
                        lambda layer, inds, side_gen, cap, **_kw: rp.calls.append((layer, side_gen, cap))
                        or ("POOL_DUAL", "LOCAL_DUAL", 32))
    pool, local, n_exp = vp.acquire(0, "INDS", 128, seq_len=4, layer_cap=32)
    assert pool == "POOL_DUAL" and local == "LOCAL_DUAL"
    assert n_exp == 32
    assert rp.calls == [(0, vp.read_gen(), 32)]      # 用读代 + cap


def test_progressive_retains_late_staging_as_speculative_cache(monkeypatch):
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "0")
    monkeypatch.setenv("PREFETCH_PROGRESSIVE", "1")
    staging = _Stg()
    VirtualPool(_RPDual(), staging, spec_slots=16)
    assert staging.late_promoter is not None


def test_nonprogressive_can_retain_late_staging(monkeypatch):
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "0")
    monkeypatch.setenv("PREFETCH_PROGRESSIVE", "0")
    staging = _Stg()
    VirtualPool(_RPDual(), staging, spec_slots=16)
    assert staging.late_promoter is not None


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


def test_dual_overcap_temporary_fetch_preserves_route_order():
    class _RPTemporary:
        stacked_batch_loader = None
        batch_loader = None

        @staticmethod
        def loader(_layer, expert):
            return {"weight": mx.array([expert, expert + 100], dtype=mx.int32)}

    vp = VirtualPool(_RPTemporary(), staging=None, spec_slots=0)
    inds = mx.array([[5, 7, 5, 9]], dtype=mx.uint32)
    pool, local, n_exp = vp._temporary_fetch(3, inds)
    assert n_exp == 3
    assert pool["weight"].tolist() == [[5, 105], [7, 107], [9, 109]]
    assert local.tolist() == [[0, 1, 0, 2]]


def test_early_rerank_acceptance_is_deferred_until_target_truth(monkeypatch):
    from mlx_streaming.core.prefetch import progressive_acceptance

    captured = []
    monkeypatch.setattr(
        progressive_acceptance, "record",
        lambda layer, **values: captured.append((layer, values)),
    )
    vp = VirtualPool(_RPRemap(), staging=None, spec_slots=0)
    vp.begin_forward(0)
    candidates = mx.array([[0, 1, 2, 3]], dtype=mx.uint32)
    selected = mx.array([0, 2, 2], dtype=mx.uint32)
    width = mx.array(2, dtype=mx.int32)
    vp.record_rerank_acceptance(
        2,
        candidate_ids=candidates,
        selected_ids=selected,
        online_width=width,
        resident=(7,),
    )

    assert captured == []
    actual = mx.array([[0, 2, 7]], dtype=mx.uint32)
    assert vp._defer_progressive_acceptance(2, actual)
    assert vp.flush_progressive_acceptance() == 1
    assert captured[0][0] == 2
    assert captured[0][1]["actual_ids"] is actual
    assert captured[0][1]["resident"] == (7,)


def test_ordinary_optimistic_prefetch_uses_gpu_only_remap(monkeypatch):
    """T-ahead rerank needs no progressive dummy to enter the replay-safe path."""
    import mlx_streaming.native_moe_ext as native

    class _RP:
        spec_gens = 1
        spec_slots = 32
        _native_demand = True
        eviction_policy = "lfu"
        lfu_decay_interval = 1

        def __init__(self):
            self._pools = {3: "POOL"}

        def _bootstrap_dual_pool(self, _layer):
            return None

        def native_real_cap_for(self, _layer):
            return 64

        def allocated_slots(self, _layer):
            return 64

    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "1")
    monkeypatch.setenv("PREFETCH_EXACT_GPU_DEMAND", "1")
    monkeypatch.setenv("PREFETCH_OPTIMISTIC_VERIFY", "1")
    monkeypatch.setenv("PREFETCH_EXACT_NO_IO", "1")
    monkeypatch.setenv("DEMAND_ASYNC", "0")
    calls = []
    monkeypatch.setattr(
        native,
        "demand_gpu_remap_only",
        lambda inds, layer, generation, cap, use_side: (
            calls.append((inds, layer, generation, cap, use_side))
            or mx.array([[-1, 7]], dtype=mx.int32)
        ),
    )

    vp = VirtualPool(_RP(), staging=object(), spec_slots=32)
    vp._native_meta = lambda _layer: ([], [], "unused", 0)
    inds = mx.array([[11, 12]], dtype=mx.uint32)
    pool, local, n_exp = vp._acquire_direct(3, inds, 0, 64, seq_len=1)

    assert pool == "POOL"
    assert local.tolist() == [[0, 7]]
    assert n_exp == 64
    assert len(calls) == 1
    assert bool(vp.take_optimistic_miss_flag().item())
