from mlx_streaming.core.cache.virtual_pool import VirtualPool


def test_ahead_for_target_cutoff():
    vp = VirtualPool(num_layers=48, cutoff=6, ahead_lo=1, ahead_hi=3)
    assert vp.ahead_for(1) == 1
    assert vp.ahead_for(6) == 1
    assert vp.ahead_for(7) == 3
    assert vp.ahead_for(47) == 3


def test_target_ahead_profile_overrides_cutoff_schedule():
    vp = VirtualPool(
        num_layers=8, cutoff=2, ahead_lo=1, ahead_hi=3,
        ahead_profile={3: 2, 7: 2},
    )
    assert vp.ahead_for(1) == 1
    assert vp.ahead_for(3) == 2
    assert vp.ahead_for(4) == 3
    assert vp.ahead_for(7) == 2
    assert vp.targets_for(1) == (2, 3, 4)


def test_targets_for_covers_every_target_once_without_shortening():
    vp = VirtualPool(num_layers=48, cutoff=6, ahead_lo=1, ahead_hi=3)
    assert vp.targets_for(0) == (1,)
    assert vp.targets_for(4) == (5, 7)
    assert vp.targets_for(5) == (6, 8)
    assert vp.targets_for(6) == (9,)
    assert vp.targets_for(44) == (47,)
    assert vp.targets_for(45) == ()
    assert vp.targets_for(47) == ()

    scheduled = [
        (source, target)
        for source in range(48)
        for target in vp.targets_for(source)
    ]
    assert sorted(target for _, target in scheduled) == list(range(1, 48))
    assert all(
        target - source == (1 if target <= 6 else 3)
        for source, target in scheduled
    )


# ---- 双源双缓冲协调器（dual-source coordinator）----

class _FakeRP:
    spec_gens = 2        # 与真实 ResidentExpertPool 一致（acquire 返回 n_experts 需要）
    spec_slots = 16
    _native_demand = True   # demand_dual 唯一权威
    def __init__(self):
        self.dual_calls = []
    def cap_for(self, layer):
        return 32


class _FakeStg:
    def __init__(self):
        self.submits = []
    def submit(self, layer, pred, resident, **metadata):
        self.submits.append((layer, pred, tuple(resident), metadata))
        return None


def test_gen_advances_once_per_forward():
    # 前向内层号递增不推进代；下一前向层号回绕(<=上次) 才 +1 一次。
    vp = VirtualPool(_FakeRP(), _FakeStg(), spec_slots=16)
    for L in (3, 7, 40):
        vp.begin_forward(L)
    g_after_A = vp._gen
    for L in (3, 7, 40):
        vp.begin_forward(L)
    assert vp._gen == g_after_A + 1


def test_read_fill_gen_disjoint():
    # 同前向内读代与填代恒不同 → 无「本前向读的物理行被本前向 fill 覆盖」竞态。
    vp = VirtualPool(_FakeRP(), _FakeStg(), spec_slots=16)
    vp.begin_forward(0)
    assert vp.read_gen() != vp.fill_gen()


def test_acquire_uses_native_and_prefetch_uses_global_staging(monkeypatch):
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "0")
    rp, stg = _FakeRP(), _FakeStg()
    vp = VirtualPool(rp, stg, spec_slots=16)
    vp.begin_forward(0)                                  # gen=0：read_gen=1, fill_gen=0
    monkeypatch.setattr(vp, "_acquire_native",
                        lambda layer, inds, side_gen, cap, **_kw: rp.dual_calls.append((layer, side_gen))
                        or ("POOL", "LOCAL", 64))
    vp.acquire(5, "INDS", 128)
    vp.prefetch(6, "PRED", [1, 2], ["A", "B"])
    assert rp.dual_calls == [(5, vp.read_gen())]         # 取用走读代
    # 全局 staging 不再计算/写入每层侧区 base_row。
    assert stg.submits == [(
        6, "PRED", (1, 2), {"source_layer": -1, "forward_id": vp._gen},
    )]


def test_single_gen_read_equals_fill():
    class _RP1:
        spec_gens = 1
        _native_demand = True
        def cap_for(self, layer): return 32
    class _Stg:
        def submit_pool_sideregion(self, *a, **k): return None
    vp = VirtualPool(_RP1(), _Stg(), spec_slots=16)
    vp.begin_forward(0)
    assert vp.read_gen() == 0 and vp.fill_gen() == 0      # 单代:读=填=0
    vp.begin_forward(0)                                   # 下一前向(层号回绕)推进 gen
    assert vp.read_gen() == 0 and vp.fill_gen() == 0      # 单代恒 0


def test_double_gen_still_alternates():
    class _RP2:
        spec_gens = 2
        _native_demand = True
        def cap_for(self, layer): return 32
    class _Stg:
        def submit_pool_sideregion(self, *a, **k): return None
    vp = VirtualPool(_RP2(), _Stg(), spec_slots=16)
    vp.begin_forward(0)
    assert vp.read_gen() != vp.fill_gen()                 # 双代:读填不同
