from mlx_streaming.core.cache.virtual_pool import VirtualPool


def test_ahead_for_cutoff():
    vp = VirtualPool(num_layers=48, cutoff=6, ahead_lo=1, ahead_hi=3)
    assert vp.ahead_for(0) == 1
    assert vp.ahead_for(5) == 1
    assert vp.ahead_for(6) == 3
    assert vp.ahead_for(40) == 3


def test_target_for_skip_no_clamp():
    vp = VirtualPool(num_layers=48, cutoff=6, ahead_lo=1, ahead_hi=3)
    assert vp.target_for(0) == 1        # 0+1
    assert vp.target_for(10) == 13      # 10+3
    assert vp.target_for(44) == 47      # 44+3=47 恰好命中末层 → 预读
    assert vp.target_for(45) == 0       # 45+3=48 越界 → 跳过（不再 clamp 堆叠到 47）
    assert vp.target_for(46) == 0       # 46+3=49 越界 → 跳过
    assert vp.target_for(47) == 0       # 末层无可预读 → 0


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
    def submit_pool_sideregion(self, layer, pred, resident, pool_list, base_row, gen=0):
        self.submits.append((layer, base_row, gen))
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


def test_acquire_uses_read_gen_prefetch_uses_fill_gen_base(monkeypatch):
    rp, stg = _FakeRP(), _FakeStg()
    vp = VirtualPool(rp, stg, spec_slots=16)
    vp.begin_forward(0)                                  # gen=0：read_gen=1, fill_gen=0
    monkeypatch.setattr(vp, "_acquire_native",
                        lambda layer, inds, side_gen, cap: rp.dual_calls.append((layer, side_gen))
                        or ("POOL", "LOCAL"))
    vp.acquire(5, "INDS", 128)
    vp.prefetch(6, "PRED", [1, 2], ["A", "B"])
    assert rp.dual_calls == [(5, vp.read_gen())]         # 取用走读代
    # 预取走填代，base_row = cap_for + fill_gen*spec
    assert stg.submits == [(6, 32 + vp.fill_gen() * 16, vp.fill_gen())]


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
