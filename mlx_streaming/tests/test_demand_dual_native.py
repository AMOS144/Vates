"""Phase 2 方案B native 层单测：demand_dual 全接管的槽映射/字节等价 + LFU 驱逐逐步一致。

沿用 test_pool_sideregion_native.py 的临时小 blob + 结构化池风格。native 未编译则 skip。
MLX 不支持布尔索引 arr[mask]，本文件用 host 侧列表推导/mx.where 规避。
"""
import os
import random
import struct
import tempfile
from collections import Counter, OrderedDict

import pytest
import mlx.core as mx

try:
    import mlx_streaming.native_moe_ext as N
    _HAS = hasattr(N, "demand_dual")
except Exception:
    _HAS = False

pytestmark = pytest.mark.skipif(not _HAS, reason="native_moe_ext demand_dual 未编译")

CAP, SPEC, NE = 4, 3, 16
W, S = 16, 8
SEG = [W * 4, S * 1]
STRIDE = sum(SEG)


def _blob(path):
    with open(path, "wb") as f:
        for e in range(NE):                       # 专家 e：weight 全=e+1，scales 全=(e+100)&0xff
            f.write(struct.pack(f"<{W}I", *([e + 1] * W)))
            f.write(bytes([(e + 100) & 0xFF] * S))


def _pool(cap=CAP, spec=SPEC):
    w = mx.zeros((cap + spec, W), dtype=mx.uint32)
    sc = mx.zeros((cap + spec, S), dtype=mx.uint8)
    mx.eval(w, sc)
    return [w, sc]


# ---------- (a) 槽映射 + 字节等价 ----------

def test_demand_all_miss_places_correct_bytes():
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(0, CAP)
    inds = mx.array([[5, 6, 7]], dtype=mx.uint32)
    local = N.demand_dual(inds, pool, SEG, 0, 0, path, STRIDE, CAP, True, 0)
    mx.eval(local)
    slots = [int(v) for v in local.reshape(-1).tolist()]
    assert all(0 <= s < CAP for s in slots)           # 全落真实区
    for pos, e in zip(slots, [5, 6, 7]):
        assert int(pool[0][pos][0]) == e + 1          # weight 段
        assert int(pool[1][pos][0]) == (e + 100) & 0xFF  # scales 段
    st = N.demand_last_stats()
    assert st == [0, 3, 3, 1]                          # hitpos=0, misspos=3, loads=3, fallback
    assert {N.real_region_contents(0)[i] for i in range(0, 6, 2)} == {5, 6, 7}
    os.unlink(path)


def test_demand_hit_no_reload():
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(0, CAP)
    inds = mx.array([[5, 6]], dtype=mx.uint32)
    N.demand_dual(inds, pool, SEG, 0, 0, path, STRIDE, CAP, True, 0)
    local = N.demand_dual(inds, pool, SEG, 0, 0, path, STRIDE, CAP, True, 0)  # 再取一次全命中
    mx.eval(local)
    st = N.demand_last_stats()
    assert st == [2, 0, 0, 0]                          # 全命中，无 load，fastpath
    os.unlink(path)


def test_demand_side_overrides_real():
    # 侧区有 expert 5（覆盖真实区）：demand 命中侧区行，不读盘。
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(0, CAP)
    d = N.prefetch_pool_sideregion(pool, SEG, mx.array([5], dtype=mx.uint32), 0, path, STRIDE,
                                   [], SPEC, CAP, gen=0)
    mx.eval(d)
    # 等侧区落地
    import time
    t = time.time() + 2.0
    while time.time() < t and not N.sideregion_contents(0, 0):
        time.sleep(0.01)
    inds = mx.array([[5]], dtype=mx.uint32)
    local = N.demand_dual(inds, pool, SEG, 0, 0, path, STRIDE, CAP, True, 0)
    mx.eval(local)
    row = int(local.reshape(-1)[0])
    assert row >= CAP                                  # 命中侧区行(>=cap)
    assert N.demand_last_stats() == [1, 0, 0, 0]       # 侧区命中，无 load
    os.unlink(path)


# ---------- (b) LFU 驱逐逐步一致 ----------

class RefReal:
    """C++ demand_core 的 Python 参考实现（canonical 方案B 语义）。"""
    def __init__(self, cap, lfu, decay):
        self.cap, self.lfu, self.decay = cap, lfu, decay
        self.order, self.e2r, self.free, self.freq, self.access = [], {}, list(range(cap)), {}, 0

    def choose_victim(self, current):
        victim, best = -1, 0
        for e in self.order:
            if e not in self.e2r or e in current:
                continue
            f = self.freq.get(e, 0)
            if victim < 0 or f < best:
                victim, best = e, f
        return victim

    def place(self, flat):
        seen, access_order = set(), []
        for e in flat:
            if e not in seen:
                seen.add(e); access_order.append(e)
        local = [self.e2r.get(e, -1) for e in flat]
        miss, ms = [], set()
        for e in flat:
            if e not in self.e2r and e not in ms:
                ms.add(e); miss.append(e)
        if self.lfu:
            for e in access_order:
                self.freq[e] = self.freq.get(e, 0) + 1
            self.access += len(access_order)
            if self.decay > 0 and self.access >= self.decay:
                for e in list(self.freq):
                    self.freq[e] //= 2
                    if self.freq[e] == 0:
                        del self.freq[e]
                self.access = 0
        current, new_slot = set(access_order), {}   # 护本次全部唯一路由专家(命中+miss)不被驱逐
        for e in miss:
            if self.free:
                slot = self.free.pop(0)
            else:
                v = self.choose_victim(current)
                if v < 0:
                    new_slot[e] = 0; continue
                slot = self.e2r.pop(v); self.order.remove(v)
            self.e2r[e] = slot; self.order.append(e); new_slot[e] = slot
        return [new_slot.get(flat[i], local[i]) if local[i] < 0 else local[i]
                for i in range(len(flat))]


def test_ref_matches_python_choose_victim():
    # 参考实现 choose_victim 与真实 ResidentExpertPool._choose_victim 在随机状态下逐步一致。
    from mlx_streaming.core.cache.resident_pool import ResidentExpertPool
    rng = random.Random(20260702)
    for _ in range(300):
        cap = rng.randint(2, 8)
        experts = rng.sample(range(50), cap)          # cap 个已驻专家(插入序)
        freq = {e: rng.randint(0, 5) for e in experts}
        current = set(rng.sample(experts, rng.randint(0, cap - 1)))
        ref = RefReal(cap, True, 0)
        ref.order = list(experts); ref.e2r = {e: i for i, e in enumerate(experts)}; ref.freq = dict(freq)
        rp = ResidentExpertPool(cap, loader=lambda l, e: {}, spec_slots=SPEC)
        rp._ensure_layer(0)
        rp._slot_of[0] = OrderedDict((e, i) for i, e in enumerate(experts))
        rp._freq[0] = Counter(freq)
        assert ref.choose_victim(current) == rp._choose_victim(0, current)


def test_cpp_matches_ref_over_random_sequence():
    N.real_reset()
    rng = random.Random(12345)
    cap, decay = 4, 0
    ref = RefReal(cap, True, decay)
    N.real_init(0, cap)
    for step in range(200):
        flat = [rng.randint(0, 9) for _ in range(rng.randint(1, 5))]
        want = ref.place(flat)
        got = N.real_debug_place(0, flat, cap, True, decay)
        assert got == want, f"step {step} flat={flat}: cpp {got} != ref {want}"
        # resident 集合一致
        cpp_res = {N.real_region_contents(0)[i] for i in range(0, len(N.real_region_contents(0)), 2)}
        assert cpp_res == set(ref.e2r.keys()), f"step {step}: resident 不一致"
