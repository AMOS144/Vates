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
    assert st[:4] == [0, 3, 3, 1]                      # hitpos=0, misspos=3, loads=3, fallback
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
    assert st[:4] == [2, 0, 0, 0]                      # 全命中，无 load，fastpath
    os.unlink(path)


def test_adaptive_predictor_rearms_only_after_a_true_load():
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(21, CAP)
    # Production queries the target before its demand; this also installs the
    # configured rearm budget while the pool is still below the fill floor.
    assert N.real_should_predict(21, CAP, 2) is True
    initial = mx.array([[1, 2, 3, 4]], dtype=mx.uint32)
    N.demand_dual(
        initial, pool, SEG, 21, 0, path, STRIDE, CAP, True, 0,
        use_side=False,
    )
    # Cold-fill loads arm two predictor occurrences; once consumed, a full
    # stable pool does not pay another target-gate prediction.
    assert N.real_should_predict(21, CAP, 2) is True
    assert N.real_should_predict(21, CAP, 2) is True
    assert N.real_should_predict(21, CAP, 2) is False

    # Expert 5 is genuinely absent, so demand performs one load and rearms the
    # predictor for exactly the configured number of following occurrences.
    N.demand_dual(
        mx.array([[2, 3, 4, 5]], dtype=mx.uint32),
        pool, SEG, 21, 0, path, STRIDE, CAP, True, 0,
        use_side=False,
    )
    assert N.demand_last_stats()[2] == 1
    assert N.real_should_predict(21, CAP, 2) is True
    assert N.real_should_predict(21, CAP, 2) is True
    assert N.real_should_predict(21, CAP, 2) is False
    os.unlink(path)


def test_unified_prediction_hits_fill_free_history_then_stop_promoting():
    """A small admission quota must still warm a larger physical main pool."""
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    layer = 22
    N.real_init(layer, CAP)

    first = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([1, 2], dtype=mx.uint32), layer, path,
        STRIDE, [], 2, -CAP, gen=0,
    )
    mx.eval(first); N.sideregion_drain()
    assert N.real_region_count(layer) == 2
    assert N.real_verified_contents(layer) == []

    # Both predictions are consumed while two rows remain free.  Their labels
    # become verified, reopening the two-row prediction quota without moving
    # or rereading either expert.
    local = N.demand_dual(
        mx.array([[1, 2]], dtype=mx.uint32), pool, SEG, layer, 0,
        path, STRIDE, CAP, True, 0, use_side=False,
    )
    mx.eval(local)
    assert set(N.real_verified_contents(layer)[::2]) == {1, 2}
    assert N.real_should_predict(layer, 3, 2) is True

    second = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([3, 4], dtype=mx.uint32), layer, path,
        STRIDE, [], 2, -CAP, gen=0,
    )
    mx.eval(second); N.sideregion_drain()
    assert N.real_region_count(layer) == CAP
    assert N.real_should_predict(layer, 3, 2) is False

    # At full capacity, a real hit retains its speculative label.  Otherwise
    # every hit would force a replacement pread on the next occurrence.
    local = N.demand_dual(
        mx.array([[3]], dtype=mx.uint32), pool, SEG, layer, 0,
        path, STRIDE, CAP, True, 0, use_side=False,
    )
    mx.eval(local)
    assert 3 not in set(N.real_verified_contents(layer)[::2])
    os.unlink(path)


def test_staged_demand_miss_rearms_adaptive_predictor():
    """Compatibility staging must rearm just like direct demand."""
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    layer = 23
    N.real_init(layer, CAP)
    # Production checks prediction eligibility before this target's demand;
    # that call also registers the configured cooldown budget.
    assert N.real_should_predict(layer, CAP, 2) is True

    local = N.demand_staged_multi(
        mx.array([[9]], dtype=mx.uint32), pool, SEG, layer, path,
        STRIDE, CAP, True, 0, SPEC, [], [], sequence_length=1,
    )
    mx.eval(local)
    assert N.real_should_predict(layer, CAP, 2) is True
    assert N.real_should_predict(layer, CAP, 2) is True
    assert N.real_should_predict(layer, CAP, 2) is False
    os.unlink(path)


def test_unified_physical_read_budget_caps_missing_rows_not_logical_ids(monkeypatch):
    """The rerank set stays wide while one callback pulls at most N misses."""
    monkeypatch.setenv("PREFETCH_PHYSICAL_READ_BUDGET", "1")
    N.real_reset(); N.sideregion_reset(); N.prefetch_audit_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    layer = 24
    N.real_init(layer, CAP)

    ready = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([1, 2, 3], dtype=mx.uint32), layer, path,
        STRIDE, [], 3, -CAP, gen=0, source_layer=23, forward_id=99,
    )
    mx.eval(ready); N.sideregion_drain()

    assert N.real_region_count(layer) == 1
    audit = list(N.prefetch_audit_stats())
    assert audit[10] == 3                 # logical candidate width unchanged
    assert audit[21:23] == [1, 1]        # requested/completed physical reads
    os.unlink(path)


def test_unified_physical_read_budget_profile_overrides_selected_layer(monkeypatch):
    monkeypatch.setenv("PREFETCH_PHYSICAL_READ_BUDGET", "1")
    monkeypatch.setenv("PREFETCH_PHYSICAL_READ_BUDGET_PROFILE", "24:2,30-31:3")
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool24 = _pool()
    pool25 = _pool()
    N.real_init(24, CAP)
    N.real_init(25, CAP)

    ready24 = N.prefetch_pool_sideregion(
        pool24, SEG, mx.array([1, 2, 3], dtype=mx.uint32), 24, path,
        STRIDE, [], 3, -CAP, gen=0, source_layer=23, forward_id=99,
    )
    ready25 = N.prefetch_pool_sideregion(
        pool25, SEG, mx.array([1, 2, 3], dtype=mx.uint32), 25, path,
        STRIDE, [], 3, -CAP, gen=0, source_layer=24, forward_id=99,
    )
    mx.eval(ready24, ready25); N.sideregion_drain()

    assert N.real_region_count(24) == 2
    assert N.real_region_count(25) == 1
    os.unlink(path)


def test_demand_overcap_requests_temporary_fallback_without_mutating_real():
    N.real_reset(); N.sideregion_reset(); N.demand_deadline_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(0, CAP)
    inds = mx.array([[1, 2, 3, 4, 5]], dtype=mx.uint32)
    local = N.demand_dual(
        inds, pool, SEG, 0, 0, path, STRIDE, CAP, True, 0,
    )
    mx.eval(local)
    # fallback=2 由 VirtualPool 解释为临时 stacked fetch；C++ 真实区必须原封不动。
    assert N.demand_last_stats()[:4] == [0, 5, 0, 2]
    assert N.real_region_count(0) == 0
    assert list(N.demand_deadline_stats()) == [0, 1, 5, 0, 0, 5]
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
    assert N.demand_last_stats()[:4] == [1, 0, 0, 0]   # 侧区命中，无 load
    os.unlink(path)


def test_global_staging_generations_promote_into_unified_main_pool():
    """同层 early/tail bank 分代发布，晋升后 demand 只寻址 cap 行主池。"""
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool(cap=CAP, spec=0)
    early_staging = mx.zeros((SPEC, STRIDE), dtype=mx.uint8)
    tail_staging = mx.zeros((SPEC, STRIDE), dtype=mx.uint8)
    mx.eval(early_staging, tail_staging)

    early = N.prefetch_into_staging(
        early_staging, mx.array([1, 2], dtype=mx.uint32),
        9, 101, path, STRIDE, [], SPEC, False,
    )
    tail = N.prefetch_into_staging(
        tail_staging, mx.array([3, 4], dtype=mx.uint32),
        9, 102, path, STRIDE, [], SPEC, False,
    )
    mx.eval(early, tail)
    import time
    deadline = time.time() + 2.0
    first = second = []
    while time.time() < deadline and (not first or not second):
        if not first:
            first = N.prefetch_staging_take(9, 101)
        if not second:
            second = N.prefetch_staging_take(9, 102)
        if not first or not second:
            time.sleep(0.005)
    assert first[0] == 101 and set(first[1::2]) == {1, 2}
    assert second[0] == 102 and set(second[1::2]) == {3, 4}

    N.real_init(9, CAP)
    assert N.late_promote_staged(
        pool, SEG, 9, CAP, SPEC, early_staging, first[1:],
    ) == 2
    # spec_limit=3：第二个 tail 会复用一个旧 speculative 槽；物理池始终 CAP 行。
    assert N.late_promote_staged(
        pool, SEG, 9, CAP, SPEC, tail_staging, second[1:],
    ) == 2
    resident = set(N.real_region_contents(9)[::2])
    assert {3, 4} <= resident and len(resident) == 3
    assert pool[0].shape[0] == CAP

    routes = mx.array([[3, 4]], dtype=mx.uint32)
    local = N.demand_dual(
        routes, pool, SEG, 9, 0, path, STRIDE, CAP, True, 0,
        use_side=False,
    )
    mx.eval(local)
    assert N.demand_last_stats()[:4] == [2, 0, 0, 0]
    os.unlink(path)


def test_global_staging_is_visible_to_prefetch_audit():
    N.real_reset(); N.sideregion_reset(); N.prefetch_audit_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool(cap=CAP, spec=0)
    staging = mx.zeros((SPEC, STRIDE), dtype=mx.uint8)
    mx.eval(staging)
    dummy = N.prefetch_into_staging(
        staging, mx.array([5, 6], dtype=mx.uint32),
        10, 201, path, STRIDE, [], SPEC, False,
        source_layer=7, forward_id=55,
    )
    mx.eval(dummy)
    N.prefetch_staging_drain()
    ready = N.prefetch_staging_take(10, 201)
    N.real_init(10, CAP)
    N.late_promote_staged(
        pool, SEG, 10, CAP, SPEC, staging, ready[1:],
    )
    local = N.demand_dual(
        mx.array([[5]], dtype=mx.uint32),
        pool, SEG, 10, 0, path, STRIDE, CAP, True, 0,
        forward_id=55, sequence_length=1, use_side=False,
    )
    mx.eval(local)

    row = list(N.prefetch_audit_stats())
    assert len(row) == 26
    assert row[0:4] == [55, 7, 10, 201]
    assert row[10] == 2                         # candidate width
    assert row[12:18] == [1, 1, 0, 1, 1, 1]   # actual/candidate/system coverage
    assert row[18:21] == [1, 0, 0]             # unified main hit, no side/fallback
    assert row[21:26] == [2, 2, 1, 1, 1]       # reads/submits/demand/deadline order
    os.unlink(path)


def test_progressive_staging_deduplicates_steps_and_waits_route_bytes():
    """early/tail share a logical target: repeated ids read once; demand joins route."""
    N.real_reset(); N.sideregion_reset()
    N.demand_prejoin_stats_reset()
    N.prefetch_staging_wait_stats_reset()
    N.real_init(12, CAP)
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    early_staging = mx.zeros((SPEC, STRIDE), dtype=mx.uint8)
    tail_staging = mx.zeros((SPEC, STRIDE), dtype=mx.uint8)
    mx.eval(early_staging, tail_staging)

    early = N.prefetch_into_staging(
        early_staging, mx.array([1, 2], dtype=mx.uint32),
        12, 401, path, STRIDE, [], SPEC, False,
        source_layer=9, forward_id=77,
    )
    tail = N.prefetch_into_staging(
        tail_staging, mx.array([2, 3], dtype=mx.uint32),
        12, 402, path, STRIDE, [], SPEC, False,
        source_layer=11, forward_id=77, priority=1,
    )
    mx.eval(early, tail)
    N.prefetch_staging_wait_experts(
        77, 12, mx.array([[2, 3]], dtype=mx.uint32),
    )
    prejoin = list(N.demand_prejoin_stats())
    assert prejoin[:3] == [12, 1, 2]
    assert sum(prejoin[3:6]) == 2
    wait_stats = list(N.prefetch_staging_wait_stats())
    assert wait_stats[:3] == [12, 1, 2]
    assert 0 <= wait_stats[3] <= 2
    assert 0 <= wait_stats[4] <= 2
    assert wait_stats[5] >= 0
    first = N.prefetch_staging_take(12, 401)
    second = N.prefetch_staging_take(12, 402)
    experts = list(first[1::2]) + list(second[1::2])
    # The route wait does not wait for false-positive expert 1.
    assert sorted(experts) == [2, 3]
    assert experts.count(2) == 1
    N.prefetch_staging_finish_demand(77, 12)
    N.prefetch_staging_drain()
    late_first = N.prefetch_staging_take(12, 401)
    late_second = N.prefetch_staging_take(12, 402)
    all_experts = (
        experts + list(late_first[1::2]) + list(late_second[1::2])
    )
    assert sorted(all_experts) == [1, 2, 3]
    N.prefetch_staging_forget(12, 401)
    N.prefetch_staging_forget(12, 402)
    os.unlink(path)


def test_demand_aware_promotion_is_not_blocked_by_full_verified_pool():
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool(cap=CAP, spec=0)
    N.real_init(11, CAP)
    N.real_debug_place(11, [1, 2, 3, 4], CAP, True, 0)
    staging = mx.zeros((SPEC, STRIDE), dtype=mx.uint8)
    dummy = N.prefetch_into_staging(
        staging, mx.array([5, 6], dtype=mx.uint32),
        11, 301, path, STRIDE, [], SPEC, False,
    )
    mx.eval(dummy)
    N.prefetch_staging_drain()
    ready = N.prefetch_staging_take(11, 301)

    promoted = N.demand_promote_staged(
        pool, SEG, 11, CAP, SPEC, staging, ready[1:],
        mx.array([[5]], dtype=mx.uint32),
    )
    assert promoted >= 1
    assert 5 in set(N.real_region_contents(11)[::2])
    local = N.demand_dual(
        mx.array([[5]], dtype=mx.uint32),
        pool, SEG, 11, 0, path, STRIDE, CAP, True, 0,
        use_side=False,
    )
    mx.eval(local)
    assert N.demand_last_stats()[:4] == [1, 0, 0, 0]
    os.unlink(path)


def test_deadline_stats_count_unique_complete_bytes_by_source():
    if not hasattr(N, "demand_deadline_stats_reset"):
        pytest.skip("native extension 尚未编译 deadline stats")
    N.real_reset(); N.sideregion_reset(); N.demand_deadline_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(0, CAP)

    # 第一次：5/6 都只能 demand fallback；重复的 5 不能重复计 union。
    N.demand_dual(
        mx.array([[5, 6, 5]], dtype=mx.uint32),
        pool, SEG, 0, 0, path, STRIDE, CAP, True, 0,
    )
    assert list(N.demand_deadline_stats()) == [0, 1, 2, 0, 0, 2]

    # 完整 publish 一个侧区专家。第二次 demand 时 5 在 real，7 在 side；
    # 二者均已在 deadline 前拥有完整字节，不发生 fallback。
    dummy = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([7], dtype=mx.uint32), 0, path, STRIDE,
        [], SPEC, CAP, gen=0,
    )
    mx.eval(dummy)
    N.sideregion_drain()
    N.demand_dual(
        mx.array([[5, 7]], dtype=mx.uint32),
        pool, SEG, 0, 0, path, STRIDE, CAP, True, 0,
    )
    assert list(N.demand_deadline_stats()) == [0, 2, 4, 1, 1, 2]
    os.unlink(path)


def test_deadline_snapshot_is_not_overwritten_by_post_wait_state():
    """The strict metric must describe target entry, not rows arriving later."""
    N.real_reset(); N.sideregion_reset(); N.demand_deadline_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(14, CAP)
    ids = mx.array([[9]], dtype=mx.uint32)

    # Entry snapshot sees a miss.
    N.demand_deadline_snapshot(ids, 14, 0, True)
    # The row becomes directly addressable only after that snapshot.
    dummy = N.prefetch_pool_sideregion(
        pool, SEG, ids, 14, path, STRIDE, [], SPEC, CAP, gen=0,
    )
    mx.eval(dummy)
    N.sideregion_drain()
    N.demand_dual(
        ids, pool, SEG, 14, 0, path, STRIDE, CAP, True, 0,
        record_deadline=False,
    )
    assert list(N.demand_deadline_stats()) == [14, 1, 1, 0, 0, 1]
    os.unlink(path)


def test_async_demand_maps_and_loads_only_after_graph_evaluation():
    if not hasattr(N, "demand_dual_async"):
        pytest.skip("native extension 尚未编译 async demand")
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(15, CAP)
    ids = mx.array([[5, 6]], dtype=mx.uint32)

    local = N.demand_dual_async(
        ids, pool, SEG, 15, 0, path, STRIDE, CAP, True, 0,
    )
    # Construction is lazy: neither route ids nor misses reached the CPU.
    assert list(N.demand_async_stats()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert not any(N.demand_async_miss_histogram())
    mx.eval(local)
    N.demand_async_check()
    assert list(N.demand_async_stats())[:8] == [1, 0, 1, 2, 0, 2, 1, 0]
    assert list(N.demand_async_miss_histogram())[2] == 1
    mapping = dict(zip(
        N.real_region_contents(15)[::2],
        N.real_region_contents(15)[1::2],
    ))
    assert local.tolist() == [[mapping[5], mapping[6]]]

    # The second occurrence is an all-hit event-gated remap with no I/O.
    again = N.demand_dual_async(
        ids, pool, SEG, 15, 0, path, STRIDE, CAP, True, 0,
    )
    mx.eval(again)
    N.demand_async_check()
    assert list(N.demand_async_stats())[:8] == [2, 1, 1, 2, 2, 4, 1, 0]
    hist = list(N.demand_async_miss_histogram())
    assert hist[0] == 1 and hist[2] == 1 and sum(hist) == 2
    assert again.tolist() == local.tolist()
    os.unlink(path)


def test_async_demand_reads_persistent_side_slot_table_without_fallback():
    """Published direct rows must be visible to the GPU remap table itself."""
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(16, CAP)
    ids = mx.array([[5, 6]], dtype=mx.uint32)
    dummy = N.prefetch_pool_sideregion(
        pool, SEG, ids.reshape(-1), 16, path, STRIDE,
        [], SPEC, CAP, gen=0,
    )
    mx.eval(dummy)
    N.sideregion_drain()

    local = N.demand_dual_async(
        ids, pool, SEG, 16, 0, path, STRIDE, CAP, True, 0,
    )
    mx.eval(local)
    N.demand_async_check()
    assert all(slot >= CAP for slot in local.reshape(-1).tolist())
    assert list(N.demand_async_stats())[:8] == [1, 1, 0, 0, 2, 2, 0, 0]
    os.unlink(path)


def test_async_demand_waits_predicted_pending_rows_without_duplicate_fallback():
    """Only experts absent from both ready and pending prediction may reread."""
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(18, CAP)
    ids = mx.array([[5, 6]], dtype=mx.uint32)

    predicted = N.prefetch_pool_sideregion(
        pool, SEG, ids.reshape(-1), 18, path, STRIDE,
        [], SPEC, CAP, gen=0,
    )
    # Keep submit before target demand in one lazy graph.  Entry remap still
    # sees -1 because publication happens in the completion callback, then the
    # demand callback waits just these two reservations and reuses side rows.
    dependent_ids = ids + predicted.astype(mx.uint32).reshape(()) * 0
    local = N.demand_dual_async(
        dependent_ids, pool, SEG, 18, 0, path, STRIDE, CAP, True, 0,
        wait_for_pending=True,
    )
    mx.eval(local)
    N.demand_async_check()

    assert all(slot >= CAP for slot in local.reshape(-1).tolist())
    assert list(N.demand_async_stats())[:8] == [1, 0, 1, 0, 0, 2, 0, 2]
    assert N.real_region_count(18) == 0
    os.unlink(path)


def test_async_demand_leases_side_rows_until_consumer_completion(monkeypatch):
    """A direct row cannot be evicted while a GPU MoE may still read it."""
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    monkeypatch.setenv("SIDEREGION_ROW_LEASES", "1")
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    layer = 19
    N.real_init(layer, CAP)
    ids = mx.array([[5, 6]], dtype=mx.uint32)
    first = N.prefetch_pool_sideregion(
        pool, SEG, ids.reshape(-1), layer, path, STRIDE,
        [], SPEC, CAP, gen=0,
    )
    mx.eval(first); N.sideregion_drain()

    local = N.demand_dual_async(
        ids, pool, SEG, layer, 0, path, STRIDE, CAP, True, 0,
    )
    mx.eval(local); N.demand_async_check()
    assert all(row >= CAP for row in local.reshape(-1).tolist())

    replacement_ids = mx.array([7, 8, 9], dtype=mx.uint32)
    blocked = N.prefetch_pool_sideregion(
        pool, SEG, replacement_ids, layer, path, STRIDE,
        [], SPEC, CAP, gen=0,
    )
    mx.eval(blocked); N.sideregion_drain()
    leased_contents = set(N.sideregion_contents(layer, 0)[::2])
    assert {5, 6}.issubset(leased_contents)

    # Entering the next decoder layer proves the previous layer's MoE has
    # finished and releases its rows without adding a GPU release primitive.
    N.real_init(layer + 1, CAP)
    next_local = N.demand_dual_async(
        mx.array([[1]], dtype=mx.uint32), pool, SEG, layer + 1, 0,
        path, STRIDE, CAP, True, 0,
    )
    mx.eval(next_local); N.demand_async_check()
    admitted = N.prefetch_pool_sideregion(
        pool, SEG, replacement_ids, layer, path, STRIDE,
        [], SPEC, CAP, gen=0,
    )
    mx.eval(admitted); N.sideregion_drain()
    assert set(N.sideregion_contents(layer, 0)[::2]) == {7, 8, 9}
    os.unlink(path)


def test_split_async_exposes_entry_misses_and_event_gated_final_slots():
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(17, CAP)
    ids = mx.array([[5, 6]], dtype=mx.uint32)

    entry, final = N.demand_dual_split_async(
        ids, pool, SEG, 17, 0, path, STRIDE, CAP, True, 0,
    )
    mx.eval(entry, final)
    N.demand_async_check()
    assert entry.tolist() == [[-1, -1]]
    assert all(0 <= slot < CAP for slot in final.reshape(-1).tolist())

    hit_entry, hit_final = N.demand_dual_split_async(
        ids, pool, SEG, 17, 0, path, STRIDE, CAP, True, 0,
    )
    mx.eval(hit_entry, hit_final)
    assert hit_entry.tolist() == hit_final.tolist()
    os.unlink(path)


def test_prefetch_audit_pairs_rerank_width_recall_and_complete_bytes():
    if not hasattr(N, "prefetch_audit_stats_reset"):
        pytest.skip("native extension 尚未编译 strict prefetch audit")
    N.real_reset(); N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(1, CAP)
    # 在审计轮前用真实 demand 把专家7完整写进 real pool；它代表 source-time resident。
    N.demand_dual(
        mx.array([[7]], dtype=mx.uint32), pool, SEG, 1, 0, path,
        STRIDE, CAP, True, 0,
    )
    N.prefetch_audit_stats_reset()

    dummy = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([5, 6, 6], dtype=mx.uint32), 1, path, STRIDE,
        [7], SPEC, CAP, gen=0, source_layer=0, forward_id=42,
    )
    mx.eval(dummy)
    N.sideregion_drain()
    N.demand_dual(
        mx.array([[5, 7, 8]], dtype=mx.uint32),
        pool, SEG, 1, 0, path, STRIDE, CAP, True, 0,
        forward_id=42, sequence_length=1,
    )
    row = list(N.prefetch_audit_stats())
    assert len(row) == 26
    assert row[:5] == [42, 0, 1, 0, 1]
    assert all(value >= 0 for value in row[5:10])
    assert row[10:] == [
        2,  # candidate_width: compact duplicate 6 only counts once
        1,  # source_resident_count
        3,  # actual_unique
        1,  # candidate_hits (5)
        1,  # source_resident_hits (7)
        2,  # system_prediction_hits
        1,  # candidate_complete_hits (5 arrived in side)
        2,  # system_complete_hits (5 side + 7 real)
        1,  # deadline_real
        1,  # deadline_side
        1,  # fallback (8)
        2,  # pread_requested
        2,  # pread_completed
        1,  # submission_count
        1,  # demand_count
        1,  # callback_before_demand
    ]
    os.unlink(path)


def test_prefetch_audit_late_callback_stays_with_same_forward():
    if not hasattr(N, "prefetch_audit_stats_reset"):
        pytest.skip("native extension 尚未编译 strict prefetch audit")
    N.real_reset(); N.sideregion_reset(); N.prefetch_audit_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    N.real_init(1, CAP)
    dummy = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([9], dtype=mx.uint32), 1, path, STRIDE,
        [], SPEC, CAP, gen=0, source_layer=0, forward_id=43,
    )

    # 先到目标 demand，之后才触发预测 callback。配对必须仍是 forward43，
    # 不能把这条迟到提交算到下一 forward。
    N.demand_dual(
        mx.array([[9]], dtype=mx.uint32),
        pool, SEG, 1, 0, path, STRIDE, CAP, True, 0,
        forward_id=43, sequence_length=1,
    )
    mx.eval(dummy)
    N.sideregion_drain()
    row = list(N.prefetch_audit_stats())
    assert len(row) == 26
    assert row[0] == 43 and row[2] == 1
    assert row[10] == row[12] == row[13] == 1  # width/actual/candidate hit
    assert row[17] == 0                         # deadline 前未完整到达
    assert row[20] == 1                         # demand fallback
    assert row[23:26] == [1, 1, 0]              # one submit/demand, callback late
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
        # real_debug_place 对拍的是可容纳的 resident 状态机；真实 over-cap
        # 已由 demand_dual 的 fallback=2 专门测试并交给临时 stacked 路径。
        flat = [rng.randint(0, 9) for _ in range(rng.randint(1, cap))]
        want = ref.place(flat)
        got = N.real_debug_place(0, flat, cap, True, decay)
        assert got == want, f"step {step} flat={flat}: cpp {got} != ref {want}"
        # resident 集合一致
        cpp_res = {N.real_region_contents(0)[i] for i in range(0, len(N.real_region_contents(0)), 2)}
        assert cpp_res == set(ref.e2r.keys()), f"step {step}: resident 不一致"
