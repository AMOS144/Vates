"""验证 C++ 把 blob 段按记录直写进 C++-owned 池数组的侧区物理行（Route 3 底座）。
2 段布局：seg0 = weight uint32 [W]，seg1 = scales uint8 [S]。临时小 blob 全确定字节。
（Route 3 后侧区字节由后台线程 pread 直写进 C++-owned 池 buffer，不再经稳定缓冲 + MLX scatter。）"""
import os, struct, tempfile, time
import mlx.core as mx
import mlx_streaming.native_moe_ext as N

CAP, SPEC, NE = 4, 3, 8
W, S = 16, 8                       # seg0: 16×uint32=64B；seg1: 8×uint8=8B
SEG = [W * 4, S * 1]
STRIDE = sum(SEG)


def _blob(path):
    with open(path, "wb") as f:
        for e in range(NE):                      # 每个专家：weight 全 = e+1，scales 全 = e+100
            f.write(struct.pack(f"<{W}I", *([e + 1] * W)))
            f.write(bytes([(e + 100) & 0xFF] * S))


def _pool():
    # C++-owned buffer：地址恒定，供后台直写（与生产 spec/dual 池一致）。
    w = N.pool_owned_zeros([CAP + SPEC, W], "uint32")
    sc = N.pool_owned_zeros([CAP + SPEC, S], "uint8")
    mx.eval(w, sc)
    return [w, sc]


def _wait(layer, want, timeout=2.0):
    t = time.time() + timeout
    while time.time() < t:
        c = N.sideregion_contents(layer)
        if len({c[i] for i in range(0, len(c), 2)}) == want:
            break
        time.sleep(0.01)
    flat = N.sideregion_contents(layer)
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


def test_sideregion_segment_scatter():
    N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    pred = mx.array([5, 6, 7], dtype=mx.uint32)
    d = N.prefetch_pool_sideregion(pool, SEG, pred, 0, path, STRIDE, [], SPEC, CAP)
    mx.eval(d)
    m = _wait(0, 3)                                  # {expert: phys_row}
    assert set(m.keys()) == {5, 6, 7}
    # 新机制：字节由 C++ 后台直写进池数组的侧区物理行；直接读池验证真值（e2r 已发布 ⇒ 字节已就绪）。
    w, sc = pool
    mx.eval(w, sc)
    for e, row in m.items():
        assert CAP <= row < CAP + SPEC              # 落在侧区
        assert int(w[row][0]) == e + 1              # weight 段直写正确
        assert int(sc[row][0]) == (e + 100) & 0xFF  # scales 段直写正确
    os.unlink(path)


def test_direct_prefetch_publishes_only_unified_real_table():
    """Negative base writes final main rows and never creates side ownership."""
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    total_cap = CAP + SPEC
    pool = [
        N.pool_owned_zeros([total_cap, W], "uint32"),
        N.pool_owned_zeros([total_cap, S], "uint8"),
    ]
    layer = 23
    N.real_init(layer, total_cap)
    ids = mx.array([5, 6], dtype=mx.uint32)
    ready = N.prefetch_pool_sideregion(
        pool, SEG, ids, layer, path, STRIDE,
        [], SPEC, -total_cap, gen=0,
    )
    mx.eval(ready); N.sideregion_drain()

    assert N.sideregion_contents(layer, 0) == []
    flat = N.real_region_contents(layer)
    real = dict(zip(flat[::2], flat[1::2]))
    assert set(real) == {5, 6}
    assert all(0 <= row < total_cap for row in real.values())
    assert int(pool[0][real[5]][0]) == 6
    assert int(pool[1][real[6]][0]) == 106

    local = N.demand_dual_async(
        ids.reshape(1, -1), pool, SEG, layer, 0, path, STRIDE,
        total_cap, True, 0, use_side=False,
    )
    mx.eval(local); N.demand_async_check()
    assert local.tolist() == [[real[5], real[6]]]
    stats = list(N.demand_async_stats())
    assert stats[3] == 0  # no SSD fallback loads
    assert stats[1] == 1  # one all-hit layer
    os.unlink(path)


def test_direct_async_demand_waits_only_for_pending_miss_rows():
    """Unified direct rows may publish after entry remap without side ownership."""
    N.real_reset(); N.sideregion_reset(); N.demand_async_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    total_cap = CAP + SPEC
    pool = [
        N.pool_owned_zeros([total_cap, W], "uint32"),
        N.pool_owned_zeros([total_cap, S], "uint8"),
    ]
    layer = 24
    N.real_init(layer, total_cap)
    ids = mx.array([5, 6], dtype=mx.uint32)
    predicted = N.prefetch_pool_sideregion(
        pool, SEG, ids, layer, path, STRIDE,
        [], SPEC, -total_cap, gen=0,
    )
    dependent = ids.reshape(1, -1) + predicted.astype(mx.uint32).reshape(()) * 0
    local = N.demand_dual_async(
        dependent, pool, SEG, layer, 0, path, STRIDE,
        total_cap, True, 0, use_side=False, wait_for_pending=True,
    )
    mx.eval(local); N.demand_async_check()
    flat = N.real_region_contents(layer)
    real = dict(zip(flat[::2], flat[1::2]))
    assert local.tolist() == [[real[5], real[6]]]
    assert list(N.demand_async_stats())[:8] == [1, 0, 1, 0, 0, 2, 0, 2]
    os.unlink(path)


def test_sideregion_guard_seg_stride_mismatch():
    import pytest
    pool = _pool()
    pred = mx.array([1], dtype=mx.uint32)
    with pytest.raises(Exception):                   # 段数 != 池数组个数 → 抛异常
        N.prefetch_pool_sideregion(pool, [W * 4], pred, 9, "/nonexistent.blob", STRIDE, [], SPEC, CAP)


def test_sideregion_resident_filter_and_budget():
    N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    pred = mx.array([1, 2, 3, 4, 5], dtype=mx.uint32)   # 5 个 > spec 3；排除常驻 2
    d = N.prefetch_pool_sideregion(pool, SEG, pred, 1, path, STRIDE, [2], SPEC, CAP)
    mx.eval(d)
    m = _wait(1, 3)
    assert 2 not in m and len(m) == SPEC
    os.unlink(path)


def test_sideregion_prefetch_stats_count_real_reads_and_hits():
    assert hasattr(N, "sideregion_prefetch_stats_reset"), "native 预取统计重置接口尚未实现"
    assert hasattr(N, "sideregion_prefetch_stats"), "native 预取统计读取接口尚未实现"
    N.sideregion_reset()
    N.sideregion_prefetch_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    pred = mx.array([1, 2, 2, 3], dtype=mx.uint32)

    d = N.prefetch_pool_sideregion(pool, SEG, pred, 2, path, STRIDE, [1], SPEC, CAP)
    mx.eval(d)
    _wait(2, 2)
    N.sideregion_drain()
    first = N.sideregion_prefetch_stats()
    assert first == [4, 2, 0, 2, 0, 2, 0, 2]

    d = N.prefetch_pool_sideregion(pool, SEG, pred, 2, path, STRIDE, [1], SPEC, CAP)
    mx.eval(d)
    N.sideregion_drain()
    second = N.sideregion_prefetch_stats()
    assert second == [8, 4, 2, 2, 0, 2, 0, 2]
    os.unlink(path)


def test_sideregion_reclaims_complete_copy_when_expert_becomes_real_resident():
    """Real+side tables must not waste two physical rows on one expert."""
    N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()

    first = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([1, 2, 3], dtype=mx.uint32),
        3, path, STRIDE, [], SPEC, CAP,
    )
    mx.eval(first)
    assert set(_wait(3, 3)) == {1, 2, 3}

    # Expert 3 is now authoritative in the real pool.  Its stale side row is
    # reclaimed for expert 4 while the other useful side rows survive.
    second = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([4], dtype=mx.uint32),
        3, path, STRIDE, [3], SPEC, CAP,
    )
    mx.eval(second)
    N.sideregion_drain()
    contents = _wait(3, 3)
    assert set(contents) == {1, 2, 4}
    os.unlink(path)


def test_sideregion_progressive_submissions_do_not_duplicate_inflight_rows():
    """early core 与 late fill 同批完成时，同一 expert 只能预留一次。"""
    N.sideregion_reset()
    N.sideregion_prefetch_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()

    early = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([1, 2], dtype=mx.uint32),
        4, path, STRIDE, [], SPEC, CAP, forward_id=123,
    )
    refined = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([1, 2, 3], dtype=mx.uint32),
        4, path, STRIDE, [], SPEC, CAP, forward_id=123, priority=1,
    )
    # 两个 primitive 进入同一 command buffer，第二个 callback 很可能在第一个
    # 后台 pread 尚未 publish 时 reserve；这是线上 progressive rerank 的竞态。
    mx.eval(early, refined)
    N.sideregion_wait_experts(
        123, 4, 0, mx.array([3], dtype=mx.uint32),
    )
    ready = N.sideregion_contents(4, 0)
    assert 3 in {ready[i] for i in range(0, len(ready), 2)}
    N.sideregion_drain()

    flat = N.sideregion_contents(4, 0)
    contents = {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}
    assert set(contents) == {1, 2, 3}
    assert len(set(contents.values())) == 3
    stats = N.sideregion_prefetch_stats()
    assert stats[3] == 3  # reserved reads；不能把 1/2 重复预留
    assert stats[5] == 3  # successful full-blob reads
    os.unlink(path)


def test_sideregion_progressive_cross_stream_preserves_early_then_full_union():
    """Independent-stream tail depends on early callback and never duplicates."""
    N.sideregion_reset()
    N.sideregion_prefetch_stats_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    early_stream = mx.new_stream(mx.default_device())
    late_stream = mx.new_stream(mx.default_device())

    early = N.prefetch_pool_sideregion(
        pool, SEG, mx.array([1, 2], dtype=mx.uint32),
        5, path, STRIDE, [], SPEC, CAP, forward_id=124,
        stream=early_stream,
    )
    # Production submits the complete core-preserving union and attaches a
    # zero-valued graph dependency so its callback cannot overtake early.
    dependency = mx.sum(early.astype(mx.uint32)) * 0
    final_ids = mx.array([1, 2, 3], dtype=mx.uint32) + dependency
    refined = N.prefetch_pool_sideregion(
        pool, SEG, final_ids,
        5, path, STRIDE, [], SPEC, CAP, forward_id=124, priority=1,
        stream=late_stream,
    )
    mx.async_eval(refined)
    N.sideregion_wait_experts(
        124, 5, 0, mx.array([1, 2, 3], dtype=mx.uint32),
    )
    N.sideregion_drain()

    flat = N.sideregion_contents(5, 0)
    contents = {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}
    assert set(contents) == {1, 2, 3}
    assert len(set(contents.values())) == 3
    stats = N.sideregion_prefetch_stats()
    assert stats[3] == 3
    assert stats[5] == 3
    os.unlink(path)
