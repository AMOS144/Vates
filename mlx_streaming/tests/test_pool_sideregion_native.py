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
