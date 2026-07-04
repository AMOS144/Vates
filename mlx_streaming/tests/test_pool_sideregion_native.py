"""验证 C++ 把 blob 段按记录写进侧区稳定缓冲，sideregion_publish 取回得到正确字节。
2 段布局：seg0 = weight uint32 [W]，seg1 = scales uint8 [S]。临时小 blob 全确定字节。
（Route 1 后侧区字节不再旁路写池数组，改由稳定缓冲 + 消费侧 MLX scatter 发布。）"""
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
    w = mx.zeros((CAP + SPEC, W), dtype=mx.uint32)
    sc = mx.zeros((CAP + SPEC, S), dtype=mx.uint8)
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
    row2exp = {int(row): e for e, row in m.items()}
    # 新机制：字节写进 C++ 稳定缓冲、不再写池；用 sideregion_publish 取回脏行字节验证真值。
    rows, segs = N.sideregion_publish(0, 0, SEG)
    assert int(rows.shape[0]) == 3
    w = segs[0].view(mx.uint32)                       # seg0 uint8 (m,64) → (m,16) uint32
    for i, row in enumerate(rows.tolist()):
        e = row2exp[int(row)]
        assert CAP <= row < CAP + SPEC              # 落在侧区
        assert int(w[i][0]) == e + 1                # weight 段正确
        assert int(segs[1][i][0]) == (e + 100) & 0xFF  # scales 段正确
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
