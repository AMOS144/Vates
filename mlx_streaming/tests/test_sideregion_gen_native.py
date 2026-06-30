"""双缓冲：同层 gen0(base=CAP) 与 gen1(base=CAP+SPEC) 互相独立，行落各自区间。"""
import os, struct, tempfile, time
import mlx.core as mx
import mlx_streaming.native_moe_ext as N

CAP, SPEC, NE = 4, 3, 8
W, S = 16, 8
SEG = [W * 4, S * 1]
STRIDE = sum(SEG)


def _blob(path):
    with open(path, "wb") as f:
        for e in range(NE):
            f.write(struct.pack(f"<{W}I", *([e + 1] * W)))
            f.write(bytes([(e + 100) & 0xFF] * S))


def _pool():
    w = mx.zeros((CAP + 2 * SPEC, W), dtype=mx.uint32)   # 两代物理行
    sc = mx.zeros((CAP + 2 * SPEC, S), dtype=mx.uint8)
    mx.eval(w, sc)
    return [w, sc]


def _wait(layer, gen, want, timeout=2.0):
    t = time.time() + timeout
    while time.time() < t:
        c = N.sideregion_contents(layer, gen)
        if len({c[i] for i in range(0, len(c), 2)}) == want:
            break
        time.sleep(0.01)
    flat = N.sideregion_contents(layer, gen)
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


def test_two_gens_independent_disjoint_rows():
    N.sideregion_reset()
    path = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
    _blob(path)
    pool = _pool()
    # gen0：base=CAP，行落 [CAP, CAP+SPEC)
    d0 = N.prefetch_pool_sideregion(pool, SEG, mx.array([5, 6], dtype=mx.uint32),
                                    0, path, STRIDE, [], SPEC, CAP, gen=0)
    mx.eval(d0)
    m0 = _wait(0, 0, 2)
    # gen1：base=CAP+SPEC，行落 [CAP+SPEC, CAP+2*SPEC)
    d1 = N.prefetch_pool_sideregion(pool, SEG, mx.array([7], dtype=mx.uint32),
                                    0, path, STRIDE, [], SPEC, CAP + SPEC, gen=1)
    mx.eval(d1)
    m1 = _wait(0, 1, 1)

    assert set(m0.keys()) == {5, 6}
    assert set(m1.keys()) == {7}
    for r in m0.values():
        assert CAP <= r < CAP + SPEC               # gen0 行区间
    for r in m1.values():
        assert CAP + SPEC <= r < CAP + 2 * SPEC    # gen1 行区间，与 gen0 不相交
    # gen0 内容不被 gen1 提交干扰
    assert set(_wait(0, 0, 2).keys()) == {5, 6}
    os.unlink(path)
