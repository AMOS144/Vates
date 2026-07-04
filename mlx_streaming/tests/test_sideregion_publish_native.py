import os
import struct
import tempfile
import numpy as np
import mlx.core as mx
import pytest

nmoe = pytest.importorskip("mlx_streaming.native_moe_ext")


def _make_blob(path, n_experts, stride):
    with open(path, "wb") as f:
        for e in range(n_experts):
            f.write(bytes([(e * 7 + i) & 0xFF for i in range(stride)]))


def test_sideregion_publish_returns_written_bytes():
    nmoe.sideregion_reset()
    layer, gen, cap, spec = 0, 0, 4, 3
    stride = 16
    seg_nbytes = [8, 8]
    n_experts = 10
    with tempfile.TemporaryDirectory() as d:
        blob = os.path.join(d, "layer0.blob")
        _make_blob(blob, n_experts, stride)
        base = cap
        pool = [mx.zeros((cap + spec, s), dtype=mx.uint8) for s in seg_nbytes]
        mx.eval(pool)
        ids = mx.array([5, 7], dtype=mx.uint32)
        tok = nmoe.prefetch_pool_sideregion(
            pool, seg_nbytes, ids, layer, blob, stride,
            [], spec, base, gen)
        mx.eval(tok)
        import time
        for _ in range(200):
            if len(nmoe.sideregion_contents(layer, gen)) >= 4:
                break
            time.sleep(0.01)
        rows, seg_arrays = nmoe.sideregion_publish(layer, gen, seg_nbytes)
        assert int(rows.shape[0]) == 2
        assert len(seg_arrays) == 2
        kv = dict(zip(*[a.tolist() for a in nmoe.sideregion_kv(layer, gen)]))
        row2exp = {int(r): int(e) for e, r in kv.items()}
        for i, r in enumerate(rows.tolist()):
            e = row2exp[int(r)]
            truth = bytes([(e * 7 + j) & 0xFF for j in range(stride)])
            off = 0
            for k, nb in enumerate(seg_nbytes):
                got = bytes(np.array(seg_arrays[k][i]).tobytes())
                assert got == truth[off:off + nb], (e, k, got, truth[off:off+nb])
                off += nb


def test_sideregion_publish_clears_dirty():
    nmoe.sideregion_reset()
    layer, gen, cap, spec, stride = 0, 0, 4, 3, 16
    seg_nbytes = [8, 8]
    with tempfile.TemporaryDirectory() as d:
        blob = os.path.join(d, "layer0.blob")
        _make_blob(blob, 10, stride)
        pool = [mx.zeros((cap + spec, s), dtype=mx.uint8) for s in seg_nbytes]
        mx.eval(pool)
        tok = nmoe.prefetch_pool_sideregion(pool, seg_nbytes, mx.array([5], dtype=mx.uint32),
                                            layer, blob, stride, [], spec, cap, gen)
        mx.eval(tok)
        import time
        for _ in range(200):
            if len(nmoe.sideregion_contents(layer, gen)) >= 2:
                break
            time.sleep(0.01)
        rows1, _ = nmoe.sideregion_publish(layer, gen, seg_nbytes)
        assert int(rows1.shape[0]) == 1
        rows2, _ = nmoe.sideregion_publish(layer, gen, seg_nbytes)
        assert int(rows2.shape[0]) == 0
