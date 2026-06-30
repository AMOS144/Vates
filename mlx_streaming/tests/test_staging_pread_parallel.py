"""验证 staging.submit 正确把 STAGING_PREAD_PARALLEL 开关透传给 native prefetch_into_staging。"""
import sys
import types

import mlx.core as mx

import mlx_streaming
from mlx_streaming.core.prefetch.native_staging import NativeStagingManager


class _FakeSrc:
    dir = "/tmp"
    stride = 8
    _segs = []


def _patch_native(monkeypatch, captured):
    fake = types.SimpleNamespace()

    def _prefetch_into_staging(buf, ids, layer, gen, path, stride, res, cap, parallel):
        captured["parallel"] = parallel
        return mx.zeros((1,), dtype=mx.uint8)

    fake.prefetch_into_staging = _prefetch_into_staging
    monkeypatch.setitem(sys.modules, "mlx_streaming.native_moe_ext", fake)
    monkeypatch.setattr(mlx_streaming, "native_moe_ext", fake, raising=False)


def test_submit_passes_parallel_true(monkeypatch):
    monkeypatch.setenv("STAGING_PREAD_PARALLEL", "1")
    captured = {}
    _patch_native(monkeypatch, captured)
    mgr = NativeStagingManager(_FakeSrc(), budget=4)
    mgr.submit(0, mx.zeros((4,), dtype=mx.uint32))
    assert captured["parallel"] is True


def test_submit_passes_parallel_false(monkeypatch):
    monkeypatch.setenv("STAGING_PREAD_PARALLEL", "0")
    captured = {}
    _patch_native(monkeypatch, captured)
    mgr = NativeStagingManager(_FakeSrc(), budget=4)
    mgr.submit(0, mx.zeros((4,), dtype=mx.uint32))
    assert captured["parallel"] is False
