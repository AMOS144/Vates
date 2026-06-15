import os

import pytest

from mlx_streaming.core.cache.blob_loader import BlobExpertSource

BLOB_DIR = "/tmp/cb_2bit_blob"


@pytest.mark.skipif(not os.path.exists(os.path.join(BLOB_DIR, "layer15.blob")),
                    reason="需要 /tmp/cb_2bit_blob")
def test_prefetch_then_acquire_no_repread(monkeypatch):
    monkeypatch.setenv("STREAM_BLOB_WINDOW", "3")
    src = BlobExpertSource(BLOB_DIR, 2048, 512, 128, 2, num_experts=512, workers=4)
    try:
        ids = [3, 7, 100, 200]
        src.prefetch_async(15, ids)
        src.wait_prefetch()
        src.preads = 0
        src.prefetch_hits = 0
        src.acquire(15, ids)                      # 应全部命中预取缓存
        assert src.preads == 0
        assert src.prefetch_hits == len(ids)
    finally:
        src.close()


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(BLOB_DIR, "layer15.blob"))
         and os.path.exists(os.path.join(BLOB_DIR, "layer25.blob"))),
    reason="需要 layer15/25 blob")
def test_rolling_window_evicts_old_layer(monkeypatch):
    monkeypatch.setenv("STREAM_BLOB_WINDOW", "1")
    src = BlobExpertSource(BLOB_DIR, 2048, 512, 128, 2, num_experts=512, workers=4)
    try:
        src.prefetch_async(15, [1, 2])
        src.wait_prefetch()
        src.prefetch_async(25, [1, 2])            # window=1 → 应淘汰 layer15
        src.wait_prefetch()
        keys = list(src._pf_cache.keys())
        assert all(k[0] != 15 for k in keys)
        assert any(k[0] == 25 for k in keys)
    finally:
        src.close()
