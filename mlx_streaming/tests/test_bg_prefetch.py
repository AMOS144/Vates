import os
import time

import mlx.core as mx
import pytest

from mlx_streaming.core.cache.blob_loader import BlobExpertSource
from mlx_streaming.core.prefetch.bg_prefetch import BackgroundExpertPrefetcher

BLOB = "/tmp/cb_2bit_blob"


@pytest.mark.skipif(not os.path.exists(os.path.join(BLOB, "layer15.blob")), reason="需要 /tmp/cb_2bit_blob")
def test_bg_prefetch_materializes_and_handoff():
    src = BlobExpertSource(BLOB, 2048, 512, 128, 2, num_experts=512)
    pf = BackgroundExpertPrefetcher(src)
    try:
        pf.submit(15, [3, 7, 100])
        deadline = time.time() + 3
        while pf.ready_count(15) < 3 and time.time() < deadline:
            time.sleep(0.01)
        assert pf.ready_count(15) == 3
        got = pf.take_ready(15, 7)
        assert got is not None
        ref = src.load_experts(15, [7])[7]
        for k in ref:
            assert bool(mx.all(got[k] == ref[k]).item())
        # take_ready 取走后不再就绪
        assert pf.take_ready(15, 7) is None
    finally:
        pf.close()
        src.close()


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(BLOB, "layer15.blob"))
         and os.path.exists(os.path.join(
             os.path.dirname(__file__), "..", "..", "models",
             "qwen3_next_experts_2bit_g128", "layer15_expert003.safetensors"))),
    reason="需要 blob + per-expert 2bit 目录")
def test_promote_prefetched_fills_pool():
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    expert_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")
    store = FileExpertStore(expert_dir, capacity=32)
    src = BlobExpertSource(BLOB, 2048, 512, 128, 2, num_experts=512)
    store._bg = BackgroundExpertPrefetcher(src)
    try:
        store._bg.submit(15, [3, 7, 100])
        deadline = time.time() + 3
        while store._bg.ready_count(15) < 3 and time.time() < deadline:
            time.sleep(0.01)
        n = store.promote_prefetched(15)
        assert n == 3
        resident = store._resident.resident_experts(15)
        assert {3, 7, 100} <= resident
        # 促进后 acquire_gpu 应全命中（miss=0）
        inds = mx.array([[3, 7, 100]], dtype=mx.uint32)
        store._resident.misses = 0
        store.acquire_gpu(15, inds, 512)
        assert store._resident.misses == 0
    finally:
        store._bg.close()
        src.close()


@pytest.mark.skipif(not os.path.exists(os.path.join(BLOB, "layer15.blob")), reason="需要 /tmp/cb_2bit_blob")
def test_bg_stats_ready_on_time():
    src = BlobExpertSource(BLOB, 2048, 512, 128, 2, num_experts=512)
    pf = BackgroundExpertPrefetcher(src)
    try:
        pf.submit(15, [1, 2, 3])
        deadline = time.time() + 3
        while pf.ready_count(15) < 3 and time.time() < deadline:
            time.sleep(0.01)
        # 全部物化后 promote → ready_on_time=3、not_ready=0
        got = pf.take_ready_layer(15)
        pf.note_promote(15, len(got))
        s = pf.stats()
        assert s["ready_on_time"] == 3
        assert s["not_ready"] == 0
    finally:
        pf.close()
        src.close()


def test_choose_victim_never_evicts_current():
    from mlx_streaming.core.cache.resident_pool import ResidentExpertPool
    pool = ResidentExpertPool(2, loader=lambda l, e: {"w": mx.zeros((2,))})
    pool._ensure_layer(0)
    pool._place_expert(0, 10, {"w": mx.zeros((2,))})
    pool._place_expert(0, 11, {"w": mx.zeros((2,))})
    # 池满 {10,11}；放 12 且 current 含全部常驻 → 无非 current 可驱逐 → 必须报错，不能驱逐 10/11
    with pytest.raises(ValueError):
        pool._place_expert(0, 12, {"w": mx.zeros((2,))}, current={10, 11, 12})
    # 10、11 仍在（没被误驱逐）
    assert set(pool.resident_experts(0)) == {10, 11}


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(BLOB, "layer15.blob"))
         and os.path.exists(os.path.join(
             os.path.dirname(__file__), "..", "..", "models",
             "qwen3_next_experts_2bit_g128", "layer15_expert003.safetensors"))),
    reason="需要 blob + per-expert 2bit 目录")
def test_submit_missing_filters_resident():
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    from mlx_streaming.core.prefetch.cross_layer import _submit_missing_prefetch
    expert_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")
    store = FileExpertStore(expert_dir, capacity=32)
    src = BlobExpertSource(BLOB, 2048, 512, 128, 2, num_experts=512)
    store._bg = BackgroundExpertPrefetcher(src)
    try:
        # 预置 {1,2,3} 为常驻（用 bg 物化 + promote 进池）
        store._bg.submit(15, [1, 2, 3])
        deadline = time.time() + 3
        while store._bg.ready_count(15) < 3 and time.time() < deadline:
            time.sleep(0.01)
        store.promote_prefetched(15)
        assert {1, 2, 3} <= store.resident_experts(15)
        # 预测 [1,2,3,7,8] → 只应预取缺失的 {7,8}
        missing = _submit_missing_prefetch(store, 15, [1, 2, 3, 7, 8])
        assert set(missing) == {7, 8}
        deadline = time.time() + 3
        while store._bg.ready_count(15) < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert store._bg.take_ready(15, 7) is not None
        assert store._bg.take_ready(15, 1) is None     # 常驻的不预取
    finally:
        store._bg.close()
        src.close()


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(BLOB, "layer15.blob"))
         and os.path.exists(os.path.join(
             os.path.dirname(__file__), "..", "..", "models",
             "qwen3_next_experts_2bit_g128", "layer15_expert003.safetensors"))),
    reason="需要 blob + per-expert 2bit 目录")
def test_stream_blob_bg_matches_resident(monkeypatch):
    from mlx_streaming.core.cache.expert_store import FileExpertStore
    from mlx_streaming.core.moe.block import FileStreamingMoeBlock
    H, I, G, B, NE, LAYER = 2048, 512, 128, 2, 512, 15
    expert_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")

    class _Gate:
        def __init__(self):
            mx.random.seed(0)
            self.w = mx.random.normal((NE, H)) * 0.02
            mx.eval(self.w)

        def __call__(self, x):
            return x @ self.w.T

    store = FileExpertStore(expert_dir, capacity=256)
    block = FileStreamingMoeBlock(gate=_Gate(), top_k=8, norm_topk_prob=True, store=store,
                                  layer_idx=LAYER, hidden=H, moe_inter=I, group_size=G, bits=B)
    mx.random.seed(1)
    x = (mx.random.normal((1, 1, H)) * 0.1).astype(mx.float32)
    mx.eval(x)

    monkeypatch.setenv("GPU_REMAP", "0")
    monkeypatch.setenv("NATIVE_MOE", "0")
    monkeypatch.setenv("STREAM_BLOB", "0")
    monkeypatch.setenv("STREAM_BLOB_BG", "0")
    y_res = block(x)
    mx.eval(y_res)

    src = BlobExpertSource(BLOB, H, I, G, B, num_experts=NE)
    store._bg = BackgroundExpertPrefetcher(src)
    store._blob_loader = src
    try:
        store._bg.submit(LAYER, list(range(64)))
        deadline = time.time() + 3
        while store._bg.ready_count(LAYER) < 64 and time.time() < deadline:
            time.sleep(0.01)
        monkeypatch.setenv("STREAM_BLOB_BG", "1")
        y_bg = block(x)
        mx.eval(y_bg)
        assert float(mx.max(mx.abs(y_res - y_bg))) < 1e-4
    finally:
        store._bg.close()
        src.close()
