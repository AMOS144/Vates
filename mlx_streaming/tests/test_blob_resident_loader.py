import os

import mlx.core as mx
import pytest

from mlx_streaming.core.cache.blob_loader import BlobExpertSource
from mlx_streaming.core.cache.expert_store import FileExpertStore

BLOB_DIR = "/tmp/cb_2bit_blob"
EXPERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(BLOB_DIR, "layer15.blob"))
         and os.path.exists(os.path.join(EXPERT_DIR, "layer15_expert007.safetensors"))),
    reason="需要 blob 与 per-expert 2bit 目录")
def test_raw_load_one_blob_matches_safetensors():
    store = FileExpertStore(EXPERT_DIR, capacity=32)
    store._blob_loader = BlobExpertSource(BLOB_DIR, 2048, 512, 128, 2, num_experts=512)
    try:
        a = store._raw_load_one(15, 7)                       # blob 路径
        b = mx.load(os.path.join(EXPERT_DIR, "layer15_expert007.safetensors"))
        assert set(a.keys()) == set(b.keys())
        for k in b:
            assert a[k].dtype == b[k].dtype
            assert bool(mx.all(a[k] == b[k]).item())
    finally:
        store._blob_loader.close()
