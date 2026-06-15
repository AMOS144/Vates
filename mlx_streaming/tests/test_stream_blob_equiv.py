import os

import mlx.core as mx
import pytest

from mlx_streaming.core.cache.blob_loader import BlobExpertSource
from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.moe.block import FileStreamingMoeBlock

BLOB_DIR = "/tmp/cb_2bit_blob"
EXPERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")
HIDDEN, INTER, GROUP, BITS, NE = 2048, 512, 128, 2, 512
LAYER = 15


class _FakeGate:
    """固定权重 gate：x[...,H] -> logits[...,NE]，两条路径共用同一 gate 保证路由一致。"""
    def __init__(self):
        mx.random.seed(0)
        self.w = mx.random.normal((NE, HIDDEN)) * 0.02
        mx.eval(self.w)

    def __call__(self, x):
        return x @ self.w.T


@pytest.mark.skipif(
    not os.path.exists(os.path.join(BLOB_DIR, f"layer{LAYER:02d}.blob"))
    or not os.path.exists(os.path.join(EXPERT_DIR, f"layer{LAYER:02d}_expert000.safetensors")),
    reason="需要 blob 与 per-expert 2bit 目录")
def test_stream_blob_matches_resident(monkeypatch):
    store = FileExpertStore(EXPERT_DIR, capacity=256)
    block = FileStreamingMoeBlock(
        gate=_FakeGate(), top_k=8, norm_topk_prob=True, store=store, layer_idx=LAYER,
        hidden=HIDDEN, moe_inter=INTER, group_size=GROUP, bits=BITS)
    block._blob = BlobExpertSource(BLOB_DIR, HIDDEN, INTER, GROUP, BITS, NE)

    mx.random.seed(1)
    x = (mx.random.normal((1, 1, HIDDEN)) * 0.1).astype(mx.float32)
    mx.eval(x)

    # resident host 路径（关 GPU_REMAP 走 acquire）
    monkeypatch.setenv("GPU_REMAP", "0")
    monkeypatch.setenv("STREAM_BLOB", "0")
    monkeypatch.setenv("NATIVE_MOE", "0")
    y_res = block(x)
    mx.eval(y_res)

    # blob 路径
    monkeypatch.setenv("STREAM_BLOB", "1")
    y_blob = block(x)
    mx.eval(y_blob)

    block._blob.close()
    assert y_res.shape == y_blob.shape
    assert float(mx.max(mx.abs(y_res - y_blob))) < 1e-4
