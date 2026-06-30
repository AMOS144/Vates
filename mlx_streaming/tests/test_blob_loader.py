import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_streaming.core.cache.blob_loader import BlobExpertSource


def test_load_experts_stacked_matches_per_expert(tmp_path):
    # 自包含（不需外部 blob fixture）：BlobExpertSource 构造仅算段表，patch read_raw 喂合成字节，
    # 断言批量物化 stacked[k][i] 与逐专家 _materialize 字节级一致。
    src = BlobExpertSource(str(tmp_path), hidden=64, inter=64, group=32, bits=4,
                           num_experts=8, nocache=False, quant_mode="mxfp4")
    try:
        rng = np.random.default_rng(0)
        raws = {e: rng.integers(0, 256, size=src.stride, dtype=np.uint8).tobytes()
                for e in (3, 5)}
        src.read_raw = lambda layer, ids: [raws[int(e)] for e in ids]
        stacked = src.load_experts_stacked(0, [3, 5])
        per = {e: src._materialize(raws[e]) for e in (3, 5)}
        for i, e in enumerate([3, 5]):
            for k in per[e]:
                assert stacked[k].dtype == per[e][k].dtype
                assert mx.array_equal(stacked[k][i], per[e][k]).item()
    finally:
        src.close()

BLOB_DIR = "/tmp/cb_2bit_blob"
EXPERT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128")
G, B = 128, 2


def _moe(x, w):
    g = mx.quantized_matmul(x, w["gate_proj.weight"], w["gate_proj.scales"], w["gate_proj.biases"],
                            transpose=True, group_size=G, bits=B)
    u = mx.quantized_matmul(x, w["up_proj.weight"], w["up_proj.scales"], w["up_proj.biases"],
                            transpose=True, group_size=G, bits=B)
    a = g * mx.sigmoid(g) * u
    return mx.quantized_matmul(a, w["down_proj.weight"], w["down_proj.scales"], w["down_proj.biases"],
                               transpose=True, group_size=G, bits=B)


@pytest.mark.skipif(not os.path.exists(os.path.join(BLOB_DIR, "layer15.blob")),
                    reason="需要 /tmp/cb_2bit_blob")
def test_acquire_stacks_and_slots_roundtrip():
    src = BlobExpertSource(BLOB_DIR, hidden=2048, inter=512, group=G, bits=B, num_experts=512)
    try:
        pool, slots = src.acquire(15, [7, 3, 7, 100])
        assert slots == [0, 1, 0, 2]          # unique 顺序 [7,3,100]
        assert set(pool.keys()) == {
            "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
            "up_proj.weight", "up_proj.scales", "up_proj.biases",
            "down_proj.weight", "down_proj.scales", "down_proj.biases"}
        # round-trip：pool[k][slot] 应等于单独 load 的该专家
        one = src.load_experts(15, [7])[7]
        for k in pool:
            assert bool(mx.all(pool[k][0] == one[k]).item())
        assert pool["gate_proj.scales"].dtype == mx.bfloat16
        assert pool["gate_proj.weight"].dtype == mx.uint32
    finally:
        src.close()


@pytest.mark.skipif(not os.path.exists(os.path.join(BLOB_DIR, "layer15.blob")),
                    reason="需要 /tmp/cb_2bit_blob（先跑 repack_expert_blobs LAYERS=15,25）")
def test_blob_source_matches_safetensors():
    src = BlobExpertSource(BLOB_DIR, hidden=2048, inter=512, group=G, bits=B, num_experts=512)
    x = (mx.random.normal((1, 2048)) * 0.1).astype(mx.float32)
    mx.eval(x)
    try:
        experts = src.load_experts(15, [3, 7, 100])
        for e, wb in experts.items():
            wr = mx.load(os.path.join(EXPERT_DIR, f"layer15_expert{e:03d}.safetensors"))
            yb, yr = _moe(x, wb), _moe(x, wr)
            mx.eval(yb, yr)
            assert float(mx.max(mx.abs(yb - yr))) < 1e-5
    finally:
        src.close()
