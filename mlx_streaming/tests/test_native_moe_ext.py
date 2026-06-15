import importlib
import json
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_streaming.core.moe import native_moe


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "native" / "ext"


def _bf16_bits(value: float) -> np.uint16:
    return np.uint16(np.array([value], dtype=np.float32).view(np.uint32)[0] >> 16)


def _write_compute_projection(root: Path, layer: int, proj: str,
                              num_experts: int, out_dim: int, in_dim: int,
                              group: int, bits: int):
    words = in_dim * bits // 32
    groups = in_dim // group
    base = root / f"layer{layer:02d}.{proj}"
    weight = np.arange(num_experts * out_dim * words, dtype=np.uint32).reshape(
        num_experts, out_dim, words)
    scales = np.full((num_experts, out_dim, groups), _bf16_bits(0.001), dtype=np.uint16)
    biases = np.zeros((num_experts, out_dim, groups), dtype=np.uint16)
    weight.tofile(str(base) + ".weight.bin")
    scales.tofile(str(base) + ".scales.bin")
    biases.tofile(str(base) + ".biases.bin")
    Path(str(base) + ".index.json").write_text(
        json.dumps({
            "format": "mlx_streaming_compute_buffer_v1",
            "layer": layer,
            "proj": proj,
            "num_experts": num_experts,
            "tensors": {
                "weight": {"shape_per_expert": [out_dim, words]},
                "scales": {"shape_per_expert": [out_dim, groups]},
                "biases": {"shape_per_expert": [out_dim, groups]},
            },
        }),
        encoding="utf-8",
    )


def test_native_moe_slot_pool_reuses_pool_arrays(tmp_path, monkeypatch):
    compute_dir = tmp_path / "compute"
    compute_dir.mkdir()
    hidden = 64
    inter = 32
    group = 32
    bits = 6
    num_experts = 8
    _write_compute_projection(compute_dir, 0, "gate_proj", num_experts, inter, hidden, group, bits)
    _write_compute_projection(compute_dir, 0, "up_proj", num_experts, inter, hidden, group, bits)
    _write_compute_projection(compute_dir, 0, "down_proj", num_experts, hidden, inter, group, bits)
    monkeypatch.setenv("NATIVE_MOE_SLOT_CAP", "4")

    pool = native_moe._slot_pool(
        str(compute_dir), 0, hidden, inter, group, bits, num_experts)
    first_local, first_arrays = pool.acquire([1, 2, 1])
    assert first_local == [0, 1, 0]
    assert pool.stats()["rebuilds"] == 1

    second_local, second_arrays = pool.acquire([2, 1])
    assert second_local == [1, 0]
    assert second_arrays[0] is first_arrays[0]
    assert pool.stats()["rebuilds"] == 1


def test_native_moe_ext_fused_moe_returns_mlx_array():
    subprocess.run(["make", "native_moe_ext"], cwd=BENCH_DIR, check=True)
    mod = importlib.import_module("mlx_streaming.native_moe_ext")
    x = mx.ones((1, 64), dtype=mx.float32) * 0.1
    expert_ids = mx.array([0, 1, 2, 3], dtype=mx.uint32)
    scores = mx.array([0.25, 0.25, 0.25, 0.25], dtype=mx.float32).reshape(1, 4)
    y = mod.fused_moe(
        x,
        expert_ids,
        scores,
        "/tmp/unused",
        0,
        64,
        32,
        32,
        6,
        8,
        True,
    )
    mx.eval(y)
    assert isinstance(y, mx.array)
    assert y.shape == (1, 64)
    assert bool(mx.all(mx.isfinite(y)).item())
    assert float(mx.max(mx.abs(y))) > 0


def test_native_moe_ext_fused_moe_staged_returns_mlx_array():
    subprocess.run(["make", "native_moe_ext"], cwd=BENCH_DIR, check=True)
    mod = importlib.import_module("mlx_streaming.native_moe_ext")
    hidden = 64
    inter = 32
    group = 32
    bits = 6
    active = 4
    gu_words = hidden * bits // 32
    down_words = inter * bits // 32
    gu_groups = hidden // group
    down_groups = inter // group
    x = mx.ones((1, hidden), dtype=mx.float32) * 0.1
    scores = mx.array([0.25, 0.25, 0.25, 0.25], dtype=mx.float32).reshape(1, 4)
    gate_w = mx.ones((active, inter, gu_words), dtype=mx.uint32)
    up_w = mx.ones((active, inter, gu_words), dtype=mx.uint32)
    down_w = mx.ones((active, hidden, down_words), dtype=mx.uint32)
    scale = int(_bf16_bits(0.001))
    gate_s = mx.array(np.full((active, inter, gu_groups), scale, dtype=np.uint16))
    up_s = mx.array(np.full((active, inter, gu_groups), scale, dtype=np.uint16))
    down_s = mx.array(np.full((active, hidden, down_groups), scale, dtype=np.uint16))
    gate_b = mx.zeros((active, inter, gu_groups), dtype=mx.uint16)
    up_b = mx.zeros((active, inter, gu_groups), dtype=mx.uint16)
    down_b = mx.zeros((active, hidden, down_groups), dtype=mx.uint16)
    y = mod.fused_moe_staged(
        x, scores,
        gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        down_w, down_s, down_b,
        hidden, inter, group, bits,
    )
    mx.eval(y)
    assert isinstance(y, mx.array)
    assert y.shape == (1, hidden)
    assert bool(mx.all(mx.isfinite(y)).item())


def test_native_moe_ext_fused_moe_staged_reduces_all_scored_experts():
    subprocess.run(["make", "native_moe_ext"], cwd=BENCH_DIR, check=True)
    mod = importlib.import_module("mlx_streaming.native_moe_ext")
    hidden = 64
    inter = 32
    group = 32
    bits = 6
    active = 4
    gu_words = hidden * bits // 32
    down_words = inter * bits // 32
    gu_groups = hidden // group
    down_groups = inter // group
    x = mx.ones((1, hidden), dtype=mx.float32) * 0.1
    gate_w = mx.ones((active, inter, gu_words), dtype=mx.uint32)
    up_w = mx.ones((active, inter, gu_words), dtype=mx.uint32)
    down_w = mx.ones((active, hidden, down_words), dtype=mx.uint32)
    scales = np.array([_bf16_bits(0.001 * (i + 1)) for i in range(active)], dtype=np.uint16)
    gate_s = mx.array(np.broadcast_to(scales[:, None, None], (active, inter, gu_groups)).copy())
    up_s = mx.array(np.broadcast_to(scales[:, None, None], (active, inter, gu_groups)).copy())
    down_s = mx.array(np.broadcast_to(scales[:, None, None], (active, hidden, down_groups)).copy())
    gate_b = mx.zeros((active, inter, gu_groups), dtype=mx.uint16)
    up_b = mx.zeros((active, inter, gu_groups), dtype=mx.uint16)
    down_b = mx.zeros((active, hidden, down_groups), dtype=mx.uint16)

    y0 = mod.fused_moe_staged(
        x, mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32),
        gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        down_w, down_s, down_b,
        hidden, inter, group, bits,
    )
    y1 = mod.fused_moe_staged(
        x, mx.array([[0.0, 1.0, 0.0, 0.0]], dtype=mx.float32),
        gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        down_w, down_s, down_b,
        hidden, inter, group, bits,
    )
    mx.eval(y0, y1)
    assert float(mx.max(mx.abs(y1))) > 0
    assert float(mx.max(mx.abs(y0 - y1))) > 1e-8


def test_native_moe_ext_fused_moe_slots_uses_local_slot_indices():
    subprocess.run(["make", "native_moe_ext"], cwd=BENCH_DIR, check=True)
    mod = importlib.import_module("mlx_streaming.native_moe_ext")
    hidden = 64
    inter = 32
    group = 32
    bits = 6
    cap = 4
    gu_words = hidden * bits // 32
    down_words = inter * bits // 32
    gu_groups = hidden // group
    down_groups = inter // group
    x = mx.ones((1, hidden), dtype=mx.float32) * 0.1
    local = mx.array([1, 3], dtype=mx.uint32)
    gate_w = mx.ones((cap, inter, gu_words), dtype=mx.uint32)
    up_w = mx.ones((cap, inter, gu_words), dtype=mx.uint32)
    down_w = mx.ones((cap, hidden, down_words), dtype=mx.uint32)
    scales = np.array([_bf16_bits(0.001 * (i + 1)) for i in range(cap)], dtype=np.uint16)
    gate_s = mx.array(np.broadcast_to(scales[:, None, None], (cap, inter, gu_groups)).copy())
    up_s = mx.array(np.broadcast_to(scales[:, None, None], (cap, inter, gu_groups)).copy())
    down_s = mx.array(np.broadcast_to(scales[:, None, None], (cap, hidden, down_groups)).copy())
    gate_b = mx.zeros((cap, inter, gu_groups), dtype=mx.uint16)
    up_b = mx.zeros((cap, inter, gu_groups), dtype=mx.uint16)
    down_b = mx.zeros((cap, hidden, down_groups), dtype=mx.uint16)

    y0 = mod.fused_moe_slots(
        x, local, mx.array([[1.0, 0.0]], dtype=mx.float32),
        gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        down_w, down_s, down_b,
        hidden, inter, group, bits,
    )
    y1 = mod.fused_moe_slots(
        x, local, mx.array([[0.0, 1.0]], dtype=mx.float32),
        gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        down_w, down_s, down_b,
        hidden, inter, group, bits,
    )
    mx.eval(y0, y1)
    assert float(mx.max(mx.abs(y0 - y1))) > 1e-8
