import os

import mlx.core as mx

from mlx_streaming.prep.pack_compute_buffers import pack_projection
from mlx_streaming.prep.pack_expert_ranges import read_safetensors_payloads


def test_pack_projection_writes_contiguous_buffers(tmp_path):
    src = tmp_path / "experts"
    out = tmp_path / "buffers"
    src.mkdir()
    for expert in range(2):
        mx.save_safetensors(
            str(src / f"layer00_expert{expert:03d}.safetensors"),
            {
                "gate_proj.weight": mx.arange(8, dtype=mx.uint32).reshape(2, 4) + expert,
                "gate_proj.scales": mx.ones((2, 1), dtype=mx.float32) * expert,
                "gate_proj.biases": mx.zeros((2, 1), dtype=mx.float32) + expert,
            },
        )

    meta = pack_projection(str(src), str(out), layer=0, proj="gate_proj", num_experts=2)
    weight = meta["tensors"]["weight"]
    assert weight["offsets"] == [0, weight["nbytes_per_expert"]]
    weight_file = out / weight["file"]
    assert os.path.getsize(weight_file) == 2 * weight["nbytes_per_expert"]

    original = read_safetensors_payloads(str(src / "layer00_expert001.safetensors"))
    payload = {rec["key"]: rec["payload"] for rec in original}["gate_proj.weight"]
    with open(weight_file, "rb") as f:
        f.seek(weight["offsets"][1])
        assert f.read(weight["nbytes_per_expert"]) == payload
