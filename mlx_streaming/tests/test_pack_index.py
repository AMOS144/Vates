import os

import mlx.core as mx

from mlx_streaming.prep.pack_expert_ranges import pack_layer, read_safetensors_payloads


def test_pack_layer_preserves_raw_tensor_payloads(tmp_path):
    src = tmp_path / "experts"
    out = tmp_path / "packs"
    src.mkdir()
    for expert in range(2):
        mx.save_safetensors(
            str(src / f"layer00_expert{expert:03d}.safetensors"),
            {
                "weight": mx.arange(8, dtype=mx.uint32).reshape(2, 4) + expert,
                "scales": mx.ones((2, 1), dtype=mx.float32) * expert,
            },
        )

    index = pack_layer(str(src), str(out), layer=0, num_experts=2)
    pack_path = out / "layer00.pack"
    assert pack_path.exists()
    assert (out / "layer00.index.json").exists()
    assert (out / "layer00.idx").exists()
    assert len(index["experts"]) == 2

    by_key = {}
    for expert in range(2):
        original = read_safetensors_payloads(str(src / f"layer00_expert{expert:03d}.safetensors"))
        for rec in original:
            by_key[(expert, rec["key"])] = rec["payload"]

    with open(pack_path, "rb") as f:
        for rec in index["tensors"]:
            f.seek(rec["offset"])
            got = f.read(rec["nbytes"])
            assert got == by_key[(rec["expert_id"], rec["key"])]
            assert rec["offset"] % 64 == 0
            blob = index["experts"][rec["expert_id"]]
            assert blob["offset"] <= rec["offset"]
            assert rec["offset"] + rec["nbytes"] <= blob["offset"] + blob["nbytes"]


def test_read_safetensors_payloads_returns_dtype_shape_and_bytes(tmp_path):
    path = tmp_path / "one.safetensors"
    mx.save_safetensors(str(path), {"x": mx.array([[1, 2], [3, 4]], dtype=mx.uint32)})
    recs = read_safetensors_payloads(str(path))
    assert len(recs) == 1
    assert recs[0]["key"] == "x"
    assert recs[0]["dtype"] == "U32"
    assert recs[0]["shape"] == [2, 2]
    assert len(recs[0]["payload"]) == os.path.getsize(path) - 8 - int.from_bytes(open(path, "rb").read(8), "little")
