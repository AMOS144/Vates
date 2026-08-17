import json
from argparse import Namespace
from pathlib import Path

import mlx.core as mx

from mlx_streaming.core.cache.blob_loader import BlobExpertSource
from mlx_streaming.prep.runtime_bundle import (
    COMPACT_MARKER,
    MANIFEST_NAME,
    _safetensors_header,
    prepare,
    verify_bundle,
)


def _quantized_experts(experts: int, out_dims: int, in_dims: int):
    values = {"weight": [], "scales": [], "biases": []}
    for expert in range(experts):
        dense = mx.full((out_dims, in_dims), float(expert + 1), dtype=mx.bfloat16)
        weight, scales, biases = mx.quantize(dense, group_size=64, bits=4)
        values["weight"].append(weight)
        values["scales"].append(scales)
        values["biases"].append(biases)
    return {name: mx.stack(arrays) for name, arrays in values.items()}


def _write_tiny_sources(root: Path) -> tuple[Path, Path]:
    main = root / "main"
    mtp_dir = root / "mtp"
    main.mkdir(parents=True)
    mtp_dir.mkdir(parents=True)
    config = {
        "hidden_size": 64,
        "moe_intermediate_size": 64,
        "num_experts": 2,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    (main / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (main / "tokenizer.json").write_text("{}", encoding="utf-8")

    arrays = {"model.embed_tokens.weight": mx.ones((8, 64), dtype=mx.bfloat16)}
    for projection, out_dims, in_dims in (
        ("gate_proj", 64, 64),
        ("up_proj", 64, 64),
        ("down_proj", 64, 64),
    ):
        quantized = _quantized_experts(2, out_dims, in_dims)
        for tensor, value in quantized.items():
            arrays[f"model.layers.0.mlp.switch_mlp.{projection}.{tensor}"] = value
    shard = main / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), arrays)
    (main / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {},
        "weight_map": {key: shard.name for key in arrays},
    }), encoding="utf-8")

    mtp = {
        "mtp.norm.weight": mx.zeros((64,), dtype=mx.bfloat16),
        "mtp.fc.weight": mx.ones((64, 128), dtype=mx.bfloat16),
    }
    for expert in range(2):
        for projection, out_dims, in_dims in (
            ("gate_proj", 64, 64),
            ("up_proj", 64, 64),
            ("down_proj", 64, 64),
        ):
            mtp[
                f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
            ] = mx.full(
                (out_dims, in_dims), float(expert + 1), dtype=mx.bfloat16,
            )
    mtp_shard = mtp_dir / "model-00041-of-00041.safetensors"
    mx.save_safetensors(str(mtp_shard), mtp)
    return main, mtp_shard


def test_prepare_builds_compact_verified_bundle_without_per_expert_files(tmp_path):
    source = tmp_path / "source"
    main, mtp = _write_tiny_sources(source)
    output = tmp_path / "runtime"
    args = Namespace(
        output=str(output),
        source_dir=str(source),
        download=False,
        main_source=str(main),
        mtp_source=str(mtp),
        keep_source=True,
        skip_full_hash=False,
    )
    assert prepare(args) == 0

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "verified"
    assert manifest["main"]["layers"] == [0]
    assert (output / "model" / COMPACT_MARKER).is_file()
    assert not list(output.rglob("*expert*.safetensors"))
    core_header = _safetensors_header(
        output / "model" / "model-00001-of-00001.safetensors",
    )
    assert "model.embed_tokens.weight" in core_header
    assert not any("switch_mlp" in key for key in core_header)

    source_reader = BlobExpertSource(
        str(output / "experts" / "blobs"), 64, 64, 64, 4, 2,
        workers=1, nocache=False,
    )
    mtp_reader = BlobExpertSource(
        str(output / "mtp" / "experts"), 64, 64, 64, 4, 2,
        workers=1, nocache=False,
    )
    try:
        assert source_reader.load_experts(0, [1])[1]["gate_proj.weight"].shape == (64, 8)
        assert mtp_reader.load_experts(100, [1])[1]["down_proj.weight"].shape == (64, 8)
    finally:
        source_reader.close()
        mtp_reader.close()
    assert verify_bundle(output, full_hash=True)["files"] > 0
    assert main.joinpath("model-00001-of-00001.safetensors").is_file()
    assert mtp.is_file()


def test_prepare_deletes_only_source_weight_files_after_verification(tmp_path):
    source = tmp_path / "source"
    main, mtp = _write_tiny_sources(source)
    output = tmp_path / "runtime"
    args = Namespace(
        output=str(output),
        source_dir=str(source),
        download=False,
        main_source=str(main),
        mtp_source=str(mtp),
        keep_source=False,
        skip_full_hash=True,
    )
    assert prepare(args) == 0
    assert not main.joinpath("model-00001-of-00001.safetensors").exists()
    assert not mtp.exists()
    # 配置和 tokenizer 不是大权重，手工源目录中应保留，避免过度删除。
    assert main.joinpath("config.json").is_file()
    assert main.joinpath("tokenizer.json").is_file()
