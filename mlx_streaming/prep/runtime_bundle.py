"""把原始 Qwen3-Next 分片直接整理成 Vates 的紧凑运行目录。

主模型路由专家直接写入逐层 blob，不产生 24,576 个 per-expert 中间文件；
非专家张量保留为可由 mlx-lm 加载的紧凑核心。MTP 同时拆成紧凑核心与
一个 4-bit 专家 blob。全部输出经结构检查和 SHA-256 复核后，才删除源权重。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

from mlx_streaming.prep.blob_layout import BLOB_V1_AFFINE, layout_for
from mlx_streaming.prep.extract_mtp import SHARD as MTP_SHARD
from mlx_streaming.prep.extract_mtp import bump_mtp_norms


MAIN_REPO = "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit"
MTP_REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MANIFEST_NAME = "vates_manifest.json"
COMPACT_MARKER = "vates_compact_model.json"
_MAIN_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$",
)
_MTP_EXPERT = re.compile(
    r"^mtp\.layers\.0\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$",
)
_MTP_STACKED = re.compile(
    r"^mtp\.layers\.0\.mlp\.switch_mlp\."
    r"(gate_proj|up_proj|down_proj)\.weight$",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> dict:
    with path.open("rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError(f"safetensors 头不完整: {path}")
        size = struct.unpack("<Q", raw)[0]
        if size <= 0 or size > path.stat().st_size - 8:
            raise ValueError(f"safetensors 头长度非法: {path}")
        return json.loads(file.read(size))


def _raw_bytes(array: mx.array) -> memoryview:
    """保留 MLX 量化张量的原始位，不把 bf16 转成 float32。"""
    value = array.view(mx.uint16) if array.dtype == mx.bfloat16 else array
    return memoryview(np.asarray(value)).cast("B")


def _tensor_nbytes(array: mx.array) -> int:
    return int(array.size) * int(array.itemsize)


def _source_shards(source: Path) -> tuple[list[Path], dict[str, str]]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        document = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = document.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"模型索引缺少 weight_map: {index_path}")
        names = sorted(set(str(name) for name in weight_map.values()))
        shards = [source / name for name in names]
        return shards, {str(key): str(value) for key, value in weight_map.items()}
    shards = sorted(source.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"没有找到主模型分片: {source}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        for key in _safetensors_header(shard):
            if key != "__metadata__":
                weight_map[key] = shard.name
    return shards, weight_map


def _copy_model_metadata(source: Path, target: Path) -> None:
    """复制 tokenizer/config 等小文件，权重和旧索引由转换流程重建。"""
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file():
            continue
        if path.suffix == ".safetensors" or path.name == "model.safetensors.index.json":
            continue
        shutil.copy2(path, target / path.name)


def _main_dimensions(config: dict) -> tuple[int, int, int, int, int]:
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    experts = int(config["num_experts"])
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if quant.get("mode", "affine") != "affine":
        raise ValueError("vates prepare 当前只支持 affine 量化主模型")
    bits = int(quant.get("bits", 4))
    group = int(quant.get("group_size", 64))
    return hidden, intermediate, experts, bits, group


def _estimate_output_bytes(main_source: Path, mtp_source: Path) -> int:
    """按 safetensors 数据段与最终 MTP stride 估算输出，写入前检查磁盘。"""
    shards, _weight_map = _source_shards(main_source)
    main_bytes = 0
    for shard in shards:
        header = _safetensors_header(shard)
        main_bytes += sum(
            int(value["data_offsets"][1]) - int(value["data_offsets"][0])
            for key, value in header.items() if key != "__metadata__"
        )
    config = json.loads((main_source / "config.json").read_text(encoding="utf-8"))
    hidden, intermediate, experts, _bits, _group = _main_dimensions(config)
    _segments, mtp_stride = layout_for(
        BLOB_V1_AFFINE, hidden, intermediate, 4, 64,
    )
    mtp_header = _safetensors_header(mtp_source)
    mtp_core_bytes = sum(
        int(value["data_offsets"][1]) - int(value["data_offsets"][0])
        for key, value in mtp_header.items()
        if key != "__metadata__" and not _is_mtp_expert_key(key)
    )
    return main_bytes + mtp_core_bytes + mtp_stride * experts


def _segment_offsets(segments: list[tuple]) -> dict[tuple[str, str], tuple[int, int, tuple]]:
    offsets = {}
    cursor = 0
    for projection, tensor, _dtype, shape, size in segments:
        offsets[(projection, tensor)] = (cursor, int(size), tuple(shape))
        cursor += int(size)
    return offsets


def build_main_core_and_blobs(source: Path, bundle: Path) -> dict:
    """逐源分片保存非专家核心，并把堆叠专家直接写到最终 blob 偏移。"""
    config_path = source / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"主模型缺少 config.json: {source}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden, intermediate, experts, bits, group = _main_dimensions(config)
    segments, stride = layout_for(
        BLOB_V1_AFFINE, hidden, intermediate, bits, group,
    )
    offsets = _segment_offsets(segments)
    shards, source_map = _source_shards(source)
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"主模型分片缺失: {missing[0]}")

    expected = {
        (int(match.group(1)), match.group(2), match.group(3))
        for key in source_map
        if (match := _MAIN_EXPERT.match(key)) is not None
    }
    layers = sorted({layer for layer, _, _ in expected})
    if not layers:
        raise ValueError("主模型索引中没有 switch_mlp 路由专家")
    configured_layers = int(config.get("num_hidden_layers", len(layers)))
    if layers != list(range(configured_layers)):
        raise ValueError(
            f"主模型路由专家层不完整：找到 {layers}，期望 0..{configured_layers - 1}",
        )
    required = {
        (layer, projection, tensor)
        for layer in layers
        for projection, tensor in offsets
    }
    if expected != required:
        absent = sorted(required - expected)
        raise ValueError(f"主模型专家张量不完整，首个缺失项: {absent[:1]}")

    model_out = bundle / "model"
    expert_out = bundle / "experts"
    blob_out = expert_out / "blobs"
    _copy_model_metadata(source, model_out)
    blob_out.mkdir(parents=True, exist_ok=True)
    handles = {}
    try:
        for layer in layers:
            path = blob_out / f"layer{layer:02d}.blob"
            handle = path.open("w+b")
            handle.truncate(stride * experts)
            handles[layer] = handle

        compact_map: dict[str, str] = {}
        compact_bytes = 0
        seen: set[tuple[int, str, str]] = set()
        compact_files = []
        for position, shard in enumerate(shards, 1):
            print(f"[主模型 {position}/{len(shards)}] {shard.name}", flush=True)
            arrays = dict(mx.load(str(shard)))
            core = {key: value for key, value in arrays.items() if not _MAIN_EXPERT.match(key)}
            if core:
                output_name = shard.name
                output_path = model_out / output_name
                mx.save_safetensors(str(output_path), core)
                compact_files.append(output_name)
                for key, value in core.items():
                    compact_map[key] = output_name
                    compact_bytes += _tensor_nbytes(value)

            for key, array in arrays.items():
                match = _MAIN_EXPERT.match(key)
                if match is None:
                    continue
                layer = int(match.group(1))
                projection, tensor = match.group(2), match.group(3)
                offset, expected_size, expected_shape = offsets[(projection, tensor)]
                if tuple(array.shape) != (experts, *expected_shape):
                    raise ValueError(
                        f"{key} shape={tuple(array.shape)}，期望 "
                        f"{(experts, *expected_shape)}",
                    )
                mx.eval(array)
                for expert in range(experts):
                    raw = _raw_bytes(array[expert])
                    if len(raw) != expected_size:
                        raise ValueError(f"{key}[{expert}] 字节数错误")
                    handles[layer].seek(expert * stride + offset)
                    handles[layer].write(raw)
                seen.add((layer, projection, tensor))
            del arrays, core
            mx.clear_cache()
    finally:
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    if seen != required:
        absent = sorted(required - seen)
        raise ValueError(f"主模型专家写入不完整: {absent[:1]}")
    for layer in layers:
        path = blob_out / f"layer{layer:02d}.blob"
        if path.stat().st_size != stride * experts:
            raise ValueError(f"主模型专家 blob 大小错误: {path}")
        _write_json(blob_out / f"layer{layer:02d}.blob.index.json", {
            "format": BLOB_V1_AFFINE,
            "quant_mode": "affine",
            "layer": layer,
            "num_experts": experts,
            "stride": stride,
            "page_aligned": stride % 16384 == 0,
            "segments": [
                {"proj": projection, "tensor": tensor, "nbytes": size}
                for projection, tensor, _dtype, _shape, size in segments
            ],
        })

    _write_json(model_out / "model.safetensors.index.json", {
        "metadata": {"total_size": compact_bytes},
        "weight_map": compact_map,
    })
    _write_json(model_out / COMPACT_MARKER, {
        "format": "vates_compact_model_v1",
        "omitted": "model.layers.*.mlp.switch_mlp.*",
        "source_repo": MAIN_REPO,
        "source_shards": [path.name for path in shards],
        "core_shards": compact_files,
        "expert_layers": layers,
    })
    split_meta = {
        "format": BLOB_V1_AFFINE,
        "blob_format": BLOB_V1_AFFINE,
        "dims": {
            "num_experts": experts,
            "hidden": hidden,
            "moe_intermediate": intermediate,
            "bits": bits,
            "group_size": group,
            "quant_mode": "affine",
        },
    }
    _write_json(expert_out / "_split_meta.json", split_meta)
    _write_json(blob_out / "blob_index.json", {
        "format": BLOB_V1_AFFINE,
        "quant_mode": "affine",
        "stride": stride,
        "page_aligned": stride % 16384 == 0,
        "num_experts": experts,
        "layers": layers,
        "bits": bits,
        "group_size": group,
    })
    return {
        "source_shards": shards,
        "layers": layers,
        "num_experts": experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "bits": bits,
        "group_size": group,
        "stride": stride,
        "core_tensor_count": len(compact_map),
    }


def _mtp_dense_weight(
    arrays: dict[str, mx.array], expert: int, projection: str,
) -> mx.array:
    key = f"mtp.layers.0.mlp.experts.{expert}.{projection}.weight"
    if key in arrays:
        return arrays[key]
    stacked = f"mtp.layers.0.mlp.switch_mlp.{projection}.weight"
    if stacked in arrays:
        return arrays[stacked][expert]
    raise KeyError(f"MTP 缺少专家张量: {key}")


def _is_mtp_expert_key(key: str) -> bool:
    return _MTP_EXPERT.match(key) is not None or _MTP_STACKED.match(key) is not None


def build_mtp_core_and_experts(source: Path, bundle: Path, config: dict) -> dict:
    """从原始 MTP 分片直接生成 norm-correct 核心和 4-bit 专家 blob。"""
    arrays = {key: value for key, value in mx.load(str(source)).items() if key.startswith("mtp.")}
    if not arrays:
        raise ValueError(f"MTP 分片中没有 mtp.* 张量: {source}")
    hidden, intermediate, experts, _main_bits, _main_group = _main_dimensions(config)
    bits, group = 4, 64
    segments, stride = layout_for(BLOB_V1_AFFINE, hidden, intermediate, bits, group)

    mtp_out = bundle / "mtp"
    expert_out = mtp_out / "experts"
    expert_out.mkdir(parents=True, exist_ok=True)
    core = bump_mtp_norms({
        key: value for key, value in arrays.items() if not _is_mtp_expert_key(key)
    })
    if not core:
        raise ValueError("MTP 紧凑核心为空")
    mx.save_safetensors(str(mtp_out / "core.safetensors"), core)

    blob_path = expert_out / "layer100.blob"
    with blob_path.open("wb") as blob:
        for expert in range(experts):
            if expert % 32 == 0:
                print(f"[MTP 专家] {expert}/{experts}", flush=True)
            quantized = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                dense = _mtp_dense_weight(arrays, expert, projection)
                weight, scales, biases = mx.quantize(
                    dense, group_size=group, bits=bits, mode="affine",
                )
                quantized[(projection, "weight")] = mx.contiguous(weight)
                quantized[(projection, "scales")] = mx.contiguous(scales)
                quantized[(projection, "biases")] = mx.contiguous(biases)
            mx.eval(list(quantized.values()))
            for projection, tensor, _dtype, shape, size in segments:
                value = quantized[(projection, tensor)]
                if tuple(value.shape) != tuple(shape):
                    raise ValueError(
                        f"MTP expert {expert} {projection}.{tensor} "
                        f"shape={tuple(value.shape)}，期望 {tuple(shape)}",
                    )
                raw = _raw_bytes(value)
                if len(raw) != size:
                    raise ValueError(f"MTP expert {expert} 量化字节数错误")
                blob.write(raw)
            del quantized
        blob.flush()
        os.fsync(blob.fileno())
    if blob_path.stat().st_size != stride * experts:
        raise ValueError("MTP 专家 blob 大小错误")
    _write_json(expert_out / "meta.json", {
        "format": BLOB_V1_AFFINE,
        "bits": bits,
        "group_size": group,
        "num_experts": experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "stride": stride,
    })
    del arrays, core
    mx.clear_cache()
    return {
        "source_shard": source,
        "num_experts": experts,
        "bits": bits,
        "group_size": group,
        "stride": stride,
    }


def _file_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return records


def verify_bundle(root: Path, *, full_hash: bool = True) -> dict:
    """复核目录结构、blob 尺寸，并可重新读取所有输出校验 SHA-256。"""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少 {MANIFEST_NAME}: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("manifest 没有文件校验记录")
    total = 0
    for record in records:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise ValueError(f"文件缺失或大小变化: {record['path']}")
        if full_hash and _sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 不匹配: {record['path']}")
        total += path.stat().st_size

    main = manifest["main"]
    for layer in main["layers"]:
        path = root / "experts" / "blobs" / f"layer{int(layer):02d}.blob"
        if path.stat().st_size != int(main["stride"]) * int(main["num_experts"]):
            raise ValueError(f"主专家 blob 尺寸错误: {path.name}")
    mtp = manifest["mtp"]
    mtp_blob = root / "mtp" / "experts" / "layer100.blob"
    if mtp_blob.stat().st_size != int(mtp["stride"]) * int(mtp["num_experts"]):
        raise ValueError("MTP 专家 blob 尺寸错误")
    _safetensors_header(root / "mtp" / "core.safetensors")
    core_shards = sorted((root / "model").glob("model*.safetensors"))
    if not core_shards:
        raise ValueError("主模型紧凑核心为空")
    for shard in core_shards:
        _safetensors_header(shard)
    return {"files": len(records), "bytes": total, "full_hash": full_hash}


def _remove_empty_parents(path: Path, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    if stop != current.resolve() and stop not in current.resolve().parents:
        return
    while current.exists() and current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def cleanup_source_weights(
    main_shards: Iterable[Path], mtp_shard: Path, *, source_root: Path,
) -> list[str]:
    """只删除已经转换并验证过的明确分片，不递归删除用户目录。"""
    targets = list(dict.fromkeys([*(Path(path) for path in main_shards), Path(mtp_shard)]))
    removed = []
    for path in targets:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            continue
        resolved.unlink()
        removed.append(str(resolved))
        _remove_empty_parents(resolved.parent, source_root)
    return removed


def download_sources(source_root: Path) -> tuple[Path, Path]:
    """用 Hugging Face 官方下载器取得主模型与原版 MTP 末分片。"""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as error:  # pragma: no cover - 正常安装已由依赖提供
        raise RuntimeError("缺少 huggingface-hub，请先运行 uv sync") from error
    main = source_root / "main"
    mtp = source_root / "mtp"
    main.mkdir(parents=True, exist_ok=True)
    mtp.mkdir(parents=True, exist_ok=True)
    print(f"下载主模型 {MAIN_REPO} ...", flush=True)
    snapshot_download(repo_id=MAIN_REPO, local_dir=main)
    print(f"下载 MTP 分片 {MTP_REPO}/{MTP_SHARD} ...", flush=True)
    downloaded = hf_hub_download(
        repo_id=MTP_REPO, filename=MTP_SHARD, local_dir=mtp,
    )
    return main, Path(downloaded)


def prepare(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve()
    source_root = Path(args.source_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
    if args.download:
        main_source, mtp_source = download_sources(source_root)
    else:
        main_source = Path(args.main_source or source_root / "main").expanduser().resolve()
        mtp_source = Path(
            args.mtp_source or source_root / "mtp" / MTP_SHARD,
        ).expanduser().resolve()
    main_source = main_source.resolve()
    mtp_source = mtp_source.resolve()
    if not main_source.is_dir():
        raise FileNotFoundError(f"主模型源目录不存在: {main_source}")
    if not mtp_source.is_file():
        raise FileNotFoundError(f"MTP 源分片不存在: {mtp_source}")
    if output == main_source or output in main_source.parents or main_source in output.parents:
        raise ValueError("输出目录不能与主模型源目录重叠")
    if output == mtp_source or output in mtp_source.parents:
        raise ValueError("输出目录不能包含 MTP 源分片")

    staging = output.parent / f".{output.name}.partial-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"暂存目录已存在: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    estimate = _estimate_output_bytes(main_source, mtp_source)
    free = shutil.disk_usage(staging.parent).free
    reserve = 256 * 1024 * 1024
    print(
        f"预计最终输出 {estimate / 2**30:.2f} GiB；"
        f"目标磁盘可用 {free / 2**30:.2f} GiB。",
        flush=True,
    )
    if free < estimate + reserve:
        raise RuntimeError(
            f"目标磁盘空间不足：至少还需 {(estimate + reserve) / 2**30:.2f} GiB",
        )
    staging.mkdir()
    try:
        main = build_main_core_and_blobs(main_source, staging)
        config = json.loads((main_source / "config.json").read_text(encoding="utf-8"))
        mtp = build_mtp_core_and_experts(mtp_source, staging, config)
        manifest = {
            "format": "vates_runtime_bundle_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "hashing",
            "sources": {
                "main_repo": MAIN_REPO,
                "mtp_repo": MTP_REPO,
                "mtp_shard": MTP_SHARD,
            },
            "main": {key: value for key, value in main.items() if key != "source_shards"},
            "mtp": {key: value for key, value in mtp.items() if key != "source_shard"},
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        print("计算最终目录 SHA-256（会完整读取一次约 44 GB 输出）...", flush=True)
        manifest["files"] = _file_records(staging)
        manifest["status"] = "ready_for_verification"
        _write_json(staging / MANIFEST_NAME, manifest)
        result = verify_bundle(staging, full_hash=not args.skip_full_hash)
        manifest["status"] = "verified"
        manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
        manifest["verified_bytes"] = result["bytes"]
        _write_json(staging / MANIFEST_NAME, manifest)
        staging.rename(output)
    except Exception:
        print(f"准备失败；保留暂存目录供排查: {staging}", file=sys.stderr)
        raise

    removed = []
    if not args.keep_source:
        removed = cleanup_source_weights(
            main["source_shards"], mtp["source_shard"], source_root=source_root,
        )
    gib = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) / 2**30
    print(f"Vates 运行目录已就绪: {output} ({gib:.2f} GiB)")
    if removed:
        print(f"完整性验证通过后已删除 {len(removed)} 个原始权重分片。")
    elif args.keep_source:
        print("按 --keep-source 保留了原始权重。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vates prepare",
        description="下载/转换 Qwen3-Next，并生成约 44 GB 的 Vates 紧凑运行目录",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="从 Hugging Face 下载官方主模型和原版 MTP 分片",
    )
    parser.add_argument(
        "--source-dir", default="models/.vates-source",
        help="下载源目录（默认 models/.vates-source）",
    )
    parser.add_argument("--main-source", help="已有 4-bit MLX 主模型目录")
    parser.add_argument("--mtp-source", help="已有原版 MTP safetensors 分片")
    parser.add_argument(
        "--output", default="models/vates-runtime",
        help="最终运行目录（默认 models/vates-runtime）",
    )
    parser.add_argument(
        "--keep-source", action="store_true",
        help="验证后仍保留原始权重（默认会删除以回收磁盘）",
    )
    parser.add_argument(
        "--skip-full-hash", action="store_true", help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return prepare(parser.parse_args(argv))
    except (FileNotFoundError, FileExistsError, KeyError, ValueError, RuntimeError) as error:
        print(f"vates prepare: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
