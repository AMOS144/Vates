"""共享装配层:把"文件后端流式加载主模型 + 取 hidden + 贪心生成"等跨入口复用的逻辑
集中在这里,而不是藏在某个入口脚本(原 validate_mtp)里。

依赖关系:本模块位于 core(mem/expert_store/streaming_moe)与 mtp 之上,
是把它们粘合成可运行模型的装配层,被 cli/ 各入口与测试复用。

环境变量:
  MODEL          主模型路径(4-bit MLX)
  EXPERT_DIR     拆分/重量化后的 per-expert safetensors 目录
  EXPERT_SLOTS   每层常驻池容量
  EXPERT_POOL_PROFILE  每层池预算 JSON(无损省内存,可选)
  HIDDEN_VARIANT pre_final_norm(默认)| post_final_norm(排错时切换)
"""
import json
import os

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from mlx_streaming import config
from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.prefetch.patch import patch_model_filebacked

MODEL = config.model_path()
EXPERT_DIR = config.expert_dir()
EXPERT_SLOTS = config.expert_slots()
# pre_final_norm(默认)| post_final_norm(排错时切换)
HIDDEN_VARIANT = config.hidden_variant()

# 默认 profile 文件名:放在 EXPERT_DIR 下随专家目录一起走,常用跑法自动启用(无损省内存)
DEFAULT_PROFILE_NAME = "pool_profile.json"


def load_pool_profile(expert_dir: str) -> "dict[int, int] | None":
    """解析每层池预算 profile,返回 layer_caps 或 None。

    优先级:环境变量 EXPERT_POOL_PROFILE 显式指定路径 > {expert_dir}/pool_profile.json 默认。
    EXPERT_POOL_PROFILE=none/0/off 显式关闭(回到 uniform capacity)。
    profile 无损:仅按各层真实工作集分配,命中率/输出/吞吐不变(caps 仍被 capacity 上限钳制)。
    """
    p = config.expert_pool_profile()
    if p.lower() in ("none", "0", "off"):
        return None
    if not p:                                   # 未显式指定 → 默认找专家目录下的 profile
        cand = os.path.join(expert_dir, DEFAULT_PROFILE_NAME)
        p = cand if os.path.exists(cand) else ""
    if p and os.path.exists(p):
        with open(p) as f:
            caps = json.load(f).get("layer_caps", {})
        return {int(k): int(v) for k, v in caps.items()}
    return None


def build_streaming_model():
    """用文件后端流式 patch 加载主模型(32GB 机器装不下 41GB 非流式)。"""
    model, tok = load(MODEL, lazy=True)
    # 取首个 MoE 维度
    dims = None
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            dims = {"hidden": gp.input_dims, "moe_inter": gp.output_dims,
                    "group_size": getattr(gp, "group_size", 64),
                    "bits": getattr(gp, "bits", 4)}
            break
    bits, group, proj_bits, layer_proj_bits, rotated = (
        dims["bits"], dims["group_size"], None, None, False)
    meta_path = os.path.join(EXPERT_DIR, "_split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        ed = meta.get("dims", {})
        bits = ed.get("bits", bits)
        group = ed.get("group_size", group)
        proj_bits = ed.get("proj_bits")
        # 旋转重量化产出在 meta 顶层标 rotated=True;runtime 必须走 Hadamard 旋转前向(RotatedSubGLU),
        # 否则等于算 W'·x(未旋转输入)→ 数值错误。
        rotated = bool(meta.get("rotated", False))
        if "per_layer_proj_bits" in ed:
            layer_proj_bits = {int(k): v for k, v in ed["per_layer_proj_bits"].items()}
    # 每层池预算 profile(pool_footprint 产出):默认从 {EXPERT_DIR}/pool_profile.json 自动启用,
    # 无损省内存(命中率/输出/吞吐不变，仅不再为低占用层预留满 capacity)。
    layer_caps = load_pool_profile(EXPERT_DIR)
    store = FileExpertStore(EXPERT_DIR, capacity=EXPERT_SLOTS, layer_caps=layer_caps)
    if config.stream_blob_loader():
        # blob 接入常驻池 miss-loader：复用 GPU-remap 快路径，小 EXPERT_SLOTS 即低内存。
        store._blob_loader = _make_blob_source(dims, group, bits)
    # 主动预取（native-fused-prefetch miss→hit）：opt-in（NATIVE_FUSED_PREFETCH=1）。
    # 实测干净交错对比 = 比 demand 慢 ~25%（每层 gate 前向+prefetch+promote 开销 > 命中收益），
    # 且命中只影响速度不影响质量、还多占 staging 内存 → 默认关，不作默认路径。
    if config.native_fused_prefetch() and getattr(store, "_blob_loader", None) is not None:
        try:
            import mlx_streaming.native_moe_ext  # noqa: F401  确认扩展已编译
            from mlx_streaming.core.prefetch.native_staging import NativeStagingManager
            store._staging = NativeStagingManager(
                store._blob_loader, budget=config.stream_blob_bg_budget(default=8))
        except Exception:
            store._staging = None   # 扩展不可用 → 关闭，不影响主路径
    if config.stream_blob_bg():
        # 后台预取池预填：bg 在独立 stream 物化预测专家，promote 写进池槽（需 CROSS_LAYER_PREFETCH=1）。
        from mlx_streaming.core.prefetch.bg_prefetch import BackgroundExpertPrefetcher
        src = _make_blob_source(dims, group, bits)
        store._blob_loader = src
        store._bg = BackgroundExpertPrefetcher(
            src, window=config.stream_blob_window())
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           group, bits, rotated=rotated, proj_bits=proj_bits,
                           layer_proj_bits=layer_proj_bits)
    if config.stream_blob():
        _attach_blob_source(model, dims, group, bits)
    return model, tok, store


def _make_blob_source(dims, group, bits):
    from mlx_streaming.core.cache.blob_loader import BlobExpertSource
    blob_dir = config.blob_dir() or os.path.join(EXPERT_DIR, "blobs")
    workers = config.stream_blob_workers()
    nocache = config.stream_blob_nocache(default="0")
    num_experts = 512
    idx_path = os.path.join(blob_dir, "blob_index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            num_experts = int(json.load(f).get("num_experts", num_experts))
    return BlobExpertSource(blob_dir, dims["hidden"], dims["moe_inter"], group, bits,
                            num_experts=num_experts, workers=workers, nocache=nocache)


def _attach_blob_source(model, dims, group, bits):
    """STREAM_BLOB=1：给每个流式 MoE 块注入共享 BlobExpertSource（全流式低内存路径）。"""
    from mlx_streaming.core.cache.blob_loader import BlobExpertSource
    from mlx_streaming.core.moe.block import FileStreamingMoeBlock

    blob_dir = config.blob_dir() or os.path.join(EXPERT_DIR, "blobs")
    workers = config.stream_blob_workers()
    nocache = config.stream_blob_nocache(default="1")
    num_experts = 512
    idx_path = os.path.join(blob_dir, "blob_index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            num_experts = int(json.load(f).get("num_experts", num_experts))
    src = BlobExpertSource(blob_dir, dims["hidden"], dims["moe_inter"], group, bits,
                           num_experts=num_experts, workers=workers, nocache=nocache)
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FileStreamingMoeBlock):
            mlp._blob = src


def capture_prenorm_hidden(model, input_ids: mx.array) -> mx.array:
    """跑主模型层循环但跳过最后的 model.norm,返回 last-layer hidden(norm 前)。

    HIDDEN_VARIANT=post_final_norm 时返回 norm 之后(用于消歧排错)。
    """
    inner = model.model
    h = inner.embed_tokens(input_ids)
    layers = inner.layers
    if not layers:
        return h
    cache = model.make_cache()
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx])
    ssm_mask = create_ssm_mask(h, cache[ssm_idx])
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
    if HIDDEN_VARIANT == "post_final_norm":
        h = inner.norm(h)
    return h


def greedy(model, input_ids: mx.array, n: int) -> mx.array:
    """主模型贪心生成 n 个 token,返回拼接后的完整序列(用作自投机参考)。"""
    cache = model.make_cache()
    cur = input_ids
    out = []
    for _ in range(n):
        logits = model(cur, cache=cache)
        nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        out.append(nxt)
        cur = nxt
        mx.eval(nxt)
    return mx.concatenate([input_ids] + out, axis=1)
