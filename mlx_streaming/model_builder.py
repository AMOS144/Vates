"""共享装配层:把"文件后端流式加载主模型 + 取 hidden + 贪心生成"等跨入口复用的逻辑
集中在这里,而不是藏在某个入口脚本(原 validate_mtp)里。

依赖关系:本模块位于 core(mem/expert_store/streaming_moe)与 mtp 之上,
是把它们粘合成可运行模型的装配层,被 cli/ 各入口与测试复用。

环境变量:
  MODEL          主模型路径(4-bit MLX)
  EXPERT_DIR     拆分/重量化后的 per-expert safetensors 目录
  EXPERT_SLOTS   每层常驻池容量
  EXPERT_POOL_PROFILE  每层池预算 JSON(无损省内存,可选)
  PREFETCH_PIN_PROFILE rerank 安全集 JSON；只在 dual C++ 真实区启用
  HIDDEN_VARIANT pre_final_norm(默认)| post_final_norm(排错时切换)
"""
import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.utils import load_model, load_tokenizer
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


def merged_pool_capacity(
    expert_slots: int,
    former_side_slots: int,
    *,
    zerocopy: bool,
) -> int:
    """取消逐层侧区后，把旧侧区容量合并进统一主池。"""
    return int(expert_slots) + (
        int(former_side_slots) if zerocopy else 0
    )


def merged_layer_caps(
    layer_caps: "dict[int, int] | None",
    former_side_slots: int,
    *,
    zerocopy: bool,
) -> "dict[int, int] | None":
    """保留逐层 profile，同时补回原侧区贡献的物理槽。"""
    if not layer_caps or not zerocopy:
        return layer_caps
    side = int(former_side_slots)
    return {int(layer): int(cap) + side for layer, cap in layer_caps.items()}


def apply_layer0_capacity(
    *,
    num_layers: int,
    base_capacity: int,
    layer_caps: "dict[int, int] | None",
    layer0_capacity: int,
) -> "tuple[int, dict[int, int]]":
    """Apply the no-predecessor layer-0 exception without widening all layers."""
    base = int(base_capacity)
    caps = {
        layer: min(int((layer_caps or {}).get(layer, base)), base)
        for layer in range(int(num_layers))
    }
    if caps:
        caps[0] = max(caps[0], int(layer0_capacity))
    return max(caps.values(), default=base), caps


def apply_layer_capacity_overrides(
    *,
    num_layers: int,
    base_capacity: int,
    layer_caps: "dict[int, int]",
    overrides: "dict[int, int]",
) -> "tuple[int, dict[int, int]]":
    """Expand only explicitly hot layers while preserving one pool schema."""
    invalid = sorted(
        layer for layer in overrides
        if layer < 0 or layer >= int(num_layers)
    )
    if invalid:
        raise ValueError(f"POOL_LAYER_CAP_OVERRIDES 超出层范围: {invalid}")
    caps = dict(layer_caps)
    for layer, capacity in overrides.items():
        # This knob is an expansion override for known-hot layers.  Silently
        # shrinking a profile/L0 exception would contradict that contract and
        # can reintroduce avoidable eviction churn.
        caps[int(layer)] = max(int(caps[int(layer)]), int(capacity))
    return max([int(base_capacity), *caps.values()]), caps


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


def load_prefetch_pin_profile(path: str) -> "tuple[int, dict[int, list[int]]]":
    """读取冻结的逐层 rerank 安全集；空路径返回空配置。"""
    if not path:
        return 0, {}
    with open(path, encoding="utf-8") as file:
        document = json.load(file)
    raw_layers = document.get("layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("PREFETCH_PIN_PROFILE 必须包含 layers 对象")
    output = {}
    for raw_layer, raw_value in raw_layers.items():
        layer = int(raw_layer)
        values = raw_value.get("pins") if isinstance(raw_value, dict) else raw_value
        if not isinstance(values, list):
            raise ValueError(f"pin profile layer {layer} 不是列表")
        pins = list(dict.fromkeys(int(expert) for expert in values))
        if any(expert < 0 or expert >= 512 for expert in pins):
            raise ValueError(f"pin profile layer {layer} 含越界专家")
        output[layer] = pins
    return int(document.get("required_base_capacity", 0)), output


def expanded_pin_caps(
    *,
    num_layers: int,
    base_capacity: int,
    layer_caps: "dict[int, int] | None",
    pins: "dict[int, list[int]]",
) -> "tuple[int, dict[int, int]]":
    """把 pin 作为额外物理槽加入每层 cap，未 pin 层仍保持原 cap。"""
    effective = {}
    for layer in range(int(num_layers)):
        base = min(
            int((layer_caps or {}).get(layer, base_capacity)),
            int(base_capacity),
        )
        effective[layer] = base + len(pins.get(layer, ()))
    return max(effective.values(), default=int(base_capacity)), effective


def build_streaming_model():
    """用文件后端流式 patch 加载主模型(32GB 机器装不下 41GB 非流式)。"""
    compact_marker = os.path.join(MODEL, "vates_compact_model.json")
    if os.path.exists(compact_marker):
        with open(compact_marker, encoding="utf-8") as file:
            marker = json.load(file)
        if marker.get("format") != "vates_compact_model_v1":
            raise ValueError(f"不支持的 Vates 紧凑核心标记: {compact_marker}")
        manifest_path = Path(MODEL).parent / "vates_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"紧凑核心缺少运行目录 manifest: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if (
            manifest.get("format") != "vates_runtime_bundle_v1"
            or manifest.get("status") != "verified"
        ):
            raise ValueError(f"紧凑核心尚未通过 prepare 验证: {manifest_path}")
        # 只有 prepare 生成并标记的核心允许缺少 switch_mlp；随后必由文件专家替换。
        model, _model_config = load_model(
            Path(MODEL), lazy=True, strict=False,
        )
        tok = load_tokenizer(MODEL)
    else:
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
    bits, group, proj_bits, layer_proj_bits = (
        dims["bits"], dims["group_size"], None, None)
    meta_path = os.path.join(EXPERT_DIR, "_split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        ed = meta.get("dims", {})
        bits = ed.get("bits", bits)
        group = ed.get("group_size", group)
        proj_bits = ed.get("proj_bits")
        if "per_layer_proj_bits" in ed:
            layer_proj_bits = {int(k): v for k, v in ed["per_layer_proj_bits"].items()}
    # 每层池预算 profile(pool_footprint 产出):默认从 {EXPERT_DIR}/pool_profile.json 自动启用,
    # 无损省内存(命中率/输出/吞吐不变，仅不再为低占用层预留满 capacity)。
    former_side_slots = config.pool_spec_slots()
    zerocopy = config.zerocopy_dual_source()
    direct_slots = zerocopy and config.prefetch_direct_slots()
    admission_slots = (
        config.pool_admission_slots() if direct_slots
        else config.global_staging_slots()
    )
    pool_capacity = merged_pool_capacity(
        EXPERT_SLOTS, former_side_slots, zerocopy=zerocopy,
    )
    merged_base_capacity = pool_capacity
    layer_caps = merged_layer_caps(
        load_pool_profile(EXPERT_DIR), former_side_slots,
        zerocopy=zerocopy,
    )
    pool_capacity, layer_caps = apply_layer0_capacity(
        num_layers=len(model.layers),
        base_capacity=pool_capacity,
        layer_caps=layer_caps,
        # Layer 0 has no predecessor that can hide I/O. Its modest default
        # exception is isolated to this layer; all other layers retain the
        # normal merged capacity.
        layer0_capacity=config.layer0_slots(),
    )
    pool_capacity, layer_caps = apply_layer_capacity_overrides(
        num_layers=len(model.layers),
        base_capacity=pool_capacity,
        layer_caps=layer_caps,
        overrides=config.pool_layer_cap_overrides(),
    )
    required_pin_base, prefetch_pins = load_prefetch_pin_profile(
        config.prefetch_pin_profile(),
    )
    if prefetch_pins and not config.zerocopy_dual_source():
        raise ValueError("PREFETCH_PIN_PROFILE 目前只支持 ZEROCOPY_DUAL_SOURCE=1")
    if required_pin_base and merged_base_capacity < required_pin_base:
        raise ValueError(
            "pin profile 要求统一主池基础容量"
            f">={required_pin_base}，当前为 {merged_base_capacity}",
        )
    store = FileExpertStore(EXPERT_DIR, capacity=pool_capacity, layer_caps=layer_caps)
    if zerocopy:
        # Both direct and staging modes expose one merged real pool.  In direct
        # mode prediction reserves rows in that pool and SSD writes them in
        # place. POOL_SPEC_SLOTS contributes physical rows, while the optional
        # POOL_ADMISSION_SLOTS controls how many may remain speculative.
        from mlx_streaming.core.cache.resident_pool import ResidentExpertPool
        _old = store._resident
        native_capacity, native_layer_caps = expanded_pin_caps(
            num_layers=len(model.layers),
            base_capacity=_old.capacity,
            layer_caps=_old.layer_caps,
            pins=prefetch_pins,
        )
        store._resident = ResidentExpertPool(
            native_capacity, loader=_old.loader, layer_caps=native_layer_caps,
            batch_loader=_old.batch_loader,
            stacked_batch_loader=_old.stacked_batch_loader,
            spec_slots=admission_slots,
            spec_gens=1)
    if config.stream_blob_loader():
        # blob 接入常驻池 miss-loader：复用 GPU-remap 快路径，小 EXPERT_SLOTS 即低内存。
        store._blob_loader = _make_blob_source(dims, group, bits)
    # 主动预取（native-fused-prefetch miss→hit）：opt-in（NATIVE_FUSED_PREFETCH=1）。
    # 经"promote 只写真实路由命中专家"修正后已是净正：易缓存基座上 +15.5% tok/s
    # （demand 11.86→13.70，hit 0.731→0.851，读盘 −45%；见 active-prefetch-turnaround-2026-06-17.md）。
    # 默认关只因收益依赖场景（基座可缓存性/是否磁盘受限）且只影响速度不影响质量、多占少量 staging 内存，
    # 故作 opt-in 而非默认路径，落地配方见上述报告 §6。
    if config.native_fused_prefetch() and getattr(store, "_blob_loader", None) is not None:
        try:
            import mlx_streaming.native_moe_ext  # noqa: F401  确认扩展已编译
            from mlx_streaming.core.prefetch.native_staging import NativeStagingManager
            _budget = (admission_slots if zerocopy
                       else config.stream_blob_bg_budget(default=16))
            store._staging = NativeStagingManager(store._blob_loader, budget=_budget)
        except Exception:
            store._staging = None   # 扩展不可用 → 关闭，不影响主路径
    # pin 是额外真实区槽，不计入每步 rerank width。分批同步落池，限制启动期
    # 临时张量峰值；g_real pin 账本会跨批保留并阻止后续 LFU 驱逐。
    for layer, experts in sorted(prefetch_pins.items()):
        for start in range(0, len(experts), 32):
            store.pin(layer, experts[start:start + 32])
    if config.stream_blob_bg():
        # 后台预取池预填：bg 在独立 stream 物化预测专家，promote 写进池槽（需 CROSS_LAYER_PREFETCH=1）。
        from mlx_streaming.core.prefetch.bg_prefetch import BackgroundExpertPrefetcher
        src = _make_blob_source(dims, group, bits)
        store._blob_loader = src
        store._bg = BackgroundExpertPrefetcher(
            src, window=config.stream_blob_window())
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           group, bits, proj_bits=proj_bits,
                           layer_proj_bits=layer_proj_bits)
    # Keep the real router untouched for exact model output, but optionally
    # give cross-layer prediction a narrower quantized copy.  The predictor is
    # evaluated on every routed layer and is otherwise a duplicate 2048x512
    # matmul; a 4-bit copy halves that read traffic without changing the real
    # gate or the eventual fallback-corrected MoE result.
    predictor_bits = config.prefetch_predict_gate_bits()
    if predictor_bits < 8 and config.native_fused_prefetch():
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            if not isinstance(mlp, FileStreamingMoeBlock):
                continue
            if not isinstance(gate, nn.QuantizedLinear):
                continue
            if int(gate.bits) <= predictor_bits or gate.mode != "affine":
                continue
            dense = mx.dequantize(
                gate["weight"], gate["scales"], gate["biases"],
                group_size=int(gate.group_size), bits=int(gate.bits),
                mode=gate.mode,
            )
            weight, scales, biases = mx.quantize(
                dense, group_size=int(gate.group_size),
                bits=predictor_bits, mode=gate.mode,
            )
            mx.eval(weight, scales, biases)
            predictor = nn.QuantizedLinear(
                int(dense.shape[1]), int(dense.shape[0]), bias=False,
                group_size=int(gate.group_size), bits=predictor_bits,
                mode=gate.mode,
            )
            predictor.update({
                "weight": weight,
                "scales": scales,
                "biases": biases,
            })
            object.__setattr__(mlp, "_prefetch_predict_gate", predictor)
    # A source-hidden forecast router ranks only inside the unchanged raw
    # target-gate top64 candidate set.  It is a single 2048x512 projection at
    # the early source boundary: no target attention/cache clone and no late
    # shadow gate.  Keep it separate from ``_prefetch_predict_gate`` because
    # the latter remains the authoritative candidate-membership baseline.
    rerank_router_paths = config.prefetch_rerank_router_paths()
    if rerank_router_paths:
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        rerank_weights = {}
        for path in rerank_router_paths:
            arrays = mx.load(path)
            for key, weight in arrays.items():
                if not key.startswith("layer") or not key.endswith(".weight"):
                    continue
                layer = int(key[5:-7])
                if (
                    layer in rerank_weights
                    and not config.prefetch_rerank_router_allow_override()
                ):
                    raise ValueError(f"rerank router 重复定义 layer {layer}")
                if tuple(weight.shape) != (512, dims["hidden"]):
                    raise ValueError(
                        f"rerank router layer {layer} shape={weight.shape}，"
                        f"期望 (512, {dims['hidden']})",
                    )
                rerank_weights[layer] = weight
        for layer_idx, layer in enumerate(model.layers):
            mlp = getattr(layer, "mlp", None)
            weight = rerank_weights.get(layer_idx)
            if weight is None or not isinstance(mlp, FileStreamingMoeBlock):
                continue
            router = nn.Linear(dims["hidden"], 512, bias=False)
            router.update({"weight": weight})
            object.__setattr__(mlp, "_prefetch_rerank_gate", router)
    source_correction_profile = {}
    if config.prefetch_source_correction_profile():
        from mlx_streaming.core.prefetch.source_correction_runtime import (
            load_source_correction_profile,
        )
        source_correction_profile = load_source_correction_profile(
            config.prefetch_source_correction_profile(),
        )
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                object.__setattr__(
                    mlp, "_prefetch_source_correction_profile",
                    source_correction_profile,
                )
    # Shared lazy arrays from the most recent *normal* target gate execution.
    # This is request-local state: layer 0 clears it at the next long prefill.
    gate_history = {}
    proxy_history = {}
    residual_history = {}
    route_history = {}
    from mlx_streaming.core.moe.block import FileStreamingMoeBlock
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FileStreamingMoeBlock):
            object.__setattr__(mlp, "_prefetch_gate_history", gate_history)
            object.__setattr__(mlp, "_prefetch_proxy_history", proxy_history)
            object.__setattr__(
                mlp, "_prefetch_residual_history", residual_history,
            )
            object.__setattr__(mlp, "_prefetch_route_history", route_history)
    transition_profile = {}
    if config.prefetch_transition_profile():
        from mlx_streaming.core.prefetch.source_transition_runtime import (
            load_runtime_transition_profile,
        )
        transition_profile = load_runtime_transition_profile(
            config.prefetch_transition_profile(),
        )
    if transition_profile:
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                mlp._prefetch_transition_profile = transition_profile
    transition_only_profile = {}
    if config.prefetch_transition_only_profile():
        from mlx_streaming.core.prefetch.transition_only_runtime import (
            load_transition_only_profile,
        )
        transition_only_profile = load_transition_only_profile(
            config.prefetch_transition_only_profile(),
            config.prefetch_target_layers(),
        )
        if transition_profile:
            raise ValueError(
                "PREFETCH_TRANSITION_PROFILE and transition-only are exclusive"
            )
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                object.__setattr__(
                    mlp, "_prefetch_transition_only_profile",
                    transition_only_profile,
                )
    if config.prefetch_online_transition():
        if transition_only_profile or transition_profile:
            raise ValueError("online transition is exclusive with frozen transitions")
        from mlx_streaming.core.prefetch.online_transition import (
            OnlineRouteTransition,
        )
        online_transition = OnlineRouteTransition(
            lambda target: (
                config.cross_layer_ahead_profile().get(int(target))
                or (config.cross_layer_ahead_lo()
                    if int(target) <= config.cross_layer_cutoff()
                    else config.cross_layer_ahead_hi())
            ),
        )
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                object.__setattr__(
                    mlp, "_prefetch_online_transition", online_transition,
                )
    oracle_route_data = os.environ.get("PREFETCH_ORACLE_ROUTE_DATA", "").strip()
    if oracle_route_data:
        from mlx_streaming.core.prefetch.oracle_route_replay import (
            OracleRouteReplay,
        )
        oracle_replay = OracleRouteReplay(oracle_route_data)
        object.__setattr__(model, "_prefetch_oracle_replay", oracle_replay)
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                object.__setattr__(mlp, "_prefetch_oracle_replay", oracle_replay)
    if config.prefetch_target_cache():
        raise ValueError(
            "PREFETCH_TARGET_CACHE clone/replay 已移除：它会重复目标层 "
            "attention/gate；请使用 PREFETCH_PROGRESSIVE 的相邻提前复用",
        )
    target_cache_profile = {}
    if config.prefetch_target_cache_profile():
        if not config.prefetch_target_cache():
            raise ValueError(
                "PREFETCH_TARGET_CACHE_PROFILE 需要 PREFETCH_TARGET_CACHE=1",
            )
        if transition_profile:
            raise ValueError(
                "target-cache correction 与旧 PREFETCH_TRANSITION_PROFILE 不能叠加",
            )
        from mlx_streaming.core.prefetch.target_cache_correction_runtime import (
            load_target_cache_profile,
        )
        target_cache_profile = load_target_cache_profile(
            config.prefetch_target_cache_profile(),
        )
        expected_targets = set(range(1, len(model.layers)))
        if set(target_cache_profile) != expected_targets:
            missing = sorted(expected_targets - set(target_cache_profile))
            extra = sorted(set(target_cache_profile) - expected_targets)
            raise ValueError(
                "target-cache profile 必须覆盖每个目标层 1..last: "
                f"missing={missing}, extra={extra}",
            )
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                object.__setattr__(
                    mlp, "_prefetch_target_cache_profile", target_cache_profile,
                )
    # 双源双缓冲：构造一个共享 VirtualPool（gen 跨层全局、每前向 +1）挂到每个流式 MoE 块。
    if config.zerocopy_dual_source() and getattr(store, "_staging", None) is not None:
        from mlx_streaming.core.cache.virtual_pool import VirtualPool
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        # 双源模式仍需 ahead 调度：block._native_fused_prefetch 靠 target_for 选目标层，
        # 不传调度参数会让 target_for 恒返回 0（_num_layers=0）→ 预取全跳过、侧区永远空。
        _vpool = VirtualPool(store._resident, store._staging, admission_slots,
                             num_layers=len(model.layers),
                             cutoff=config.cross_layer_cutoff(),
                             ahead_lo=config.cross_layer_ahead_lo(),
                             ahead_hi=config.cross_layer_ahead_hi(),
                             ahead_profile=config.cross_layer_ahead_profile())
        store._vpool = _vpool
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                mlp._vpool = _vpool
    if config.prefetch_progressive():
        mode = config.prefetch_progressive_mode()
        progressive_targets = config.prefetch_progressive_target_layers()
        configured_targets = list(
            range(1, len(model.layers))
            if progressive_targets is None
            else sorted(progressive_targets)
        )
        invalid_targets = [
            target for target in configured_targets
            if target < 1 or target >= len(model.layers)
        ]
        if invalid_targets:
            raise ValueError(
                "PREFETCH_PROGRESSIVE_TARGET_LAYERS 超出目标层范围: "
                f"{invalid_targets}",
            )
        validated_targets = [
            target for target in configured_targets
            if config.prefetch_progressive_for(target)
        ]
        cores = {
            config.prefetch_progressive_core_for(target)
            for target in validated_targets
        }
        if mode not in {"k1", "k3"}:
            raise ValueError("PREFETCH_PROGRESSIVE_MODE 必须是 k1 或 k3")
        if config.prefetch_target_cache():
            raise ValueError(
                "PREFETCH_PROGRESSIVE 与 callback 前的 PREFETCH_TARGET_CACHE 互斥",
            )
        if not config.zerocopy_dual_source() or getattr(store, "_staging", None) is None:
            raise ValueError(
                "PREFETCH_PROGRESSIVE 需要可用的 ZEROCOPY_DUAL_SOURCE native staging",
            )
        max_core = 26 if mode == "k3" else 15
        if any(not 1 <= core <= max_core for core in cores):
            raise ValueError(
                "PREFETCH_PROGRESSIVE_CORE 超过模式上限："
                f"mode={mode}, legal=[1,{max_core}]",
            )
        physical_prefetch_slots = (
            admission_slots if direct_slots
            else config.global_staging_slots()
        )
        if config.prefetch_progressive_max_width() > physical_prefetch_slots:
            raise ValueError(
                "PREFETCH_PROGRESSIVE_MAX_WIDTH 不能超过物理预取槽位",
            )
    # 主动预取（非 zerocopy）：挂 per-layer ahead 调度器 vpool（cutoff），让晚层预读更早发起。
    if (config.native_fused_prefetch() and not config.zerocopy_dual_source()
            and getattr(store, "_staging", None) is not None):
        from mlx_streaming.core.cache.virtual_pool import VirtualPool
        from mlx_streaming.core.moe.block import FileStreamingMoeBlock
        _sched = VirtualPool(num_layers=len(model.layers),
                             cutoff=config.cross_layer_cutoff(),
                             ahead_lo=config.cross_layer_ahead_lo(),
                             ahead_hi=config.cross_layer_ahead_hi(),
                             ahead_profile=config.cross_layer_ahead_profile())
        for layer in model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, FileStreamingMoeBlock):
                mlp._vpool = _sched
    if config.stream_blob():
        _attach_blob_source(model, dims, group, bits)
    # KV 量化(IsoQuant K4/V3 + SO(4) 旋转):仅作用于 12 个全注意力层,128k KV 3.0→~0.68 GiB。
    if config.kv_quant():
        from mlx_streaming.core.cache.kv_quant_patch import patch_kv_quant
        patch_kv_quant(model,
                       group_size=config.kv_group_size(),
                       k_bits=config.kv_k_bits(),
                       v_bits=config.kv_v_bits(),
                       rotate=config.kv_rotate(),
                       seed=config.kv_rot_seed())
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
