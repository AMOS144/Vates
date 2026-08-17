"""集中配置：所有运行时开关从环境变量读取（env 名与默认值与历史一致，便于现有跑法）。

设计目的：把原先散落在 core/ 各文件的 `os.environ.get` 全部收口到这里，单点可查、
避免默认值漂移。读取均为运行时（每次调用读 env），与原行为一致——测试/probe 用
monkeypatch env 仍生效，热路径每 token 读 env 的成本与原先相同。

对「同一 env 名在不同调用方有不同默认」的项（如 CROSS_LAYER_PREFETCH_AHEAD：hook=0、
native 预取=1），accessor 暴露 `default` 参数，由调用方传入，绝不擅自统一。
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar


_prefetch_exact_no_io_override = ContextVar(
    "prefetch_exact_no_io_override", default=None,
)
_demand_async_override = ContextVar("demand_async_override", default=None)


def _b(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _i(name: str, default) -> int:
    return int(os.environ.get(name, str(default)))


def _f(name: str, default) -> float:
    return float(os.environ.get(name, str(default)))


def _s(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ============================ 模型 / 目录 ============================
# 默认放持久目录 models/（不要放 /tmp，会被系统清空）。运行从仓库根目录起，相对路径即可。
def model_path() -> str: return _s("MODEL", "models/qwen3_next_80b_4bit")
def qn_config() -> str: return _s("QN_CONFIG", "models/qwen3_next_80b_4bit/config.json")
def mtp_out() -> str: return _s("MTP_OUT", "models/qn_mtp_weights.safetensors")
def mtp_bits() -> int: return max(2, min(8, _i("MTP_BITS", 4)))
def mtp_group_size() -> int: return max(32, _i("MTP_GROUP_SIZE", 64))
def mtp_stream_experts() -> bool: return _b("MTP_STREAM_EXPERTS", "0")
def mtp_expert_dir() -> str: return _s("MTP_EXPERT_DIR", "models/qn_mtp_experts_4bit_g64")
def mtp_expert_slots() -> int: return max(10, _i("MTP_EXPERT_SLOTS", 32))
def expert_dir(default: str = "models/qwen3_next_experts_4bit_g64") -> str: return _s("EXPERT_DIR", default)
def expert_slots() -> int: return _i("EXPERT_SLOTS", 64)
# 长期运行内存防御:封顶 MLX 可回收缓冲(默认 1GB),防长会话缓存膨胀;
# 双源侧区池的专家 buffer 走 C++ owned pool、不经 MLX 缓冲缓存,故 1GB 缓冲复用额度已够,
# 再大只是白占常驻。wired limit 默认 0=关(opt-in),设 >0 则 wire 该 GB 数的 GPU 缓冲防 macOS
# 压缩器,须 < 系统建议工作集(本机 26.8GB)。
def mlx_cache_limit_gb() -> float: return _f("MLX_CACHE_LIMIT_GB", 1.0)
def mlx_wired_limit_gb() -> float: return _f("MLX_WIRED_LIMIT_GB", 0.0)
def expert_pool_profile() -> str: return _s("EXPERT_POOL_PROFILE", "")
def hidden_variant() -> str: return _s("HIDDEN_VARIANT", "pre_final_norm")
def blob_dir() -> str: return _s("BLOB_DIR", "")
def compute_buffer_dir() -> str: return _s("COMPUTE_BUFFER_DIR", "")


# ============================ MoE 热路径 ============================
def resident_pool_enabled() -> bool: return _b("RESIDENT_POOL", "1")
def gpu_remap_enabled() -> bool: return _b("GPU_REMAP", "1")
# 实验：让 verify(seq>1)也走 GPU 侧 slot 重映射(acquire_gpu),消掉 host 路径每层 .tolist() 栅栏。
def verify_gpu_remap() -> bool: return _b("VERIFY_GPU_REMAP", "0")
# A2：GPU 重映射路径下 promote 是否用 GPU membership 现算 used 过滤假阳性（默认开；回退设 0）。
def gpu_remap_promote_filter() -> bool: return _b("GPU_REMAP_PROMOTE_FILTER", "1")
def moe_topk_override() -> "str | None": return os.environ.get("MOE_TOPK_OVERRIDE")
def eager_expert_load() -> bool: return _b("EAGER_EXPERT_LOAD", "0")


# ============================ Expert-major prefill ============================
# The production algorithm and its tuned constants are intentionally fixed in
# core/attention/expert_major.py and core/moe/block.py.  This clean branch has
# no environment-selectable fallback implementations.
def expert_major_mem_trace() -> bool:
    return _b("EXPERT_MAJOR_MEM_TRACE", "0")


# ============================ 缓存 / 驱逐 ============================
def evict_policy() -> str: return _s("EVICT_POLICY", "lfu")
# 专家读盘是否绕过 OS page cache(F_NOCACHE):默认开,保证基准每次都是真实 NVMe 读、
# 结果可复现(不被页缓存冷热污染);EXPERT_NOCACHE=0 恢复 mmap/page-cache(重复跑更快但飘)。
def expert_nocache() -> bool: return _b("EXPERT_NOCACHE", "1")
def lfu_decay_interval() -> int: return _i("LFU_DECAY_INTERVAL", 0)
def expert_bundle() -> bool: return _b("EXPERT_BUNDLE", "0")
def expert_bundle_dir(default: str) -> str: return _s("EXPERT_BUNDLE_DIR", default)
def expert_bundle_cache() -> int: return _i("EXPERT_BUNDLE_CACHE", 4)
def expert_stack_cache() -> int: return _i("EXPERT_STACK_CACHE", 0)
def async_prefetch() -> bool: return _b("ASYNC_PREFETCH", "0")
def prefetch_buffer_experts() -> int: return _i("PREFETCH_BUFFER_EXPERTS", 2048)


# ============================ 自定义 Metal 算子 ============================
def custom_qproj() -> bool: return _b("CUSTOM_QPROJ", "0")
def custom_qproj_bits() -> int: return _i("CUSTOM_QPROJ_BITS", 6)
def custom_qproj_targets() -> str: return _s("CUSTOM_QPROJ_TARGETS", "gate,up")
def custom_qproj_max_seq() -> int: return _i("CUSTOM_QPROJ_MAX_SEQ", 4)
def custom_qproj_tile() -> int: return _i("CUSTOM_QPROJ_TILE", 4)
def custom_fused_moe() -> bool: return _b("CUSTOM_FUSED_MOE", "0")
def custom_fused_moe_bits() -> int: return _i("CUSTOM_FUSED_MOE_BITS", 6)
def custom_fused_moe_lanes() -> int: return _i("CUSTOM_FUSED_MOE_LANES", 8)
def custom_fused_moe_block() -> int: return _i("CUSTOM_FUSED_MOE_BLOCK", 256)
def custom_fused_moe_max_seq() -> int: return _i("CUSTOM_FUSED_MOE_MAX_SEQ", 4)
# Single-token decode can consume the async demand primitive's event-ordered
# final remap directly and run one standard MoE pass.  MTP batches retain the
# split hit/miss overlap path, where waiting before all useful compute is less
# attractive.
def demand_async_single_pass() -> bool: return _b("DEMAND_ASYNC_SINGLE_PASS", "0")
def demand_async_single_pass_for(layer: int) -> bool:
    selected = parse_layers_env("DEMAND_ASYNC_SINGLE_PASS_LAYERS")
    return demand_async_single_pass() and (
        selected is None or int(layer) in selected
    )


# ============================ native MoE 后端 ============================
def native_moe() -> bool: return _b("NATIVE_MOE", "0")
def native_moe_mlx_op() -> bool: return _b("NATIVE_MOE_MLX_OP", "0")
def native_moe_synthetic() -> bool: return _b("NATIVE_MOE_SYNTHETIC", "0")
def native_moe_slot_pool() -> bool: return _b("NATIVE_MOE_SLOT_POOL", "0")
def native_moe_raise() -> bool: return _b("NATIVE_MOE_RAISE", "0")
def native_moe_slot_cap() -> int: return _i("NATIVE_MOE_SLOT_CAP", 96)
def native_moe_stage_cache() -> bool: return _b("NATIVE_MOE_STAGE_CACHE", "1")
def native_moe_stage_cache_experts() -> int: return _i("NATIVE_MOE_STAGE_CACHE_EXPERTS", 96)
def native_moe_stage_bundle_cache() -> bool: return _b("NATIVE_MOE_STAGE_BUNDLE_CACHE", "1")
def native_moe_stage_cache_bundles() -> int: return _i("NATIVE_MOE_STAGE_CACHE_BUNDLES", 16)
def native_moe_stage_prefetch() -> bool: return _b("NATIVE_MOE_STAGE_PREFETCH", "0")


# ============================ 流式 blob ============================
def stream_blob() -> bool: return _b("STREAM_BLOB", "0")
def stream_blob_loader() -> bool: return _b("STREAM_BLOB_LOADER", "0")
def stream_blob_bg() -> bool: return _b("STREAM_BLOB_BG", "0")
def stream_blob_workers() -> int: return _i("STREAM_BLOB_WORKERS", 8)
def stream_blob_window() -> int: return _i("STREAM_BLOB_WINDOW", 3)
def stream_blob_nocache(default: str = "1") -> bool: return _b("STREAM_BLOB_NOCACHE", default)
# 默认 12:双源侧区(cap32+侧区32)K=3 MTP 甜点扫描结果(REPEAT=2)。瓶颈是 SSD 读盘时序而非容量,
# 加大 budget 只会灌满共享读队列→预取到得更晚(timing miss↑)。W=24/B=12 在 hit(0.901)/disk(3874)/
# tok·s(11.61)/确定性(nmis=2)全面优于旧 W=32/B=16(10.0 tok·s、nmis=43),tok·s +16%。
def stream_blob_bg_budget(default: int = 12) -> int: return _i("STREAM_BLOB_BG_BUDGET", default)
# 按需 miss 批量并行读：把本层所有 miss 专家收成一批，一次 load_experts(8-worker 并行 pread)，
# 取代逐专家串行 pread（实测串行 6.4GB/s vs 并行 8 路 22GB/s）。仅影响读取调度，结果等价。
# 默认开：剖析实测 decode 时间 ~92% 花在每层 host 回退路径，其中字节物化是大头；批量堆叠
# (每段一次 mx.array 取代逐专家 6N 次构造)在 cap=12 把 decode 从 ~466ms 降到 ~408ms(~12%)。
def batch_miss_read() -> bool: return _b("BATCH_MISS_READ", "1")
# demand miss 走 C++ 原生物化(load_experts_native)：blob_load 把字节 pread 直进 MLX 数组、
# eval 时在 C++ 跑、绕 GIL、惰性。比 stacked 再快 ~6%(把物化从 acquire 挪到 eval)。
# 默认关(opt-in)：实测它偶发把末尾 token 推进双稳态慢挡(2-3s/token 的"悬崖")，
# 其 lazy 物化推高内存压力,稳定性不如 stacked(纯 numpy 同步、4 次跑 0 慢挡)。仅在确认
# 环境内存充裕、需要那 6% 时手动开。不消费 async prefetch buffer，故仅在 not async_prefetch 接入。
def native_demand_loader() -> bool: return _b("NATIVE_DEMAND_LOADER", "0")
def staging_ring() -> int: return _i("STAGING_RING", 2)  # 安全下限=2：MTP 每步 verify+replay 各对同层 submit 一次
def global_staging_slots() -> int: return max(1, _i("GLOBAL_STAGING_SLOTS", 24))
# Direct-slot mode reserves final rows in the one merged main pool.  SSD writes
# them in place and publishes the same GPU expert->slot table only after every
# segment is complete; there is no per-layer side ownership/table.
# Global staging remains available with PREFETCH_DIRECT_SLOTS=0.
def prefetch_direct_slots() -> bool: return _b("PREFETCH_DIRECT_SLOTS", "1")
def prefetch_partial_projections() -> bool:
    return _b("PREFETCH_PARTIAL_PROJECTIONS", "0")
def prefetch_partial_demand_tail() -> bool:
    """Keep false-positive prefixes partial; complete only demanded routes."""
    return _b("PREFETCH_PARTIAL_DEMAND_TAIL", "0")
# L0 has no preceding decoder layer that can hide an expert prefetch.  Allow a
# small per-model exception without multiplying every layer's pool footprint.
def layer0_slots(default: int = 256) -> int:
    return min(512, max(1, _i("LAYER0_SLOTS", default)))
# Event-gated demand keeps route materialization and expert->slot remap off the
# Python/main thread. It is meaningful only with the directly addressable pool.
def demand_async() -> bool:
    override = _demand_async_override.get()
    return _b("DEMAND_ASYNC", "0") if override is None else bool(override)


@contextmanager
def override_demand_async(enabled: bool):
    token = _demand_async_override.set(bool(enabled))
    try:
        yield
    finally:
        _demand_async_override.reset(token)
# ``mx.async_eval(entry_local)`` does not guarantee that the entry remap ends
# its Metal command buffer before the event-gated final mapping is assembled.
# On a long-lived decode graph that can let later work cross the CPU completion
# callback which publishes demand-loaded pool rows, corrupting the target
# model's cache.  The native submission boundary is exact; keep the evaluator
# path available only as an explicit diagnostic opt-in.
def demand_async_python_submit() -> bool: return _b("DEMAND_ASYNC_PY_SUBMIT", "0")
def demand_async_eval_boundary() -> bool:
    return _b("DEMAND_ASYNC_EVAL_BOUNDARY", "0")
# Evaluate the source predictor as an extra input of the source layer's native
# demand primitive.  Its completion handler submits the target reads after
# resolving source demand, removing a separate Metal completion callback
# without moving prediction later than source-MoE compute.
def prefetch_fuse_with_demand() -> bool:
    return _b("PREFETCH_FUSE_WITH_DEMAND", "0")
# Standard-MLX sparse correction budget for event-gated demand.  Zero keeps
# the mature one-pass path. Positive values compute resident hits immediately
# and recompute only this many missing route positions after SSD completion;
# overflow is detected lazily and causes an exact whole-verify replay.
def demand_sparse_miss_budget() -> int:
    return max(0, _i("DEMAND_SPARSE_MISS_BUDGET", 0))
def demand_sparse_miss_budget_overrides() -> "dict[int, int]":
    """Parse ``layer[-layer]:budget`` sparse-correction overrides."""
    spec = _s("DEMAND_SPARSE_MISS_BUDGET_OVERRIDES", "").strip()
    if not spec:
        return {}
    output: "dict[int, int]" = {}
    try:
        for item in spec.split(","):
            layer_spec, budget_spec = item.strip().split(":", 1)
            budget = int(budget_spec)
            if budget < 0:
                raise ValueError
            if "-" in layer_spec:
                start, end = (int(value) for value in layer_spec.split("-", 1))
                layers = range(start, end + 1)
            else:
                layers = (int(layer_spec),)
            for layer in layers:
                if layer < 0:
                    raise ValueError
                output[layer] = budget
    except ValueError as error:
        raise ValueError(
            "DEMAND_SPARSE_MISS_BUDGET_OVERRIDES 必须是 layer[-layer]:budget 列表",
        ) from error
    return output
def demand_sparse_miss_budget_by_sequence() -> "dict[int, int]":
    """Parse ``sequence_length:budget`` fixed-tail overrides."""
    spec = _s("DEMAND_SPARSE_MISS_BUDGET_BY_SEQ", "").strip()
    if not spec:
        return {}
    output: "dict[int, int]" = {}
    try:
        for item in spec.split(","):
            sequence_spec, budget_spec = item.strip().split(":", 1)
            sequence = int(sequence_spec)
            budget = int(budget_spec)
            if sequence <= 0 or budget < 0:
                raise ValueError
            output[sequence] = budget
    except ValueError as error:
        raise ValueError(
            "DEMAND_SPARSE_MISS_BUDGET_BY_SEQ 必须是 sequence:budget 列表",
        ) from error
    return output
def demand_sparse_miss_budget_for(
    layer: int, sequence_length: "int | None" = None,
) -> int:
    layer_overrides = demand_sparse_miss_budget_overrides()
    layer_budget = layer_overrides.get(int(layer), demand_sparse_miss_budget())
    if sequence_length is None:
        return layer_budget
    sequence_budget = demand_sparse_miss_budget_by_sequence().get(
        int(sequence_length), layer_budget,
    )
    if _b("DEMAND_SPARSE_SEQ_LAYER_MAX", "0") and int(layer) in layer_overrides:
        return max(sequence_budget, layer_budget)
    return sequence_budget
def demand_sparse_enabled() -> bool:
    return demand_sparse_miss_budget() > 0 or any(
        budget > 0 for budget in demand_sparse_miss_budget_overrides().values()
    ) or any(budget > 0 for budget in demand_sparse_miss_budget_by_sequence().values())
def demand_sparse_partition() -> bool:
    return _b("DEMAND_SPARSE_PARTITION", "0")
def demand_sparse_local_correction() -> bool:
    """Repair rare fixed-tail overflows in-layer instead of replaying verify."""
    return _b("DEMAND_SPARSE_LOCAL_CORRECTION", "0")
def demand_sparse_suffix_replay() -> bool:
    """Replay only from the first overflowed decoder layer."""
    return _b("DEMAND_SPARSE_SUFFIX_REPLAY", "0")
def demand_sparse_hit_aux_stream() -> bool:
    """Submit the entry-ready half on an independent Metal stream."""
    return _b("DEMAND_SPARSE_HIT_AUX_STREAM", "0")
# Run the always-resident shared expert on a separate device stream after the
# prefetch graph is attached but before routed-expert demand can wait on SSD.
def shared_expert_overlap() -> bool: return _b("SHARED_EXPERT_OVERLAP", "1")
def global_staging_banks() -> int:
    # Progressive has an immutable early core plus one refinement submission;
    # keep enough shared banks for overlapping ahead=3 targets without going
    # back to per-layer buffers.
    default = 8 if prefetch_progressive() else 2
    return max(2, _i("GLOBAL_STAGING_BANKS", default))
def prefetch_staging_late_promote() -> bool:
    """Keep a prefetch that completed only after its target demand.

    Disabled by default: demand has already loaded the routed experts at that
    point, so admitting an unverified late candidate can only duplicate work
    or evict a useful resident row.
    """
    return _b("PREFETCH_STAGING_LATE_PROMOTE", "0")
def staging_pread_parallel() -> bool:
    # Multi-step early/refinement must leave the Metal completion thread and
    # use the priority-aware background queue. Legacy single-stage staging
    # keeps its conservative default; explicit env values override both.
    return _b("STAGING_PREAD_PARALLEL", "1" if prefetch_progressive() else "0")
def stg_verify() -> bool: return _b("STG_VERIFY", "0")  # 诊断:acquire_gpu 命中后池槽字节真值校验,默认关、对主路径零影响
def stream_blob_prefetch_budget(default: int) -> int: return _i("STREAM_BLOB_PREFETCH_BUDGET", default)


# ============================ 跨层 / 预取 ============================
# 注意 AHEAD 默认按调用方传：跨层 hook=0、native 预取=1。
def cross_layer_prefetch() -> bool: return _b("CROSS_LAYER_PREFETCH", "0")
def cross_layer_ahead(default: int = 0) -> int: return _i("CROSS_LAYER_PREFETCH_AHEAD", default)
def cross_layer_mult() -> int: return max(1, _i("CROSS_LAYER_PREFETCH_MULT", 2))
# 预测宽度（方案B）：预测 top-N 候选用于"减常驻"，N 大→recall 高且不占内存（只是 gate argpartition）。
# 真正占 staging 的是过滤常驻后、按分截断到 staging budget 的缺口子集。
# 默认 24:双源侧区 K=3 甜点扫描(REPEAT=2)。realized hit 在 W≥24 就到顶(~0.90),再加宽只抬"名义
# recall"——多出的候选在 SSD 时序上落不进侧区且抢带宽,W=48→hit 0.85、W=64→0.75、tok·s 一路下滑。
# 收窄到刚好覆盖可及工作集(W=24)最快最省。详见 benchmarks/reports/prefetch-width-budget-sweep-2026-07-01.md。
def cross_layer_predict_width() -> int: return _i("CROSS_LAYER_PREDICT_WIDTH", 24)
# per-layer 自适应 ahead：早层用小 ahead 保召回、晚层用大 ahead 保时序（默认来自 Phase 0 实测）。
def cross_layer_cutoff() -> int: return _i("CROSS_LAYER_CUTOFF", 6)        # 切点：层号 <cutoff 用 lo，否则 hi
def cross_layer_ahead_lo() -> int: return _i("CROSS_LAYER_AHEAD_LO", 1)    # 早层 ahead（保召回）
def cross_layer_ahead_hi() -> int: return _i("CROSS_LAYER_AHEAD_HI", 3)    # 晚层 ahead（保时序）
def cross_layer_ahead_profile() -> "dict[int, int]":
    """Parse ``target[-target]:ahead`` overrides separated by commas."""
    spec = _s("CROSS_LAYER_AHEAD_PROFILE", "").strip()
    if not spec:
        return {}
    out: "dict[int, int]" = {}
    for item in spec.split(","):
        layer_spec, ahead_spec = item.strip().split(":", 1)
        ahead = int(ahead_spec)
        if "-" in layer_spec:
            start, end = (int(value) for value in layer_spec.split("-", 1))
            targets = range(start, end + 1)
        else:
            targets = (int(layer_spec),)
        for target in targets:
            if target < 1 or ahead < 1 or ahead > target:
                raise ValueError(
                    f"invalid target:ahead override {target}:{ahead}",
                )
            out[target] = ahead
    return out


def prefetch_target_layers() -> "set[int] | None":
    """Optional ordinary-prefetch target allowlist; unset keeps every layer."""
    return parse_layers_env("PREFETCH_TARGET_LAYERS")
def predict_use_x() -> bool: return _b("PREDICT_USE_X", "1")  # 默认用本层 MoE 输入 x（更新鲜，+3.6pp recall）；=0 回退旧 norm 路径
def predict_agg() -> str: return _s("PREDICT_AGG", "max")  # K+1 token 聚合：max|mean|union
def predict_union_k() -> int: return _i("PREDICT_UNION_K", 8)  # union 时每 token 取的 top-k（控候选数）
# 无训练预取重排：默认关闭，便于与现有固定 width 路径做严格 A/B。
def prefetch_rerank() -> str: return _s("PREFETCH_RERANK", "off").strip().lower()
# 候选宽度与最终 side-region 输出宽度是两个独立约束。top64 指每个
# token 独立取 64；最终输出仍由物理 side budget 和 width policy 截断。
def prefetch_rerank_candidate_width() -> int: return max(1, _i("PREFETCH_RERANK_CANDIDATE_WIDTH", 64))
# Logical rerank output is deliberately independent of the physical side-pool
# capacity.  A 32-row side pool is useful for persistence, but treating all 32
# rows as one occurrence's prediction budget causes false-positive reads and
# LFU churn.  Callers provide the mode-specific production default (15 for a
# single token, 26 for K=3); the env remains available for controlled sweeps.
def prefetch_rerank_max_width(default: int) -> int: return max(1, _i("PREFETCH_RERANK_MAX_WIDTH", default))
def prefetch_rerank_max_width_overrides() -> "dict[int, int]":
    """Parse per-target logical width caps (``layer[-layer]:width``)."""
    spec = _s("PREFETCH_RERANK_MAX_WIDTH_OVERRIDES", "").strip()
    if not spec:
        return {}
    output: "dict[int, int]" = {}
    try:
        for item in spec.split(","):
            layer_spec, width_spec = item.strip().split(":", 1)
            width = int(width_spec)
            if width < 1:
                raise ValueError
            if "-" in layer_spec:
                start, end = (int(value) for value in layer_spec.split("-", 1))
                layers = range(start, end + 1)
            else:
                layers = (int(layer_spec),)
            for layer in layers:
                if layer < 1:
                    raise ValueError
                output[layer] = width
    except ValueError as error:
        raise ValueError(
            "PREFETCH_RERANK_MAX_WIDTH_OVERRIDES 必须是 layer[-layer]:width 列表",
        ) from error
    return output
def prefetch_rerank_max_width_for(target_layer: int, default: int) -> int:
    return prefetch_rerank_max_width_overrides().get(
        int(target_layer), prefetch_rerank_max_width(default),
    )
# K=3 实测 0.97 在 width、命中与吞吐间最均衡；0.99 过宽并回退吞吐。
def prefetch_rerank_mass() -> float: return max(0.0, min(1.0, _f("PREFETCH_RERANK_MASS", 0.97)))
def prefetch_rerank_min_width(default: int) -> int: return max(1, _i("PREFETCH_RERANK_MIN_WIDTH", default))
def prefetch_rerank_backfill_extra() -> int:
    """Additional nonresident candidates after resident filtering."""
    return max(0, _i("PREFETCH_RERANK_BACKFILL_EXTRA", 0))
def prefetch_rerank_backfill_extra_for(target_layer: int) -> int:
    extra = prefetch_rerank_backfill_extra()
    layers = parse_layers_env("PREFETCH_RERANK_BACKFILL_LAYERS")
    if layers is not None and int(target_layer) not in layers:
        return 0
    return extra
def prefetch_rerank_width_policy() -> str: return _s("PREFETCH_RERANK_WIDTH_POLICY", "mass").strip().lower()
def prefetch_rerank_ranking_policy() -> str: return _s("PREFETCH_RERANK_RANKING_POLICY", "noisy_or").strip().lower()
def prefetch_rerank_ranking_policy_overrides() -> "dict[int, str]":
    spec = _s("PREFETCH_RERANK_RANKING_POLICY_OVERRIDES", "").strip()
    if not spec:
        return {}
    allowed = {"max", "noisy_or", "topk_union", "topk_union_fast"}
    output: "dict[int, str]" = {}
    for item in spec.split(","):
        layer_spec, policy_spec = item.strip().split(":", 1)
        policy = policy_spec.strip().lower()
        if policy not in allowed:
            raise ValueError(f"未知 rerank ranking policy: {policy}")
        if "-" in layer_spec:
            start, end = (int(value) for value in layer_spec.split("-", 1))
            targets = range(start, end + 1)
        else:
            targets = (int(layer_spec),)
        for target in targets:
            if target < 1:
                raise ValueError(f"invalid rerank target layer {target}")
            output[target] = policy
    return output
def prefetch_rerank_union_margin() -> int: return max(0, _i("PREFETCH_RERANK_UNION_MARGIN", 4))
def prefetch_rerank_union_margin_overrides() -> "dict[int, int]":
    """Parse ``target[-target]:margin`` overrides separated by commas."""
    spec = _s("PREFETCH_RERANK_UNION_MARGIN_OVERRIDES", "").strip()
    if not spec:
        return {}
    output: "dict[int, int]" = {}
    for item in spec.split(","):
        layer_spec, margin_spec = item.strip().split(":", 1)
        margin = int(margin_spec)
        if margin < 0:
            raise ValueError("rerank union margin 不能为负")
        if "-" in layer_spec:
            start, end = (int(value) for value in layer_spec.split("-", 1))
            targets = range(start, end + 1)
        else:
            targets = (int(layer_spec),)
        for target in targets:
            if target < 1:
                raise ValueError(f"invalid rerank target layer {target}")
            output[target] = margin
    return output
def prefetch_rerank_prof() -> bool: return _b("PREFETCH_RERANK_PROF", "0")
# Optional source-hidden forecast projections used only to rank members of the
# frozen raw target-gate top64. Multiple safetensors files may be comma-separated
# (for example layers 1..12 and 13..47); later files may not redefine a layer.
def prefetch_rerank_router_paths() -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in _s("PREFETCH_RERANK_ROUTER_PATHS", "").split(",")
        if value.strip()
    )
def prefetch_rerank_router_allow_override() -> bool:
    """Allow a later checkpoint to replace selected per-layer weights."""
    return _b("PREFETCH_RERANK_ROUTER_ALLOW_OVERRIDE", "0")
# Comma-separated source-router correction checkpoints or a directory holding
# layerXX.npz files.  The correction consumes the source gate logits already
# computed by the main model and therefore adds no target attention/gate pass.
def prefetch_source_correction_profile() -> str:
    return _s("PREFETCH_SOURCE_CORRECTION_PROFILE", "").strip()
def prefetch_rerank_history_beta_overrides() -> "dict[int, float]":
    """Parse per-target previous-real-gate blend coefficients."""
    spec = _s("PREFETCH_RERANK_HISTORY_BETA_OVERRIDES", "").strip()
    if not spec:
        return {}
    output: "dict[int, float]" = {}
    for item in spec.split(","):
        layer_spec, beta_spec = item.strip().split(":", 1)
        beta = float(beta_spec)
        if beta < 0:
            raise ValueError("rerank history beta 不能为负")
        if "-" in layer_spec:
            start, end = (int(value) for value in layer_spec.split("-", 1))
            targets = range(start, end + 1)
        else:
            targets = (int(layer_spec),)
        for target in targets:
            if target < 1:
                raise ValueError(f"invalid rerank target layer {target}")
            output[target] = beta
    return output
def prefetch_rerank_residual_scale_overrides() -> "dict[int, float]":
    """Parse per-target previous ``actual - proxy`` correction scales."""
    spec = _s("PREFETCH_RERANK_RESIDUAL_SCALE_OVERRIDES", "").strip()
    if not spec:
        return {}
    output: "dict[int, float]" = {}
    for item in spec.split(","):
        layer_spec, scale_spec = item.strip().split(":", 1)
        scale = float(scale_spec)
        if scale < 0:
            raise ValueError("rerank residual scale 不能为负")
        if "-" in layer_spec:
            start, end = (int(value) for value in layer_spec.split("-", 1))
            targets = range(start, end + 1)
        else:
            targets = (int(layer_spec),)
        for target in targets:
            if target < 1:
                raise ValueError(f"invalid rerank target layer {target}")
            output[target] = scale
    return output
def prefetch_rerank_residual_decay() -> float:
    return max(0.0, min(0.999, _f("PREFETCH_RERANK_RESIDUAL_DECAY", 0.0)))
# The real router remains at the model's configured precision.  This controls
# only the extra cross-layer predictor copy, so lowering it cannot change model
# logits; it trades predictor ranking fidelity for substantially less gate
# bandwidth.  8 preserves the loaded Qwen checkpoint exactly.
def prefetch_predict_gate_bits() -> int: return max(2, min(8, _i("PREFETCH_PREDICT_GATE_BITS", 8)))
# Evaluate the predictor/callback on the VirtualPool's auxiliary Metal stream.
# Demand already joins pending rows at the target boundary, so the source
# layer need not serialize its own expert compute behind this speculative work.
def prefetch_async_predict() -> bool:
    return _b("PREFETCH_ASYNC_PREDICT", "0")
# Submit an adjacent target prediction from the completed source decoder
# output. Its gate runs beside the target attention/GDN, retaining an I/O
# window while using a fresher hidden state than the source MoE-entry proxy.
def prefetch_post_moe() -> bool: return _b("PREFETCH_POST_MOE", "0")
# Experimental two-stage mode: retain the ordinary early callback and use the
# fresher post-MoE prediction only as a bounded second fill.  The default keeps
# the historical replacement semantics.
def prefetch_post_moe_refinement() -> bool:
    return _b("PREFETCH_POST_MOE_REFINEMENT", "0")
def prefetch_post_moe_refine_width() -> int:
    return max(1, _i("PREFETCH_POST_MOE_REFINE_WIDTH", 1))
def prefetch_post_moe_refinement_layers() -> "set[int] | None":
    """Optional targets for the costly late refinement submission."""
    return parse_layers_env("PREFETCH_POST_MOE_REFINEMENT_LAYERS")
def prefetch_post_moe_replacement_layers() -> "set[int] | None":
    """Optional targets whose single early prediction moves post-MoE.

    Unlike refinement, replacement does not add a second predictor.  Targets
    outside this set retain their ordinary MoE-entry submission.  An unset
    value preserves the historical all-target post-MoE behaviour.
    """
    return parse_layers_env("PREFETCH_POST_MOE_REPLACEMENT_LAYERS")
def prefetch_multistage_early() -> bool:
    """Add one T-3 prediction before the ordinary T-2 submission."""
    return _b("PREFETCH_MULTISTAGE_EARLY", "0")
def prefetch_multistage_early_ahead() -> int:
    return max(2, _i("PREFETCH_MULTISTAGE_EARLY_AHEAD", 3))
def prefetch_multistage_early_layers() -> "set[int] | None":
    return parse_layers_env("PREFETCH_MULTISTAGE_EARLY_LAYERS")
def prefetch_multistage_history() -> bool:
    """Use the previous exact target route for T-3 instead of another gate."""
    return _b("PREFETCH_MULTISTAGE_HISTORY", "0")
def prefetch_multistage_history_width() -> int:
    return max(1, _i("PREFETCH_MULTISTAGE_HISTORY_WIDTH", 10))
# Cheap late correction: the ordinary early full gate retains the SSD window,
# then the completed source output scores only its frozen top-64 candidates.
def prefetch_late_candidate_rerank() -> bool:
    return _b("PREFETCH_LATE_CANDIDATE_RERANK", "0")
def prefetch_late_candidate_width() -> int:
    return max(1, _i("PREFETCH_LATE_CANDIDATE_WIDTH", 2))
def prefetch_late_candidate_layers() -> "set[int] | None":
    return parse_layers_env("PREFETCH_LATE_CANDIDATE_LAYERS")
# Stop rebuilding a target predictor once that layer's unified pool has a
# stable working set.  A true demand load rearms prediction for a few forwards.
def prefetch_adaptive() -> bool: return _b("PREFETCH_ADAPTIVE", "0")
def prefetch_adaptive_fill() -> float: return max(0.0, min(1.0, _f("PREFETCH_ADAPTIVE_FILL", 0.85)))
def prefetch_adaptive_cooldown() -> int: return max(1, _i("PREFETCH_ADAPTIVE_COOLDOWN", 32))
# 两阶段预取：原 main source callback 先锁定小 core；到目标 T-1 后把下一层
# 的真实 attention/GDN+gate 提前执行，并由正式 decoder 调用直接复用，只补剩余槽。
# 它不移动第一次 callback，也不允许不能被正式调用复用的 shadow gate/replay。
def prefetch_progressive() -> bool: return _b("PREFETCH_PROGRESSIVE", "0")
def prefetch_progressive_exact_only() -> bool:
    return _b("PREFETCH_PROGRESSIVE_EXACT_ONLY", "0")
def prefetch_progressive_mode() -> str: return _s("PREFETCH_PROGRESSIVE_MODE", "k1").strip().lower()
def prefetch_progressive_target_layers() -> "set[int] | None":
    """Targets using early-core + refinement; empty config means all."""
    return parse_layers_env("PREFETCH_PROGRESSIVE_TARGET_LAYERS")
def prefetch_progressive_for(target_layer: int) -> bool:
    if not prefetch_progressive():
        return False
    targets = prefetch_progressive_target_layers()
    target = int(target_layer)
    selected = targets is None or target in targets
    # Production progressive refinement is legal only for an adjacent moved
    # target computation.  A non-adjacent/post-MoE refinement necessarily
    # evaluates another target gate that the decoder cannot reuse, recreating
    # the shadow-gate overhead this path exists to remove.  Such targets fall
    # back to the ordinary one-shot early rerank instead of using a narrow core.
    return (
        selected
        and prefetch_progressive_signal_for(target) == "target_cache"
        and prefetch_progressive_refine_ahead_for(target) == 1
    )
def prefetch_progressive_core() -> int:
    default = 15 if prefetch_progressive_mode() == "k3" else 10
    return _i("PREFETCH_PROGRESSIVE_CORE", default)
def prefetch_progressive_max_width() -> int:
    """Logical rerank output cap, independent of persistent side-cache rows."""
    default = 24 if prefetch_progressive_mode() == "k3" else 15
    return max(1, _i("PREFETCH_PROGRESSIVE_MAX_WIDTH", default))
def prefetch_progressive_hybrid_cutoff() -> int:
    return max(0, _i(
        "PREFETCH_PROGRESSIVE_HYBRID_CUTOFF", cross_layer_cutoff(),
    ))
def prefetch_progressive_core_for(target_layer: int) -> int:
    if prefetch_progressive_signal() != "hybrid":
        return prefetch_progressive_core()
    # K3 的真实并集下限为 10，15-wide early core 始终满足 1.5x
    # 合同；ahead=1 的早层没有额外 I/O 窗口，尽早提交这一行比留给
    # T-1 tail 更有效。真实短跑将 deadline fallback 1395→1343。
    low_default = 15 if prefetch_progressive_mode() == "k3" else 10
    high_default = 8 if prefetch_progressive_mode() == "k3" else 12
    if int(target_layer) <= prefetch_progressive_hybrid_cutoff():
        return _i("PREFETCH_PROGRESSIVE_CORE_LO", low_default)
    return _i("PREFETCH_PROGRESSIVE_CORE_HI", high_default)
# Debug/compatibility switch only. Production defaults to non-blocking tail:
# target demand consumes whatever has completed and falls back normally.
def prefetch_progressive_wait() -> bool: return _b("PREFETCH_PROGRESSIVE_WAIT", "0")
# Production wait point for an asynchronous tail.  Unlike
# PREFETCH_PROGRESSIVE_WAIT this does not stall at the T-1 refinement
# boundary: target demand waits only route experts that the tail already has
# in flight, after all intervening decoder work has had a chance to overlap
# the SSD reads.  This also prevents demand_dual from issuing duplicate reads
# for the same pending experts.
def prefetch_progressive_demand_wait() -> bool:
    return _b("PREFETCH_PROGRESSIVE_DEMAND_WAIT", "0")
# At a normal target boundary, wait only for actual route experts that already
# own an in-flight direct side row.  Truly unpredicted experts still enter the
# fallback reader immediately; predicted rows are never read from SSD twice.
def prefetch_wait_predicted_pending() -> bool:
    return _b("PREFETCH_WAIT_PREDICTED_PENDING", "1")
def prefetch_progressive_callback_wait() -> bool:
    return _b("PREFETCH_PROGRESSIVE_CALLBACK_WAIT", "1")
# Exact adjacent refinement can make the target route fully ready before
# demand.  In that case use a GPU-only real+direct table remap and create no
# target-boundary CPU completion handler. Opt-in until long-run validation.
def prefetch_exact_gpu_demand() -> bool:
    return _b("PREFETCH_EXACT_GPU_DEMAND", "0")
# Diagnostic: keep the exact route's device dependency but skip its native
# CPU callback/read submission.  Valid only when the working set is already
# resident; used to isolate empty callback/event overhead.
def prefetch_exact_no_io(layer: "int | None" = None) -> bool:
    override = _prefetch_exact_no_io_override.get()
    if override is not None:
        if isinstance(override, (set, frozenset)):
            return layer is not None and int(layer) in override
        return bool(override)
    return _b("PREFETCH_EXACT_NO_IO", "0")


@contextmanager
def override_prefetch_exact_no_io(enabled):
    """Temporarily select the handler-free exact-demand implementation.

    The override is context-local so speculative verify can use it without
    making prefill, baseline decode, or a concurrent request optimistic.
    """
    value = frozenset(int(layer) for layer in enabled) \
        if isinstance(enabled, (set, frozenset, tuple, list)) else bool(enabled)
    token = _prefetch_exact_no_io_override.set(value)
    try:
        yield
    finally:
        _prefetch_exact_no_io_override.reset(token)


def prefetch_optimistic_verify() -> bool:
    # Run MTP batch verify without per-layer completion callbacks. Any missing
    # GPU table row is detected after the forward and causes an exact safe
    # replay through the ordinary native I/O path.
    return _b("PREFETCH_OPTIMISTIC_VERIFY", "0")


def prefetch_optimistic_reprobe() -> bool:
    # Diagnostic: keep probing layers after a miss so a short exact run can
    # reveal the per-layer miss distribution. Production keeps failed layers
    # on the safe callback path for the rest of the request.
    return _b("PREFETCH_OPTIMISTIC_REPROBE", "0")
# Diagnostic/certified-working-set mode.  The caller must provide a pin profile
# covering every route that can occur; demand then becomes a pure GPU table
# lookup with no callback.  This gives a numerically correct resident-speed
# anchor, unlike the historical dirty-slot PROBE_ALL_HIT_LAZY measurement.
def prefetch_pinned_gpu_demand() -> bool:
    return _b("PREFETCH_PINNED_GPU_DEMAND", "0")
def prefetch_progressive_signal() -> str:
    # Only target_cache + ahead=1 is eligible for progressive execution: it
    # moves the real adjacent attention/gate and reuses it. ``post_moe`` and
    # the non-target half of ``hybrid`` are retained as configuration aliases
    # for experiments, but prefetch_progressive_for routes them through the
    # ordinary one-shot early rerank and never constructs a shadow gate.
    value = _s("PREFETCH_PROGRESSIVE_SIGNAL", "hybrid").strip().lower()
    if value not in {"target_cache", "post_moe", "hybrid"}:
        raise ValueError(f"unsupported PREFETCH_PROGRESSIVE_SIGNAL={value!r}")
    return value
def prefetch_progressive_signal_for(target_layer: int) -> str:
    signal = prefetch_progressive_signal()
    if signal != "hybrid":
        return signal
    explicit = parse_layers_env("PREFETCH_PROGRESSIVE_TARGET_CACHE_LAYERS")
    if explicit is not None:
        return "target_cache" if int(target_layer) in explicit else "post_moe"
    return (
        "target_cache"
        if int(target_layer) <= prefetch_progressive_hybrid_cutoff()
        else "post_moe"
    )
def prefetch_progressive_refine_ahead() -> int:
    return max(1, _i("PREFETCH_PROGRESSIVE_REFINE_AHEAD", 2))
def prefetch_progressive_refine_aux_stream() -> bool:
    # The adjacent target gate depends on the source MoE output and is already
    # next on the decoder's critical path.  Keep the historical auxiliary
    # placement available, but allow avoiding its cross-stream fence.
    return _b("PREFETCH_PROGRESSIVE_REFINE_AUX_STREAM", "1")
def prefetch_progressive_late_layers() -> "set[int]":
    configured = parse_layers_env("PREFETCH_PROGRESSIVE_LATE_LAYERS")
    if configured is not None:
        return configured
    if prefetch_progressive_mode() == "k3":
        return {7, 8, 9, 10, 44, 47}
    return {7, 9, 47}
def prefetch_progressive_refine_ahead_for(target_layer: int) -> int:
    target = int(target_layer)
    # Targets in the original low-ahead region cannot refine before T-1
    # because their early state itself is created there.  A frozen exception
    # set lets weak high layers retain exact T-1 refinement while the rest
    # use an earlier signal and keep a larger SSD window.
    if (
        target <= cross_layer_cutoff()
        or target in prefetch_progressive_late_layers()
    ):
        return 1
    return prefetch_progressive_refine_ahead()
def prefetch_progressive_union_margin() -> int:
    default = 4 if prefetch_progressive_mode() == "k3" else 5
    return max(0, _i("PREFETCH_PROGRESSIVE_UNION_MARGIN", default))
def prefetch_progressive_union_margin_for(target_layer: int) -> int:
    if prefetch_progressive_signal() != "hybrid":
        margin = prefetch_progressive_union_margin()
    elif int(target_layer) <= prefetch_progressive_hybrid_cutoff():
        margin = max(0, _i("PREFETCH_PROGRESSIVE_UNION_MARGIN_LO", 2))
    else:
        margin = max(0, _i("PREFETCH_PROGRESSIVE_UNION_MARGIN_HI", 7))
    extra = parse_layers_env("PREFETCH_PROGRESSIVE_EXTRA_MARGIN_LAYERS")
    if extra is None:
        extra = {45} if prefetch_progressive_mode() == "k3" else set()
    return margin + int(int(target_layer) in extra)
# 目标层旧 attention/GDN cache 诊断路径：在原 source MoE 入口前用 source
# post-attention residual 近似执行目标 attention，再把所得 gate logits 交给原 native
# callback。默认关闭；它保持逻辑 source 不变，但会增加 callback 前计算，必须通过真实
# window/deadline A/B 才能判断是否可用。
def prefetch_target_cache() -> bool: return _b("PREFETCH_TARGET_CACHE", "0")
def prefetch_target_cache_alpha_lo() -> float: return max(0.0, min(1.0, _f("PREFETCH_TARGET_CACHE_ALPHA_LO", 1.0)))
def prefetch_target_cache_alpha_hi() -> float: return max(0.0, min(1.0, _f("PREFETCH_TARGET_CACHE_ALPHA_HI", 0.25)))
def prefetch_target_cache_max_seq() -> int: return max(1, _i("PREFETCH_TARGET_CACHE_MAX_SEQ", 4))
def prefetch_target_cache_layers() -> "set[int] | None": return parse_layers_env("PREFETCH_TARGET_CACHE_LAYERS")
# 逐层 source-route residual correction。profile 自带 ranking/width policy；空路径关闭。
def prefetch_target_cache_profile() -> str: return _s("PREFETCH_TARGET_CACHE_PROFILE", "").strip()
# 在目标 demand 的精确边界统计逐层唯一专家：real resident / 已完整 publish
# 的 side prefetch / 同步 fallback。默认关闭，开启时不改变提交时机或缓存策略。
def prefetch_deadline_prof() -> bool: return _b("PREFETCH_DEADLINE_PROF", "0")
# 严格验收探针：用逻辑 forward id 把 source-time rerank 提交与目标 demand
# 一一配对，统计逐次 width、recall、1.5x 违规和真实 I/O 时间线。
def prefetch_audit_prof() -> bool: return _b("PREFETCH_AUDIT_PROF", "0")
# Lightweight logical-set acceptance only; unlike PREFETCH_AUDIT_PROF this
# does not collect native callback/I/O timelines for every layer.
def prefetch_acceptance_prof() -> bool: return _b("PREFETCH_ACCEPTANCE_PROF", "0")
# Optional decode-only training trace. Each row stores the exact raw proxy
# gate logits used to form top64 plus the target layer's true routed experts.
def prefetch_rerank_data_out() -> str:
    return _s("PREFETCH_RERANK_DATA_OUT", "").strip()
def prefetch_rerank_data_active() -> bool:
    return bool(prefetch_rerank_data_out()) and _b(
        "PREFETCH_RERANK_DATA_ACTIVE", "0",
    )
def prefetch_pin_profile() -> str: return _s("PREFETCH_PIN_PROFILE", "").strip()
def prefetch_transition_profile() -> str: return _s("PREFETCH_TRANSITION_PROFILE", "").strip()
def prefetch_transition_only_profile() -> str: return _s("PREFETCH_TRANSITION_ONLY_PROFILE", "").strip()
def prefetch_online_transition() -> bool: return _b("PREFETCH_ONLINE_TRANSITION", "0")
def prefetch_online_host_submit() -> bool: return _b("PREFETCH_ONLINE_HOST_SUBMIT", "0")
def prefetch_host_ready_submit() -> bool: return _b("PREFETCH_HOST_READY_SUBMIT", "0")
def transition_trace() -> bool: return _b("TRANSITION_TRACE", "0")
def transition_trace_width() -> int: return max(1, _i("TRANSITION_TRACE_WIDTH", 64))
def residual_hidden_trace() -> bool: return _b("RESIDUAL_HIDDEN_TRACE", "0")
def route_delta_trace() -> bool: return _b("ROUTE_DELTA_TRACE", "0")
def route_delta_trace_width() -> int: return max(1, _i("ROUTE_DELTA_TRACE_WIDTH", 64))
def route_delta_trace_target_layers() -> frozenset[int] | None:
    """返回诊断 trace 要保留的目标层；空配置表示保留全部层。"""
    raw = _s("ROUTE_DELTA_TRACE_TARGET_LAYERS", "").strip()
    if not raw:
        return None
    try:
        layers = frozenset(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("ROUTE_DELTA_TRACE_TARGET_LAYERS 必须是逗号分隔整数") from error
    if not layers or any(layer < 0 for layer in layers):
        raise ValueError("ROUTE_DELTA_TRACE_TARGET_LAYERS 必须包含非负层号")
    return layers
def expert_output_trace() -> bool: return _b("EXPERT_OUTPUT_TRACE", "0")
def trajectory_trace() -> bool: return _b("TRAJECTORY_TRACE", "0")
def online_event_trace() -> bool: return _b("ONLINE_EVENT_TRACE", "0")
def distributional_full_trace() -> bool: return _b("DISTRIBUTIONAL_FULL_TRACE", "0")
def native_fused_prefetch() -> bool: return _b("NATIVE_FUSED_PREFETCH", "0")
# 统一主池模式。POOL_SPEC_SLOTS 保留旧配置名称，但它的容量会并入每层主池；
# 预测字节先进入少量全局 staging bank，随后晋升主池，不再建立 per-layer side region。
def zerocopy_dual_source() -> bool: return _b("ZEROCOPY_DUAL_SOURCE")
def pool_spec_slots() -> int: return max(0, _i("POOL_SPEC_SLOTS", 3))
def pool_admission_slots() -> int:
    """Maximum speculative rows in the merged pool.

    ``POOL_SPEC_SLOTS`` remains the physical capacity contribution for
    compatibility.  Keeping admission separately tunable lets a compact pool
    devote its remaining rows to verified history instead of treating every
    added row as speculative cache.
    """
    return max(0, _i("POOL_ADMISSION_SLOTS", pool_spec_slots()))
def pool_layer_cap_overrides() -> "dict[int, int]":
    """Parse ``layer[-layer]:physical-cap`` overrides for hot layers."""
    spec = _s("POOL_LAYER_CAP_OVERRIDES", "").strip()
    if not spec:
        return {}
    output: "dict[int, int]" = {}
    try:
        for item in spec.split(","):
            layer_spec, cap_spec = item.strip().split(":", 1)
            cap = int(cap_spec)
            if cap < 1:
                raise ValueError
            if "-" in layer_spec:
                start, end = (int(value) for value in layer_spec.split("-", 1))
                layers = range(start, end + 1)
            else:
                layers = (int(layer_spec),)
            for layer in layers:
                if layer < 0:
                    raise ValueError
                output[layer] = cap
    except ValueError as error:
        raise ValueError(
            "POOL_LAYER_CAP_OVERRIDES 必须是 layer[-layer]:cap 列表",
        ) from error
    return output
def sideregion_lfu() -> bool: return _b("SIDEREGION_LFU", "1")  # 兼容旧环境变量；统一主池使用同一 LFU 策略
def sideregion_row_leases() -> bool: return _b("SIDEREGION_ROW_LEASES", "0")
def native_no_submit() -> bool: return _b("NATIVE_NO_SUBMIT", "0")
def native_no_promote() -> bool: return _b("NATIVE_NO_PROMOTE", "0")
def native_materialize() -> bool: return _b("NATIVE_MATERIALIZE", "0")
def prefetch_physical_read_budget() -> int: return _i("PREFETCH_PHYSICAL_READ_BUDGET", 0)
def prefetch_physical_read_budget_profile() -> str:
    return _s("PREFETCH_PHYSICAL_READ_BUDGET_PROFILE", "").strip()
def prefetch_k1_physical_rank_limit() -> int:
    """Only read a K=1 miss when it appears this high in proxy order.

    Zero keeps the full logical rerank output.  The logical top64/top15 audit
    remains unchanged; this is solely an SSD precision gate.
    """
    return max(0, _i("PREFETCH_K1_PHYSICAL_RANK_LIMIT", 0))
def prefetch_k3_physical_rank_limit() -> int:
    """K=2/3 counterpart, ranked by max proxy logit across verify tokens."""
    return max(0, _i("PREFETCH_K3_PHYSICAL_RANK_LIMIT", 0))
def prefetch_host_ready_protect_logical() -> bool:
    """Keep resident logical candidates in the native eviction protect set."""
    return _b("PREFETCH_HOST_READY_PROTECT_LOGICAL", "0")
def prefetch_protect_logical() -> bool:
    """Keep resident winners in the async logical prefix (no miss backfill)."""
    return _b("PREFETCH_PROTECT_LOGICAL", "0")
def prefetch_isolated_side() -> bool:
    """Reserve the allocation tail for non-competing prediction rows."""
    return _b("PREFETCH_ISOLATED_SIDE", "0")
def prefetch_isolated_side_for(layer: int) -> bool:
    selected = prefetch_target_layers()
    return prefetch_isolated_side() and (
        selected is None or int(layer) in selected
    )
def file_prefetch_global_budget() -> int: return _i("FILE_PREFETCH_GLOBAL_BUDGET", 64)
def stage_prefetch_global_budget() -> int: return _i("STAGE_PREFETCH_GLOBAL_BUDGET", 64)
def stage_prefetch_min_score() -> float: return _f("STAGE_PREFETCH_MIN_SCORE", 0)
def stage_prefetch_per_layer_budget(default: int = 4) -> int: return _i("STAGE_PREFETCH_PER_LAYER_BUDGET", default)


# ============================ MTP ============================
def mtp_verify_mode() -> str: return _s("MTP_VERIFY_MODE", "batch")
# 喂给 MTP drafter 的主模型 hidden:pre_norm(默认,与训练/验证一致,接受率高)| post_norm(旧行为)
def mtp_hidden() -> str: return _s("MTP_HIDDEN", "pre_norm")
# Fixed production Expert-major superblock. It is deliberately not an env
# switch on the clean branch.
def prefill_chunk() -> int: return 32768


# ============================ KV 量化(IsoQuant K4/V3)============================
# 只作用于 12 个全注意力层。SO(4) 块旋转去相关 + 非对称 K4/V3 仿射量化:128k KV 3.0→~0.68 GiB。
# 线性层(Gated DeltaNet 递归态)不动。质量验收:token 一致率≥95% + logits cosine≥0.99。
def kv_quant() -> bool: return _b("KV_QUANT", "0")
def kv_k_bits() -> int: return _i("KV_K_BITS", 4)            # Key 位宽(默认 4)
def kv_v_bits() -> int: return _i("KV_V_BITS", 3)            # Value 位宽(默认 3,非对称)
def kv_group_size() -> int: return _i("KV_GROUP_SIZE", 64)   # 仿射量化分组(整除 head_dim=256)
def kv_rotate() -> bool: return _b("KV_ROTATE", "1")          # SO(4) 块旋转去相关(默认开)
def kv_rot_seed() -> int: return _i("KV_ROT_SEED", 0)        # 旋转随机种子(data-oblivious、固定)


# ============================ profiling / 诊断 ============================
def window_prof() -> bool: return _b("WINDOW_PROF", "0")
def predict_recall_prof() -> bool: return _b("PREDICT_RECALL_PROF", "0")
def miss_attrib() -> bool: return _b("MISS_ATTRIB", "0")  # miss 归因：A(预测到没到位)/B(没预测到)
def route_trace_enabled() -> bool: return _b("ROUTE_TRACE", "0")
def stream_prof() -> bool: return _b("STREAM_PROF", "0")
def probe_predict_only() -> bool: return _b("PROBE_PREDICT_ONLY", "0")
def probe_perlayer_sync() -> bool: return _b("PROBE_PERLAYER_SYNC", "0")
# 预取 host 墙钟探针：量 predict/submit/promote 各段主线程不可重叠的 CPU 时间。默认关、零开销。
def prefetch_tprof() -> bool: return _b("PREFETCH_TPROF", "0")
# 并集专家数探针:按前向 seq 分桶记每层路由专家并集大小(seq=K 即 MTP verify 的专家并集)。默认关。
def union_prof() -> bool: return _b("UNION_PROF", "0")
# 接受率 top-k 覆盖探针(>0 且 profile 时):记每个草稿位置 MTP 的 top-k 候选,量模型真实 token
# 是否落在 top-2/top-3 里 = 树形展开的救回上界。默认 0=关。
def accept_topk() -> int: return _i("ACCEPT_TOPK", 0)
# 最小树:位置1 展开 top-2。仅当 A 链首草稿被拒且 B 候选=真实 token 时,额外跑一次 B 链前向救回。
# 两分支各为独立 batch=1 seq=K 前向(线性层不能批处理树,故不拍平),per-forward union 不变。默认关。
def tree_top2() -> bool: return _b("TREE_TOP2", "0")
# 第2 草稿位置(pos1)top-2 救回:在 tree_top2 基础上,额外抽 chainC(第2 位次选分支),当第1 位
# 命中但第2 位被拒、且 chainC 第2 token=模型真值时改验 chainC。探针实测 pos1「首选错次选对」比例
# (~11%)高于 pos0(~7%),消融证实接受长度确实 +2.74%(bit-lossless);但每次救回要多一次主模型
# 前向,该成本在当前硬件/批量下恰好抵消 token 收益,净 tok/s 无提升(见 benchmarks/_bench_p1_ablation)。
# 故默认关,保护已验证的 pos0 纯路径 tok/s 收益;留作前向变廉价/批量变大时收益翻正的储备,一行 env 开。
def tree_top2_p1() -> bool: return _b("TREE_TOP2_P1", "0")
# 完整树形验证(batch-of-paths):把 P 条候选路径拍到 batch 维,一次 batched 前向并行验证所有路径,
# 选接受最长的路径提交(提取赢家 row)。每条路径是普通线性序列,故线性层/全注意力层都走成熟的
# batch 前向(无需改 kernel);单次前向的 batch=P 计算加宽了预取窗口,同时多路径提升接受长度
# ——一举拿下 accept_len 与 hit_rate 两只鸟。默认关(与 tree_top2/普通链验证互斥,优先级最高)。
def tree_verify() -> bool: return _b("TREE_VERIFY", "0")
# 树分支数(候选路径条数 P)。当前 drafter 在位置1 展开 top-P,P=2 即 top-2。
def tree_branches() -> int: return _i("TREE_BRANCHES", 2)
# 置信度门控动态深度(P-MTP 风格):逐位抽草稿时累计置信度 C_i=p0·…·p_i,C_i≥tau 且未到 depth_max
# 就继续加深,否则本步只 verify 到当前深度。低置信步收缩(省专家加载/cap 压力),高置信步至多到
# depth_max。探针实测置信度对接受率区分度极强(高低置信接受率差 +42~59pp)。消融结论(见
# benchmarks/reports,_bench_adaptive):收益全部来自"向下收缩",τ=0.3、depth_max=3 即 +5~6% tok/s
# 且 bit-lossless、零额外显存;向上扩到 K=4 反而更慢(第4 位专家加载成本 > 多接受的 token),且在
# 生产 EXPERT_SLOTS=32 下 seq=4·top_k=40>cap 会溢出致有损。故 depth_max 默认 3(=基础 K,slots=32 安全),
# 要扩到 4 必须同时把 EXPERT_SLOTS 提到 ≥40。默认关,作可选加速路径(与 tree_top2 互斥)。
def adaptive_depth() -> bool: return _b("MTP_ADAPTIVE_DEPTH", "0")
def conf_tau() -> float: return _f("MTP_CONF_TAU", 0.3)
def depth_max() -> int: return _i("MTP_DEPTH_MAX", 3)
# 合并路径:在动态深度基础上,对"被保留成深链(n>=2)"的步叠加 pos0 top-2 救回(首 token 被拒且
# top-2=模型真值时改验 B 链)。两机理正交(动态深度压每步成本、救回抬接受长度),但都作用于低置信步、
# 方向相反(depth=1 的浅步无位置可救),故叠加非简单相加,须实测。默认关。
def adaptive_rescue() -> bool: return _b("MTP_ADAPTIVE_RESCUE", "0")


def parse_layers_env(name: str) -> "set[int] | None":
    """解析 "0-3,5,8" 形式的层集合环境变量；为空返回 None（表示全部层）。"""
    spec = os.environ.get(name, "").strip()
    if not spec:
        return None
    out: "set[int]" = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out
