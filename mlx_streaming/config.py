"""集中配置：所有运行时开关从环境变量读取（env 名与默认值与历史一致，便于现有跑法）。

设计目的：把原先散落在 core/ 各文件的 `os.environ.get` 全部收口到这里，单点可查、
避免默认值漂移。读取均为运行时（每次调用读 env），与原行为一致——测试/probe 用
monkeypatch env 仍生效，热路径每 token 读 env 的成本与原先相同。

对「同一 env 名在不同调用方有不同默认」的项（如 CROSS_LAYER_PREFETCH_AHEAD：hook=0、
native 预取=1），accessor 暴露 `default` 参数，由调用方传入，绝不擅自统一。
"""
import os


def _b(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _i(name: str, default) -> int:
    return int(os.environ.get(name, str(default)))


def _f(name: str, default) -> float:
    return float(os.environ.get(name, str(default)))


def _s(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# ============================ 模型 / 目录 ============================
def model_path() -> str: return _s("MODEL", "/tmp/qwen3_next_80b_4bit")
def expert_dir(default: str = "/tmp/qwen3_next_experts") -> str: return _s("EXPERT_DIR", default)
def expert_slots() -> int: return _i("EXPERT_SLOTS", 96)
def expert_pool_profile() -> str: return _s("EXPERT_POOL_PROFILE", "")
def hidden_variant() -> str: return _s("HIDDEN_VARIANT", "pre_final_norm")
def blob_dir() -> str: return _s("BLOB_DIR", "")
def compute_buffer_dir() -> str: return _s("COMPUTE_BUFFER_DIR", "")


# ============================ MoE 热路径 ============================
def resident_pool_enabled() -> bool: return _b("RESIDENT_POOL", "1")
def gpu_remap_enabled() -> bool: return _b("GPU_REMAP", "1")
def moe_topk_override() -> "str | None": return os.environ.get("MOE_TOPK_OVERRIDE")
def eager_expert_load() -> bool: return _b("EAGER_EXPERT_LOAD", "0")


# ============================ 缓存 / 驱逐 ============================
def evict_policy() -> str: return _s("EVICT_POLICY", "lru")
def lfu_decay_interval() -> int: return _i("LFU_DECAY_INTERVAL", 512)
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
def stream_blob_bg_budget(default: int = 8) -> int: return _i("STREAM_BLOB_BG_BUDGET", default)
def stream_blob_prefetch_budget(default: int) -> int: return _i("STREAM_BLOB_PREFETCH_BUDGET", default)


# ============================ 跨层 / 预取 ============================
# 注意 AHEAD 默认按调用方传：跨层 hook=0、native 预取=1。
def cross_layer_prefetch() -> bool: return _b("CROSS_LAYER_PREFETCH", "0")
def cross_layer_ahead(default: int = 0) -> int: return _i("CROSS_LAYER_PREFETCH_AHEAD", default)
def cross_layer_mult() -> int: return max(1, _i("CROSS_LAYER_PREFETCH_MULT", 2))
def native_fused_prefetch() -> bool: return _b("NATIVE_FUSED_PREFETCH", "0")
def native_no_submit() -> bool: return _b("NATIVE_NO_SUBMIT", "0")
def native_no_promote() -> bool: return _b("NATIVE_NO_PROMOTE", "0")
def native_materialize() -> bool: return _b("NATIVE_MATERIALIZE", "0")
def file_prefetch_global_budget() -> int: return _i("FILE_PREFETCH_GLOBAL_BUDGET", 64)
def stage_prefetch_global_budget() -> int: return _i("STAGE_PREFETCH_GLOBAL_BUDGET", 64)
def stage_prefetch_min_score() -> float: return _f("STAGE_PREFETCH_MIN_SCORE", 0)
def stage_prefetch_per_layer_budget(default: int = 4) -> int: return _i("STAGE_PREFETCH_PER_LAYER_BUDGET", default)


# ============================ MTP ============================
def mtp_verify_mode() -> str: return _s("MTP_VERIFY_MODE", "batch")


# ============================ profiling / 诊断 ============================
def window_prof() -> bool: return _b("WINDOW_PROF", "0")
def predict_recall_prof() -> bool: return _b("PREDICT_RECALL_PROF", "0")
def route_trace_enabled() -> bool: return _b("ROUTE_TRACE", "0")
def stream_prof() -> bool: return _b("STREAM_PROF", "0")
def probe_predict_only() -> bool: return _b("PROBE_PREDICT_ONLY", "0")
def probe_perlayer_sync() -> bool: return _b("PROBE_PERLAYER_SYNC", "0")


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
