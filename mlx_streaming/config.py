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
# 默认放持久目录 models/（不要放 /tmp，会被系统清空）。运行从仓库根目录起，相对路径即可。
def model_path() -> str: return _s("MODEL", "models/qwen3_next_80b_4bit")
def qn_config() -> str: return _s("QN_CONFIG", "models/qwen3_next_80b_4bit/config.json")
def mtp_out() -> str: return _s("MTP_OUT", "models/qn_mtp_weights.safetensors")
def expert_dir(default: str = "models/qwen3_next_experts_4bit_g64") -> str: return _s("EXPERT_DIR", default)
def expert_slots() -> int: return _i("EXPERT_SLOTS", 32)
# 长期运行内存防御:封顶 MLX 可回收缓冲(默认 2GB),防长会话缓存膨胀;
# wired limit 默认 0=关(opt-in),设 >0 则 wire 该 GB 数的 GPU 缓冲防 macOS 压缩器,
# 须 < 系统建议工作集(本机 26.8GB)。
def mlx_cache_limit_gb() -> float: return _f("MLX_CACHE_LIMIT_GB", 2.0)
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


# ============================ 缓存 / 驱逐 ============================
def evict_policy() -> str: return _s("EVICT_POLICY", "lru")
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
# 默认 16:实测 64 槽/K=3 甜点(配合 predict_width=32 满 recall=0.931,promote/内存最省、最快)。
def stream_blob_bg_budget(default: int = 16) -> int: return _i("STREAM_BLOB_BG_BUDGET", default)
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
def staging_pread_parallel() -> bool: return _b("STAGING_PREAD_PARALLEL", "0")  # staging fill 派后台池并行;默认关:实测不降 timing miss(IO 受限)且弱化 buffer 新鲜度不变量,详见 benchmarks/reports/staging-pread-parallel-2026-06-25.md
def stg_verify() -> bool: return _b("STG_VERIFY", "0")  # 诊断:acquire_gpu 命中后池槽字节真值校验,默认关、对主路径零影响
def stream_blob_prefetch_budget(default: int) -> int: return _i("STREAM_BLOB_PREFETCH_BUDGET", default)


# ============================ 跨层 / 预取 ============================
# 注意 AHEAD 默认按调用方传：跨层 hook=0、native 预取=1。
def cross_layer_prefetch() -> bool: return _b("CROSS_LAYER_PREFETCH", "0")
def cross_layer_ahead(default: int = 0) -> int: return _i("CROSS_LAYER_PREFETCH_AHEAD", default)
def cross_layer_mult() -> int: return max(1, _i("CROSS_LAYER_PREFETCH_MULT", 2))
# 预测宽度（方案B）：预测 top-N 候选用于"减常驻"，N 大→recall 高且不占内存（只是 gate argpartition）。
# 真正占 staging 的是过滤常驻后、按分截断到 staging budget 的缺口子集。
def cross_layer_predict_width() -> int: return _i("CROSS_LAYER_PREDICT_WIDTH", 32)
# per-layer 自适应 ahead：早层用小 ahead 保召回、晚层用大 ahead 保时序（默认来自 Phase 0 实测）。
def cross_layer_cutoff() -> int: return _i("CROSS_LAYER_CUTOFF", 6)        # 切点：层号 <cutoff 用 lo，否则 hi
def cross_layer_ahead_lo() -> int: return _i("CROSS_LAYER_AHEAD_LO", 1)    # 早层 ahead（保召回）
def cross_layer_ahead_hi() -> int: return _i("CROSS_LAYER_AHEAD_HI", 3)    # 晚层 ahead（保时序）
def predict_use_x() -> bool: return _b("PREDICT_USE_X", "1")  # 默认用本层 MoE 输入 x（更新鲜，+3.6pp recall）；=0 回退旧 norm 路径
def predict_agg() -> str: return _s("PREDICT_AGG", "max")  # K+1 token 聚合：max|mean|union
def predict_union_k() -> int: return _i("PREDICT_UNION_K", 8)  # union 时每 token 取的 top-k（控候选数）
def native_fused_prefetch() -> bool: return _b("NATIVE_FUSED_PREFETCH", "0")
def zerocopy_dual_source() -> bool: return _b("ZEROCOPY_DUAL_SOURCE")  # 池侧区零拷贝双源，opt-in
def pool_spec_slots() -> int: return _i("POOL_SPEC_SLOTS", 3)          # 每层侧区投机槽数（cap=16 时每层 miss≈2.6 → 3 够）
def native_no_submit() -> bool: return _b("NATIVE_NO_SUBMIT", "0")
def native_no_promote() -> bool: return _b("NATIVE_NO_PROMOTE", "0")
def native_materialize() -> bool: return _b("NATIVE_MATERIALIZE", "0")
def file_prefetch_global_budget() -> int: return _i("FILE_PREFETCH_GLOBAL_BUDGET", 64)
def stage_prefetch_global_budget() -> int: return _i("STAGE_PREFETCH_GLOBAL_BUDGET", 64)
def stage_prefetch_min_score() -> float: return _f("STAGE_PREFETCH_MIN_SCORE", 0)
def stage_prefetch_per_layer_budget(default: int = 4) -> int: return _i("STAGE_PREFETCH_PER_LAYER_BUDGET", default)


# ============================ MTP ============================
def mtp_verify_mode() -> str: return _s("MTP_VERIFY_MODE", "batch")
# 喂给 MTP drafter 的主模型 hidden:pre_norm(默认,与训练/验证一致,接受率高)| post_norm(旧行为)
def mtp_hidden() -> str: return _s("MTP_HIDDEN", "pre_norm")
# 分块 prefill 的每块 token 数:整段 prefill 一次前向的激活峰值 ∝ prompt 长度(每 MoE 层瞬时
# 物化大量唯一专家 + 长序列激活),把 prompt 切成小块逐块喂入(KV/SSM 按 offset 因果累积、末块
# 末位 logits 与整段等价),峰值压回 ∝chunk,使 prefill 与 decode 同稳态。默认 2(沿用 DeepSeek
# 实证:每块唯一专家少、稳走 resident 池便宜路径);PREFILL_CHUNK=0 关闭分块(整段 prefill)。
def prefill_chunk() -> int: return _i("PREFILL_CHUNK", 2)


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
