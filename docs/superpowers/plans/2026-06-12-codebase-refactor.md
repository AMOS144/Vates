# 代码重构实现计划（增量 + 测试守护）

> **执行原则**：每个阶段 = 纯搬移/重命名/拆函数/补注释，**绝不改逻辑**；每阶段结束跑 `回归门`（见下）通过才进下一阶段。用 `- [ ]` 跟踪。

**Goal:** 按职责把项目重组为高内聚低耦合结构，运行时行为零变化（exact_match 逐位一致）。

**Tech Stack:** Python / MLX，现有 pytest 套件 + `run_mtp_spec` exact_match 作回归门。

## 回归门（每阶段后必跑）

```bash
# 1) 单测全绿
uv run pytest mlx_streaming/tests -q -p no:cacheprovider
# 2) 端到端等价（小样本，exact_match 必须 true）
MODEL=/tmp/qwen3_next_80b_4bit EXPERT_DIR=$PWD/models/qwen3_next_experts_2bit_g128 \
RESIDENT_POOL=1 MTP_VERIFY_MODE=batch K=2 MAXTOK=64 EXPERT_SLOTS=64 \
  uv run python -m mlx_streaming.runtime.run_mtp_spec 2>/dev/null | grep -E '"exact_match"|"spec_tok_per_s"'
```
任一不过 → 回滚该阶段改动，定位后重做。

---

## 阶段 0：config.py 集中 env 常量

**Files:** Create `mlx_streaming/config.py`；Modify 读取方改为引用。

- [ ] Step 1: 扫描所有 `os.environ.get(...)`，在 `config.py` 定义带默认值的访问器（函数或惰性属性），按模块分组、命名常量化（如 `resident_pool_enabled()`、`gpu_remap_enabled()`、`cross_layer_ahead()`、`bg_budget()`）。保留 env 名不变（兼容现有跑法）。
- [ ] Step 2: 先只迁移**无歧义**的常量（K/MAXTOK 不动，那些在 cli）。core/ 里的开关逐个替换为 `config.xxx()`。
- [ ] Step 3: 跑回归门。

## 阶段 1：拆 streaming_moe.py 巨石

**Files:** Create `core/moe/{__init__,block,gate,compute,custom_kernel}.py`、`core/profiling.py`；`core/streaming_moe.py` 改为 re-export shim。

- [ ] Step 1: `core/profiling.py` ← 移 PROF/WINDOW_PROF/PREDICT_RECALL_PROF/prof_reset/_tick。streaming_moe import 回去保持引用。跑回归门。
- [ ] Step 2: `core/moe/custom_kernel.py` ← 移 `_custom_qlinear_indexed`/`_custom_fused_moe_indexed`/`_custom_*_enabled`/`_custom_qproj_targets`。跑回归门。
- [ ] Step 3: `core/moe/compute.py` ← 移 `PersistentSubGLU`/`RotatedSubGLU`/`SwitchGLU` 相关 forward 助手（`_build_qsl`/`_update_qsl`/`streaming_switch_glu_forward*`/`_slice_switch_linear`/`_unique_and_local`）。跑回归门。
- [ ] Step 4: `core/moe/gate.py` ← 移 `_effective_top_k`/`_predict_layer_experts`。跑回归门。
- [ ] Step 5: `core/moe/block.py` ← 移 `FileStreamingMoeBlock`/`StreamingMoeBlock`（含 `__call__`/`_native_fused_prefetch`/`_try_native_forward`/`_call_prof`）。跑回归门。
- [ ] Step 6: `core/streaming_moe.py` 仅留 `from .moe.block import *; from .prefetch.cross_layer import *; ...`（shim）。跑回归门。

## 阶段 2：拆预取 + patch

**Files:** Create `core/prefetch/{__init__,cross_layer}.py`、`core/patch.py`；移动 `native_staging.py`/`bg_prefetch.py` 到 `core/prefetch/`。

- [ ] Step 1: 移 `core/native_staging.py`→`core/prefetch/native_staging.py`、`core/bg_prefetch.py`→`core/prefetch/bg_prefetch.py`，旧路径留 shim。顶部补"多线程/GPU 完成回调 + gen-匹配 + 环形 buffer"不变量注释。跑回归门。
- [ ] Step 2: `core/prefetch/cross_layer.py` ← 移 `enable_cross_layer_prefetch`/`patched_call`/`_submit_missing_prefetch`/预取预算助手（`_prefetch_budget`/`_consume_prefetch_budget`/`_prefetch_layers`/`_parse_layers_env`/`_cross_layer_prefetch_mult`）。跑回归门。
- [ ] Step 3: `core/patch.py` ← 移 `patch_model_filebacked`/`patch_model`。跑回归门。

## 阶段 3：拆缓存

**Files:** Create `core/cache/{__init__,resident_pool,expert_store,blob_loader}.py`，旧 `core/expert_store.py`/`core/blob_loader.py` 留 shim。

- [ ] Step 1: `core/cache/resident_pool.py` ← 移 `ResidentExpertPool` + 其私有助手（`_stack_*`/`_PerLayerLru`）。跑回归门。
- [ ] Step 2: `core/cache/expert_store.py` ← 移 `FileExpertStore`/`LruExpertStore`。`core/cache/blob_loader.py` ← 移 `BlobExpertSource`。旧路径 shim。跑回归门（重点 test_resident_pool / test_file_streaming）。

## 阶段 4：拆 MTP

**Files:** Create `mtp/{drafter,generate}.py`；`qwen3_next_mtp.py`→`mtp/model.py`（旧名 shim）；`mtp_generate.py` 留 shim。

- [ ] Step 1: `mtp/drafter.py` ← 移 checkpoint/forward_with_hidden/snapshot/restore/commit/accept 系列。跑回归门（test_mtp_generate/spike）。
- [ ] Step 2: `mtp/generate.py` ← 移 `mtp_generate`。`mtp/mtp_generate.py` 留 shim。跑回归门。

## 阶段 5：runtime/tools 分离 + 收尾

**Files:** 新建 `mlx_streaming/runtime/`（移 run_mtp_spec/run_streaming/run_spec/run_baseline）、`mlx_streaming/tools/`（移 probe_*）。

- [ ] Step 1: 移动 runner 与 probe，旧 `cli` 路径留 shim（或更新文档/调用）。跑回归门。
- [ ] Step 2: 统一命名收尾（语义模糊的私有名改清晰）、给主线热路径与特殊逻辑补注释（seq=K+1 并集预取 / cap 容量墙 / 预测抖动误驱逐 / 跨线程回调 / gen-匹配）。跑回归门。
- [ ] Step 3: 一轮稳定后清理无用 shim（确认无外部引用）。跑回归门 + 提交。

## 自检
- 每阶段仅搬移/重命名/注释，零逻辑改动；回归门是 exact_match 兜底。
- shim 保证 import 不断；import 环用延迟 import 化解。
- 终态：无 >400 行巨石、env 集中、热路径注释清晰、多线程隔离标注。
