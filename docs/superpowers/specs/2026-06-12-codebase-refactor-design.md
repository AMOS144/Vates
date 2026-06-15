# 设计：MLX 流式 MoE + MTP 项目代码重构

日期：2026-06-12
状态：设计待评审

## 目标（一句话）

把当前能跑通但职责混杂的代码（尤其 1059 行的 `core/streaming_moe.py` 巨石）按职责重组为高内聚、低耦合的模块结构，**不改变任何运行时行为**（`exact_match` 与重构前逐位一致），并补齐核心热路径/特殊逻辑的注释。

## 背景与约束

- 系统：基于 MLX 的 MTP 多令牌投机解码 + MoE 流式推理（专家从盘流式、常驻池缓存、跨层预取、GPU 完成回调零拷贝物化）。含 native C++/Metal 扩展、多 MLX stream、多线程。
- **硬约束**：这是 ~15k 行、验证过的复杂系统。重构是**纯结构搬移 + 重命名 + 注释 + 拆函数**，**绝不在搬移的同时改逻辑**（边搬边改是回归头号来源）。每步用测试 + `run_mtp_spec` 的 `exact_match` 守护。
- **不做**：不改算法、不改数值路径、不动 native 扩展 C++、不删 opt-in 实验路径（prefetch 等保留，默认关）、不动 prep/ 离线工具。

## 核心病灶

`core/streaming_moe.py` 一个文件混 6 类职责：① SubGLU 专家计算；② gate 门控+选专家；③ 自定义量化算子；④ MoE block 热路径；⑤ 跨层预取 + native 预取；⑥ decoder patch + profiling。导致跨文件混乱调用、难维护。其余文件（expert_store/mtp_generate）大但职责相对清晰。

## 目标结构

```
mlx_streaming/
  config.py                集中 env 常量（现散落 ~40 处 os.environ）
  core/
    moe/block.py           FileStreamingMoeBlock / StreamingMoeBlock（MoE 热路径）
    moe/gate.py            gate 前向 + argpartition 选 top-k + _predict_layer_experts + _effective_top_k
    moe/compute.py         PersistentSubGLU / RotatedSubGLU / SwitchGLU 前向
    moe/custom_kernel.py   _custom_qlinear_indexed / _custom_fused_moe_indexed（混合精度算子）
    cache/resident_pool.py ResidentExpertPool（放置/驱逐/promote/slot 表 host↔GPU）
    cache/expert_store.py  FileExpertStore / LruExpertStore（加载/pin/缓存装配）
    cache/blob_loader.py   BlobExpertSource
    prefetch/cross_layer.py  enable_cross_layer_prefetch hook + _native_fused_prefetch + _submit_missing_prefetch + 预取预算
    prefetch/native_staging.py  NativeStagingManager（gen-匹配 + 环形 buffer）
    prefetch/bg_prefetch.py     BackgroundExpertPrefetcher（多线程，顶部标注 stream/线程不变量）
    patch.py               patch_model_filebacked / patch_model（装配流式块 + hook）
    profiling.py           PROF / WINDOW_PROF / PREDICT_RECALL_PROF / prof_reset / _tick
    route_trace.py / mem.py  保持
  mtp/
    drafter.py             speculative checkpoints + forward_with_hidden + accept/commit
    generate.py            mtp_generate 主循环
    model.py               qwen3_next_mtp（现 qwen3_next_mtp.py 改名）
  runtime/                 run_mtp_spec / run_streaming / run_spec / run_baseline（现 cli 里的 runner）
  tools/                   probe_* 诊断脚本（与生产代码隔离）
  prep/  tests/            保持（tests 随 import 更新）
```

向后兼容：保留 `core/streaming_moe.py`、`core/expert_store.py` 等原模块名为**re-export shim**（`from .moe.block import *`），避免外部 import 全断；一轮稳定后再清理 shim。

## 主线热路径（注释要重点覆盖）

```
mtp/generate.mtp_generate → draft → verify forward_with_hidden
  └ 每 decoder 层 patch.patched_call
      ├ prefetch/cross_layer: 用上层 hidden 预测下层(gate) → 提交 native staging（seq=K+1 预取并集）
      └ moe/block.__call__: gate→argpartition → native_staging.promote(miss→hit) → resident_pool.acquire_gpu(GPU slot 重映射) → moe/compute(quantized_matmul + SwiGLU)
```
特殊逻辑注释点：seq=K+1 并集预取（避免越界）、缓存容量限制（cap 满置换抵消）、预测错误抖动（recall<1 → promote 误驱逐）、GPU 完成回调跨线程不变量、gen-匹配防 buffer/映射错配。

## 执行策略（增量 + 测试守护）

每阶段 = 纯搬移/重命名/拆函数/补注释 → 跑测试 + `run_mtp_spec exact_match` → 绿才下一步：
- 阶段 0：`config.py`（集中 env）
- 阶段 1：拆 `streaming_moe.py` → `moe/{block,gate,compute,custom_kernel}` + `profiling.py` + shim
- 阶段 2：拆 `prefetch/{cross_layer,native_staging,bg_prefetch}` + `patch.py`
- 阶段 3：`cache/{resident_pool,expert_store,blob_loader}` + shim
- 阶段 4：`mtp/{drafter,generate,model}`
- 阶段 5：`runtime/` 与 `tools/` 分离 + 统一命名 + 注释收尾

## 成功标准

- 每阶段后：现有 pytest 全绿；`run_mtp_spec`（小 MAXTOK）`exact_match=true`、`n_mismatch` 与重构前一致。
- 终态：无 >400 行的"巨石"文件；env 常量集中；核心热路径有清晰注释；多线程/stream 代码隔离标注；调用链可一眼看清。

## 显式风险

- import 环：moe/cache/prefetch 互相引用 → 用 shim + 延迟 import 化解。
- native 扩展路径：不动 C++，只动 Python 调用点的归属文件。
- 测试覆盖盲区：prefetch/native_staging 的端到端正确性靠 `run_mtp_spec exact_match` 兜底（已有用例不足）。
