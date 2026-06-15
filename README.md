# mlx-streaming-moe

Apple Silicon(统一内存)上的**显存外置流式 MoE 推理** + **Qwen3-Next MTP 自投机解码**,基于 [MLX](https://github.com/ml-explore/mlx)。

目标:让超大 MoE 模型(如 `Qwen3-Next-80B-A3B`)在小内存机器上跑起来——专家权重按需从 NVMe 流式加载、只把工作集常驻统一内存,同时用模型自带的 MTP 做自投机解码提速。

## 核心结论(详见 `benchmarks/reports/`)

- **MTP 自投机在本机有效**:`Qwen3-Next-80B-A3B` 2-bit 专家 + 256 槽常驻池,K=2 批量验证 → **~24–28 tok/s(2.4–2.5×)**,投机模式下 `disk_load_ratio≈0.14`(I/O 基本消除)。
- **内存自动右尺寸(无损,grow-on-demand,无需 profile)**:每层池起步小、随实际工作集按需增长到全局 capacity 才开始 LRU 淘汰,因此**默认就把内存收敛到≈实际工作集**——峰值 **13.5GB → ~10.6GB**(实测 256 槽、未启用任何 profile),命中率/吞吐不变,且对任意 prompt 自适应、capacity 内永不超预算(超了只是更慢,不崩)。profile 仅作为可选的「更紧上限」,不再是省内存的前提。
- **专家 `group_size=128` 近无损省内存**:2-bit 专家用 g128(而非 g64)重量化,scales/biases 元数据减半 → 实测峰值 **12.0GB → 11.1GB(-0.95GB)**、磁盘 23G→20G,吞吐/命中不变(`avg_accept_len` 仅 -2%)。推荐默认用 g128。
- **fused MoE Metal kernel = NO-GO**:手写 fused 量化 SwiGLU kernel 数值正确但慢于 MLX 调优的 `gather_qmm`,冲 30 tok/s 不划算(已止损,见 spec)。

## 安装

用 [uv](https://docs.astral.sh/uv/) 管理:

```bash
uv sync              # 创建 .venv 并安装依赖(mlx / mlx-lm / numpy)
uv run pytest        # 跑测试
```

## 准备数据(离线一次性)

脚本通过环境变量指向本地模型与拆分后的专家目录(大文件不入库):

```bash
# 1) 拆分专家为 per-expert safetensors
uv run python -m mlx_streaming.prep.split_experts          # 产出 EXPERT_DIR

# 2) 重量化专家(推荐 2-bit + group_size 128:近无损,比 g64 再省 ~1GB 峰值 / 3GB 磁盘)
#    用法:requantize_experts <4bit源目录> <输出目录> <bits> <group_size>
uv run python -m mlx_streaming.prep.requantize_experts $SRC_4BIT_DIR $EXPERT_DIR 2 128

# 3) 抽取 MTP 权重(自投机用)
uv run python -m mlx_streaming.prep.extract_mtp            # 产出 MTP_OUT / QN_CONFIG
```

## 运行

```bash
# 基线流式解码
EXPERT_DIR=/tmp/qwen3_next_experts_2bit_g128 EXPERT_SLOTS=256 RESIDENT_POOL=1 \
  uv run python -m mlx_streaming.runtime.run_streaming

# MTP 自投机基准(推荐配置)
EXPERT_DIR=/tmp/qwen3_next_experts_2bit_g128 EXPERT_SLOTS=256 RESIDENT_POOL=1 \
MTP_VERIFY_MODE=batch K=2 \
  uv run python -m mlx_streaming.runtime.run_mtp_spec
```

### 无损省内存:grow-on-demand(默认,无需任何配置)

每层常驻池**按需增长**:起步 16 槽,工作集扩大时按 ~1.5× 增长到 `EXPERT_SLOTS`(全局天花板)才开始 LRU
淘汰。因此**不设任何环境变量、不需要 profile**,内存就自动收敛到≈各层真实工作集(实测 256 槽峰值
13.5GB → ~10.6GB,命中率/吞吐不变)。换任意 prompt 自适应、天花板内永不超预算(超了只是更慢,不崩),
增长只在预热期偶发、稳态零拷贝。

可选:若想给某些层设**更紧的硬上限**(低于全局 capacity),仍可用 profile。它现在的语义是「增长天花板」,
而非一次性预分配:

```bash
# 产出 profile 到默认位置(margin 给未见 prompt 留冗余)
EXPERT_POOL_PROFILE_OUT=$EXPERT_DIR/pool_profile.json EXPERT_POOL_MARGIN=1.15 \
EXPERT_DIR=/tmp/qwen3_next_experts_2bit_g128 EXPERT_SLOTS=256 RESIDENT_POOL=1 \
MTP_VERIFY_MODE=batch K=2 \
  uv run python -m mlx_streaming.tools.pool_footprint
```

约定 **`{EXPERT_DIR}/pool_profile.json`** 为默认 profile(存在即作为天花板加载);`EXPERT_POOL_PROFILE=none`
彻底关闭、纯靠 grow-on-demand;也可指定其它路径覆盖默认。

## 关键环境变量

| 变量 | 含义 |
|---|---|
| `MODEL` | 主模型路径(4-bit MLX) |
| `EXPERT_DIR` | 拆分/重量化后的 per-expert safetensors 目录 |
| `EXPERT_SLOTS` | 每层常驻池**增长天花板**(甜点 256);池按需增长到此才淘汰 |
| `RESIDENT_POOL` | `1` 用连续常驻池(零拷贝 gather);否则旧 stack 路径 |
| `EXPERT_POOL_PROFILE` | 可选的每层**更紧硬上限** JSON;**缺省自动读 `{EXPERT_DIR}/pool_profile.json`**,`none` 关闭(纯 grow-on-demand) |
| `EXPERT_POOL_PROFILE_OUT` / `EXPERT_POOL_MARGIN` | 产出 profile / 冗余系数 |
| `EAGER_EXPERT_LOAD` | `0`(默认)惰性加载专家(读盘并入异步图,~6–11% 提速);`1` 恢复旧的逐专家强制物化 |
| `GPU_REMAP` | `1`(默认)decode 时 GPU 侧 slot 重映射(精确,中性);`0` 走 host 路径 |
| `MTP_OUT` / `QN_CONFIG` | MTP 权重 / 原始 config |
| `MTP_VERIFY_MODE` | `batch`(默认,一次前向验证;递归 cache 自动重放已接受前缀，逐位精确) / `step`(逐 token 解码,亦精确,速度相当) |
| `K` | 每步草稿 token 数(K=2 最优) |
| `MAXTOK` | 生成 token 上限 |

## 目录结构

```
mlx_streaming/
  config.py             集中的环境变量常量访问器(consts/开关)
  core/                 运行时核心库
    mem.py                内存快照/清理工具(最底层)
    profiling.py          PROF/WINDOW_PROF/PREDICT_RECALL_PROF 埋点(跨切面诊断)
    route_trace.py        MoE 路由 trace 记录(ROUTE_TRACE=1,跨切面诊断,probe 消费)
    moe/                  MoE 推理热路径
      gate.py               门控选专家 + 跨层专家预测
      custom_kernel.py      indexed 量化线性 / fused MoE 的自定义 Metal 算子
      compute.py            专家切片计算(PersistentSubGLU / RotatedSubGLU)
      block.py              流式 MoE 块(StreamingMoeBlock / FileStreamingMoeBlock)
      native_moe.py         native fused-MoE 扩展的 Python 集成层(NATIVE_MOE 路径)
    cache/                专家缓存
      resident_pool.py      ResidentExpertPool(每层连续池/slot LRU/GPU 重映射)
      expert_store.py       LruExpertStore / FileExpertStore(每层 LRU + 文件后端)
      blob_loader.py        BlobExpertSource(blob 流式专家源)
    prefetch/             专家预取
      cross_layer.py        跨层预测预取 + 预算 + enable_cross_layer_prefetch
      patch.py              patch_model / patch_model_filebacked
      native_staging.py     native-fused-prefetch staging 管理
      bg_prefetch.py        后台物化预取
  mtp/                  Qwen3-Next MTP 自投机
    qwen3_next_mtp.py     MTP 模块加载(模型)
    kv_cache.py           verify checkpoint + cache 快照/恢复/提交
    generate.py           主模型前向 + 接受判定 + 自投机解码主循环
    drafter.py            MTPDrafter(把 Qwen3NextMTP 包成 draft/sync 接口)
  model_builder.py      共享装配层:build_streaming_model / capture_prenorm_hidden / greedy
  prep/                 离线数据准备(一次性,产出大文件)
    split_experts.py / requantize_experts.py / rotate_requantize_experts.py
    extract_mtp.py / sensitivity_probe.py / derisk_lloydmax.py / download_retry.py
  runtime/              生产推理入口(uv run python -m mlx_streaming.runtime.<name>)
    run_streaming.py / run_mtp_spec.py / run_spec.py / run_baseline.py / run_mmap.py
  tools/                诊断与实验工具(uv run python -m mlx_streaming.tools.<name>)
    probe_*.py            各类性能/正确性探针
    validate_*.py         正确性验证
    simulate_*.py         驱逐/预取预算模拟(被单测复用)
    crosstoken_recall.py  离线分析:从 route_trace 事件算跨-occurrence 预测 recall(被单测复用)
    bench_seqlen.py / profile_streaming.py / pool_footprint.py / env_probe.py
  tests/                pytest 单测
native/                 native 构建源码(与 Python 包区分,产物不入库)
  ext/                  生产 MLX 扩展 native_moe_ext(make -C native/ext native_moe_ext)
  bench/                一次性 de-risk 基准程序(make -C native/bench all)
benchmarks/reports/     实验报告(结论与数据)
docs/superpowers/       spec(设计)与 plan(实现计划)
```

## 平台

仅 Apple Silicon(MLX + Metal,统一内存)。
