# mlx-streaming-moe

Apple Silicon(统一内存)上的**显存外置流式 MoE 推理** + **Qwen3-Next MTP 自投机解码**,基于 [MLX](https://github.com/ml-explore/mlx)。

目标:让超大 MoE 模型(如 `Qwen3-Next-80B-A3B`)在小内存机器上跑起来——专家权重按需从 NVMe 流式加载、只把工作集常驻统一内存,同时用模型自带的 MTP 做自投机解码提速。

## 核心结论(详见 `benchmarks/reports/`)

- **MTP 自投机在本机有效**:`Qwen3-Next-80B-A3B` 2-bit 专家 + 256 槽常驻池,K=2 批量验证 → **~24–28 tok/s(2.4–2.5×)**,投机模式下 `disk_load_ratio≈0.14`(I/O 基本消除)。
- **内存右尺寸(无损)**:各层专家工作集不均匀,按 profile 给每层独立池容量后,**常驻专家权重 11.25GB → ~7GB**,峰值 14.6GB → ~10–11GB,命中率/吞吐不变。
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
uv run python -m mlx_streaming.split_experts          # 产出 EXPERT_DIR

# 2) 重量化专家(2-bit / 混合精度，省 SSD 与 I/O)
uv run python -m mlx_streaming.requantize_experts

# 3) 抽取 MTP 权重(自投机用)
uv run python -m mlx_streaming.extract_mtp            # 产出 MTP_OUT / QN_CONFIG
```

## 运行

```bash
# 基线流式解码
EXPERT_DIR=/tmp/qwen3_next_experts_2bit EXPERT_SLOTS=256 RESIDENT_POOL=1 \
  uv run python -m mlx_streaming.run_streaming

# MTP 自投机基准(推荐配置)
EXPERT_DIR=/tmp/qwen3_next_experts_2bit EXPERT_SLOTS=256 RESIDENT_POOL=1 \
MTP_VERIFY_MODE=batch MTP_ARRAY_COMMIT=1 K=2 \
  uv run python -m mlx_streaming.run_mtp_spec
```

### 无损省内存:每层池 profile

```bash
# 1) 产出每层池预算(margin 给未见 prompt 留冗余)
EXPERT_POOL_PROFILE_OUT=/tmp/qn_pool_profile.json EXPERT_POOL_MARGIN=1.15 \
EXPERT_DIR=/tmp/qwen3_next_experts_2bit EXPERT_SLOTS=256 RESIDENT_POOL=1 \
MTP_VERIFY_MODE=batch MTP_ARRAY_COMMIT=1 K=2 \
  uv run python -m mlx_streaming.probe_pool_footprint

# 2) 之后运行加上 profile 即可右尺寸内存
EXPERT_POOL_PROFILE=/tmp/qn_pool_profile.json ...(其余同上)
```

## 关键环境变量

| 变量 | 含义 |
|---|---|
| `MODEL` | 主模型路径(4-bit MLX) |
| `EXPERT_DIR` | 拆分/重量化后的 per-expert safetensors 目录 |
| `EXPERT_SLOTS` | 每层常驻池容量(甜点 256) |
| `RESIDENT_POOL` | `1` 用连续常驻池(零拷贝 gather);否则旧 stack 路径 |
| `EXPERT_POOL_PROFILE` | 每层池预算 JSON(无损省内存) |
| `EXPERT_POOL_PROFILE_OUT` / `EXPERT_POOL_MARGIN` | 产出 profile / 冗余系数 |
| `MTP_OUT` / `QN_CONFIG` | MTP 权重 / 原始 config |
| `MTP_VERIFY_MODE` | `batch`(快,一次前向验证) / `step`(逐位精确,慢) |
| `MTP_ARRAY_COMMIT` | `1` 在 batch 模式直接提交 ArraysCache 检查点 |
| `K` | 每步草稿 token 数(K=2 最优) |
| `MAXTOK` | 生成 token 上限 |

## 目录结构

```
mlx_streaming/          核心包(import 路径仍为 mlx_streaming.*)
  streaming_moe.py        流式 MoE block + patch_model_filebacked
  expert_store.py         FileExpertStore + ResidentExpertPool(每层池/LRU/profile)
  mtp_generate.py         MTP 自投机生成(快照/重放/检查点)
  qwen3_next_mtp.py       MTP 模块加载
  *_experts.py            专家拆分/重量化/旋转
  run_*.py / validate_*.py基准与验证入口
  probe_*.py              诊断探针(I/O、命中率、池高水位、kernel 等)
  tests/                  pytest 单测
benchmarks/reports/     实验报告(结论与数据)
docs/superpowers/       spec(设计)与 plan(实现计划)
```

## 平台

仅 Apple Silicon(MLX + Metal,统一内存)。
