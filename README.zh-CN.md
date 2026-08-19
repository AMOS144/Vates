<div align="center">

# vates

[English](README.md) | **简体中文**

**在 Apple Silicon 上以可复现的磁盘流式 K=3 路径运行 Qwen3-Next-80B-A3B**

面向 MLX 的显存外置流式 MoE 推理引擎，集成 Qwen3-Next MTP 自投机解码。

[![GitHub Stars](https://img.shields.io/github/stars/AMOS144/Vates?style=flat&logo=github&label=Stars)](https://github.com/AMOS144/Vates/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/AMOS144/Vates?style=flat&logo=github&label=Forks)](https://github.com/AMOS144/Vates/forks)
[![GitHub Issues](https://img.shields.io/github/issues/AMOS144/Vates?style=flat&logo=github&label=Issues)](https://github.com/AMOS144/Vates/issues)
[![Last Commit](https://img.shields.io/github/last-commit/AMOS144/Vates?style=flat&logo=git&label=Last%20Commit)](https://github.com/AMOS144/Vates/commits/main)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Required-000000?style=flat&logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-report-mchlp1176/mac)
[![MLX](https://img.shields.io/badge/MLX-0.31%2B-8A2BE2?style=flat)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[快速开始](#快速开始) · [详细使用](#详细使用) · [常见问题](#常见问题-faq) · [参与贡献](#开发与贡献)

</div>

---

## 项目简介

大型混合专家模型（Mixture of Experts，MoE）的完整权重通常远大于 Mac 的可用统一内存。vates 将绝大部分专家权重保留在磁盘，仅在运行时按需加载并预测式预取当前需要的专家，从而显著降低常驻内存。

项目面向 **Qwen3-Next-80B-A3B 4-bit MLX**：512 个专家、每个 token 激活 10 个专家、48 个 MoE 层和 12 个全注意力层。配合模型自带的 MTP 头，vates 通过自投机解码进一步提高生成吞吐。

> [!NOTE]
> 按当前源分片头精确估算，准备完成的运行目录约占 42.7 GiB（`du` 约显示 43 GB，低于约 44 GB 的目标），下载的 4-bit 主模型源权重本身约为 41.8 GiB。[PR #1](https://github.com/AMOS144/Vates/pull/1) 合入的固定配置实测 MLX 活跃内存约 10.96 GiB、MLX 峰值约 11.51 GiB。它们都不是进程 RSS 或系统总内存，也不代表同等容量的统一内存足以运行。macOS、映射文件、原生分配、文件系统缓存和其他应用仍需要额外余量。实际内存与速度会受到硬件、提示词长度、模型文件和缓存热度影响。仓库不包含模型权重。

## 功能亮点

- **低内存推理**：专家权重按需从磁盘流式读取，内存中仅保留有界的主模型与 MTP 专家池。
- **有界 Expert-major Prefill**：按路由专家聚合 prompt token，避免旧 token-major 大激活路径。
- **MTP 自投机解码**：使用 Qwen3-Next 自带的 MTP 头生成草稿，由主模型批量验证并接受或回退。
- **固定 K=3 Decode 配置**：将主模型池、4-bit 流式 MTP 池、物理读取预算、重排策略和预取层固定为一套可复现配置。
- **原生高性能路径**：C++ 扩展负责并行 `pread`、池状态管理、预测式预取与融合 MoE 计算。
- **长上下文优化**：IsoQuant K4/V3 与 SO(4) 块旋转将 128k 上下文 KV 从约 3.0 GiB 压缩至约 0.68 GiB。
- **正确性检查**：提供定向回归测试、容量不变性、逐字节池 oracle 与 32K Prefill 验证器。
- **非阻塞交互终端**：分词解码与 Textual 渲染不进入推理热路径，TUI 分别显示 Prefill 和 Decode 指标。

## 工作原理

每轮包含两个独立计时、I/O 策略不同的阶段：

```text
prompt → PREFILL → 首 token 边界 → DECODE → token 流
           │                           │
           ├─ Expert-major MoE          ├─ K=3 MTP 草拟 + 批量验证
           ├─ 同步专家读取             ├─ 异步 demand + 原生预取
           └─ 构建 KV/递归状态        └─ 非阻塞 detokenize + TUI
```

两个阶段共用同一套磁盘流式原生运行栈：

```text
┌──────────────────────────────────────────────────────────────┐
│  vates TUI / CLI                                             │
├──────────────────────────────────────────────────────────────┤
│  MTP 自投机解码                                               │
│  drafter 草稿 → 主模型批量验证 → 接受或回退                   │
├──────────────────────────────────────────────────────────────┤
│  流式 MoE 专家池                                              │
│  真实区（cap）∪ 侧区（LFU）→ 单次 GPU gather                  │
│  miss → C++ demand 按需 pread → 落池；跨层预测式预取          │
├──────────────────────────────────────────────────────────────┤
│  C++ 原生扩展                                                 │
│  统一池状态 · 后台并行 pread · 融合 MoE · KV 量化             │
├──────────────────────────────────────────────────────────────┤
│  磁盘：per-expert blob，每个专家对应一段连续字节               │
└──────────────────────────────────────────────────────────────┘
```

核心机制：

1. **有界专家池**：主模型使用仓库内固定的逐层容量配置，MTP 使用独立的 256 槽 4-bit 池；缺失专家从 SSD 读入对应池。
2. **C++ 统一池状态**：槽状态、驱逐、按需读取和预取均由原生扩展管理，降低 Python 主线程同步开销。
3. **预测式跨层预取**：根据当前层路由结果预测后续层所需专家，在计算期间提前读取。
4. **动态深度 MTP**：根据草稿置信度调整验证深度，减少低置信步骤中的无效专家加载。
5. **KV 量化**：只压缩 12 个全注意力层的 KV，保持线性注意力层递归状态不变。

### Prefill 阶段：摄入提示词

Prefill 处理尚未缓存的完整提示词，并构建后续生成需要的 KV cache 和线性注意力递归状态。它的路由专家并集远宽于 Decode，因此生产路径在这一阶段刻意使用**同步**专家读取。

Expert-major 实现按照路由专家聚合 token，并复用一个有界临时 bank，避免旧 token-major 大激活路径，使内存由固定 superblock 约束，而不是随提示词长度无界增长。确定性 reduction 保持 canonical route-rank 累加顺序。仓库还提供 32K 边界验证工具，对比多次运行的 logits、hidden state、argmax、cache offset 和内存。

Prefill 延迟决定首 token 等待时间，不是 Decode 吞吐。CLI 和 TUI 会单独显示 Prefill token 数、耗时和 tok/s。多轮聊天中，只有在旧 cache 是新提示词的权威前缀时才复用，并且只 Prefill 新增后缀；前缀不成立时会丢弃 cache 并重新构建。

### Decode 阶段：生成新 token

在精确的 Prefill/Decode 边界，固定配置才启用异步专家 demand 和预测式跨层预取。Decode 使用 Qwen3-Next 的 4-bit 流式 MTP 专家一次草拟最多 K=3 个 token，再由主模型批量验证；只有通过 target 验证的 token 才会提交，低置信步骤可以提前停止在更浅的动态深度。

主模型专家池使用仓库内固定的逐层容量配置、C++ 原生池状态、批量 `preadv` demand 读取、候选重排和物理 SSD 读取预算。K4/V3 旋转 KV 量化只作用于 12 个全注意力层，36 个线性注意力层的递归状态保持未量化。

Decode 吞吐使用引擎在 Prefill 结束后启动的 decode-only `wall_s`。token detokenize 在独立线程完成，只把最新文本快照发布给 TUI，不向推理线程施加反压；最终文本再使用完整权威 token 序列校正。

## 技术栈

| 类别 | 技术 | 用途 |
| --- | --- | --- |
| 运行时 | Python 3.11+ | CLI、模型装配、推理流程与工具 |
| 推理框架 | MLX 0.31+、mlx-lm 0.31+ | Apple Silicon 统一内存推理 |
| 数值计算 | NumPy 2.0+ | 数据准备与数值处理 |
| 终端界面 | Textual 0.80+ | 全屏交互式 TUI |
| 原生扩展 | C++、nanobind、MLX Primitive | 专家池、I/O、预取与融合计算 |
| 构建工具 | uv、CMake、Make | 依赖管理与原生扩展构建 |
| 测试 | pytest、pytest-asyncio | 单元测试、集成测试与正确性验证 |

## 快速开始

### 环境要求

- Apple Silicon Mac
- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- CMake、Make 与可用的 C++ 编译环境
- 真实推理所需的模型文件和足够的本地磁盘空间
- 超出 MLX 分配高水位的充足统一内存余量

### 安装与原生构建

```bash
# 克隆仓库
git clone https://github.com/AMOS144/Vates.git
cd Vates

# 按 uv.lock 创建虚拟环境并安装依赖
uv sync
source .venv/bin/activate

# 传入虚拟环境 Python 的绝对路径（make -C 会切换工作目录）
VATES_PYTHON="$PWD/.venv/bin/python"
make -C native/ext PYTHON="$VATES_PYTHON" native_moe_ext
```

> [!TIP]
> 推荐使用 `uv sync`，以复用仓库锁定的依赖版本，避免传递依赖升级后产生兼容性问题。

### 最简使用示例

无需准备模型即可启动演示界面：

```bash
vates --demo
```

准备好下文的紧凑运行目录后，在项目根目录启动实测固定配置：

```bash
vates --stats
```

公开 `vates` 命令会在导入模型运行时之前锁定完整配置。内部实验和 benchmark runner 仍供开发者使用，但不是支持的用户入口。

## 数据准备

现在推荐只运行一条命令。它会下载官方 4-bit MLX 主模型和 Qwen 原版 MTP 分片，直接完成转换，完整读取输出计算并复核 SHA-256，验证成功后再删除原始权重分片：

```bash
vates prepare --download
```

最终生成可直接运行的 `models/vates-runtime`：

```text
models/vates-runtime/
├── model/                 # 删除路由专家后的紧凑主模型核心
├── experts/blobs/         # 从原始分片直写的 48 层主专家 blob
├── mtp/core.safetensors   # 紧凑 MTP 核心
├── mtp/experts/           # 单个 4-bit MTP 专家 blob
└── vates_manifest.json    # 文件大小与 SHA-256 完整性记录
```

`vates prepare` 不会再生成原来的 24,576 个 per-expert 中间文件。它在保存非专家主模型核心的同时，将堆叠的 `switch_mlp` 张量直接写到最终 blob 的对应偏移；MTP 也直接拆成紧凑核心和 4-bit/group-64 专家 blob。按当前源分片头估算，最终目录约为 42.7 GiB（`du` 约显示 43 GB，少量差异取决于模型元数据）。

如果希望自己先下载，使用以下 Hugging Face CLI 命令：

```bash
hf download mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit \
  --local-dir models/.vates-source/main

hf download Qwen/Qwen3-Next-80B-A3B-Instruct \
  model-00041-of-00041.safetensors \
  --local-dir models/.vates-source/mtp

vates prepare
```

也可以直接指定已有权重：

```bash
vates prepare \
  --main-source /path/to/Qwen3-Next-80B-A3B-Instruct-4bit \
  --mtp-source /path/to/model-00041-of-00041.safetensors
```

> [!CAUTION]
> 默认会清理源权重，但只在结构检查和输出哈希复核全部成功后执行。需要保留原始分片时加 `--keep-source`。命令不会覆盖已经存在的输出目录。

> [!NOTE]
> 下载与转换在同一轮执行时，请预留约 90 GiB 临时空间，因为验证前源权重与最终输出会短暂共存。成功清理后只保留约 42.7 GiB 的运行目录（另有可忽略的下载元数据）。旧流程会额外复制 per-expert 文件，准备目录一度约 128 GB；新流程已去掉这部分重复占用。

## 详细使用

### 交互式对话

所有命令均应在项目根目录执行：

```bash
# 以固定 K=3 配置启动全屏 TUI
vates

# 调整生成长度，并分别输出 Prefill/Decode 统计
vates -n 800 --stats

# 设置系统提示词
vates --system "你是一个简洁的助手"

# 不加载模型，直接预览界面
vates --demo

# 终端不兼容时使用纯文本 REPL
vates --plain --stats
```

未激活虚拟环境时，也可以直接使用：

```bash
.venv/bin/vates --demo
# 或以 Python 模块方式运行
.venv/bin/python -m mlx_streaming.cli --demo
```

TUI 支持以下操作：

| 操作 | 说明 |
| --- | --- |
| `Enter` | 发送消息 |
| `Esc` | 中断当前生成 |
| `Ctrl+C` | 退出程序 |
| `/help` | 显示帮助 |
| `/reset` | 清空对话历史 |
| `/clear` | 清空屏幕 |
| `/exit` | 退出程序 |

### 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | 4-bit MLX 紧凑主模型核心路径 | `models/vates-runtime/model` |
| `--expert-dir` | 专家根目录；默认在其 `blobs/` 子目录读取 blob。若直接指定 blob 目录，请设置 `BLOB_DIR` | `models/vates-runtime/experts` |
| `--mtp-out` | 紧凑 MTP 核心文件 | `models/vates-runtime/mtp/core.safetensors` |
| `--qn-config` | Qwen3-Next 配置文件 | `models/vates-runtime/model/config.json` |
| `-k`, `--k` | MTP 投机宽度；由公开 CLI 固定 | `3` |
| `-n`, `--max-tokens` | 每轮最多生成的新 token 数 | `4096` |
| `--expert-slots` | 主模型专家池容量；由公开 CLI 固定 | `152` |
| `--spec-slots` | 旧侧区行数；由公开 CLI 固定 | `0` |
| `--system` | 系统提示词 | 无 |
| `--stats` | 输出 token 数、吞吐与接受长度 | 关闭 |
| `--plain` | 使用纯文本 REPL | 关闭 |
| `--demo` | 使用模拟后端预览 TUI | 关闭 |

使用以下命令查看当前版本的完整参数：

```bash
vates --help
```

## 配置说明

公开 `vates` 入口是实测固定配置的权威入口。它会在导入推理运行时前安装配置，并主动覆盖性能相关环境变量，避免旧 shell 变量悄然改变结果。关键约束是：

- **Prefill**：Expert-major 提示词摄入、同步 demand 与固定有界 superblock；
- **Decode**：K=3 批量 MTP 验证、最大深度 3，并在阶段边界后启用异步 demand；
- 4-bit 流式 MTP 专家与 256 个 MTP 槽；
- 152 个主池槽与仓库内固定的逐层容量覆盖；
- K4/V3 旋转 KV 量化；
- 原生预测式预取与固定候选重排、目标层、物理读取预算。

模型路径、system prompt 和最大生成 token 数仍然是 CLI 参数。公开命令会固定池容量和关键配置；开发者测试变体时应使用内部 benchmark runner，且不能将结果标成固定配置的实测数据。

## 项目结构

```text
.
├── mlx_streaming/
│   ├── cli.py                 # vates 命令入口
│   ├── config.py              # 环境变量与默认值
│   ├── model_builder.py       # 流式模型装配
│   ├── core/
│   │   ├── cache/             # 专家池、blob loader 与 KV 量化
│   │   ├── moe/               # 流式 MoE、gate 与融合计算
│   │   ├── prefetch/          # 跨层预测与后台预取
│   │   └── linear_attn/       # Qwen3-Next 线性注意力
│   ├── mtp/                   # MTP 草稿、验证与 KV 复用
│   ├── prep/                  # 专家拆分、blob 打包与权重提取
│   ├── runtime/               # 基准与运行入口
│   ├── tools/                 # 分析和诊断工具
│   ├── tui/                   # Textual 全屏界面
│   └── tests/                 # Python 测试
├── native/
│   ├── ext/                   # 生产 C++/nanobind 扩展
│   └── bench/                 # 原生微基准
├── benchmarks/
│   └── reports/               # 消融实验与性能报告
├── docs/
│   └── superpowers/           # 设计规范与实施计划
├── pyproject.toml             # 项目元数据和依赖
└── uv.lock                    # 锁定的依赖版本
```

## 性能与关键成果

当前生产数据来自 [PR #1](https://github.com/AMOS144/Vates/pull/1) 合入的固定配置。`benchmarks/reports/` 中的旧报告是不同池容量、提示词、预热长度和时钟口径下的独立消融，其收益**不能直接相加**。

| 项目 | 结果 |
| --- | --- |
| 存储与内存 | 准备完成的运行目录约 42.7 GiB（`du` 约显示 43 GB）；固定配置的 MLX 活跃内存约 10.96 GiB、MLX 峰值约 11.51 GiB，都不是进程 RSS 或系统总内存 |
| Prefill | 单独报告 prompt token 数、秒数和 tok/s；同步执行，不计入 Decode 速度。首 token 延迟取决于未缓存提示词长度和 SSD 状态 |
| 稳态 Decode | 已有固定 128-token 稳态实测约 31–37 tok/s；缓存热度和 prompt 会显著影响结果 |
| TUI 开销 A/B | 同一热池对比中，异步 TUI 流式输出 31.65 tok/s，完全关闭流式输出 31.76 tok/s，差异约 0.35% |
| Decode 时钟 | 使用 Prefill 结束后才启动的引擎 decode-only `wall_s`；detokenize 和 UI 渲染不运行在推理热路径 |
| KV 占用 | 128k 上下文由约 3.0 GiB 降至约 0.68 GiB |
| 长 Prefill 验证 | 提供 32K 边界工具，对比 logits、hidden state、argmax、cache offset 和内存 |

正确性验证包括：

- 容量不变性报告覆盖的确定性贪心 prompts 与配置在不同池容量下输出逐字节一致；
- 使用 `DUAL_VERIFY` 与 `STG_VERIFY` 的报告以逐字节池 oracle 的 `0 BAD` 为验收条件；
- Expert-major 确定性 reduction 保持 canonical route-rank 累加顺序；
- 主模型批量验证只提交 target 已验证 token；
- TUI 最终文本使用完整权威 token 序列校正；
- Python 与原生路径均有测试覆盖。

这些 oracle 仅覆盖文档记录的确定性贪心 prompts、池状态和基准配置，是对应案例的回归证据；它们不是对所有 prompt、解码模式、硬件状态或配置均逐字节一致的数学保证。

完整实验记录位于 [`benchmarks/reports/`](benchmarks/reports/)。

已经验证但未纳入生产路径的方向包括完整树形验证、事件门控异步 demand、滑动窗口专家池和逐层容量回收。相关实现或实验因吞吐回归、I/O 预算不足或当前配置无收益而被否决，详情请参阅基准报告。

## 测试

安装开发依赖后运行完整 Python 测试集：

```bash
.venv/bin/python -m pytest
```

测试覆盖 Expert-major Prefill、专家池、blob 布局、MTP、KV 量化、预测式预取、原生 I/O、公开命令入口、TUI 和流式 detokenize。性能路径还使用原生测试、容量不变性、逐字节真值校验与 32K Prefill 验证器。

## 常见问题 FAQ

<details>
<summary><strong>为什么只支持 Apple Silicon？</strong></summary>

项目基于 MLX，并依赖 Apple Silicon 的统一内存架构及 MLX 原生能力。当前没有 CUDA、ROCm 或 CPU-only 后端。

</details>

<details>
<summary><strong>为什么提示 <code>vates: command not found</code>？</strong></summary>

`vates` 安装在 `.venv/bin/` 中。请先运行 `source .venv/bin/activate`，或直接执行 `.venv/bin/vates`。如果虚拟环境创建后被移动或重命名，请使用 `uv venv --clear && uv sync` 重建。

</details>

<details>
<summary><strong>不编译 native 扩展可以运行吗？</strong></summary>

可以降级运行，但预测式预取、统一池等生产快路径会被关闭，速度会明显下降。正式推理应将虚拟环境的 Python 可执行文件传给 Make：

```bash
VATES_PYTHON="$PWD/.venv/bin/python"
make -C native/ext PYTHON="$VATES_PYTHON" native_moe_ext
```

</details>

<details>
<summary><strong>为什么推荐使用 <code>uv sync</code>？</strong></summary>

`uv sync` 会按 `uv.lock` 安装经过验证的依赖组合，避免 `transformers` 等传递依赖漂移到不兼容版本。

</details>

<details>
<summary><strong>模型文件应该放在哪里？</strong></summary>

运行 `vates prepare --download`，默认会生成 `models/vates-runtime`。需要整体换位置时设置 `VATES_RUNTIME_DIR`；也可以通过 `--model`、`--expert-dir`、`--mtp-out` 和 `--qn-config` 分别指定路径。

</details>

<details>
<summary><strong>如何在没有模型的情况下检查界面？</strong></summary>

运行 `vates --demo`。该模式使用模拟后端，不读取模型文件，可用于检查 TUI、流式显示和状态栏。

</details>

## 开发与贡献

欢迎提交 Issue 和 Pull Request。开始前请阅读 [贡献指南](CONTRIBUTING.md) 和 [行为准则](CODE_OF_CONDUCT.md)。

1. 在开始较大改动前，建议先创建 Issue，说明问题、目标和预期方案。
2. 从 `main` 创建独立分支，避免在一个 PR 中混入无关改动。
3. 保持改动聚焦，并为行为变化补充或更新测试。
4. 提交前运行 `.venv/bin/python -m pytest`。
5. PR 描述应包含改动背景、实现方式、验证结果和潜在影响。
6. 性能优化应附可复现的基准命令、对照数据和正确性验证结果。
7. Bug 报告请提供设备型号、macOS 版本、Python 版本、复现命令和完整错误信息。

## License

本项目采用 [Apache License 2.0](LICENSE)。

Copyright 2026 AMOS144

## 作者与联系方式

- 作者：[AMOS144](https://github.com/AMOS144)
- 仓库：[github.com/AMOS144/Vates](https://github.com/AMOS144/Vates)
- Issue：[github.com/AMOS144/Vates/issues](https://github.com/AMOS144/Vates/issues)
- 邮箱：[3108424075@qq.com](mailto:3108424075@qq.com)

---

欢迎通过 [Issue](https://github.com/AMOS144/Vates/issues) 提交技术反馈或参与贡献。
