<div align="center">

# vates

[English](README.md) | **简体中文**

**在 Apple Silicon 上以约 8.23–8.27 GiB 的 MLX 张量分配在途高水位运行 80B MoE 模型**

面向 MLX 的显存外置流式 MoE 推理引擎，集成 Qwen3-Next MTP 自投机解码。

[![GitHub Stars](https://img.shields.io/github/stars/AMOS144/Vates?style=flat&logo=github&label=Stars)](https://github.com/AMOS144/Vates/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/AMOS144/Vates?style=flat&logo=github&label=Forks)](https://github.com/AMOS144/Vates/forks)
[![GitHub Issues](https://img.shields.io/github/issues/AMOS144/Vates?style=flat&logo=github&label=Issues)](https://github.com/AMOS144/Vates/issues)
[![Last Commit](https://img.shields.io/github/last-commit/AMOS144/Vates?style=flat&logo=git&label=Last%20Commit)](https://github.com/AMOS144/Vates/commits/main)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Required-000000?style=flat&logo=apple&logoColor=white)](https://support.apple.com/guide/mac-help/about-this-mac-system-report-mchlp1176/mac)
[![MLX](https://img.shields.io/badge/MLX-0.31%2B-8A2BE2?style=flat)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[演示](#演示) · [快速开始](#快速开始) · [详细使用](#详细使用) · [常见问题](#常见问题-faq) · [参与贡献](#开发与贡献)

</div>

---

## 项目简介

大型混合专家模型（Mixture of Experts，MoE）的完整权重通常远大于 Mac 的可用统一内存。vates 将绝大部分专家权重保留在磁盘，仅在运行时按需加载并预测式预取当前需要的专家，从而显著降低常驻内存。

项目面向 **Qwen3-Next-80B-A3B 4-bit MLX**：512 个专家、每个 token 激活 10 个专家、48 个 MoE 层和 12 个全注意力层。配合模型自带的 MTP 头，vates 通过自投机解码进一步提高生成吞吐。

> [!NOTE]
> 4-bit 主模型权重在磁盘上约占 41 GB。约 8.23–8.27 GiB 来自仓库基准报告中的 MLX `get_peak_memory` 结果，表示报告配置下的 MLX 张量分配在途高水位；它不是进程 RSS、系统总内存占用，也不证明仅配备相同容量统一内存的设备足以运行。端到端配置在配备 Apple M5（10 核 CPU）、32 GB 物理统一内存和 1 TB 内置 Apple SSD 的 MacBook Pro 上测试。macOS 和非 MLX 分配仍需要额外内存余量。实际占用与速度会受到硬件、上下文长度、模型文件和配置影响。仓库不包含主模型、专家数据或 MTP 权重，也没有同时提供这三类资产的单一下载位置。

## 演示

点击下方封面播放演示视频：

[![vates 演示](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo-poster.png)](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo.mp4)

## 功能亮点

- **低内存推理**：专家权重按需从磁盘流式读取，内存中仅保留小型常驻池与 LFU 侧区缓存。
- **MTP 自投机解码**：使用 Qwen3-Next 自带的 MTP 头生成草稿，由主模型批量验证并接受或回退。
- **零拷贝双源专家池**：真实区与侧区共享统一池，减少 Host 与 Device 之间的重复搬运。
- **原生高性能路径**：C++ 扩展负责并行 `pread`、池状态管理、预测式预取与融合 MoE 计算。
- **长上下文优化**：IsoQuant K4/V3 与 SO(4) 块旋转将 128k 上下文 KV 从约 3.0 GiB 压缩至约 0.68 GiB。
- **正确性检查**：对仓库报告覆盖的确定性贪心 prompts、池状态与配置执行容量不变性、逐字节真值校验及完整测试；这些 oracle 不构成对所有输入和配置普遍“无损”的保证。
- **交互式终端**：提供 Textual 全屏 TUI、纯文本 REPL、流式输出、吞吐与内存状态显示。

## 工作原理

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

1. **零拷贝双源池**：每层仅维护真实区和跨步持久的 LFU 侧区，命中后直接按槽位执行 GPU gather。
2. **C++ 统一池状态**：槽状态、驱逐、按需读取和预取均由原生扩展管理，降低 Python 主线程同步开销。
3. **预测式跨层预取**：根据当前层路由结果预测后续层所需专家，在计算期间提前读取。
4. **动态深度 MTP**：根据草稿置信度调整验证深度，减少低置信步骤中的无效专家加载。
5. **KV 量化**：只压缩 12 个全注意力层的 KV，保持线性注意力层递归状态不变。

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

# 动态获取当前虚拟环境的 Python 库目录并编译原生扩展
PY_SITE="$("./.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
make -C native/ext PY_SITE="$PY_SITE" native_moe_ext
```

> [!TIP]
> 推荐使用 `uv sync`，以复用仓库锁定的依赖版本，避免传递依赖升级后产生兼容性问题。

### 最简使用示例

无需准备模型即可启动演示界面：

```bash
vates --demo
```

准备好模型与专家数据后，在项目根目录运行：

```bash
vates
```

> [!IMPORTANT]
> 仓库不包含模型权重。真实推理前需要自行准备兼容的 Qwen3-Next-80B-A3B 4-bit MLX 主模型、专家数据和 MTP 权重；本项目当前未提供统一下载地址。

## 数据准备

下面分别使用中间 per-expert 文件目录和最终运行时 blob 目录：

```text
models/
├── qwen3_next_80b_4bit/                  # 4-bit MLX 主模型
├── qwen3_next_expert_files_4bit_g64/     # 中间 per-expert 文件
├── qwen3_next_experts_4bit_g64/          # CLI 默认专家目录
│   └── blobs/                            # 最终运行时 blob
└── qn_mtp_weights.safetensors            # MTP 权重
```

专家权重需要转换为每个专家对应一段连续字节的 blob，使运行时读取一个专家只需一次 `pread`：

```bash
# 将堆叠的 switch_mlp 权重拆分为 per-expert 文件
.venv/bin/python -m mlx_streaming.prep.split_experts \
  models/qwen3_next_80b_4bit \
  models/qwen3_next_expert_files_4bit_g64

# 将 per-expert 文件打包到 CLI 默认目录的 blobs 子目录，并生成 blob_index.json
EXPERT_DIR=models/qwen3_next_expert_files_4bit_g64 \
BLOB_DIR=models/qwen3_next_experts_4bit_g64/blobs \
BITS=4 GROUP=64 LAYERS=all \
.venv/bin/python -m mlx_streaming.prep.pack_blob_from_experts

# 从原始模型分片提取并整理 MTP 权重
.venv/bin/python -m mlx_streaming.prep.extract_mtp
```

字节布局由 `mlx_streaming/prep/blob_layout.py` 统一定义，并与运行时 blob loader 保持一致。

> [!WARNING]
> 数据准备所需磁盘空间明显多于约 41 GB 的主模型目录：打包期间中间文件与最终 blob 会同时存在，MTP 源分片还会增加约 3.30 GB（3.07 GiB）。验证 `models/qwen3_next_experts_4bit_g64/blobs` 及其 `blob_index.json` 后，可以删除中间目录 `models/qwen3_next_expert_files_4bit_g64`。41 GB 仅指主模型权重，不是数据准备期间的峰值磁盘占用。

## 详细使用

### 交互式对话

所有命令均应在项目根目录执行：

```bash
# 启动全屏 TUI
vates

# 调整 MTP 投机宽度、生成长度并输出统计信息
vates -k 4 -n 800 --stats

# 设置系统提示词
vates --system "你是一个简洁的助手"

# 不加载模型，直接预览界面
vates --demo

# 终端不兼容时使用纯文本 REPL
vates chat --plain
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
| `--model` | 4-bit MLX 主模型路径 | `models/qwen3_next_80b_4bit` |
| `--expert-dir` | 专家根目录；默认在其 `blobs/` 子目录读取 blob。若直接指定 blob 目录，请设置 `BLOB_DIR` | `models/qwen3_next_experts_4bit_g64` |
| `--mtp-out` | MTP 权重文件 | `models/qn_mtp_weights.safetensors` |
| `--qn-config` | Qwen3-Next 配置文件 | `models/qwen3_next_80b_4bit/config.json` |
| `-k`, `--k` | MTP 投机宽度 | `3` |
| `-n`, `--max-tokens` | 每轮最多生成的新 token 数 | `4096` |
| `--expert-slots` | 常驻专家池容量 | `32` |
| `--spec-slots` | 侧区行数 | 跟随 `--expert-slots` |
| `--system` | 系统提示词 | 无 |
| `--stats` | 输出 token 数、吞吐与接受长度 | 关闭 |
| `--plain` | 使用纯文本 REPL | 关闭 |
| `--demo` | 使用模拟后端预览 TUI | 关闭 |

使用以下命令查看当前版本的完整参数：

```bash
vates chat --help
```

## 配置说明

CLI 会使用 `setdefault` 设置经基准验证的生产快路径，因此用户显式设置的环境变量具有更高优先级。

| 环境变量 | 默认生产值 | 作用 |
| --- | --- | --- |
| `STREAM_BLOB_LOADER` | `1` | 使用 blob 直读处理专家池 miss |
| `ZEROCOPY_DUAL_SOURCE` | `1` | 启用零拷贝双源专家池 |
| `NATIVE_FUSED_PREFETCH` | `1` | 启用原生预测式跨层预取 |
| `SIDEREGION_LFU` | `1` | 启用持久 LFU 侧区缓存 |
| `KV_QUANT` | `1` | 启用 K4/V3 KV 量化与 SO(4) 旋转 |
| `MTP_ADAPTIVE_DEPTH` | `1` | 启用置信度门控动态深度 |
| `MTP_CONF_TAU` | `0.3` | 动态深度置信度阈值 |
| `MTP_DEPTH_MAX` | `3` | 动态深度上限 |

示例：

```bash
# 显式覆盖常驻池和侧区容量
EXPERT_SLOTS=32 POOL_SPEC_SLOTS=16 vates --stats
```

更多实验性开关及默认值以 `mlx_streaming/config.py` 为准。

> [!WARNING]
> `EXPERT_SLOTS` 会影响内存、速度和正确性。当前 K=3、top-k=10 的生产路径将 `32` 作为经过验证的容量下限；修改后应重新执行容量不变性和字节真值校验。下表 cap=48 的 C++ 统一池消融是独立实验，不是 CLI 默认 cap=32 的统一基准。

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

以下数据来自仓库中的消融报告。每项消融均来自独立实验，并非同一轮硬件测试，**不能直接相加**。13–15 tok/s 是多组端到端实验中观察到的大致范围；各实验的配置、提示词、预热长度和输出长度不同，并非单一标准化 benchmark。实际性能取决于设备、模型文件、上下文和配置。

**本文端到端生产配置结果的测试设备**：MacBook Pro，Apple M5（10 核 CPU），32 GB 统一内存，1 TB 内置 Apple SSD。

| 项目 | 结果 |
| --- | --- |
| 存储与内存 | 4-bit 主模型权重在磁盘上约占 41 GB；报告配置的 MLX `get_peak_memory` 约为 8.23–8.27 GiB，表示 MLX 张量分配在途高水位，不是进程 RSS 或系统总内存占用；仍需额外系统内存余量 |
| 生成速度 | 多组端到端实验中观察到约 13–15 tok/s；配置、提示词、预热长度和输出长度不同，因此不是标准化 benchmark |
| KV 占用 | 128k 上下文由约 3.0 GiB 降至约 0.68 GiB |
| 持久 LFU | 命中率由 0.76 提升至约 0.81，实测吞吐提升约 8%–12% |
| C++ 统一池消融 | cap=48、K=3、MAXTOK=48、WARMUP=48、REPEAT=2 时为 13.70 → 14.80 tok/s，侧区由双缓冲改为单缓冲；该 cap 与 CLI 默认 cap=32 不同，不是默认配置的统一基准 |
| MTP top-2 救回 | 在 [top-2 rescue 报告](benchmarks/reports/tree-top2-rescue-2026-07-05.md)覆盖的对应确定性贪心 prompts 与配置上输出逐字节一致，吞吐提升约 10.8% |
| 动态深度 MTP | 在 [动态深度报告](benchmarks/reports/adaptive-depth-2026-07-05.md)覆盖的对应确定性贪心 prompts 与配置上输出逐字节一致，吞吐提升约 5%–6% |
| 峰值优化 | 避免无用 KV 快照后，MLX 分配高水位降低约 0.18–0.22 GiB |

正确性验证包括：

- 容量不变性报告覆盖的确定性贪心 prompts 与配置在不同池容量下输出逐字节一致；
- 使用 `DUAL_VERIFY` 与 `STG_VERIFY` 的报告以逐字节池 oracle 的 `0 BAD` 为验收条件；
- 单次前向单层最大专家并集实测为 30，因此生产配置使用 32 个槽；
- Python 与原生路径均有测试覆盖。

这些 oracle 仅覆盖文档记录的确定性贪心 prompts、池状态和基准配置，是对应案例的回归证据；它们不是对所有 prompt、解码模式、硬件状态或配置均逐字节一致的数学保证。

完整实验记录位于 [`benchmarks/reports/`](benchmarks/reports/)。

已经验证但未纳入生产路径的方向包括完整树形验证、事件门控异步 demand、滑动窗口专家池和逐层容量回收。相关实现或实验因吞吐回归、I/O 预算不足或当前配置无收益而被否决，详情请参阅基准报告。

## 测试

安装开发依赖后运行完整 Python 测试集：

```bash
.venv/bin/python -m pytest
```

仓库当前包含 60 个测试文件，覆盖专家池、双源缓存、blob 布局、MTP、KV 量化、TUI 和配置等模块。性能路径还使用原生测试、容量不变性与逐字节真值校验。

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

可以降级运行，但预测式预取、统一池等生产快路径会被关闭，速度会明显下降。正式推理建议动态获取虚拟环境的 Python 库目录并传给 Make：

```bash
PY_SITE="$("./.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
make -C native/ext PY_SITE="$PY_SITE" native_moe_ext
```

</details>

<details>
<summary><strong>为什么推荐使用 <code>uv sync</code>？</strong></summary>

`uv sync` 会按 `uv.lock` 安装经过验证的依赖组合，避免 `transformers` 等传递依赖漂移到不兼容版本。

</details>

<details>
<summary><strong>模型文件应该放在哪里？</strong></summary>

默认路径位于仓库根目录的 `models/`。也可以通过 `--model`、`--expert-dir`、`--mtp-out` 和 `--qn-config` 指定其他位置。

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
