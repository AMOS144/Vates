# vates

**把 80B MoE 的运行内存压到 ~8 GB —— 相对 4-bit 全量 41 GB 约 5× 压缩，几乎任何型号的 Apple Silicon Mac 都能跑。**

vates 是一套面向 Apple Silicon（MLX）的**显存外置流式 MoE 推理引擎** + **MTP 自投机解码**系统。核心思路：把 80B MoE 里绝大部分体积的专家权重留在磁盘（4-bit ≈ 41GB），运行时按需流式加载 + 预测式预取，配合 Qwen3-Next 自带的 MTP 头做自投机解码，让一台 32GB 的 Mac 也能跑完整的 80B 模型。

> 模型规格：Qwen3-Next-80B-A3B（4-bit MLX）——512 专家 / 每 token top_k=10 / 48 个 MoE 层 + 12 个全注意力层。

---

## 演示

[![vates 演示 · 点击播放](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo-poster.png)](https://github.com/AMOS144/Vates/releases/download/v0.1.0/vates-demo.mp4)

> ▶ 点击上图播放完整演示（约 2.5 分钟）。截图为实时生成画面，状态栏可见 `14.1 tok/s · 内存 7.96 GB`。

---

## 亮点

- **装得下**：常驻内存峰值 ≈ **8 GB**（`EXPERT_SLOTS=32`），而模型 4-bit 全量 ≈ 41 GB。专家权重字节在磁盘，内存里只留一个小 LFU 缓存池。
- **跑得动**：生产快路径 ≈ **13–15 tok/s**，叠加 MTP 投机优化后更高。
- **不掉质量**：解码/验证走**逐字节无损**路径——容量不变性（换 cap 输出逐字节一致）+ `STG_VERIFY` 0 BAD 双重把关。
- **长上下文友好**：KV 量化（IsoQuant K4/V3 + SO(4) 块旋转）把 128k 上下文 KV 从 3.0 GiB 压到 **~0.68 GiB**。
- **开箱即用**：`vates` 一条命令进全屏 TUI，状态栏实时显示 token/s 与内存占用。

---

## 工作原理

```
┌──────────────────────────────────────────────────────────────┐
│  vates TUI / CLI  (MTP 自投机快路径)                            │
├──────────────────────────────────────────────────────────────┤
│  MTP 自投机解码   drafter 草稿 → 主模型批量验证 → 接受/回退      │
│    · 置信度门控动态深度   · 最小树 top-2 救回   · 跨轮 KV 复用   │
├──────────────────────────────────────────────────────────────┤
│  流式 MoE 专家池 (零拷贝双源)                                    │
│    真实区(cap) ∪ 侧区(LFU 二级缓存)  ── 单次 GPU gather          │
│    miss → C++ demand 按需 pread → 落池；预测式跨层预取           │
├──────────────────────────────────────────────────────────────┤
│  C++ native 扩展 (nanobind + MLX Primitive)                     │
│    统一池权威 · 后台并行 pread · 融合 MoE kernel · KV 量化       │
├──────────────────────────────────────────────────────────────┤
│  磁盘：per-expert blob（每专家一段连续字节，1 次 pread 取一个）  │
└──────────────────────────────────────────────────────────────┘
```

几个关键设计：

- **零拷贝双源专家池**：每层内存里只保留一个小池 = 真实区（本轮路由命中）∪ 侧区（跨步持久 LFU 缓存）。命中直接用池内槽位做一次 GPU gather，专家字节全程只存一份，不做 host↔device 来回拷贝。
- **C++ 统一池权威**：真实区槽状态、LFU 驱逐、按需 `pread`、预取全部下沉到 C++（`demand_dual`），主线程零落池工作，避免 per-layer 同步打断 MLX 流水线。
- **MTP 自投机解码**：用 Qwen3-Next 自带的 MTP 头预测多个 token，主模型一次前向批量验证，接受则跳步。叠加置信度门控动态深度（低置信步少加载专家）与最小树 top-2 救回。
- **KV 量化**：仅作用于 12 个全注意力层（线性注意力层的递归态不动），长上下文 KV 极致压缩。

---

## 目录结构

```
mlx_streaming/
├── cli.py               # vates 命令入口（默认进 TUI，MTP 快路径）
├── config.py            # 所有环境变量默认值的单一真相源
├── model_builder.py     # 装配层：流式加载主模型 + 挂专家池/预取/KV量化
├── core/
│   ├── cache/           # 专家池：resident_pool / virtual_pool / expert_store
│   │                    #        blob_loader / KV 量化（quant_kv / kv_quant_patch）
│   ├── moe/             # 流式 MoE 块、gate、融合计算、native kernel 封装
│   ├── prefetch/        # 跨层预测式预取、native staging、后台预取
│   ├── linear_attn/     # Qwen3-Next 门控 delta 多态线性注意力
│   └── mem.py           # 内存度量 + 长会话内存防御（cache/wired 封顶）
├── mtp/                 # MTP 自投机：drafter / generate / kv_cache / 裁决
├── prep/                # 离线数据准备：拆专家、打 blob、抽 MTP 权重
├── runtime/             # 各基准入口（环境变量驱动）
├── tools/               # 分析探针：池容量曲线、逐层 profile、召回等
├── tui/                 # Textual 全屏 TUI（opencode 风格）
└── tests/               # 60 个测试文件

native/
├── ext/                 # 生产 C++ 扩展（nanobind + MLX）
│   ├── pool/            #   demand / owned_pool / side_region（统一池权威）
│   ├── io/              #   blob 直读 / 后台并行 pread / IO 计量
│   ├── compute/         #   融合 MoE kernel
│   └── prefetch/        #   预取
└── bench/               # C++ 微基准

docs/superpowers/        # 设计文档（spec）+ 实施计划（plan）
benchmarks/reports/      # 每次优化的消融报告与 GO/NO-GO 裁决
```

---

## 安装

```bash
# 0) 进项目根目录（后续所有命令都在这里跑；默认模型路径是相对路径）
cd /path/to/vates/mlx-streaming-moe

# 1) 按锁文件安装依赖并创建虚拟环境（uv sync 会自动生成 .venv 并装 uv.lock 里验证过的版本）
uv sync

# 2) 激活虚拟环境（vates 命令依赖此步才在 PATH 上）
source .venv/bin/activate

# 3) 编译 native 扩展（生产快路径必需：统一池权威 / 融合 kernel / 并行 pread）
cd native/ext && make native_moe_ext && cd ../..
```

依赖：Python ≥ 3.11、MLX ≥ 0.31、mlx-lm ≥ 0.31、textual ≥ 0.80；编译扩展需 `nanobind`（在 dev 依赖组）与 CMake。

> 用 `uv sync` 而非 `uv pip install -e .`：前者严格按 `uv.lock` 锁定版本，避免传递依赖（如 `transformers`）漂移到不兼容的新版导致加载失败。

> 扩展未编译时会自动降级（关掉预取/统一池），但会明显变慢；生产使用请务必编译。

> **`vates: command not found`？** `vates` 是装进 `.venv/bin/` 的入口脚本，只有**激活 venv**（`source .venv/bin/activate`）后才在 PATH 上；没激活时用全路径 `.venv/bin/vates`。若 venv 是在别的目录名下创建后又移动/改名过，绝对路径会失效，用 `uv venv --clear && uv sync` 重建即可。

---

## 数据准备

主模型（4-bit MLX）正常放 `models/`；专家权重需要离线拆成"每专家一段连续 blob"，这样运行时读一个专家 = 1 次 `pread`。

```bash
# 拆专家：把堆叠的 switch_mlp 权重拆成 per-expert 小文件
.venv/bin/python -m mlx_streaming.prep.split_experts

# 打 blob：per-expert 小文件 → 每层一个连续 blob 文件（+ blob_index.json）
.venv/bin/python -m mlx_streaming.prep.pack_blob_from_experts

# 抽 MTP 权重：从原版末分片抽取并整理成单文件
.venv/bin/python -m mlx_streaming.prep.extract_mtp
```

字节布局由 `prep/blob_layout.py` 统一描述（v1 affine / v2 mxfp4），与运行时 `blob_loader` 完全一致。

---

## 使用：交互式对话

在项目根目录、激活 venv 后，直接 `vates` 进入全屏 TUI（opencode 风格），默认走 MTP 自投机 + 零拷贝双源快路径：

```bash
cd /path/to/vates/mlx-streaming-moe   # 默认模型路径是相对路径，须在项目根目录运行
source .venv/bin/activate             # 每开一个新终端都要先激活

vates                              # 进入交互式对话
vates -k 4 -n 800 --stats          # 调宽投机 / 加长生成 / 打印吞吐
vates --system "你是一个简洁的助手"
vates --demo                       # 免模型秒开界面（验证 UI/流式/状态栏）
vates chat --plain                 # 终端不兼容时用纯文本 REPL
```

未安装入口脚本时可用模块方式：`.venv/bin/python -m mlx_streaming.cli`。

交互：回车发送、`Esc` 中断当前生成、`Ctrl+C` 退出；斜杠命令 `/help`、`/reset`（清历史）、`/clear`（清屏）、`/exit`。状态栏实时显示 token 数 / tok·s / **内存占用（含峰值）**。

### 命令行参数

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--model` | 主模型路径（4-bit MLX） | `models/qwen3_next_80b_4bit` |
| `--expert-dir` | 拆分后的 per-expert / blob 目录 | `models/qwen3_next_experts_4bit_g64` |
| `--mtp-out` | MTP 权重文件 | `models/qn_mtp_weights.safetensors` |
| `-k, --k` | MTP 投机宽度 | `3` |
| `-n, --max-tokens` | 每轮最多生成 token 数 | `4096` |
| `--expert-slots` | 常驻专家池容量（同时作侧区行数默认） | `32` |
| `--spec-slots` | 侧区行数（默认跟随 `--expert-slots`） | 跟随 |
| `--system` | 可选 system 提示词 | 无 |
| `--stats` | 每轮打印 token 数 / tok·s / 接受长度 | 关 |

### 生产快路径默认（`cli._FASTPATH_ENV`）

`vates` 启动时用 `setdefault` 兜底以下经消融验证的最优组合（用户显式导出的环境变量优先级更高）：

| 环境变量 | 作用 |
| --- | --- |
| `STREAM_BLOB_LOADER=1` | blob 直读接入池 miss-loader（低内存路径） |
| `ZEROCOPY_DUAL_SOURCE=1` | 零拷贝双源专家池 |
| `NATIVE_FUSED_PREFETCH=1` | native 预测式跨层预取 |
| `SIDEREGION_LFU=1` | 侧区持久 LFU（单缓冲，省一半侧区内存） |
| `KV_QUANT=1` | KV 极致压缩（K4/V3 + SO(4) 旋转） |
| `MTP_ADAPTIVE_DEPTH=1` + `MTP_CONF_TAU=0.3` + `MTP_DEPTH_MAX=3` | 置信度门控动态深度 |

启动时还会调用内存防御（`setup_memory_hygiene`）封顶 MLX 可回收缓冲（默认 1 GB），防长会话缓存膨胀。更多调优项见 `mlx_streaming/config.py`。

---

## 基准测试

各基准入口在 `mlx_streaming/runtime/`（环境变量驱动），例如生产配方端到端跑：

```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```

`benchmarks/` 下有各专项消融脚本，`benchmarks/reports/` 存放每次优化的报告与裁决。

---

## 研发历程与关键成果

这个项目是一系列"先量化瓶颈、再改、再用消融和字节真值校验裁决 GO/NO-GO"的迭代。以下是主要里程碑（数字均来自 `benchmarks/reports/`）：

### 1. 流式专家池基座

- **侧区持久 LFU 二级缓存**：把侧区从"每步全清的一次性预取"改成跨步 LFU 持久缓存，命中率 0.76 → **0.81**，tok/s **+8~12%**；`spec_slots=8` 时可再省 **~1.9 GB** 内存。
- **预取 width/budget 扫描**：量化 recall ↔ hit ↔ tok/s 的权衡曲线，为专家池容量下限提供数据支撑。

### 2. 瓶颈定位与 C++ 统一池权威

- **VirtualPool Phase 0 探针**：证实主瓶颈是 **per-layer 同步 + 主线程落池**（消除后理论上界 +174%~+387%），而非物理 I/O。
- **demand_dual 异步版**：C++ worker 线程并行 `pread`，cap=32 **+5.9%**、cap=64 **+38.7%**。
- **C++ 统一池权威（Phase 4）**：真实区 + 侧区 + demand + prefetch 全部收进 C++，主线程零落池工作 → tok/s **+8.0%**（13.70 → 14.80），侧区内存**减半**（双缓冲→单缓冲）。随后彻底退役 Python decode 权威路径（净删 ~316 行，tok/s 无回归）。

### 3. 正确性工程体系

- **容量不变性 oracle**：同 prompt、greedy，换 `cap` 输出必须逐字节一致；坐实错槽类 bug。
- **字节真值校验**：`DUAL_VERIFY` / `STG_VERIFY` 以"0 BAD"为硬验收线。
- **并集实测定 cap 下限**：单前向单层最大专家并集 **U_max=30**（=K3×top_k10），故 `EXPERT_SLOTS=32` 是正确性下限（仅 2 槽裕度）。
- **错槽修复链**：侧区 Route 1（稳定缓冲 + scatter 发布）→ Route 3（C++ 拥有池 buffer 直写，修复 Route 1 引入的 **15× 性能回归**：0.89 → 13.22 tok/s）→ `demand_dual` 修 `contiguous(inds)`（修复 seq≥2 按连续内存读非连续 `inds` 导致装错专家的隐蔽回归）。

### 4. MTP 投机解码提速

- **最小树 top-2 救回（`TREE_TOP2`）**：串行第二次 1×K 前向、不放大专家并集，修 bug 后 **+10.8% tok/s，逐字节无损**。
- **置信度门控动态深度（`MTP_ADAPTIVE_DEPTH`）**：低置信步向下收缩草稿深度、少加载专家，**+5~6% tok/s，逐字节无损** → 设为默认。
- **峰值内存优化**：批量验证路径跳过无用的 cache 快照（每步 72 MiB 深拷贝），峰值 **−0.2 GB**，零速度代价。

### 5. 已探索但否决的方向（诚实存档）

- **完整树形验证（多路径 batch）**：并集放大致 cap 溢出，cap=32 时 tok/s 仅 **0.46×** 且有损 → **NO-GO**。
- **事件门控异步 demand**：机制正确性成立（spike 0/200），但端到端零加速——真瓶颈是 **~17s 的 miss 磁盘 I/O**，非 per-layer 同步 → 实现回滚。
- **滑动窗口专家池**：想靠"只常驻 N 层、其余按需搬运"进一步省内存，实测 SSD 只有 ~25% 空闲带宽、无可利用"气泡"，重载量超预算 11–22× → **NO-GO**。
- **逐层容量 profile（`pool_profile.json`）**：当年 cap=256 时代的省内存工具；实测当前每层真实工作集 122–339（均值 192）远超 cap=32，无"用不满"的层可回收，在 cap=32 下是 no-op → 不启用。

> 各优化收益为独立测得、非严格可加；且部分互斥（如动态深度与 pos0 救回不能叠加，`−5.34%`），生产路径需按场景取舍。

---

## 测试

```bash
.venv/bin/python -m pytest        # 60 个测试文件，覆盖池/双源/MTP/KV量化/TUI 等
```

C++ 侧关键路径另有 native 单测（如 `io_meter`、`demand_dual` 布线）与字节真值校验。

---

## 设计理念

- **测量优先**：每个优化先做探针量化瓶颈上界，再决定要不要做。
- **字节不变量为准**：性能优化一律以逐字节无损 / 容量不变性为验收线，不接受"看起来差不多"。
- **诚实存档**：否决的方向连同数字一起记进 `benchmarks/reports/`，避免重复踩坑。
