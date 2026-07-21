# vates 知乎与小红书营销内容实施计划

> **面向执行代理：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，按任务逐项实施。所有步骤使用复选框（`- [ ]`）跟踪。

**目标：** 产出可直接发布的知乎长文、小红书正文、标题、标签、行动号召和配图逐页文案。

**架构：** 以 README 和基准报告为唯一事实来源，先建立共用的数据与措辞边界，再为知乎和小红书分别重写。知乎采用工程叙事和技术论证，小红书采用短段落、类比和视觉数据卡。

**技术栈：** 中文 Markdown、知乎长文、小红书图文、GitHub 项目资料

---

### 任务 1：建立事实与传播素材清单

**文件：**
- 读取：`README.md`
- 读取：`benchmarks/reports/sideregion-lfu-2026-07-01.md`
- 读取：`benchmarks/reports/cpp-unified-pool-final-2026-07-04.md`
- 读取：`benchmarks/reports/tree-top2-rescue-2026-07-05.md`
- 读取：`benchmarks/reports/adaptive-depth-2026-07-05.md`
- 读取：`benchmarks/reports/peak-shrink-2026-07-03.md`

- [ ] **步骤 1：锁定可使用的数据**

两个版本仅使用以下数字：

- 4-bit 权重约 41 GB；
- 运行内存峰值约 8 GB；
- 生产路径约 13–15 tok/s；
- 128k KV 约 3.0 GiB → 0.68 GiB；
- LFU 命中率约 0.76 → 0.81；
- C++ 统一池 13.70 → 14.80 tok/s；
- MTP top-2 独立实验约 +10.8%；
- 动态深度 MTP 独立实验约 +5%–6%；
- 峰值优化约降低 0.18–0.22 GB。

- [ ] **步骤 2：锁定限定语**

每个包含性能数据的版本必须说明“项目实测，结果因设备、模型和配置而异”；不得将独立优化收益相加，不得宣称所有 Mac 都能运行。

### 任务 2：撰写知乎长文

**文件：**
- 新建：`docs/marketing/zhihu-vates-launch.md`

- [ ] **步骤 1：完成标题与摘要**

主标题使用：

```text
41GB 的 80B MoE，8GB 峰值跑起来了：Vates 开源
```

副标题使用“把 512 个专家留在 SSD，重新设计 Apple Silicon 上的 MoE 推理数据路径”。同时提供 5 个备用标题；正文不使用论文摘要，而是以挑战默认假设的冷开场直接进入故事。

- [ ] **步骤 2：完成正文**

正文长度控制在 2500–3500 个中文字符，按以下顺序写作：

1. 质疑“模型大小约等于所需内存”的默认假设；
2. 首屏展示 `41GB → SSD Streaming → 约 8GB → 13–15 tok/s`；
3. MoE 每个 token 只激活部分专家，以及“专家留在 SSD”的核心方案；
4. 零拷贝双源池、LFU、C++ `pread`、MTP 和 KV 量化；
5. 以“模型悄悄变笨”为切口讲容量不变性与逐字节校验；
6. 实测数据；
7. 以“聪明方案实测后反而更蠢”为切口讲 NO-GO 实验；
8. 适用人群、限制和模型准备要求；
9. 用“本地 AI 可能缺少的是另一种运行方式”收束，再给出演示、GitHub 和行动号召。

- [ ] **步骤 3：补充发布素材**

文章末尾提供：

- 3 个文中插图位置及图意；
- 10 个知乎话题标签；
- 一段可单独用于动态转发的 100 字推荐语。

### 任务 3：撰写小红书图文

**文件：**
- 新建：`docs/marketing/xiaohongshu-vates-launch.md`

- [ ] **步骤 1：完成标题与正文**

主标题使用：

```text
Mac 跑 80B 大模型，我把运行内存压到了约 8 GB
```

正文控制在 600–900 个中文字符，每段 1–3 句，使用不超过 6 个 emoji。正文必须包含结果反差、通俗原理、四个亮点、适用限制和搜索 GitHub 项目的行动号召。

- [ ] **步骤 2：完成 8 张配图文案**

逐页输出：

1. 封面：80B / 41 GB → 约 8 GB；
2. TUI 演示；
3. 512 个专家住在 SSD；
4. 运行内存对比；
5. 13–15 tok/s；
6. KV 3.0 GiB → 0.68 GiB；
7. 工作原理；
8. GitHub 开源与行动号召。

每页包含主标题、副标题、建议画面和数据脚注。

- [ ] **步骤 3：补充发布信息**

提供 5 个备用标题、10 个小红书标签和一条置顶评论文案。

### 任务 4：验证内容

**文件：**
- 验证：`docs/marketing/zhihu-vates-launch.md`
- 验证：`docs/marketing/xiaohongshu-vates-launch.md`

- [ ] **步骤 1：检查事实**

搜索两个文件中的所有数字，逐项与任务 1 的素材清单核对。

- [ ] **步骤 2：检查禁用表达**

确认不存在“所有 Mac”“绝对无损”“全球首个”“吊打”“封神”“保姆级”等无依据或低可信表达。

- [ ] **步骤 3：检查平台差异**

确认知乎版不是 README 罗列，小红书版不是知乎版的机械缩写；两个版本都包含平台适配的标题、结构和行动号召。

- [ ] **步骤 4：检查 Markdown**

运行 `git diff --check`，并检查代码围栏和本地链接完整。

### 任务 5：制作知乎插图

**文件：**
- 新建：`docs/marketing/assets/generate_zhihu_images.py`
- 新建：`docs/marketing/assets/zhihu/01-vates-memory-wall.png`
- 新建：`docs/marketing/assets/zhihu/02-vates-cross-layer-prefetch.png`
- 新建：`docs/marketing/assets/zhihu/03-vates-benchmark-results.png`
- 修改：`docs/marketing/zhihu-vates-launch.md`

- [ ] **步骤 1：编写可重复生成脚本**

使用 Pillow 和 macOS 中文系统字体生成 1600×900 PNG。脚本统一定义背景、网格、卡片、箭头、标题、脚注与 Vates 标识，不依赖外部图片。

- [ ] **步骤 2：生成三张信息图**

运行：

```bash
uv run --with pillow python docs/marketing/assets/generate_zhihu_images.py
```

预期生成三张 1600×900 RGB PNG，分别表达内存结果、跨层预取和实验数据。

- [ ] **步骤 3：插入知乎文章**

将三张图片分别插入首屏数据块之后、跨层预取说明之后和性能结果段落之后，使用描述性 alt 文本和仓库相对路径。

- [ ] **步骤 4：验证图片**

使用 Pillow 检查三张图片均为 1600×900、RGB/RGBA 且文件大小大于 20 KB；检查文章中的三个图片链接均指向实际文件。

### 任务 6：制作小红书发布成品

**文件：**
- 修改：`docs/marketing/xiaohongshu-vates-launch.md`
- 新建：`docs/marketing/xiaohongshu-vates-paste-ready.txt`
- 新建：`docs/marketing/assets/generate_xiaohongshu_images.py`
- 新建：`docs/marketing/assets/xiaohongshu/01-cover.png`
- 新建：`docs/marketing/assets/xiaohongshu/02-moe-routing.png`
- 新建：`docs/marketing/assets/xiaohongshu/03-ssd-expert-pool.png`
- 新建：`docs/marketing/assets/xiaohongshu/04-cross-layer-prefetch.png`
- 新建：`docs/marketing/assets/xiaohongshu/05-memory-and-speed.png`
- 新建：`docs/marketing/assets/xiaohongshu/06-long-context-kv.png`
- 新建：`docs/marketing/assets/xiaohongshu/07-correctness-and-no-go.png`
- 新建：`docs/marketing/assets/xiaohongshu/08-open-source.png`

- [x] **步骤 1：润色正文**

统一项目名为 `Vates`，保留现有短段落和轻量 emoji；将预取说明改为“用未来层自己的 gate 预测候选专家，再由 C++ 在中间层计算期间异步 `pread`”，避免只写成模糊的“提前猜”。

- [x] **步骤 2：生成可复制纯文本**

输出只包含主标题、正文、10 个标签和置顶评论的 UTF-8 纯文本，不包含备用标题、配图说明或 Markdown 标记。

- [x] **步骤 3：生成 8 张竖版信息图**

使用 Pillow 和 macOS 中文系统字体生成 1200×1600 PNG，统一 Vates 品牌、页码、脚注和 GitHub 搜索入口。

- [x] **步骤 4：验证发布成品**

确认 8 张图片均为 1200×1600、RGB/RGBA、文件大小大于 20 KB；纯文本不包含 Markdown 标题或辅助章节；正文与图片中的全部数字均来自已验证素材清单。
