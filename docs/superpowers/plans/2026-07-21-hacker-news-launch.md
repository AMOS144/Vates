# Vates Hacker News 发布实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 GitHub 首页国际化，并生成一套可直接用于 `Show HN` 的英文发布材料。

**Architecture:** 当前中文 README 原样迁移到 `README.zh-CN.md`，默认 `README.md` 改为结构对等的自然英文版本，两者通过顶部语言入口互相链接。独立的 HN 发布包只负责标题、首评、检查清单和评论回复，不与项目使用文档混杂。

**Tech Stack:** Markdown、GitHub README、Hacker News 文本提交、Python 标准库验证脚本

---

### Task 1：保留中文 README 并补齐基准环境

**Files:**
- Create: `README.zh-CN.md`

- [x] **Step 1：复制当前中文首页**

将当前 `README.md` 的完整内容复制到 `README.zh-CN.md`，不得删减安装、配置、基准、FAQ、贡献、许可证或联系方式。

- [x] **Step 2：添加语言切换入口**

在项目标题下方加入：

```markdown
[English](README.md) | **简体中文**
```

- [x] **Step 3：记录基准硬件与事实边界**

在性能表附近加入以下信息：

```markdown
基准设备：MacBook Pro，Apple M5（10 核），32 GB 统一内存，1 TB 内置 Apple SSD。
```

同时明确“约 8.23–8.27 GiB”为 MLX `get_peak_memory` 的张量分配在途高水位，不是进程 RSS 或系统总占用，也不代表 8 GB 统一内存设备具备足够系统余量。

- [x] **Step 4：检查中文版本完整性**

运行：

```bash
python - <<'PY'
from pathlib import Path

text = Path("README.zh-CN.md").read_text()
required = [
    "快速开始",
    "数据准备",
    "配置说明",
    "性能与关键成果",
    "常见问题",
    "开发与贡献",
    "Apple M5",
    "32 GB",
]
for item in required:
    assert item in text, item
print("Chinese README sections: OK")
PY
```

预期输出：`Chinese README sections: OK`

### Task 2：制作默认英文 README

**Files:**
- Modify: `README.md`

- [x] **Step 1：完整翻译项目首页**

保持中文版本的模块顺序与全部技术事实，将正文改写为自然英文。必须包含：

- project overview and problem statement；
- demo；
- feature highlights；
- architecture and data flow；
- requirements, installation and native extension build；
- expert blob preparation；
- CLI and environment configuration；
- repository layout；
- benchmarks, correctness and rejected experiments；
- tests, FAQ, contributing, license and contact。

- [x] **Step 2：添加语言切换入口**

在项目标题下方加入：

```markdown
**English** | [简体中文](README.zh-CN.md)
```

- [x] **Step 3：明确基准环境**

英文性能章节必须包含：

```text
MacBook Pro, Apple M5 (10-core), 32 GB unified memory,
1 TB internal Apple SSD
```

并明确区分 41 GB 磁盘权重、约 8.23–8.27 GiB 的 MLX `get_peak_memory` 张量分配在途高水位与 32 GB 设备物理内存；明确该高水位不是进程 RSS 或系统总占用。

- [x] **Step 4：检查英文完整性与残余中文**

运行 Python 检查所有核心英文标题、硬件与语言入口均存在。再使用 Unicode 正则扫描中文字符；允许范围仅包括语言切换入口中的“简体中文”。

预期结果：核心模块全部存在，不存在未翻译的中文正文。

### Task 3：编写 Hacker News 发布包

**Files:**
- Create: `docs/marketing/hacker-news-vates-launch.md`

- [x] **Step 1：编写提交信息**

写入推荐标题：

```text
Show HN: Vates – SSD-backed expert streaming for Qwen3-Next on Apple Silicon
```

提交 URL 使用：

```text
https://github.com/AMOS144/Vates
```

另提供两个克制、准确的备用标题。

- [x] **Step 2：编写作者首评事实提纲**

HN 官方规则禁止生成文本和 AI 润色评论，因此不输出可直接发布的首评。改为提供中文事实提纲，由作者本人用自己的语言从零撰写，内容包含：

- 512 experts / 10 active per token per MoE layer；
- per-expert SSD blobs and demand loading；
- future-layer gate prefetch and C++ asynchronous `pread`；
- M5 MacBook Pro / 32 GB / internal SSD；
- about 8.23–8.27 GiB MLX `get_peak_memory` in-flight tensor-allocation high-water mark / 13–15 tok/s，并明确前者不是 RSS 或系统总占用；
- correctness verification and fallback behavior；
- current Apple Silicon, MLX and model limitations；
- 具体的技术反馈邀请。

- [x] **Step 3：编写评论区事实卡**

为设计文档列出的八类常见问题分别提供中文事实核对卡、仓库来源和禁止声称的内容，不生成可直接复制的回复；不承诺其他设备或模型上的结果。

- [x] **Step 4：添加发布前检查清单**

检查项至少包括：

- GitHub 默认 README 已为英文；
- 演示与许可证链接可访问；
- 标题没有误导为 8 GB Mac；
- 若作者选择发布首评，必须由作者本人从零撰写，不得使用生成或 AI 润色文本；
- 不在 HN 标题或评论中请求 upvote、comment 或 GitHub Star，不跨平台组织投票，也不把 GitHub Star 与 HN 投票混淆；
- 后续评论必须由作者亲自撰写，不使用生成文本或 AI 润色文本；
- 评论中披露作者身份和测试硬件；
- 对质疑提供报告链接或明确说明尚未测试。

### Task 4：一致性与发布验证

**Files:**
- Verify: `README.md`
- Verify: `README.zh-CN.md`
- Verify: `docs/marketing/hacker-news-vates-launch.md`

- [x] **Step 1：验证数字和硬件一致**

检查三个文件都使用同一组事实：

```text
41 GB weights
about 8.23–8.27 GiB MLX get_peak_memory in-flight tensor-allocation high-water mark (not RSS or total system memory)
13–15 tok/s
Apple M5, 10-core
32 GB unified memory
1 TB internal Apple SSD
```

- [x] **Step 2：验证本地相对链接**

使用 Python 提取两个 README 中不含 URL、锚点和图片 URL 的相对 Markdown 链接，并断言目标文件存在。

- [x] **Step 3：验证 HN 内容边界**

检查发布包不包含 `8 GB Mac`、`runs on any Mac`、`lossless`、`full speed`、`breakthrough` 等误导或夸张表达。

- [x] **Step 4：复核变更范围**

确认只修改或新增以下发布材料，不改项目实现：

```text
README.md
README.zh-CN.md
docs/marketing/hacker-news-vates-launch.md
docs/superpowers/specs/2026-07-21-hacker-news-launch-design.md
docs/superpowers/plans/2026-07-21-hacker-news-launch.md
```

- [x] **Step 5：完成代码与文档复核**

运行 IDE lint 检查并进行独立只读审查。修复所有 Critical 和 Important 问题后，再向用户交付结果。

### Task 5：修正最终审查问题

**Files:**
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `docs/marketing/hacker-news-vates-launch.md`
- Modify: `docs/superpowers/specs/2026-07-21-hacker-news-launch-design.md`
- Modify: `docs/superpowers/plans/2026-07-21-hacker-news-launch.md`

- [x] **Step 1：统一内存指标口径**

设计、计划与事实清单统一使用“约 8.23–8.27 GiB，MLX `get_peak_memory` 的张量分配在途高水位”，并明确它不是进程 RSS 或系统总占用。

- [x] **Step 2：收紧性能与正确性边界**

中文 README 对齐英文版的 C++ cap=48 消融条件，以及 top-2、动态深度和逐字节 oracle 的报告覆盖范围，不作普遍“无损”承诺。

- [x] **Step 3：修正 HN 互动与首评规则**

README 末尾仅邀请通过 Issue 提交技术反馈或贡献；HN 标题和评论不请求 upvote、comment 或 GitHub Star。若作者选择发首评，必须本人从零撰写，不使用生成或 AI 润色文本。

- [x] **Step 4：修正专家目录与可尝试性说明**

两版 README 明确 `--expert-dir` 是专家根目录，默认读取其 `blobs/` 子目录；直接指定 blob 目录时设置 `BLOB_DIR`。保留 Show HN 标题与可尝试性警告，透明披露门槛，但不把缺少第三方复现写成绝对禁止条件。

- [x] **Step 5：执行最终一致性检查**

运行数字与措辞一致性、英文 README 中文残留、相对链接、允许文件范围和 `git diff --check` 检查。
