# README 优化实施计划

> **面向执行代理：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，按任务逐项实施。所有步骤使用复选框（`- [ ]`）跟踪。

**目标：** 将 `main` 分支的 README 重构为内容真实、结构完整、可直接用于 GitHub 项目首页的专业开源文档。

**架构：** 以现有 README、`pyproject.toml`、命令行帮助和仓库文件为事实来源，只重组、精简并补全项目文档。README 采用“项目定位 → 快速上手 → 深入使用 → 开发参与”的阅读路径，缺失的许可证和联系方式明确标记为 `【待补充】`。

**技术栈：** Markdown、GitHub 徽章、GitHub Alerts、Bash 示例、Python 3.11、MLX、Textual、C++/nanobind、CMake

---

### 任务 1：核验 README 的事实来源

**文件：**
- 读取：`README.md`
- 读取：`pyproject.toml`
- 读取：`mlx_streaming/cli.py`
- 读取：`mlx_streaming/config.py`
- 读取：`native/ext/Makefile`

- [ ] **步骤 1：核验仓库状态**

运行：

```bash
git status --short --branch
```

预期：当前分支为 `main`；除本次新增的设计和计划文档外，没有其他未提交修改。

- [ ] **步骤 2：核验 CLI 参数**

运行：

```bash
uv run vates --help
```

预期：命令成功退出；README 中列出的 `--model`、`--expert-dir`、`--mtp-out`、`--k`、`--max-tokens`、`--expert-slots`、`--spec-slots`、`--system`、`--stats`、`--demo` 和 `chat --plain` 均能从帮助信息或 CLI 源码得到验证。

- [ ] **步骤 3：核验安装、构建和测试入口**

检查 `pyproject.toml` 与 `native/ext/Makefile`，确认：

- Python 最低版本为 3.11；
- 运行依赖为 MLX、mlx-lm、NumPy 和 Textual；
- 开发依赖包含 nanobind、pytest 和 pytest-asyncio；
- Python 命令入口为 `vates = "mlx_streaming.cli:main"`；
- native 扩展构建目标为 `make native_moe_ext`；
- 测试命令可使用 `.venv/bin/python -m pytest`。

### 任务 2：重构完整 README

**文件：**
- 修改：`README.md`

- [ ] **步骤 1：重建项目首页信息层级**

将 README 顶部组织为以下内容：

1. 居中的 `vates` 标题；
2. 一句话定位：在 Apple Silicon 上以约 8 GB 运行内存运行 80B MoE；
3. 使用 `AMOS144/Vates` 真实仓库地址的 Stars、Forks、Issues、Last Commit 徽章；
4. Python 3.11+、Apple Silicon、MLX 徽章；
5. 保留现有演示封面和 Release 视频链接；
6. 用简洁段落说明项目解决“模型权重大于可用统一内存”的问题。

不得添加 License、CI、Docker、PyPI 或 npm 徽章，因为仓库中没有对应的事实依据。

- [ ] **步骤 2：补齐核心开源文档模块**

按以下顺序编排 README：

1. 项目简介
2. 演示
3. 功能亮点
4. 工作原理
5. 技术栈
6. 快速开始
7. 数据准备
8. 详细使用
9. 配置说明
10. 项目结构
11. 性能与关键成果
12. 测试
13. FAQ
14. 开发与贡献
15. License
16. 作者与联系方式

每个模块必须使用标准 Markdown；命令块标记为 `bash`，目录树标记为 `text`，所有命令注释使用中文。

- [ ] **步骤 3：整理快速开始和使用示例**

快速开始必须给出可连续复制的流程：

```bash
git clone https://github.com/AMOS144/Vates.git
cd Vates
uv sync
source .venv/bin/activate
cd native/ext && make native_moe_ext && cd ../..
vates --demo
```

另行说明真实模型推理需要用户自行准备兼容的 Qwen3-Next-80B-A3B 4-bit MLX 主模型、专家 blob 和 MTP 权重，不虚构下载地址。

保留并规范以下真实使用方式：

```bash
vates
vates -k 4 -n 800 --stats
vates --system "你是一个简洁的助手"
vates --demo
vates chat --plain
```

- [ ] **步骤 4：精简性能与研发历程**

保留原文中可验证的核心数据：

- 4-bit 全量约 41 GB，运行内存峰值约 8 GB；
- 生产快路径约 13–15 tok/s；
- 128k 上下文 KV 从 3.0 GiB 降至约 0.68 GiB；
- LFU 命中率、统一池、正确性 oracle、MTP 和峰值内存优化的代表性结果。

删除逐阶段的长篇过程叙述，将被否决方向压缩为一个说明段，并明确各项收益不可简单相加。

- [ ] **步骤 5：补充 FAQ、贡献、许可证和联系方式**

FAQ 至少覆盖：

- 为什么只支持 Apple Silicon；
- `vates: command not found`；
- native 扩展未编译的影响；
- 为什么应使用 `uv sync`；
- 模型文件应放在哪里；
- 如何在不加载模型时验证界面。

贡献指南包含 Issue、分支、测试、PR 描述和性能优化需附基准数据等要求，但不虚构仓库已有模板或强制制度。

License 明确写明仓库目前没有许可证文件并标记 `【待补充】`；作者写为 `AMOS144`，仓库地址使用 `https://github.com/AMOS144/Vates`，其他联系方式标记 `【待补充】`。

### 任务 3：验证文档并发布

**文件：**
- 验证：`README.md`
- 验证：`docs/superpowers/specs/2026-07-21-readme-optimization-design.md`
- 验证：`docs/superpowers/plans/2026-07-21-readme-optimization.md`

- [ ] **步骤 1：检查 Markdown 内容和链接**

运行一个只读 Python 检查脚本，确认 README：

- 包含全部 16 个核心模块；
- 不包含 `TODO`、`TBD`；
- 所有本地相对 Markdown 链接均指向存在的文件；
- GitHub 仓库链接统一使用 `AMOS144/Vates`；
- 代码围栏成对出现。

预期：脚本退出码为 0。

- [ ] **步骤 2：检查 README 差异**

运行：

```bash
git diff --check
git diff -- README.md
```

预期：`git diff --check` 无输出并以 0 退出；README 差异仅为文档重构，没有源码或配置修改。

- [ ] **步骤 3：提交文档**

运行：

```bash
git add README.md \
  docs/superpowers/specs/2026-07-21-readme-optimization-design.md \
  docs/superpowers/plans/2026-07-21-readme-optimization.md
git commit -m "docs(readme): 重构项目首页与使用指南"
```

预期：提交成功，包含 README、设计规范和实施计划三个文档。

- [ ] **步骤 4：推送 main**

运行：

```bash
git push origin main
```

预期：远端 `origin/main` 更新到新提交，且本地分支与远端同步。
