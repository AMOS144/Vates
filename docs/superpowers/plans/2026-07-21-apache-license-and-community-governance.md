# Apache-2.0 许可证与社区治理实施计划

> **面向执行代理：** 必须使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，按任务逐项实施。所有步骤使用复选框（`- [ ]`）跟踪。

**目标：** 为项目补齐 Apache-2.0 许可证、作者邮箱、贡献指南、行为准则以及 GitHub Issue/PR 模板。

**架构：** 根目录文档负责许可证和社区规范，`.github` 下的模板负责在提交 Issue 与 Pull Request 时收集结构化信息。README 作为入口链接这些文档，所有法律文本从官方来源获取并保持版本准确。

**技术栈：** Markdown、Apache License 2.0、Contributor Covenant 2.1、GitHub Issue Forms、YAML

---

### 任务 1：添加许可证与行为准则

**文件：**
- 新建：`LICENSE`
- 新建：`CODE_OF_CONDUCT.md`

- [ ] **步骤 1：获取 Apache License 2.0 官方文本**

从 Apache Software Foundation 官方地址 `https://www.apache.org/licenses/LICENSE-2.0.txt` 获取文本，保存为根目录 `LICENSE`。文件必须以 `Apache License` 和 `Version 2.0, January 2004` 开头，并包含第 1 至第 9 条和 `APPENDIX`。

- [ ] **步骤 2：添加 Contributor Covenant 2.1 中文行为准则**

以 Contributor Covenant 2.1 官方中文译文为基础创建 `CODE_OF_CONDUCT.md`，保留以下模块：

- 我们的承诺
- 我们的准则
- 责任和权力
- 适用范围
- 监督执行
- 处理方针
- 署名与归属

将举报和执行联系人设为 `3108424075@qq.com`，不得保留未填写的邮箱占位符。

### 任务 2：添加贡献指南与 GitHub 模板

**文件：**
- 新建：`CONTRIBUTING.md`
- 新建：`.github/ISSUE_TEMPLATE/bug_report.yml`
- 新建：`.github/ISSUE_TEMPLATE/feature_request.yml`
- 新建：`.github/ISSUE_TEMPLATE/config.yml`
- 新建：`.github/pull_request_template.md`

- [ ] **步骤 1：编写中文贡献指南**

`CONTRIBUTING.md` 必须包含：

- 提交 Issue 前先搜索重复问题；
- `uv sync`、虚拟环境、原生扩展构建和完整测试命令；
- 推荐分支前缀 `feat/`、`fix/`、`docs/`、`perf/`、`test/`；
- 改动聚焦、中文代码注释和测试要求；
- PR 描述、性能基准、正确性验证和文档同步要求；
- 行为准则链接和 QQ 邮箱。

- [ ] **步骤 2：编写 Bug Issue Form**

创建 `bug_report.yml`，使用 `name`、`description`、`title`、`body` 等 GitHub Issue Forms 标准字段。表单必须收集问题描述、复现步骤、预期行为、设备型号、macOS、Python、项目提交、运行命令和完整日志，并包含行为准则确认框。

- [ ] **步骤 3：编写 Feature Issue Form**

创建 `feature_request.yml`，收集使用场景、当前痛点、建议方案、备选方案、性能或兼容性影响和补充信息，并包含行为准则确认框。

- [ ] **步骤 4：配置 Issue 与 PR 入口**

`config.yml` 设置 `blank_issues_enabled: false`，并提供名为“私密联系”的 HTTPS 链接，指向 README 的作者联系方式。GitHub 的 Issue 配置只接受 HTTP(S) URL，因此不直接使用 `mailto:`。

PR 模板必须包含摘要、变更类型、关联 Issue、验证命令、性能影响、正确性结果、文档影响和提交前检查清单。

### 任务 3：更新 README

**文件：**
- 修改：`README.md`

- [ ] **步骤 1：更新徽章和许可证章节**

在顶部徽章区添加链接到 `LICENSE` 的 Apache-2.0 徽章：

```markdown
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
```

将 License 章节改为：

```markdown
本项目采用 [Apache License 2.0](LICENSE)。

Copyright 2026 AMOS144
```

- [ ] **步骤 2：更新贡献入口与联系方式**

在“开发与贡献”章节链接 `CONTRIBUTING.md` 和 `CODE_OF_CONDUCT.md`，删除缺少贡献规范的提示。

将“其他联系方式”替换为：

```markdown
- 邮箱：[3108424075@qq.com](mailto:3108424075@qq.com)
```

### 任务 4：验证、提交与推送

**文件：**
- 验证：`LICENSE`
- 验证：`CODE_OF_CONDUCT.md`
- 验证：`CONTRIBUTING.md`
- 验证：`.github/ISSUE_TEMPLATE/*.yml`
- 验证：`.github/pull_request_template.md`
- 验证：`README.md`

- [ ] **步骤 1：验证许可证和文档链接**

运行只读脚本确认：

- `LICENSE` 包含 Apache-2.0 标题、第 1 至第 9 条和附录；
- README 包含许可证徽章、版权声明和 QQ 邮箱；
- README、贡献指南和行为准则中的仓库内相对链接均存在；
- 所有 Markdown 代码围栏成对闭合。

- [ ] **步骤 2：验证 YAML**

使用 Python 和 PyYAML 解析 `.github/ISSUE_TEMPLATE/*.yml`；确认两个表单包含 `name`、`description`、`body`，并确认 `config.yml` 关闭空白 Issue。

- [ ] **步骤 3：检查差异**

运行：

```bash
git diff --check
git status --short --branch
```

预期：格式检查通过，变更仅包含本计划约定的文档与模板。

- [ ] **步骤 4：提交**

提交消息：

```text
docs: 添加许可证与社区贡献规范
```

- [ ] **步骤 5：推送**

运行 `git push origin main`，然后确认本地 `main` 与 `origin/main` 同步。
