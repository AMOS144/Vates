# Apache-2.0 许可证、联系方式与社区治理设计

## 目标

- 为仓库补充标准 Apache License 2.0 许可证。
- 使用 `Copyright 2026 AMOS144` 作为版权声明。
- 在 README 中清晰展示许可证并提供许可证文件链接。
- 将作者联系方式更新为 `3108424075@qq.com`。
- 建立完整、可直接使用的 Issue、PR、贡献和社区行为规范。

## 变更范围

### LICENSE

在仓库根目录新增 `LICENSE`：

- 使用 Apache License 2.0 官方标准全文；
- 保留许可证原始英文内容，避免翻译产生法律含义偏差；
- 不改写附录中的示例占位符，确保 GitHub 可以正确识别许可证。

### README

- 在顶部徽章区增加 Apache-2.0 徽章，并链接到根目录 `LICENSE`；
- 将 License 章节改为项目采用 Apache License 2.0，并链接到许可证全文；
- 在 License 章节声明 `Copyright 2026 AMOS144`；
- 将“其他联系方式”替换为可点击的 QQ 邮箱 `3108424075@qq.com`；
- 删除许可证、联系方式和缺少贡献规范的提示；
- 在开发与贡献章节链接到 `CONTRIBUTING.md` 和行为准则。

### 贡献指南

新增中文 `CONTRIBUTING.md`，包含：

- 提交 Issue 前的搜索与信息准备；
- 本地环境安装、原生扩展构建和测试命令；
- 分支命名与改动范围要求；
- Commit 与 Pull Request 编写要求；
- Bug 修复、功能变更和性能优化的验证标准；
- 性能优化必须附带可复现命令、对照数据与正确性结果。

这些内容作为推荐协作规范，不虚构自动化检查或当前不存在的合并要求。

### GitHub Issue 与 PR 模板

新增：

- `.github/ISSUE_TEMPLATE/bug_report.yml`：收集问题描述、复现步骤、预期行为、设备、macOS、Python、项目版本、运行命令与日志；
- `.github/ISSUE_TEMPLATE/feature_request.yml`：收集使用场景、问题、建议方案、备选方案和补充信息；
- `.github/ISSUE_TEMPLATE/config.yml`：关闭空白 Issue，并通过 HTTPS 链接指向 README 的 QQ 邮箱联系方式；
- `.github/pull_request_template.md`：要求填写改动摘要、类型、验证、性能影响、正确性与检查清单。

所有 YAML 表单使用 GitHub Issue Forms 支持的标准字段，不引用不存在的标签、负责人或外部服务。

### 行为准则

新增 `CODE_OF_CONDUCT.md`：

- 使用 Contributor Covenant 2.1 官方中文版本；
- 明确适用范围、行为标准、执行职责和处理等级；
- 将执行与投诉联系人设置为 `3108424075@qq.com`；
- 链接 Contributor Covenant 官方来源。

## 不在范围内

- 不修改源码、依赖或运行方式；
- 不增加额外的 `NOTICE` 文件；
- 不修改 `pyproject.toml` 的包发布元数据；
- 不添加与第三方依赖许可证相关的声明。
- 不新增 `SECURITY.md`，避免承诺尚未建立的安全响应流程。

## 验证

- 核对 `LICENSE` 包含 Apache License 2.0 标题、官方条款和附录；
- 核对 README 包含 `Copyright 2026 AMOS144`；
- 检查 README 徽章、许可证链接与邮箱链接；
- 检查 README、贡献指南、行为准则和模板之间的相对链接；
- 解析两个 Issue Form 和配置文件，确认 YAML 语法有效；
- 检查 Issue Form 必填字段与 GitHub 支持的键；
- 运行 `git diff --check`；
- 确认变更仅包含许可证、README、社区治理文件、设计规范和实施计划。
