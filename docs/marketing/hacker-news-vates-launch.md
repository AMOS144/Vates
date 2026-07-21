# Vates 的 Hacker News 发布事实清单

> [!IMPORTANT]
> [Hacker News 官方规则](https://news.ycombinator.com/newsguidelines.html)明确要求评论不得使用生成文本或经 AI 编辑的文本。HN 评论必须由作者本人根据真实经历、用自己的语言从零撰写；不要让 AI 生成、改写或润色首评及后续回复。本文只提供事实核对清单，不提供任何可直接发布的评论成稿。

> [!WARNING]
> [Show HN 官方规则](https://news.ycombinator.com/showhn.html)要求作品能让其他用户实际尝试；README 已提供界面演示与真实推理的数据准备路径，但主模型、专家 blob、MTP 权重和额外磁盘空间仍构成较高门槛，发布时必须透明披露。若准备路径实际上不可执行，应先修正后再以 Show HN 发布。官方规则并未要求已有第三方复现；当前尚无第三方复现，不能虚构已有外部用户跑通，也不能把这一事实本身写成禁止 Show HN 的理由。

## 提交标题与 URL

以下标题只适用于项目已经满足 Show HN“可实际尝试”的要求时。

推荐标题：

> Show HN: Vates – SSD-backed expert streaming for Qwen3-Next on Apple Silicon

可用备用标题：

> Show HN: Vates – Out-of-core MoE inference with MLX on Apple Silicon

> Show HN: Vates – Streaming Qwen3-Next experts from SSD with MLX

不推荐标题：

> Show HN: Vates – Stream an 80B MoE model from SSD on Apple Silicon

不推荐原因：这个标题可能让读者误以为整个模型都从 SSD 流式加载。当前实现主要把 MoE 专家权重做成磁盘 blob 并按需装入专家池，其他模型组件并不是同样的流式加载路径。

提交 URL：<https://github.com/AMOS144/Vates>

## 作者首评写作提纲

以下内容不是首评草稿。作者应只把它当作写作前的事实检查，并亲自决定取舍、组织和表述。

### 个人动机

- 作者需要亲自说明：为什么开始研究统一内存装不下完整权重时的大型稀疏模型推理？
- 作者需要亲自说明：这是解决自己的什么实际问题，还是一次系统工程探索？
- 不要虚构个人经历、用户需求或项目起因。

### 核心机制

- 目标模型是 Qwen3-Next-80B-A3B 的 4-bit MLX 版本。
- 4-bit 主模型权重约 41 GB，无法在低内存路径中全量常驻；Vates 因此把大部分专家权重留在磁盘，仅按运行时需要装入专家池。
- 模型有 512 个专家；每个 token 在每个 MoE 层选择 10 个 routed experts，另有一个常驻 shared expert。
- 专家权重打包在 SSD blob 中，每个专家对应一段连续字节范围；miss 时按需读取。
- 小型常驻专家池与 LFU side region 用于复用专家。
- 当前层路由信息用于提前运行未来层 gate，预测候选专家。
- native 路径通过 C++ 异步 `pread` 让部分 I/O 与计算重叠。
- 使用 Qwen3-Next 内置 MTP head 做 self-speculative decoding。
- 来源：[README 的 Overview、Features、Architecture and data flow](../../README.md#overview)。

### 测试硬件

- MacBook Pro。
- Apple M5，10-core CPU。
- 32 GB physical unified memory。
- 1 TB internal Apple SSD。
- 作者需要明确：是否只在这一台机器上完成了当前端到端实验？
- 来源：[README 的 Benchmarks 部分](../../README.md#benchmarks-correctness-and-rejected-experiments)。

### 约 8.23–8.27 GiB 的口径

- 约 8.23–8.27 GiB 来自 MLX `get_peak_memory`。
- 它表示报告配置下的张量分配在途高水位。
- 它不是 process RSS，不是整机总内存占用，也不证明低内存容量机器足以运行。
- macOS 与非 MLX allocation 仍需要额外空间。
- 来源：[README 的口径说明](../../README.md#overview)与[峰值内存报告](../../benchmarks/reports/peak-shrink-2026-07-03.md)。

### 13–15 tok/s 的口径

- 这是多组端到端实验中观察到的大致范围。
- 不同实验的配置、prompt、warmup、输出长度和 repeat 并不统一。
- 当前报告尚未形成一套统一的最终 benchmark，不能把这个范围描述成单次标准化基准结果。
- 来源：[README 的 Benchmarks 部分](../../README.md#benchmarks-correctness-and-rejected-experiments)及 [`benchmarks/reports/`](../../benchmarks/reports/)。

### 回退与正确性

- 预测错误或未及时完成时，系统按真实 gate 结果走 demand fallback。
- prefetch 不替代、也不改变真实专家路由。
- 修复非连续路由张量读取问题后，`tree-top2-rescue-2026-07-05.md` 报告的 oracle 范围是其中 6 个 prompts 及对应的 deterministic 配置；`bench_tree.py` 记录的条件为 production env、K=3、MAXTOK=64、REPEAT=2，报告结果为 `control_mm=0`、`on_mm=0`。
- 该结果只能作为这 6 个 prompts 与对应 deterministic 配置的回归证据，不能泛化为所有 prompt、采样模式、硬件或配置下的正确性保证。
- 来源：[README 的路由与回退说明](../../README.md#architecture-and-data-flow)及[修复后的 tree top-2 rescue 报告](../../benchmarks/reports/tree-top2-rescue-2026-07-05.md)。

### 当前限制

- 目前只支持 Apple Silicon 与 MLX；没有 CUDA、ROCm 或 CPU-only backend。
- 当前准备和运行路径面向兼容的 Qwen3-Next 文件。
- 真实推理要求用户自行准备主模型、专家 blob 和 MTP 权重。
- 仓库不包含这些权重，也没有包含三类输入的单一下载位置。
- 作者需要亲自说明：自己实际验证过哪些安装和准备步骤？
- 来源：[README 的 Requirements、Data preparation 与 FAQ](../../README.md#requirements)。

### 希望讨论的问题

- 作者对 cache policy 最不确定或最希望改进的部分是什么？
- future-layer gate prefetch 在什么情况下容易失效？
- SSD contention、并发读取和系统 swap 会怎样互相影响？
- 适配其他 MoE 架构需要重新处理哪些 layout、routing、shared expert 或 speculative decoding 假设？
- 只提出作者确实愿意深入讨论的问题，不要为增加互动而堆砌话题。

## 重要透明度说明

- 当前仓库报告是多次独立实验记录，不是一套统一最终 benchmark。
- 各报告使用的 prompt、config、warmup、输出长度和 repeat 可能不同，独立消融收益不能直接相加。
- `benchmarks/reports/` 中部分报告使用中文，英文读者可能需要额外解释。
- 截至本文整理时，尚未有第三方复现；不要把仓库内自测写成外部验证。
- Show HN 强调作品可尝试；README 已提供准备路径，但资产获取、准备磁盘空间和硬件要求等门槛仍需透明披露。官方规则不以已有第三方复现作为发布前提。
- 真实推理需要用户自行准备兼容的主模型、专家 blob 和 MTP 权重，并预留明显多于主模型约 41 GB 的准备磁盘空间。
- `vates --demo` 使用 mock backend，只能体验 TUI，不加载真实模型，也不能验证内存或性能数字。

## 八类回复事实卡

作者应先完整阅读对方的问题，再根据自己掌握的事实自然作答。不要逐条复制事实卡，不要把事实卡交给 AI 拼成或润色成评论。

### 1. 低内存容量机器能否运行

- 可核对事实：8.23–8.27 GiB 是 MLX allocation high-water mark；报告硬件有 32 GB unified memory；项目尚未在 8 GB 机器上验证。
- 仓库来源：[README 的 Overview](../../README.md#overview)、[峰值内存报告](../../benchmarks/reports/peak-shrink-2026-07-03.md)。
- 不能声称：该数字代表 RSS、整机占用、最低内存要求，或证明 8 GB 机器可运行。

### 2. 与 mmap 或系统 swap 的区别

- 可核对事实：Vates 在模型层知道 expert ID，管理有界专家池、驱逐、按需读取和预测预取；每个专家在 blob 中对应连续字节范围。
- 仓库来源：[README 的 Architecture and data flow](../../README.md#architecture-and-data-flow)、[Data preparation](../../README.md#data-preparation)。
- 不能声称：完全绕过操作系统缓存、不会发生 swap，或磁盘读取是零拷贝；miss 从 SSD 进入池仍有数据传输。

### 3. SSD 为何可能达到可交互速度

- 可核对事实：每层每 token 只路由 10/512 个专家并使用常驻 shared expert；池缓存复用，future-layer gate prefetch 与 C++ 异步读取尝试隐藏部分 I/O。
- 仓库来源：[README 的 Core mechanisms](../../README.md#architecture-and-data-flow)、[C++ 统一池报告](../../benchmarks/reports/cpp-unified-pool-final-2026-07-04.md)。
- 不能声称：所有 Apple SSD、外置 SSD 或所有 prompt 都能达到相同速度；不能声称 SSD 延迟已被完全隐藏。

### 4. 基准硬件与方法

- 可核对事实：测试机为 M5 MacBook Pro、10-core CPU、32 GB unified memory、1 TB internal Apple SSD；13–15 tok/s 来自条件不同的多组端到端实验。
- 仓库来源：[README 的 Benchmarks 部分](../../README.md#benchmarks-correctness-and-rejected-experiments)、[`benchmarks/reports/`](../../benchmarks/reports/)。
- 不能声称：当前已有统一最终 benchmark、跨硬件对比、统计显著性结论或第三方复现。

### 5. 是否影响质量或真实路由

- 可核对事实：真实 gate 路由保持权威，预测 miss 或延迟时走 demand fallback；修复后的 tree top-2 rescue 报告仅对报告中的 6 个 prompts 与对应 deterministic 配置给出 `control_mm=0`、`on_mm=0`，其 `bench_tree.py` 条件为 production env、K=3、MAXTOK=64、REPEAT=2。
- 仓库来源：[README 的路由与回退说明](../../README.md#architecture-and-data-flow)、[修复后的 tree top-2 rescue 报告](../../benchmarks/reports/tree-top2-rescue-2026-07-05.md)。
- 不能声称：该 oracle 覆盖报告之外的 prompt、采样模式、硬件或配置，或已普遍证明输出一致；也不能引用被后续根因修复推翻的旧报告来扩大正确性结论。

### 6. 为何仅支持 Apple Silicon 与 MLX

- 可核对事实：当前依赖 MLX primitive、Apple unified memory 与 native MLX/C++ extension；没有其他 backend。
- 仓库来源：[README 的 Requirements](../../README.md#requirements)与[FAQ](../../README.md#faq)。
- 不能声称：移植 CUDA、ROCm 或 CPU-only 只需简单切换配置，或其他平台已在开发完成。

### 7. 其他 MoE 模型支持

- 可核对事实：当前目标与准备流程针对兼容的 Qwen3-Next 文件；其他模型可能在 tensor layout、routing、shared expert 和 MTP 上不同。
- 仓库来源：[README 的 Overview](../../README.md#overview)、[Data preparation](../../README.md#data-preparation)。
- 不能声称：已经支持其他 MoE 家族，或无需适配和重新验证即可运行。

### 8. 仓库为何没有权重

- 可核对事实：仓库提供引擎和准备工具；真实推理需自行取得并准备 4-bit MLX 主模型、专家 blob 和 MTP 权重；三者没有单一打包下载。
- 仓库来源：[README 的 Data preparation](../../README.md#data-preparation)与[Overview 口径说明](../../README.md#overview)。
- 不能声称：克隆仓库即可直接进行真实推理，或 `--demo` 能验证真实模型性能。

## 发布前检查清单

- [ ] 阅读并遵守 [HN 官方规则](https://news.ycombinator.com/newsguidelines.html)，尤其是不发布生成文本或经 AI 编辑的评论。
- [ ] 若作者选择发布首评，必须由作者本人用自己的语言从零撰写；不复制本文条目组成评论，也不使用生成或 AI 润色文本。
- [ ] 后续回复由作者看完问题后亲自自然作答，不使用预制英文回复模板。
- [ ] 默认落地页是英文 `README.md`，GitHub、演示、安装步骤、报告与许可证链接均可访问。
- [ ] 不在 HN 标题或评论中请求 upvote、comment 或 GitHub Star；不组织或暗示集中投票，也不把 GitHub Star 与 HN 投票混为一谈。
- [ ] 披露作者身份、测试硬件、内存指标口径与 benchmark 不统一的限制。
- [ ] 对未测试的芯片、SSD、内存容量、模型或配置直接说明未测试。
- [ ] 明确仓库不含权重、准备流程占用额外磁盘空间，且 `--demo` 不代表真实推理。
- [ ] 逐步核对 README 中的演示和真实推理准备路径可执行，并透明披露资产、磁盘与硬件门槛；不虚构第三方复现。
- [ ] 最好先补一组固定 prompt、config、warmup、输出长度与 repeat 的最终 benchmark；这是提高可核查性的建议，不是发布硬条件。

## 何时发布与互动

选择作者能连续留出两到三小时的时段发布，优先保证能亲自及时回答技术问题，而不是追逐某个“最佳时区”。发布后先理解问题，再用自己的语言回答事实；必要时链接到 README 或具体报告，对未测试的组合明确说未测试。HN 流量受主题、时机和讨论质量等多种因素影响，没有发布时间能够保证曝光。
