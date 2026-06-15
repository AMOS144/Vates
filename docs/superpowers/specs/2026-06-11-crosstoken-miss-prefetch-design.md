# 设计：跨-token 大窗口预取（测量门控）

日期：2026-06-11
状态：设计已认可，待写实现计划

## 一句话目标

低内存优先：在**中池**（cap≈64–96 专家）下，用**大时间窗口**的预测信号把每 token 的少量 miss 提前异步物化好，做到**内存≈中池、命中/速度≈大池**。

## 背景与已确立的硬约束

1. **物化不慢，是塞不进窗口**：实测物化一个专家 ≈ **85µs**（4 个专家 load+eval 0.34ms）。同层（AHEAD=0）预测 recall 高达 0.984，但 decode 单层 attention/GDN 窗口只有 **70µs**，物化（340µs）≈ 窗口的 5× → `ready_on_time≈0`，同层预取被物理窗口卡死。
2. **小池天然 I/O-bound**：池太小会 thrash，一个 token 跨 48 层物化上百次专家，体量逼近甚至超过算力，预测改变不了体量。
3. **中池才是能赢的战场**：cap≈96、hit≈0.80 时，每 token 仅 **~19 个 miss**，物化总量 ~1.6ms ≪ 算力（~55ms/token）→ 这些 miss **完全可被大窗口隐藏**。
4. **窗口 vs 信号天生对立**：窗口要大必须早预测（跨 token / 提前多层），信号要强必须贴近真实 router 输入（越晚越准）。

## 核心风险（本设计的命门）

**需要预取的专家（miss）很可能 ⊥ 大窗口能预测的专家。**

中池 + LRU 稳态下，高频复发专家已常驻（都是 hit）；每 token 的 miss 按定义是「新需要、最近没用过」的专家，与历史/上下文信号弱相关。

- 大窗口信号（跨 token 同层、提前多层）= 历史信号 → 准确预测**复发专家**，但它们本就常驻、不需预取。
- 真正的 miss = 新颖专家 → 只有**同层/临近层真实 hidden** 能准（0.984），而那正是小窗口。

→ 因此**绝不能假设大窗口信号对 miss 有效**。必须先用离线实验把「大窗口信号对 **miss 子集** 的 recall」测出来，过关才落运行时。这是 make-or-break。

## 架构（两段，测量门控）

### Milestone 1：判生死（离线 probe，不改热路径）

**做什么**：跑真实 decode（中池 cap∈{64,96} + LRU），逐层采集并计算大窗口信号对 miss 的命中率。

**每层采集**：
- `routed`：本层真实路由专家集（decode top_k=2，但用 MTP 时为草稿+验证的实际激活）。
- `resident`：进入本层 MoE 前、该层常驻池里的专家集。
- `miss = routed − resident`：本层真正需要预取的专家。

**预测集 `pred`**（大窗口信号，candidate）：
- 主信号：**上一个 accepted token 同层 routed**（窗口=整个上一 token）。
- 可选增强：并上 MTP 的 K 个草稿 token 同层 routed（取并集，提升对「下几个 token」的覆盖）。
- 可选增强：并上「该层最近 N 个 token 的 routed 滑动并集」（历史频率）。

**两个核心指标**（按层聚合 + 全局聚合）：
- `recall_full = |pred ∩ routed| / |routed|`（对比基线，理解信号整体强度）。
- **`recall_miss = |pred ∩ miss| / |miss|`（命门指标）**——大窗口信号能否覆盖真正需要预取的那部分。
- 辅助：`pred` 规模（避免靠超大候选集刷 recall）、miss/层均值。

**判据**：
- `recall_miss ≥ ~0.8`（在合理 `pred` 规模下）→ 信号对 miss 有效，进入 Milestone 2 做运行时。
- `recall_miss` 低 → 命门坐实：大窗口信号对 miss 无效。**不再造预测器**，转向「减少 miss 体量」的备选（见下「Plan B」），并把这个量级事实写进报告。

### Milestone 2：运行时（仅当 Milestone 1 过关）

**数据流**：
1. 每生成一个 accepted token，记录其**逐层 routed**（写入 `last_token_routing[layer]`，零额外算力——路由本来就算了）。
2. 生成下一个 token 时，在**每层进入前用整-token 窗口**，对该层取 `pred = last_token_routing[layer]`（并按 M1 结论决定是否并 MTP 草稿/滑动并集），过滤出 `pred ∩ 非常驻`，**异步提交**给后台预取器（复用现有 `BackgroundExpertPrefetcher` + `_submit_missing_prefetch`）。
3. 该层 MoE 计算前 `promote_prefetched` 写进池槽（复用现有 Task 2 的「永不驱逐 current」语义）。
4. 窗口 = 从「token 起点 / 很多层之前」到「该层 MoE」≈ ms 级 ≫ 340µs 物化，物理上盖得住。

**复用既有组件**（上一计划已建好、测试过）：
- `BackgroundExpertPrefetcher`（s2 物化 + 交接）、`promote_prefetched`、`_submit_missing_prefetch`、`bg_stats.ready_on_time/not_ready`、`WINDOW_PROF`。
- 与同层方案的唯一区别：**预测信号源** = 上一 token 路由（提前一整 token），而非同层 hidden（提前 70µs）。

### Plan B（仅当 Milestone 1 不过关，作为结论方向，不在本 spec 实现）

若 miss 不可被历史信号预测，则唯一能动的杠杆是**减少 miss 体量**而非预测时机：
- 把「跨 token 复发集」显式 **pin**（高频专家钉住），等价于把中池有效命中再抬一点；本质仍是「池容量 vs 工作集」那堵墙，收益上限有限。
- 或接受低内存=低命中=慢的内在取舍，把内存重心转回主模型基座量化（~5GB 大头）。

## 数据流图

```
accepted token t:  逐层 routed  ──记录──▶ last_token_routing[L]
                                              │ (整 token 窗口)
token t+1, 进入层 L 前: pred=last_token_routing[L] ∩ 非常驻 ──submit──▶ 后台 s2 物化
                                              │
                       层 L MoE 前: promote_prefetched(L) ──▶ 池槽(miss→hit)
                                              │
                       层 L MoE: acquire_gpu(routed)  ──▶ 命中率↑、内存=中池
```

## 测试与验证

- **Milestone 1**：新增 `probe_crosstoken_miss_recall.py`；输出 `recall_full / recall_miss / pred_size / miss_per_layer`，cap∈{64,96}。无需新单测（纯离线测量），但 probe 自身逻辑（miss 集计算、并集）要有小单测覆盖。
- **Milestone 2**（若做）：等价性测试 `exact_match` 与中池 plain 一致；端到端 A(plain 中池) vs C(跨 token 预取) 对比 `tok/s / hit_rate / mlx_peak_gb / ready_on_time率`；判净收益。

## 成功标准

- **判生死**：拿到 `recall_miss` 的确定数字（不靠假设）。
- **若过关**：中池下 C 的 hit_rate 接近大池、tok/s ≥ 中池 plain、内存仍是中池水平、`ready_on_time` 率高（窗口够）。
- **若不过关**：用数据证明「miss ⊥ 历史信号」，给出 recall_miss 量级，转 Plan B 结论。

## 显式边界（诚实预期）

- 即便成立，上限是「中池内存 + 接近大池命中/速度」，**不会超过大池吞吐**。
- 本方向只解决「在给定中池下把少量 miss 藏起来」；**降总内存**仍需量化主模型基座，**提吞吐到 24+** 仍回 MTP 接受率——与本条线正交。
- 命门 `recall_miss` 很可能偏低（这是理性预期），Milestone 1 就是为了尽早、低成本地证伪。
```
