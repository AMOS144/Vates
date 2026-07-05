# 最小树 top-2 救回评测（方案 B）设计

## 背景

目标是**通过提高 MTP 接受率来提速**。当前基线（K=3、生产配置、单 prompt 65 步）：

- `avg_accept_len = 1.969`（每步吐 1.97 token，上限 3）
- 逐位置条件接受率 α₁/α₂/α₃ = **69.2% / 42.2% / 42.1%**
- top-k 覆盖探针（真实 token 落在 MTP top-k 的比例）：
  - 位置 1：top1 69.2% / **top2 81.5%** / top3 87.7%
  - 位置 2：top1 44.6% / top2 58.5% / top3 67.7%
  - 位置 3：top1 46.2% / top2 58.5% / top3 64.6%

探针说明：真实 token 大量落在 MTP 的 top-2/top-3 里，top-1 贪心草稿丢掉了可观的可接受 token。位置 1 是接受长度主导项（α₁ 权重最大），其 top-2 覆盖比 top-1 高约 **12.3pp**，是最高杠杆的救回点。

### 为什么不用完整树形验证（tree_verify, batch-of-paths）

完整树形验证把 P 条路径拍到 batch 维，单次前向处理 **P×K** 个 token，路由专家并集从 `K×top_k` 放大到 `P×K×top_k`。模型是 512 专家、top_k=10：

- 单链 P=1：K=3 → 并集上界 30，勉强 ≤ cap=32
- 树 P=2：2×3=6 → 并集上界 60，**远超 cap=32 → 真实区溢出**

历史实测（`benchmarks/reports/tree-verify-2026-07-01.md`）印证此坑：

| 配置 | accept_len | disk_loads | tok/s | exact_match |
|---|---|---|---|---|
| baseline P=1 (slots=32) | 2.286 | 9375 | 7.91 | true |
| tree P=2 (slots=32) | 2.370 | 13634 (+45%) | 3.68 (0.46×) | **false, 14 失配** |
| tree P=2 (slots=256) | 2.370 | 560 | 11.35 (0.71×) | true |

结论：并集放大导致既慢又错（cap 溢出）；撑大 cap 到 256 虽正确但吃内存且仍 0.71×。因此本项目**约束在"不放大单次前向并集"**。

## 目标

用稳态、多 prompt 的 A/B 数据回答一个问题：

> **"首草稿被拒时串行跑 B 链救回"（最小树 top-2），在当前生产配置（cap=32, dual-source）下，净 tok/s 是赚还是亏？**

最小树救回每次前向仍是 **1×K**，并集不放大，是"不放大并集提接受率"的最直接候选。代码已存在（`draft_tree` / `TREE_TOP2`），但 `reports/` 下**从未有干净评测**（只有 tree_verify 的报告）。本项目产出这份评测与明确结论。

## 现有实现回顾

- `mlx_streaming/mtp/drafter.py::draft_tree`：位置 1 展开 top-2，返回两条链 (chainA, chainB)，各长 K；用 mtp_cache 快照保证 B 链从 pos1 态分叉、不带 A 的递归污染。
- `mlx_streaming/mtp/generate.py`（主循环，`tree_mode = config.tree_top2()`）：
  - 每步先验证 chainA（`verify_in = [x, d1a, d2a]`，1×K 前向）。
  - 救回条件：`matched == 0 and tree_b[0] == preds[0]`——即 A 首草稿被拒、且 B 的首候选恰等于主模型真实首 token（`preds[0]` 只依赖 `[x]`，两链一致）。
  - 满足则 `_restore(main_cache, snap_m)` 回滚，改验 B 链（`[x, d1b, d2b]`，又一次 1×K 前向），`tree_rescues += 1`。
- `benchmarks/bench_tree.py`：TREE_TOP2 off/on 的 A/B harness（进程内切 env，单模型加载）。

**代价结构**：
- 每步 `draft_tree` 抽两条链 → MTP draft 成本约 2×（MTP 单层、成本小）。
- 仅在救回步（预期 ~12% 步：matched==0 且 top2 命中）多一次主模型 1×K 前向的 IO/延迟。

## 正确性红线（lossless 护栏）

最小树只接受"等于主模型贪婪 argmax"的 token，理论上必然与非投机贪婪逐 token 等价。评测**必须**对每个 prompt 校验 `exact_match == true`（tree-on 输出逐 token 等于 tree-off 参考 `ref`）。

任一 prompt 失配 → **判定为实现有 bug，方案作废**，先修实现再谈收益。此点与 tree_verify 在 cap=32 下的 14 处失配严格区分：最小树是 1×K 前向、不该溢出 cap，但必须由实测证明，而非假设。

## 评测口径

对齐 `run_mtp_spec` 的稳态做法，消除冷启动/噪声：

- **多 prompt**：固定一组 5–8 条中文 prompt，覆盖问答 / 概念解释 / 代码 / 长文续写等不同风格，避免单 prompt 的 argmax 余量掩盖真实分布。
- **warmup**：正式测量前跑满一遍（热 Metal kernel + 专家常驻池 + 预取），再测。
- **repeat 取中位数**：每配置每 prompt 重复 3 次取 tok/s 中位数，抵抗 run-to-run 抖动。
- **配置**：`STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3`。
- **干净轮**：tok/s 与 IO 用**关诊断探针**的轮次测（UNION_PROF 等会扰动热路径）；探针数据（如 union、tree_rescues）单独跑或用不影响热路径的计数。

### 采集指标（每 prompt × {tree-off, tree-on}）

| 指标 | 用途 |
|---|---|
| `exact_match` / `n_mismatch` | lossless 护栏，逐 prompt 必须全 true |
| `avg_accept_len` + `accept_hist` | 接受率是否真的抬升 |
| `tree_rescues` | 救回实际触发次数（验证机制在跑、非空转）|
| `spec_tok_per_s`（中位数）| **最终裁决指标**：净 tok/s |
| `disk_loads` | B 链额外前向带来的 IO 增量 |
| `dec_hit_rate` | 命中率是否被额外前向拖累 |
| 额外前向次数 / 占步数比 | 量化"换"的代价（预期 ~12% 步触发）|

**对照基准**：先跑一次 tree-off 作 `ref`（lossless 参考），tree-on 输出必须逐 token 等于它。

## go / no-go 判据（按顺序）

1. **lossless 门（硬性）**：所有 prompt `exact_match == true`。任一失配 → 判 bug，方案作废（先修实现）。
2. **净收益门**：tree-on 相对 tree-off 的**中位 tok/s 提升 > 5%**（跨全部 prompt 的中位）。5% 阈值盖过 run-to-run 噪声地板。
   - `> +5%` → **go**：并入生产路径，进入方案 A（置信度门控）叠加。
   - `-5% ~ +5%` → **持平**：记负结论归档，转方案 A（A 省成本、独立于 B）。
   - `< -5%` → **no-go**：确认串行救回代价大于收益，归档，转方案 A。

## 可选 top-3 扩展

仅当 top-2 结果为 **go 或持平偏正**时才做：

- 位置 1 从 top-2 扩到 top-3 救回（pos1 top-3 覆盖 87.7%，比 top-2 的 81.5% 再多约 6pp）。
- 仍是串行、1×K 前向、并集不放大。
- 同一 harness 加一档 `tree-top3` 配置对比，判据同上。
- 若 top-2 已 no-go → top-3 不做（串行代价只会更大）。

## 产出物

一份 `benchmarks/reports/minimal-tree-top2-YYYY-MM-DD.md`，含完整 A/B 表 + 明确 go/no-go 结论。正/负结论都归档（符合仓库既有"负结论也留档"习惯）。

## 范围外（YAGNI）

- 不碰 tree_verify（batch 路径，已证伪）。
- 不改 cap / EXPERT_SLOTS。
- 不动方案 A（置信度门控 / 自适应 K）与方案 C（MTP 微调）——它们是独立子项目，各自走 spec → plan → 实现。

## 全局路线（上下文）

本项目是"提高 MTP 接受率提速"三方案中的 **B**，按约定 B 先行：

- **A｜省成本（推理期）**：置信度门控 + 自适应 K，MTP 低置信位置截断，缩小 verify 并集、省 IO。
- **B｜换接受（推理期，本 spec）**：最小树 top-2 串行救回，并集不放大，博更长接受。
- **C｜训练（后手）**：轻量微调/蒸馏 MTP 抬高 α₂/α₃，零额外并集/前向开销，天花板最高。

排序：先 A+B 推理期快赢（B 先出数据），再据净收益决定是否投入 C。
