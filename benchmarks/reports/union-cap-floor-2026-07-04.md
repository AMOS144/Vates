# 实测各层单前向最大并集（坐实 cap 下限）

- 日期：2026-07-04
- 分支：`perf/async-demand-offload`
- 关联：Phase 0 / Task 0.1（spec：`docs/superpowers/specs/2026-07-04-cpp-unified-pool-authority-design.md`；plan：`docs/superpowers/plans/2026-07-04-cpp-unified-pool-authority.md`）
- 原始日志：`/tmp/union_prof.log`

## 结论（一句话）

**正确性要求 cap（真实区 + 侧区可寻址）≥ U_max = 30。**

MTP verify（seq=K=3）单次前向里，某些层路由到的唯一专家并集最大达 **30**，恰好触到理论上界 `seq × top_k = 3 × 10 = 30`；p99=29。因此专家池「真实区 + 侧区」的可寻址总容量必须 ≥ 30 才能保证任意一次 verify 前向都装得下、不溢出（否则会 miss 回退甚至丢专家、破坏正确性）。

## 关键修正：verify 前向是 K 个 token，不是 K+1

任务描述里假设「MTP verify 时 seq=K+1=4」。**实测与代码均证明 verify 前向实际只喂 K 个 token**：

```296:296:mlx_streaming/mtp/generate.py
        verify_in = mx.array([[x] + drafts[: K - 1]])              # [x, d_1..d_{K-1}]
```

即 `verify_in = [x, d_1 .. d_{K-1}]`，长度 = 1 + (K-1) = **K**。K=3 时 verify 桶落在 **seq=3**，日志里也确实只出现 seq1/seq2/seq3、没有 seq4，且 seq3 的样本数 = 48 层 × 步数（32 或 28 步，随接受长度波动），与 verify 步数完全吻合。故理论上界应为 `K × top_k = 3 × 10 = 30`，而非 40。

## 采集配置

命令（cap=64 避免采集期本身溢出污染，完整输出 tee 到 `/tmp/union_prof.log`）：

```bash
UNION_PROF=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=64 \
  ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=64 K=3 MAXTOK=64 WARMUP_TOK=0 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/union_prof.log
```

- 模型：Qwen3-Next-80B-A3B，512 专家，top_k=10，48 层
- K=3、MAXTOK=64、REPEAT=1、`expert_slots=64`、`verify_mode=batch`
- 探针：`UNION_PROF=1`，`note_union(seq, union_count, layer_idx)` 按 seq 分桶、按层记录每次前向的去重并集大小（见 `mlx_streaming/core/profiling.py`）

> 说明：为产出 U_max / p99 / 分层分布，本次对 UNION_PROF 探针做了最小增强——原探针只累加 `[sum, n]`（只能给均值），现额外记录每层每次前向的原始并集大小（`UNION_SAMPLES`，仅 `UNION_PROF=1` 时开销，退出时汇总）。`_union_prof()` 相应输出 `max_union / p99_union / per_layer`。改动仅限探针与其汇总打印，关时零开销。

## 分桶总览（tee 日志，28 步那次）

| seq 桶 | 含义 | avg | p50 | p99 | **max** | 样本数 | 路径 |
|--------|------|-----|-----|-----|---------|--------|------|
| seq1 | decode（单 token） | 10.0 | 10 | 10 | 10 | 48（1 前向） | GPU remap |
| seq2 | prefill chunk（chunk=2） | 16.38 | 17 | 20 | 20 | 192（4 前向） | host |
| **seq3** | **MTP verify（K=3）** | **19.46** | 19 | **29** | **30** | 1344（28 步） | GPU/dual |

- decode（seq1）并集恒 = top_k = 10（单 token 必然），完全落在 cap 内。
- prefill（seq2）走 host 路径（有超容量 fetch 回退，不受 cap 硬约束），max=20。
- **verify（seq3）走 GPU/dual 双源路径，是 cap+侧区必须覆盖的桶，U_max=30 即 cap 下限的驱动值。**

两次独立运行 U_max 均为 **30**、p99 均为 **29**，均值 19.46~19.92，结果稳定可复现（步数因接受长度不同而在 28~32 间波动，不影响 U_max/p99）。

## verify（seq=3）分层分布

- 逐层「单前向并集最大值」范围：**21 ~ 30**，均值 25.96
- 逐层「单前向并集均值」范围：16.32 ~ 26.89，均值 19.46
- **并集最大的层集中在浅层**（浅层相邻 token 路由重叠更少、并集更大）：

| 层 | max | avg |
|----|-----|-----|
| L0 | **30** | 26.89 |
| L2 | **30** | 25.43 |
| L19 | 29 | 22.54 |
| L18 | 28 | 22.64 |
| L7 | 28 | 22.39 |
| L1 | 28 | 22.29 |
| L26 | 28 | 19.25 |
| L32 | 28 | 18.89 |
| …（其余 40 层 max ≤ 27） | | |

全 48 层逐层 max/avg 明细见 `/tmp/union_prof.log` 的 `union_experts.by_seq.seq3.per_layer`。

## 对 cap 下限的解读

- 理论上界 `K × top_k = 30`；实测 **U_max 恰好触顶 30**，说明存在「3 个 verify token 在某层路由完全不重叠」的前向。相邻 token 路由整体高度重叠（均值仅 19.46，约为上界的 65%），但**最坏情况没有任何裕度**。
- 因此 cap 下限必须按 **max 而非 avg** 取：**cap（真实区 + 侧区可寻址）≥ 30**。任务预估的 ~19-25 与**均值**吻合，但作为正确性下限不足——真正的下限是 **30**。
- 当前配置 `expert_slots=64`（真实区）+ `POOL_SPEC_SLOTS=64`（侧区）远超 30，本次采集期未触发溢出，数据未被污染。

## 自审 / 疑虑

1. **verify=K 而非 K+1**：与任务描述不符，已按代码与实测更正（seq=3，上界=30）。后续 spec/plan 里凡假设 verify 并集上界=40（K+1×top_k）的推导，下限应改用实测 U_max=30。
2. **U_max 触顶理论上界（30）**：意味着 cap 下限没有经验裕度可削；若未来 K 变大，verify 上界随 `K × top_k` 线性增长，需重测。
3. **`exact_match=false`（n_mismatch=16）**：本次 spec 输出与 baseline 贪婪不完全逐 token 一致。这是投机解码接受/回退层面的现象，**不影响本任务的并集计数正确性**（`note_union` 只统计每次前向真实路由的去重专家数，与是否 direct-commit 无关）。但它是一条独立的正确性观察，记录在此供 Phase 0 后续排查（非本 Task 范畴）。
4. 采集用 REPEAT=1、单条 prompt，样本层面覆盖 28~32 个 verify 步 × 48 层 ≈ 1344~1536 个前向样本；U_max=30 在两次运行中稳定复现，置信度较高，但更大 token 量/更多 prompt 可能出现同为 30 的持平上界（不会超过 K×top_k）。
