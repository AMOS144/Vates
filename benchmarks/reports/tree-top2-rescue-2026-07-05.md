# 最小树 top-2 救回评测报告（2026-07-05）

## 结论一句话

**NO-GO（严格 lossless 下）**。最小树 top-2 救回在本流式 MoE 后端上**无法做到 bit-lossless**，
根因是**结构性、不可修的数值差异**（后端对序列长度的数值敏感性），而非 cache 管理 bug。
沿途发现并修复了一个**真实的潜在 cache 正确性 bug**（`_restore` 别名双恢复污染递归态），予以保留。

## 方案与目标

- **方案**：最小树 top-2 救回。每步 MTP 出 chainA(top-1)/chainB(top-2)；当 chainA 首草稿被拒
  (`matched==0`) 且 chainB 首候选 == 模型真实 token (`tree_b[0]==preds[0]`) 时，恢复 cache、改验
  chainB，试图把"被拒的第一位"救回，从而提升平均接受长度、提速。
- **红线**：输出必须 lossless。参照采用"噪声地板"口径：`tree-on` 相对 `tree-off`（MTP 自投机，
  单链 spec）的失配不得超过 `tree-off` 自身的 run-to-run 噪声。

## 评测配置

```
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=96 REPEAT=3 \
  .venv/bin/python benchmarks/bench_tree.py
```

6 条多样中文 prompt，每配置每 prompt 重复 3 次取中位数；ref=非投机贪婪基线，`control_mm`=tree-off
各轮相对 ref 的最大失配（噪声地板），`on_mm`=tree-on 各轮相对 ref 的最大失配。

## 结果（修复别名 bug 后）

| prompt | off tok/s | on tok/s | Δ% | rescues | direct/fallback | control_mm | on_mm | lossless |
|---|---|---|---|---|---|---|---|---|
| 混合专家 | 13.82 | 14.21 | +2.82 | 4 | 44/0 | 0 | 28 | ✗ |
| Python 代码 | 11.55 | 11.44 | -0.95 | 3 | 38/0 | 0 | 0 | ✓ |
| 量化困惑度 | 11.94 | 18.14 | +51.9 | 3 | 36/0 | 0 | 92 | ✗ |
| 英文摘要 | 10.57 | 10.71 | +1.32 | 8 | 46/0 | 0 | 70 | ✗ |
| 短故事 | 11.06 | 11.79 | +6.60 | 3 | 49/0 | 0 | 61 | ✗ |
| JSON 解析 | 11.01 | 11.20 | +1.73 | 2 | 36/0 | 0 | 90 | ✗ |

**汇总**：`lossless_all=false`，`max_control_mm=0`，`max_on_mm=92`，**verdict=bug**。

关键观察：`control_mm=0`（本轮后端确定性，tree-off run-to-run 无漂移），但 tree-on 相对 tree-off
出现**确定性大幅失配**；生产全程走 checkpoint **直接提交**（`fallback_replays=0`），从不 fallback。

## 根因定位（systematic debugging）

用玩具递归层复现 + 一系列受控开关逐步逼近，证据链如下：

1. **别名 bug（已修，真实）**：`_restore(caches, snaps)` 用 `c.state = st` 把快照 list 直接赋给
   `ArraysCache`（其 setter 是别名 `self.cache=v`），后续前向 `cache[idx]=new` 会**原地改写快照
   元素**。救回步对同一 `snap_m` 恢复两次（救回一次 + fallback 一次），第二次恢复到的已是被
   chainB 前向污染的状态 → 递归 cache 损坏。修法：`_restore` 装入 `_copy_state(st)` 副本。
   含递归层的玩具测试可稳定复现并已回归锁定。
   - 但生产走直接提交、从不 fallback，故此修复**未改变生产评测结果**（仍发散）→ 说明还有第二个原因。

2. **probe-only 排除"专家池扰动"**：让救回步做额外 chainB 前向（扰动流式专家池）但**不采用**其结果
   → 输出**完全 lossless**（mismatch=0）。证明额外前向 + 双恢复的机制本身是干净的，发散只来自
   **采用 chainB 的结果**。

3. **接受长度 bisect**：限制救回接受长度 `cap`：
   - `cap=0`（只提交 position-0）→ **lossless**（但等价 tree-off，无收益）。
   - `cap≥1`（提交 position≥1）→ **发散**（fdi=2, mismatch≈92）。
   定位到"采纳 chainB 的 position≥1 结果"才出问题。

4. **专家池容量排除**：`EXPERT_SLOTS=64` 仍**完全相同地发散** → 非池容量溢出。

5. **seq=1 自检（决定性）**：救回步用 seq=1 逐 token 从 `snap_m` 重算续写，与 chainB 的 seq=K
   preds 对比：
   ```
   [SELFCHK] chainB preds[:2]=[108048, 101219] | seq1 t0=108048 t1=20412
   ```
   chainB 的 **seq=K position-1** logit 给出 `101219`，而**同一前缀的 seq=1 续写**给出 `20412`
   （正是 tree-off 的对应 token）。**chainB 前向本身算得没错**，是 seq=K 与 seq=1 在近平局处
   **天然不一致**。

### 为什么这是结构性、不可修的

- **tree-off** 在救回候选步（MTP top-1 错）→ `matched=0` → 只采纳 position-0 logit；该输出位的下一
  token 由**新的 seq=1/position-0 前向**产出。
- **tree-on 救回** → 在同一输出位采纳 chainB 的 **position-1(seq=K)** logit。
- 后端（MoE 专家选择 / 注意力归约顺序对序列长度敏感）使 position-0(seq=1) 与 position-1(seq=K) 的
  logit 在近平局处**确定性翻转** → 一处翻转即级联，`mismatch` 高达 92。

这与既知现象"spec(seq=K) 相对 dense(seq=1) 在 token 3 系统性发散"是**同一机制**。canonical spec 在
精确算术下是 lossless 的（seq=K 的 position-i logit 应 == seq=1），本后端的非精确性打破了这一前提。
救回一旦要拿到收益，就必须采纳 seq=K position≥1 的续写，故与朴素解码的偏离不可避免。

## 保留 / 变更

- **保留（真实修复）**：`mtp/kv_cache.py::_restore` 别名 bug 修复 + 两条含递归层的救回 lossless
  单测（fallback replay 与 direct commit 两条路径）。
- **保留**：`bench_tree.py` 稳态多 prompt A/B harness（noise-floor 口径 + direct/fallback 计数）。
- **变更**：`tree_top2` 保持**默认关闭**；救回块加注释标明"非严格 lossless、开启即接受非 lossless 输出"。
- **清理**：所有诊断脚手架（probe_only / accept_cap / force_fallback / tree_dbg / selfcheck）与临时诊断脚本已移除。

## 已知次要 bug（未修，未在生产触发）

- fallback replay 在 `matched==K` 时 `accepted_in=[x0]+drafts[:K]` 会 replay K+1 个 token（比直接提交
  路径多提交 1 个），且在真实模型上会触发专家池 `inds.size=(K+1)·top_k > cap` 溢出告警。生产恒走直接
  提交、从不 fallback，故未触发；但递归态无 checkpoint 的极端场景会命中。建议后续单独修（对齐直接
  提交路径：`matched==K` 时只 replay `drafts[:K-1]`、留 `drafts[K-1]` 作 pending）。

## 后续（若仍想要接受率收益）

严格 lossless 无解；若可放宽到"near-lossless / 质量不劣化"，可考虑训练侧提升 MTP top-1 覆盖率
（减少救回触发本身），或对救回接受的 token 事后用 seq=1 复核（但会吃掉大部分提速）。
