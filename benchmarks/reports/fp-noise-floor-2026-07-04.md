# FP 噪声地板实测（确定性 delta 的验收基准）

- 日期：2026-07-04
- 分支：`perf/async-demand-offload`
- 任务：Phase 0 · Task 0.2（plan：`docs/superpowers/plans/2026-07-04-cpp-unified-pool-authority.md`）
- 目的：量化「同一配置多次运行时输出 token 的 run-to-run 波动」，得到噪声地板 **N_floor**，作为后续 Phase 2 判定「某个确定性 delta 是否算 bug」的验收线。

> **重要修订（2026-07-04，spec 评审后补测）**：初版把 run-to-run 波动定性为「纯 FP 平局翻转（良性噪声）」。评审指出该定性是**论证而非实测**，要求用字节真值校验坐实。补测发现：**在噪声地板同一配置下，零拷贝双源快路径存在可复现的字节真值违规（`[DUAL_VERIFY] BAD`）**——即 run-to-run 波动**含真数据损坏成分，不是纯 FP 噪声**。原「纯 FP 噪声」结论据此**推翻/收窄**。详见第五、六节。

---

## 一、结论（TL;DR）

1. **token 级 run-to-run 非确定**：真实 80B、当前运行时下，baseline greedy 的输出 token run-to-run 不是逐位确定的（64-token 窗口下三次共同一致前缀仅 **19** token，之后分歧并自回归放大到最多 45/64）。
2. **波动成分：FP 噪声 + 已坐实的字节损坏（真 bug），二者叠加**：
   - 用**正确的字节校验开关 `DUAL_VERIFY=1`**（见第五节说明为何不是 `STG_VERIFY`）在同一配置下连跑 3 次，每次都打出 `[DUAL_VERIFY] BAD`：**bad = 31 / 62 / 54**（ok ≈ 2.7 万/次）。即零拷贝双源**快路径**上，某个被路由命中的专家 E→行 R 映射，其池行 R 的字节**不等于** E 的磁盘真值——专家权重装错字节的铁证。
   - 因此「run-to-run 波动 = 纯 GPU 浮点非确定性」的定性**不成立**；至少有一部分是**确定性/半确定性的字节损坏真 bug**（正是 plan Phase 2 要 root-cause 的「侧区 gen 新鲜度错槽」疑犯）。
3. **对 N_floor 的影响**：当前运行时下**无法把 token 波动当作「良性噪声地板」**——它被真 bug 污染。**干净的 N_floor 必须在修复侧区错槽之后、于字节校验 0 BAD 的路径上重新测**。在此之前，Phase 2 的验收只能以 **FP-无关的整数/字节不变量**（容量不变性 + `DUAL_VERIFY`/`STG_VERIFY` 0 BAD）为准，不能以 token 逐位一致为准。

---

## 二、判定方法与对比口径

`run_mtp_spec` 在同一进程内同时跑两条路径：**baseline greedy**（非投机，`mtp/generate.py::_baseline_greedy`，`argmax` 逐 token）与 **MTP 投机路径**（spec）。脚本原本只打印 `exact_match`（spec==baseline）与 `n_mismatch`（spec vs baseline 差异位数），**不 dump 任一路径的完整 token 序列**。

为拿真值，对测量脚本做了两处 **env 门控小改**（默认关闭、不改常规输出、随本报告提交）：
- `DUMP_IDS=1`：把 baseline / spec 完整 token 序列以 `DUMP_BASE_IDS` / `DUMP_SPEC_IDS` 打进日志，供跨进程 run-to-run 逐位对比。
- `STG_VERIFY` 或 `DUAL_VERIFY` 置位时：在结束处打印 `VERIFY_SUMMARY`——三处字节校验器（`verify_acquire_bytes` / `_verify_native_bytes` / `_verify_side_bytes`）的累计 `ok/bad/calls`。**关键作用：让「0 BAD」可判真伪**——若 `calls==0` 说明该校验器在本配置根本没触发，此时「0 BAD」是空结论。

**对比口径**：
- run-to-run 噪声：取三次独立进程日志的 `DUMP_BASE_IDS`（baseline greedy 那条）两两逐位对比（首次分歧位置 = 最长公共前缀；Hamming 差异位数）。
- 字节真值：开字节校验开关，`grep '[...] BAD'` + 读 `VERIFY_SUMMARY` 的 `ok/bad/calls`。

> **spec 序列长度澄清（评审提出）**：块级投机在末尾窗口不足一窗时可能比 baseline 少产出 1 个 token（故某些配置会见到 spec=63、baseline=64）。**但本报告全部实测中 baseline 与 spec 均为 64 token（MAXTOK=64）**，逐位对比按 64 位口径，无长度错位。

---

## 三、复现命令

### 3.1 噪声地板（无校验，测 run-to-run token 波动）

```
for i in 1 2 3; do \
  DUMP_IDS=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
    SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=0 REPEAT=1 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/noise_$i.log ; done
```

### 3.2 STG_VERIFY（评审要求的命令；实测在本配置下**不触发**校验，见 5.1）

```
for i in 1 2 3; do \
  STG_VERIFY=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
    SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=0 REPEAT=1 DUMP_IDS=1 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/noise_verify_$i.log ; done
```

### 3.3 DUAL_VERIFY（本配置**正确**的字节校验开关，坐实字节损坏）

```
for i in 1 2 3; do \
  DUAL_VERIFY=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
    SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=0 REPEAT=1 DUMP_IDS=1 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/noise_dual_$i.log ; done
```

---

## 四、baseline greedy run-to-run 一致性（原始数据，仍成立）

| 对比 | 首次分歧位置(0-based) | Hamming 差异 / 64 |
|---|---|---|
| run1 vs run2 | 19 | 45 |
| run2 vs run3 | 27 | 6 |
| run1 vs run3 | 19 | 45 |

- **三次共同一致前缀长度：19 / 64；三次 baseline 完全逐位一致：否。**

三次 baseline greedy 完整 token 序列（`DUMP_BASE_IDS`，来自 3.1）：

```
run1: [220, 16, 13, 6567, 115, 115, 39762, 101057, 104949, 9909, 44, 12735, 315, 50759, 11, 6050, 36, 7552, 101158, 104210, 17177, 69905, 100383, 9370, 102182, 100134, 106379, 3837, 99652, 44063, 102181, 88802, 107799, 17714, 101213, 44729, 88802, 3837, 67071, 101213, 2073, 101057, 854, 71356, 101127, 54542, 1773, 2303, 17, 13, 6567, 107, 237, 18947, 101057, 71356, 107782, 105149, 31905, 105918, 57191, 31196, 100144, 3837]
run2: [220, 16, 13, 6567, 115, 115, 39762, 101057, 104949, 9909, 44, 12735, 315, 50759, 11, 6050, 36, 7552, 101158, 44063, 101213, 2073, 101057, 854, 104949, 101220, 99793, 71817, 102041, 9370, 100166, 3837, 103991, 101057, 100668, 54542, 31196, 20074, 9370, 105149, 44729, 42067, 57191, 101065, 1773, 2303, 17, 13, 220, 67338, 46944, 2073, 64689, 99332, 71356, 854, 9909, 70, 1095, 3922, 7552, 104299, 50404, 31235]
run3: [220, 16, 13, 6567, 115, 115, 39762, 101057, 104949, 9909, 44, 12735, 315, 50759, 11, 6050, 36, 7552, 101158, 44063, 101213, 2073, 101057, 854, 104949, 101220, 99793, 9370, 102182, 100134, 106379, 3837, 103991, 101057, 100668, 54542, 31196, 20074, 9370, 105149, 44729, 42067, 57191, 101065, 1773, 2303, 17, 13, 220, 67338, 46944, 2073, 64689, 99332, 71356, 854, 9909, 70, 1095, 3922, 7552, 104299, 103930, 18493]
```

（run-to-run token 波动确实存在。**但下一节表明：这一波动的成因不止 FP，还含字节损坏。**）

---

## 五、字节真值校验（本次补测核心）

### 5.1 STG_VERIFY 在本配置下**是空结论**（校验器未触发）

评审建议的 `STG_VERIFY=1` 三次运行，`grep BAD` 全 0——但 `VERIFY_SUMMARY` 揭示原因是**校验器根本没跑**：

```
三次 VERIFY_SUMMARY 均为：
  STG_VERIFY.resident(verify_acquire_bytes): ok=0 bad=0 calls=0
  STG_VERIFY.virtual(_verify_native_bytes):  ok=0 bad=0 calls=0
  DUAL_VERIFY.resident(_verify_side_bytes):  ok=0 bad=0
```

代码路径审计（根因）：本配置 `ZEROCOPY_DUAL_SOURCE=1` → `block.py` 走 `VirtualPool.acquire` → 因未开 `NATIVE_DEMAND_DUAL`，`_native_demand=False` → 落 `ResidentExpertPool.acquire_gpu_dual`。而：
- `STG_VERIFY` 的 `verify_acquire_bytes` 只挂在 **非 zerocopy** 的 `else` 分支（`block.py:203-206`）；
- `STG_VERIFY` 的 `_verify_native_bytes` 只挂在 **方案B native-demand** 路径（`virtual_pool.py:142`，需 `NATIVE_DEMAND_DUAL=1`）；
- 本配置实际走的 `acquire_gpu_dual`，其字节校验挂在 **`_verify_side_bytes`，由 `DUAL_VERIFY=1` 门控**（`resident_pool.py:596-598`）。

**结论：对噪声地板这条 `acquire_gpu_dual` 路径，正确的字节校验开关是 `DUAL_VERIFY`，不是 `STG_VERIFY`。** 评审给的 `STG_VERIFY=1` 命令在此配置下不做任何校验，其「0 BAD」不能作为「无损坏」的证据。

### 5.2 DUAL_VERIFY：坐实字节损坏（`[DUAL_VERIFY] BAD`）

同一配置改用 `DUAL_VERIFY=1` 连跑 3 次，`VERIFY_SUMMARY`：

| run | ok | **bad** | exact_match | n_mismatch | baseline tok/s |
|---|---|---|---|---|---|
| 1 | 27256 | **31** | true | 0 | 0.96 |
| 2 | 26790 | **62** | false | 37 | 0.99 |
| 3 | 27941 | **54** | false | 20 | 0.98 |

**三次全部出现 BAD**（bad=31/62/54，约占 acquire 校验的 0.1%–0.23%）。每次前 12 条 BAD 明细（`layer / expert / row / key`）：

```
run1:
  L1  e453 r32 gate_proj.scales      L5  e429 r60 up_proj.scales
  L5  e438 r36 gate_proj.weight      L6  e465 r42 gate_proj.weight
  L5  e381 r41 gate_proj.biases      L4  e66  r35 up_proj.scales
  L5  e457 r63 up_proj.scales        L4  e66  r35 up_proj.scales
  L4  e66  r35 up_proj.scales        L4  e493 r54 up_proj.biases
  L4  e479 r59 gate_proj.weight      L4  e172 r61 gate_proj.weight
run2:
  L1  e453 r32 gate_proj.biases      L5  e438 r36 gate_proj.scales
  L5  e251 r43 gate_proj.weight      L6  e480 r38 gate_proj.weight
  L6  e480 r38 gate_proj.weight      L5  e381 r41 up_proj.weight
  L6  e480 r38 gate_proj.weight      L6  e227 r61 up_proj.scales
  L3  e140 r46 up_proj.scales        L6  e480 r38 gate_proj.weight
  L17 e374 r59 up_proj.scales        L3  e140 r46 up_proj.scales
run3:
  L1  e453 r32 up_proj.weight        L5  e429 r41 up_proj.scales
  L6  e480 r44 gate_proj.scales      L6  e480 r44 gate_proj.scales
  L6  e109 r38 gate_proj.weight      L6  e177 r42 up_proj.scales
  L6  e254 r37 up_proj.scales        L3  e463 r48 gate_proj.scales
  L5  e457 r63 gate_proj.scales      L6  e177 r42 up_proj.scales
  L4  e479 r48 gate_proj.weight      L4  e479 r48 gate_proj.weight
```

**结构性线索（非纯随机竞态）**：
- `L1 e453 r32` 是**三次运行各自的第 1 条 BAD**（跨 run 复现，仅 key 段不同）——高度可复现，指向该槽存在稳定的落池/映射错误。
- 损坏集中在**早层（L1、L3、L4、L5、L6）**，个别到 L17；`_verify_side_bytes` 只校验**快路径**（`n_miss==0` 且有侧区条目），故这是快路径侧区行装错字节。
- `L6 e480` 在 run2/run3 反复出现。

`_verify_side_bytes` 的作者注释即定性此为「侧区行装错字节的铁证（与 timing/gen 无关）」——即 BAD 表示被路由专家实际会 gather 到错误权重字节。

---

## 六、定性、成因与验收线（修订）

### 6.1 最终定性

**run-to-run 波动 = GPU 浮点非确定性（真实存在）+ 已坐实的字节损坏真 bug（侧区快路径装错字节），二者叠加。** 原初版「纯 FP 噪声、GPU 浮点非确定性是唯一物理来源」的定性**被推翻/收窄**：FP 噪声仍在，但绝非全部；至少存在一个可复现的数据损坏缺口。

### 6.2 成因（现有理解，待 Phase 2 root-cause）

零拷贝双源快路径把「真实区表 ∪ 侧区(读代)」叠加成 `eff` 后单次 gather。侧区为消除「本前向 gather 读的物理行在本前向 eval 被 fill 覆盖」而做了双代(gen)双缓冲；`[DUAL_VERIFY] BAD` 说明该不变量在本配置下仍被破坏——某些侧区行在被消费时装的是**别的专家/上一代**的字节（plan/spec 记为「侧区 gen 新鲜度错槽，即 61 的疑犯」）。`L1 e453 r32` 跨 run 复现，提示其中含**确定性成分**（按 plan 口径，确定性 delta 即真 bug）。

### 6.3 测量置信度与残留

- **置信（强）**：字节真值不变量在快路径被破坏——三次可复现，校验器读的是模型 gather 同一池行，非误报。
- **待厘清**：`DUAL_VERIFY` 使全程慢约 5×（baseline 0.96 vs 5.1 tok/s），会改变 fill/gen 竞态窗口，故**观测到的损坏率（0.1%–0.23%）不等于无校验噪声运行里的真实率**；「acquire 时刻校验到坏字节」与「eval 消费时刻是否仍坏」也非 1:1（run1 甚至 exact_match=true）。**结论只主张「损坏存在且可复现」，不主张具体损坏率。**
- **覆盖盲区**：`_verify_side_bytes` 仅覆盖快路径（`n_miss==0`）。本配置约 **70% 的层-acquire 走的是 fallback（host demand）慢路径**（`gpu_fallback≈1100` vs `gpu_fastpath≈470`），该路径**当前无 live 字节校验**，损坏与否未知——需在 Phase 1/2 补挂校验或用 `NATIVE_DEMAND_DUAL` 路径的 `_verify_native_bytes` 覆盖。

### 6.4 后续验收线（修订）

1. **当前运行时的 token run-to-run 波动不可作为「良性 N_floor」**：它被真 bug 污染。干净的 N_floor 必须**在修复侧区错槽、且字节校验 0 BAD 的路径上重测**后才成立。
2. **Phase 2 判定「确定性 delta 是否算 bug」以 FP-无关不变量为准**（不以 token 逐位一致为准）：
   - **容量不变性**（每层 resident 槽 ≤ cap、无溢出）；
   - **字节真值 0 BAD**：按实际启用路径选对开关——`acquire_gpu_dual` 路径用 **`DUAL_VERIFY`**、方案B native-demand 路径用 **`STG_VERIFY`（`_verify_native_bytes`）**、非 zerocopy 路径用 **`STG_VERIFY`（`verify_acquire_bytes`）**；
   - 路由 top-k 专家 **id 集合**（整数、确定）。
3. **修复达标口径**：字节校验 0 BAD 之后，再测 baseline greedy run-to-run 的稳定前缀——若前缀显著变长（趋近整窗）且波动只剩零星尾部翻转，则那才是真正的 FP 噪声地板；届时「高于该 N_floor 且可复现的 delta 才算 bug」方成立。

---

## 七、附：改动说明

对 `mlx_streaming/runtime/run_mtp_spec.py` 做了两处 env 门控小改（默认关闭、不改常规输出、随报告提交）：
1. `DUMP_IDS=1`：dump baseline/spec token 序列，供 run-to-run 逐位对比；
2. `STG_VERIFY`/`DUAL_VERIFY` 置位时打印 `VERIFY_SUMMARY`（三处校验器 ok/bad/calls），使「0 BAD」可判真伪（区分「校验通过」与「校验没跑」）。

除此之外未改动任何运行时/正确性代码；字节校验逻辑（`_verify_side_bytes` 等）为仓库既有设施，本次仅用其正确的开关触发。
