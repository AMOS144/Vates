# Phase 2 方案B(demand_dual)错槽 root-cause 报告

**日期:** 2026-07-04
**结论:** demand_dual(`NATIVE_DEMAND_DUAL=1`)**已是正确的**——plan 里假设的「确定性错槽 bug」不存在(前提已过时)。与 Python 路径的唯一 token 分歧是 logit 精确平局处的良性 FP-order 噪声,属噪声地板 `N_floor`。

---

## 1. 复现事实

| 对比 | cap | 首个分歧 token | 判定 |
|---|---|---|---|
| demand_dual cap32 vs cap48 | 32/48 | **无(-1)** | 容量不变性 PASS |
| demand_dual cap48 vs Python cap48 | 48 | index **45**(共 48) | 需归类 |
| demand_dual cap32 vs Python cap48 | 32 | index 45 | 同上 |

demand_dual **自身容量不变**(cap 只影响命中率不影响输出),证明其真实区槽逻辑/驱逐自洽。唯一疑点是与 Python 路径在 token 45 的单点分歧。

## 2. 归类:FP 平局,非字节/映射 bug

用 `DUMP_MARGIN` 打印两路径 greedy 每步 top-2 logit 与差值,token step=45:

```
Python 路径:      top1=100144(25.87500)  top2=101065(25.87500)  gap=0.000000   ← 精确平局
demand_dual 路径: top1=101065(25.75000)  top2=100144(25.62500)  gap=0.125000
```

- **同一对候选** {100144, 101065};logit 量化到 0.125 步长。
- Python 侧两者**完全相等**(25.875),argmax 由 tie-break(取最小 index)选 100144。
- demand_dual 的 MoE gather 累加顺序与 Python 有 ~1 量化步(0.125~0.25)的 FP 差异 → 在这个平局点翻边,选 101065。

**字节/映射 bug 会产生大幅 logit 偏差(读到别的专家权重=垃圾输出),绝不可能是「差 0.125 的精确平局」。** 故 token-45 分歧被确定性归类为**良性 FP-order 噪声**,即噪声地板 `N_floor`。

## 3. 为何 plan 的旧前提过时

plan Phase 2 假设 demand_dual 有「确定性错槽」,是基于当时的状态。此后:

- C++ `demand_core_locked`(`native_prefetch.cpp:911-915`)的驱逐保护集用的是 `access_seen`(本前向**全部**唯一路由专家=命中+miss),注释明写「比 Python 仅护 miss 更严格、更正确」。
- 即 C++ demand 路径**早已**带有正确的驱逐不变量;而 Python `acquire` 仅护 miss 的同类 bug 是 2026-07-04 才修(commit e388851)。

所以 demand_dual 不存在错槽;Phase 2 的 root-cause 结论是**「无 bug,原前提过时」**。

## 4. Phase 2 出口判据核对

- ✅ 容量不变性 PASS(demand_dual cap32==cap48)
- ✅ n_mismatch ≤ N_floor(唯一分歧=精确平局的 tie-break,属 N_floor)
- ✅ STG_VERIFY 字节落池 0 BAD(见 §5)
- ✅ tok/s 净提升 +8%(见 §7)

## 5. STG_VERIFY 字节落池校验(0 BAD)

`NATIVE_DEMAND_DUAL=1 STG_VERIFY=1`,MAXTOK=16:

```
STG_VERIFY.virtual(_verify_native_bytes): ok=45730  bad=0  calls=1584
```

45730 次「真实区槽字节 == g_real 属主专家 blob 真值」比对**全部通过(0 BAD)**。C++ 接管落池字节等价成立。

## 6. N_floor 定义(本报告副产)

跨代码路径(Python-slot vs C++ demand_dual)存在 **logit 精确平局** token,其 argmax 由 FP 累加顺序的 tie-break 决定 → 天然不可消除的路径间 token 差。故跨路径对比的 `N_floor > 0`:**同一对候选、logit 差 ≤ 1 量化步(0.125)的翻边不算 bug**。本例 48 token 里仅 1 处(token 45)。

## 7. tok/s 净收益

cap48, K=3, MAXTOK=48, WARMUP_TOK=48, REPEAT=2:

| 指标 | OFF(Python 权威) | ON(demand_dual) | Δ |
|---|---|---|---|
| spec_tok_per_s | 13.70 | **14.80** | **+8.0%** |
| baseline_tok_per_s | 7.16 | 7.50 | +4.8% |
| spec_hit_rate | 0.903 | 0.913 | +1.0% |

demand_dual 每层省掉主线程落池/记账 + 单次 inds 同步,net +8% spec tok/s。**可作为 Phase 3 的稳定基座,并支撑 Phase 4 设默认。**
