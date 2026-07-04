# C++ 统一池权威 —— 最终收益报告(Phase 4)

> **后续更新(2026-07-05):** 本报告写于"保留 `NATIVE_DEMAND_DUAL=0` 兜底一版"阶段。
> 之后已**彻底退役 Python decode 权威路径**:移除 opt-out、删 `acquire_gpu_dual`+`DUAL_VERIFY`、
> dual 模式弃用 PIN_HOT,demand_dual 成 decode 唯一权威(净删 316 行,tok/s 无回归 14.75)。
> prefill/host 常驻池路径与非双源 `acquire_gpu` 按需保留。详见
> `docs/superpowers/plans/2026-07-05-retire-python-decode-authority.md`。下文"显式延后"段已落地。


**日期:** 2026-07-04
**分支:** `perf/async-demand-offload`
**结论:** 生产路径的专家池管理已统一到 C++ 单一权威(demand_dual 默认接管真实区),**正确性严格验证通过、spec tok/s +8%、侧区内存减半**。Python 权威路径保留为 `NATIVE_DEMAND_DUAL=0` 兜底(计划「保留开关兜底一版」)。

---

## 1. 达成的端状态

| 维度 | 统一前(Python 权威) | 统一后(C++ demand_dual 默认) |
|---|---|---|
| 真实区槽/命中/驱逐/落池 | Python `acquire_gpu_dual` + `_slot_of/_free/_freq` | **C++ `g_real` 单一权威**(`demand_core_locked`) |
| 每层主线程同步 | n_miss + 落池记账 | **1 次 `inds.eval()`**,零落池/记账 |
| 侧区∪真实区 overlay | Python `eff[keys]=vals` | C++ `demand_core` 内 side 快照∪e2r |
| miss pread | 主线程/回退 | C++ BgReader 并行 worker |
| `resident_experts` 真值源 | Python `_slot_of` | C++ `real_region_contents` |
| 侧区缓冲 | 双缓冲(spec_gens=2) | **单缓冲持久 LFU(spec_gens=1)** |

## 2. tok/s 收益(cap48, K=3, MAXTOK=48, WARMUP=48, REPEAT=2)

| 指标 | OFF(Python 权威) | ON(demand_dual 默认) | Δ |
|---|---|---|---|
| spec_tok_per_s | 13.70 | **14.80** | **+8.0%** |
| baseline_tok_per_s | 7.16 | 7.50 | +4.8% |
| spec_hit_rate | 0.903 | 0.913 | +1.0% |

## 3. 内存

- 侧区从双缓冲(2×spec_slots 行/层)降为单缓冲(1×spec_slots 行/层)——**侧区内存减半**,现为所有默认路径(不止 cli)。

## 4. 正确性验收(严格)

| oracle | 结果 |
|---|---|
| 容量不变性(cap32≡cap48,greedy 逐位) | ✅ PASS(demand_dual 默认下 `test_capacity_invariance`) |
| 字节真值 Python 路径(DUAL_VERIFY) | ✅ 0 BAD, ok>0 |
| 字节真值 demand_dual 路径(STG_VERIFY) | ✅ 0 BAD, ok=45730 |
| demand_dual vs Python n_mismatch | ✅ ≤ N_floor(48 里 1 处,logit 精确平局的 FP tie-break;详见 schemeB 报告) |
| 单测(wiring/native/config/sideregion) | ✅ 全绿 |

## 5. 本轮完成 vs 显式延后

**完成(已提交):**
- `e388851` Route3 P1 C++ 拥有池直写 + 根治驱逐错槽 + 死代码清理
- `7d90fed` spec_gens 默认单缓冲(退役双缓冲 fallback)
- `8817111` Phase 2 root-cause:demand_dual 无 bug(FP 平局)
- `2327ffa` Phase 4.1:demand_dual 设默认(C++ 单一权威)

**显式延后(计划「保留开关兜底一版」,下一版再做):**
- Phase 3.5/4.2:删 Python `_slot_of/_free/_freq` 死影子 —— 当前仍是 `NATIVE_DEMAND_DUAL=0` 兜底路径的活状态,本版保留,不删。
- Phase 3.7:收敛 block.py 单路径 —— 与「保留兜底」冲突,同延后。
- P3-d:blob 并行读统一到 C++ `BgReader`(独立性能优化,风险最高)。

> Phase 3 的 P3-a(promote→C++)/P3-c(overlay→C++)在 dual-source 默认路径**已被 demand_dual 吸收或本就不在热路径**(`_do_promote=False` when zerocopy_dual_source),无需单独迁移。
