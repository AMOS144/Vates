# 退役 Python decode 权威路径 —— demand_dual 单一权威

**日期:** 2026-07-05
**前置:** demand_dual 已默认 + 经 Phase 2/4.1 严格验证(容量不变性 PASS、字节真值 0 BAD、tok/s +8%)。
**用户拍板:**
- **prefill 保持走常驻池**(chunk=2 分块、内存有界),host 槽机制 `_slot_of/_free/_freq/acquire/_place_expert/_alloc_slot/_choose_victim/fetch` **原样保留**,不删。
- **PIN_HOT 在 dual 模式弃用**(断言 `PIN_HOT==0`)→ 可整块删 `acquire_gpu_dual` + `DUAL_VERIFY`。

## 目标

decode/verify 双源路径**不再留 Python 退路**,demand_dual(C++ g_real)作唯一权威;移除 `NATIVE_DEMAND_DUAL` opt-out;删净只服务该退路的死代码。**不动 prefill/host 路径,不动非双源 acquire_gpu。**

## 严格边界(不变式)

1. **prefill/overcap 不受影响**:`acquire_host`→`acquire`/`fetch` 及其槽机制保持逐字节行为不变。
2. **非双源模式(ZEROCOPY_DUAL_SOURCE=0)不受影响**:`acquire_gpu` 分支保留(本轮不碰)。
3. **验收 oracle 必须持续全绿**:容量不变性(cap32≡cap48)+ demand_dual 字节真值(STG_VERIFY 0 BAD)+ 快速单测套件。
4. **PIN_HOT**:dual 模式下若检出 pinned 非空 → 明确报错(不再静默回退)。

---

## Stage 1 —— 删 opt-out + acquire_gpu_dual + DUAL_VERIFY(源码)✅

- [x] **1.1 `resident_pool.py`**:`_native_demand` = `spec_slots>0 且 hasattr(_N,"demand_dual")`(去 config 门)。删 `acquire_gpu_dual`、`_verify_side_bytes`、`_DUAL_VERIFY`/`_dual_verify_state`。`resident_experts/_count` 读 C++ 保留。
- [x] **1.2 PIN_HOT 守卫**:`pin()` 在 `_native_demand` 且非空时 `raise`。
- [x] **1.3 `virtual_pool.py`**:`acquire` dual 分支直连 `_acquire_native`;native 缺失则 `raise`;删 `_StagingSide` import。
- [x] **1.4 `config.py`**:删 `native_demand_dual()` + Phase 0 探针(`probe_all_hit_lazy/probe_no_demand`);`cli.py` 去掉 `NATIVE_DEMAND_DUAL` 兜配。
- [x] **1.5 `run_mtp_spec.py`**:VERIFY_SUMMARY 去掉 `DUAL_VERIFY.resident` 键。
- [x] **1.6 `block.py`**:更新注释。`_StagingSide` 类从 native_staging.py 删除。

## Stage 2 —— 测试对齐 ✅

- [x] 删 `test_resident_sideregion.py`、`test_dual_source_verify_shape.py`、`test_dual_fallback_gpu_miss.py`、`test_staging_side_adapter.py`;`test_virtual_pool*.py` dual 断言改为派发到 `_acquire_native`。
- [x] `test_demand_dual_wiring.py`:重写为"spec=0 不启用 / native 在则启用 / native 缺失 acquire 报错 / resident_experts 读 C++ / PIN_HOT 报错"。
- [x] `test_pool_byte_truth.py`:删 Python-path 用例,保留 demand_dual STG_VERIFY。

## Stage 3 —— 验收 ✅

- [x] `test_capacity_invariance` PASS(25.6s)
- [x] `test_pool_byte_truth`(demand_dual STG_VERIFY)0 BAD(141s)
- [x] 快速单测套件全绿(排除既存环境坏点 qlinear/qwen-mtp)
- [x] commit(Stage1+2+3 一起)

## Stage 4 —— Task 3.7 收敛 block.py(双源 decode 分支)

- [ ] 评估 decode GPU-remap 分支:双源已单一(_vpool.acquire→demand_dual);清理只服务旧退路的诊断分叉(miss_attrib/route_trace 保留为诊断)。**非双源 acquire_gpu 分支保留**。
- [ ] 出口:oracle 全绿 + commit。

## Stage 5 —— P3-d 并行读→C++ BgReader(最高风险,单独评估)

- [ ] 现状:host `fetch`/prefetch 用 `blob_loader` 的 Python `ThreadPoolExecutor`;`_materialize_native` 已用 C++ `blob_load`(eval 时 C++ pread 绕 GIL)。demand miss 已用 C++ `BgReader`。
- [ ] 评估把 host 并行 pread 也统一到 C++ `BgReader` 的收益/风险;有净收益且可灰度再做,否则记录为不做的理由。
- [ ] 出口:A/B tok/s + oracle 全绿 + commit 或明确不做。
