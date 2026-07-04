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

## Stage 4 —— Task 3.7 收敛 block.py ✅(评估:无安全高价值收敛空间)

- [x] 评估结论:block.py 生产双源路径**已是** gate→prefetch(`_native_fused_prefetch`)→acquire(`_vpool.acquire`→demand_dual)。其余分支均为**保留的运行模式**,非死代码:
  - `stream_blob`/`stream_blob_bg`(legacy 全流式,配置门控)
  - 非双源 `acquire_gpu`(用户拍板保留)
  - host prefill `acquire_host`(用户拍板保留,内存有界)
  - `route_trace`/`miss_attrib`/`union_prof`/`probe_perlayer_sync`(opt-in 可观测性)
- 在"保留非双源 + prefill"的约束下,进一步删这些分支=删用户选择保留的功能,超范围且有回归风险。故**本轮不动**;若未来要更彻底单路径,需另立"退役 legacy 模式"决策。

## Stage 5 —— P3-d 并行读→C++ BgReader ✅(评估:不做,附理由)

- [x] 现状:decode 稳态**已全走** C++ `BgReader`(demand miss,`bg_reader_start` DEMAND_WORKERS=8);`_materialize_native` 用 C++ `blob_load`(eval 时 C++ pread 绕 GIL)。`blob_loader` 的 Python `ThreadPoolExecutor` 仅服务**一次性 prefill/host fetch**,decode 稳态不触发。
- **结论:不做。** 迁移只影响一次性 prefill 的并行读,对 decode 稳态 tok/s **零收益**,而改动 blob_loader 线程模型是本计划标注的**最高风险项**。风险/收益严重倒挂。若将来要压 prefill 首 token 延迟再单独立项灰度。
