# 侧区持久 LFU 端到端数据(2026-07-01)

环境:`STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32`、`MAXTOK=64 WARMUP_TOK=64`、80B 4bit、prompt="用三句话解释什么是混合专家模型。"。
脚本:`benchmarks/bench_dual_source.py`。

## 第一期(reserve 内 LFU,回调线程发布)

| 配置 | hit_rate | disk_loads | active_gb | peak_gb | tok/s |
|---|---|---|---|---|---|
| cap=32 单池(dual off,基线) | 0.7645 | 7975 | 6.65 | 6.83 | 4.99 |
| dual on `SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32` run1 | 0.8096 | 6448 | 7.02 | 7.68 | 5.46 |
| dual on `SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32` run2 | 0.8123 | 6356 | 6.79 | 7.69 | 5.26 |
| cap=64 单池(参考,取自 spec 前测) | 0.869 | 4450 | 9.38 | — | 4.75 |

## 差异比对(bench_dual_source --diff)

| 比对 | exact_match | n_mismatch | first_mm_pos |
|---|---|---|---|
| on1 vs on2(自身确定性) | false | 26 | 38 |
| off vs on1(对基线) | false | 34 | 30 |

## fastpath / fallback(dual on run1)

- gpu_fastpath = 1198,gpu_fallback = 1874(基线 off:fastpath 666 / fallback 2406)。
