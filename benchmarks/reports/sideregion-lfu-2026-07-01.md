# 侧区持久 LFU 端到端数据(2026-07-01)

环境:`STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32`、`MAXTOK=64`、80B 4bit、
prompt="用三句话解释什么是混合专家模型。"。脚本:`benchmarks/bench_dual_source.py`。
dual on 均为 `ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1`,`POOL_SPEC_SLOTS=spec`(=侧区行数)。

## 一、spec 扫描(warmup=64)

| 配置 | hit_rate | disk_loads | active_gb | peak_gb | tok/s |
|---|---|---|---|---|---|
| cap=32 单池(dual off,基线) | 0.7645 | 7975 | 6.65 | 6.83 | 4.99 |
| LFU spec=8  run1/2 | 0.7277 / 0.7304 | 9223 / 9132 | 4.76 | 5.07 | 5.50 / 5.65 |
| LFU spec=12 run1/2 | 0.7456 / 0.7510 | 8617 / 8434 | 5.10 | 5.58 | 5.37 / 5.40 |
| LFU spec=32 run1/2 | 0.8096 / 0.8123 | 6448 / 6356 | 7.02 / 6.79 | 7.68 / 7.69 | 5.46 / 5.26 |
| cap=64 单池(参考,取自 spec 前测) | 0.869 | 4450 | 9.38 | — | 4.75 |

## 二、长 warmup 对命中(spec=32)

| warmup | hit_rate | disk_loads | active_gb | peak_gb | tok/s |
|---|---|---|---|---|---|
| 64  | 0.8096 | 6448 | 7.02 | 7.68 | 5.46 |
| 192 | 0.8019 | 6710 | 6.79 | 7.91 | 5.20 |
| 320 | 0.8025 | 6688 | 6.79 | 8.03 | 5.07 |

命中在 ~0.81 饱和,加 warmup 不再上升(LFU 工作集很快填满)。

## 三、run-to-run 差异(bench_dual_source --diff)

| 比对 | n_mismatch | first_mm_pos |
|---|---|---|
| spec=8  on1 vs on2 | 13 | 51 |
| spec=12 on1 vs on2 | 35 | 28 |
| spec=32 on1 vs on2 | 26 | 38 |
| spec=8  vs 基线 off | 14 | 45 |
| spec=12 vs 基线 off | 34 | 30 |
| spec=32 vs 基线 off | 34 | 30 |

所有 dual on 配置均非逐位确定,漂移在 ~30-50 token 后出现;字节校验(`DUAL_VERIFY`)全程 0 BAD。

## 四、默认值决策

- `SIDEREGION_LFU` 默认 **off**(dual-source 本身也 opt-in):dual on 会引入 run-to-run token 漂移
  (良性时序噪声,非字节错误),不宜作默认。
- 使用 dual-source 时**推荐开 LFU**(严格优于旧"∉P 全清"侧区:后者 hit 仅 0.709、且大 spec 触发竞态)。
- 两个推荐工作点(按需):
  - **省内存**:`POOL_SPEC_SLOTS=8` → active 4.76GB(−1.9GB vs 基线)、hit 0.73、tok/s +12%。
  - **提命中**:`POOL_SPEC_SLOTS=32` → hit 0.81(+0.045 vs 基线)、内存持平 6.9GB、tok/s +8%。
- 0.85+ 命中需真加常驻槽(cap=64→0.869),侧区 LFU 因单区容量与 warmup 饱和到 ~0.81 封顶。
