# 双源回退去 host 胶水 数据 (2026-07-02)

配置：K=3, dual+LFU, EXPERT_SLOTS=32, POOL_SPEC_SLOTS=32, W24/B12, MAXTOK=64, WARMUP_TOK=64, REPEAT=2。
只记数据，不写结论。

## Tier 1（mx.where 取 miss，砍侧区 keys.tolist + set）

| 版本 | tok/s | hit | accept | fastpath | fallback | fallback占比 | disk | n_mismatch | peak_gb |
|---|---|---|---|---|---|---|---|---|---|
| Tier1 前(def_mtp, R2) | 11.105 | 0.900 | 2.37 | - | - | - | 4159 | 13 | - |
| Tier1 前(fin_w24_b12, R2) | 11.61 | 0.901 | 2.37 | - | - | - | 3874 | 2 | 8.40 |
| Tier1 前(fp, R1) | 11.08 | 0.863 | 2.286 | 463 | 1121 | 70.8% | 5356 | - | - |
| Tier1 后(t1_mtp, R2) | 11.31 | 0.882 | 2.286 | 520 | 1064 | 67.2% | 5181 | 14 | 8.47 |

注：Tier1 前后 tok/s 均落在 ~11.1–11.6 噪声带内，无可测净提速；fallback 占比 67–71% 量级不变。

## Tier 2（map_only 下沉 gather）

（待执行）
