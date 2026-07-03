# 峰值内存收缩：跳过 batch verify 无用快照（2026-07-03）

## 目标与红线

把 MTP 自投机解码的**在途峰值** `mlx_peak_gb` 压向 `mlx_active_gb`（现状 cap32：peak≈8.45 / active≈7.72，差 ~0.7GB），
且**不碰专家池、不掉速度**。红线：不改 `EXPERT_SLOTS/POOL_SPEC_SLOTS/cap`、不显著增加 per-layer 同步、bit-exact。

口径说明：`active` 在 `clear_cache` 后量（真正持有的张量）；`peak` 是 `get_peak_memory` 的在途高水位。
本机 swap 近满（物理 32G），**绝对 tok/s 不可信**，故以 `n_mismatch/hit/disk_loads/fallback_replays` 等计数型确认无回归。

## 一、峰值来源定位（分段 reset_peak 探针）

用 `benchmarks/peak_probe.py`（throwaway）在各阶段间 `reset_peak()`，把峰值拆开：

| 阶段 | 峰值 GiB | 说明 |
|---|---|---|
| prefill（分块 chunk=2） | ~8.05 | MoE 按需 gather 的瞬时量，已被分块压到稳态 |
| decode 单步（seq=1，含 lm_head）孤立 | ~7.49 | 孤立测（工作集较小） |
| verify（seq=K）**不含** checkpoint 孤立 | ~7.58 | 前向瞬时量 ~0.31（相对孤立 active 7.27） |
| verify（seq=K）**含** checkpoint 孤立 | ~7.91 | checkpoint 路径把瞬时量抬到 ~0.64 |
| **真实解码循环** | **~8.44** | 峰值发生在此（不是 prefill） |

**单步解剖**（真实 cache 上量字节）：

| 组成 | 大小 | 是否可省 |
|---|---|---|
| `snap_m`（每步一次全 cache 深拷贝，回退兜底用） | **72 MiB** | ✅ 纯 batch 直提交路径下**永不使用** |
| `_spec_checkpoints`（K=3 个 per-token conv+ssm，跨 48 层） | 216 MiB | ❌ 零 replay 直接提交所必需 |
| MoE-gather 前向瞬时量 | ~0.32 GiB | ❌ 即 prefill 地板（8.05−7.72），绑定专家 gather |

**结论**：0.7GB ≈ `snap_m`(0.07) + `checkpoints`(0.22) + `前向瞬时量`(0.32，prefill/decode 都有)。
唯一「零速度代价 + bit-exact + 不碰池」可省的是 `snap_m`。

## 二、被证伪的候选

| 候选 | 实测 | 结论 |
|---|---|---|
| `set_cache_limit`（HYGIENE_CACHE_GB=1.0/0.5） | peak 仍 8.45 / 8.49 | MLX `get_peak_memory` 只计 **active（在用张量）**、不计可回收缓冲 → 封顶无效 |
| verify 前向分段 eval（VERIFY_SEG=8/24） | loop_peak 8.245 / 8.273 ≈ 未分段 8.258 | checkpoint 跨 48 层累积、与 eval 粒度无关 → 无削峰，反增 per-layer 同步 |
| lm_head 分块投影 | lm_head 边际峰值 ~0.008 GiB | 可忽略，无收益 |

## 三、所做改动（唯一改动）

`mlx_streaming/mtp/generate.py`：纯 batch 直接提交路径下跳过每步的 `snap_m = _snapshot(main_cache)`。

`snap_m` 只在三种分支被读：① tree 救回（`tree_top2`）② step 模式 ③ replay 回退（`commit` 失败）。
在**纯 batch 模式**且模型保证 `commit_verified_prefix` 恒成功时，这三条都走不到，`snap_m` 是纯废重量（每步一次 72MiB 深拷贝 + 一次 `mx.eval` 同步栅栏）。

**保证 commit 恒成功的判定**（`_batch_direct_commit_guaranteed`，模型结构恒定、循环外算一次）：
- 非线性层 → `KVCache` 可裁剪，必成功；
- 线性层 → 必须是被 patch 过的 `Qwen3NextGatedDeltaNet`（verify 前向才会写 `_spec_checkpoints`）。

通用/玩具递归层（未 patch，如单测里的 `_RecurLayer`）判定为 False → **保留 `snap_m` 走安全 replay 兜底**，行为与原来逐 token 等价。回退分支加了 `snap_m is None` 的显式 `RuntimeError` 守卫（理论不可达，破坏前提时立即定位而非静默错算）。

## 四、改后实测对比（cap32：EXPERT_SLOTS=32 / POOL_SPEC_SLOTS=32 / K=3 / MAXTOK=48）

| 指标 | 改前(有 snap_m) | 改后(跳过 snap_m) |
|---|---|---|
| **mlx_peak_gb** | **8.45** | **8.23 ~ 8.27** |
| mlx_active_gb | 7.72 | 7.72（不变） |
| **省下峰值** | — | **≈ 0.18 ~ 0.22 GB** |
| spec_hit_rate | 0.882 | 0.87 ~ 0.88（不变） |
| spec_disk_loads | 4190 | 4262 ~ 4407（同量级） |
| fallback_replays | 0 | **0** |
| direct_commits | 全部 | **全部** |
| spec_tok_per_s（仅参考，swap 未清） | 9.25 | 9.6 ~ 10.77（未回归，略升） |

> 实测省 0.18~0.22GB，**大于** `snap_m` 裸字节 72MiB——因为省掉的那次 `mx.eval` 同步栅栏原本还会强制物化其它中间量、抬高高水位。

## 五、bit-exact 结论

**逻辑证明**（强于经验）：`fallback_replays=0` 且 `direct_commits`=全部步数 → 每步都走**直接提交**分支，该分支字节路径与是否存在 `snap_m` 无关；`snap_m` 仅在执行了 **0 次**的 `else`（回退）分支被读。故输出与改前**逐字节相同**。

`n_mismatch`（spec vs 非投机 greedy）在 0~22 间波动，改前(0/18/21)与改后(0/21/22)**同一区间**——这是异步预取/后台读线程在 swap 压力下的**既有**路由非确定性，与本次改动无关（本改动不触碰任何数值算子）。

## 六、速度回归风险

无。改动**移除**了每步一次全 cache 深拷贝 + 同步栅栏，是**减负**；单测 `test_mtp_generate.py` 全 19 项绿（含 `arrayscache_default_uses_safe_replay` 等回退路径），`test_dual_source_verify_shape.py` / `test_pool_spec_gens.py` 绿。

## 七、为何只能省这么多

剩余 gap（peak 8.25 − active 7.72 ≈ 0.53GB）由两块构成，均触红线不可动：
- `_spec_checkpoints` 216MiB：零 replay 直接提交的根基，去掉即退化成 replay → **掉速**；
- MoE-gather 前向瞬时量 ~0.32GiB：也是 prefill 地板（8.05），绑定专家 gather/计算 → **碰池**。

因此「最简单、零速度、不碰池」的峰值收缩上限就是 `snap_m` 这 ~0.2GB。
