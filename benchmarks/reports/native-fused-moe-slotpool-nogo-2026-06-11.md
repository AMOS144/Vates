# Native Fused MoE / Slot Pool 全模型 NO-GO 复盘（2026-06-11）

## 结论（TL;DR）
手写 Metal fused MoE（gate/up/SwiGLU/down 融合）即便做成 **MLX C++ Primitive + Native Compute Slot Pool**（常驻一层、按 local slot 读权重、避免每次 active-list staging），在**真实全模型 2bit-g128 配置**下仍然**显著慢于 MLX 原生 `gather_qmm`**，并且翻倍内存。

**判定：native fused MoE 作为全模型方案 NO-GO，搁置。** 与 README 早期结论（"fused MoE Metal kernel = NO-GO"）一致，这里在 slot-pool 变体 + 全 48 层规模上再次坐实。

## 实测 A/B（同机、同配置）
- 模型：`/tmp/qwen3_next_80b_4bit`（主模型 4bit）
- 专家：`models/qwen3_next_experts_2bit_g128`（2bit/g128，512 专家 × 48 层，未旋转）
- compute buffers：`/tmp/cb_2bit_g128`（全 48 层 × 3 投影，2bit，35s 打包，~21GB）
- 运行：`run_mtp_spec`，`EXPERT_SLOTS=256 RESIDENT_POOL=1 MTP_VERIFY_MODE=batch MTP_ARRAY_COMMIT=1 K=2 MAXTOK=96`

| 变体 | spec tok/s | t_verify_s | mlx_peak_gb | exact_match |
|---|---|---|---|---|
| baseline（MLX 原生） | **17.48** | 3.47 | 11.06 | batch 近似（37 处） |
| native slot pool（全 48 层, cap=256） | **7.25** | 11.88 | 24.47 | true |

native 全层开启：**慢 2.4×**，验证耗时 3.4×，峰值内存 2.2×。

## 正确性已排除（不是 bug，是慢）
2bit native kernel 与 MLX `quantized_matmul` 参考逐专家对齐：
- `cosine = 0.999999`，`max_rel_diff = 0.0003`，`max_abs_diff ≈ 0`
- A/B 里 `exact_match=true`

数值正确，纯粹是性能问题：单 token、每专家一个 threadgroup 的 fused kernel 干不过 MLX 批量调优的 `gather_qmm`；slot pool 的 rebuild（`mx.stack` cap=256 槽 × 9 数组）+ 每层重复常驻进一步放大开销与内存。

## 为什么之前 layer43/47 看着"略赢"
早期只在 2 层（6bit compute buffer）上测，得到 14.07 vs 13.65——小样本噪声 + 仅占 2/48 层。铺满全模型后，fused kernel 的低效与 slot-pool 开销完全压过 MLX 原生路径。

## 真实瓶颈（重定向依据）
真实主线基线（2bit-g128 + 256 槽 + K=2）：
- `spec_tok_per_s=17.48`、`baseline_greedy=10.94`、`speedup=1.6×`
- `spec_hit_rate=0.989`、`disk_load_ratio=0.12` → **IO 已基本解决，预取上限有限**
- `avg_accept_len=1.745`、`t_verify_s=3.47`、`t_draft_s=1.94` → **瓶颈在 MTP 接受率与验证成本**

注：本机 ~17.5 tok/s，低于历史报告的 24-28，主要是硬件差异 + 本次 prompt/配置下接受率偏低。

## 处置
- `NATIVE_MOE` 默认 `0`，保持关闭；slot pool / fused primitive 代码保留作实验，不进默认路径。
- 临时 compute buffers `/tmp/cb_2bit_g128`（~21GB）可删，需要时 35s 重新打包。
- 下一步聚焦 MTP：接受率（草稿质量 / 主模型与专家量化配套）、batch vs step verify 的速度-质量取舍、`t_verify_s` 优化。
