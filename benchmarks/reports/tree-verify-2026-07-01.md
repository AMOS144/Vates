# 完整树形验证(batch-of-paths)实测数据

模型 qwen3-next-80b-a3b-4bit,g64 专家,STREAM_BLOB_LOADER=1,NATIVE_FUSED_PREFETCH=1,K=3。
PROMPT="用三句话解释什么是混合专家模型。",MAXTOK=64。ref=baseline 链验证输出(逐 token)。
control=baseline vs baseline(量非确定性本底)。exact_match=输出逐 token 等于 ref。

## 1. 逐层 batch 等价性(benchmarks/test_layerwise.py)

| 项 | 结果 |
|---|---|
| batch=2 tiled-prefix vs batch=1 solo,每层 h[0,-1,:] max\|Δ\| | 全 48 层 = 0.00000 |
| 最终 logits max\|Δ\| / argmax | 0.0 / match |

单层(GDN 线性 / 全注意力 / 含 MoE 的 DecoderLayer / gated_delta kernel)重复前向、batch 等价:全 0.0。

## 2. 整网前向重复确定性(benchmarks/test_determinism.py)

同输入、同 cache、连续 8 次前向,相邻 max\|Δ\|:

| run i vs i+1 | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 |
|---|---|---|---|---|---|---|---|
| max\|Δ\| | 9.375 | 3.531 | 5.062 | 6.562 | 2.594 | 5.531 | 5.219 |

- 与 EXPERT_SLOTS(64/512)无关、与 MoE 路径(native/host/remap)无关(diff 逐位相同)。
- token 级本底(control_baseline_vs_baseline_mismatch)=0(该 prompt argmax 余量掩盖 logit 噪声)。

## 3. A/B:baseline 链验证 vs TREE_VERIFY(benchmarks/bench_tree_verify.py)

### EXPERT_SLOTS=32

| 配置 | accept_len | dec_hit_rate | disk_loads | miss_A_timing | tok/s | exact_match | n_mismatch |
|---|---|---|---|---|---|---|---|
| baseline (P=1) | 2.286 | 0.6895 | 9375 | 6772 | 7.91 | true | 0 |
| tree_verify P=1 | 2.286 | 0.6959 | 9181 | 6575 | 8.59 | true | 0 |
| tree_verify P=2 | 2.370 | 0.6863 | 13634 | 7630 | 3.68 | **false** | 14 (首失配 pos 45) |

### EXPERT_SLOTS=256(池 ≥ 并集,无驱逐)

| 配置 | accept_len | dec_hit_rate | disk_loads | tok/s | exact_match | n_mismatch |
|---|---|---|---|---|---|---|
| baseline (P=1) | 2.286 | 0.9961 | 117 | 15.88 | true | 0 |
| tree_verify P=2 | 2.370 | 0.9844 | 560 | 11.35 | **true** | 0 |

## 4. 关键数量关系

- accept_len:P=2 相对 baseline +3.7%(2.286→2.370),accept_hist [5,9,4,10]→[4,8,5,10]。
- exact_match:P=1 恒 true;P=2 在 slots=32 为 false(14 失配)、slots=256 为 true(0 失配)。
- disk_loads:slots=32 时 P=2 相对 baseline +45~49%(9375→13634)。
- tok/s:slots=32 时 P=2 ≈ baseline 的 0.46×(7.91→3.68);slots=256 时 ≈ 0.71×(15.88→11.35)。
- dec_hit_rate:slots=32 P=2 vs baseline 0.686 vs 0.690(持平);slots=256 0.984 vs 0.996。
