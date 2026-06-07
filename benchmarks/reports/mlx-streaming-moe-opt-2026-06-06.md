# MLX Streaming-MoE 提速实验报告（①②④③）

> 日期：2026-06-06　分支：`feature/mlx-streaming-moe`
> 目标：在不破坏数值等价的前提下，给路线 B 文件后端流式 MoE 提速。
> 基线见同目录 `mlx-streaming-moe-final-2026-06-05.md`。模型 `/tmp/qwen3moe`（Qwen3-30B-A3B-4bit，48 层×128 专家×top8），`MAXTOK=64`。

## 改动一览

| 编号 | 改动 | 结论 |
|---|---|---|
| ① | 驱逐时不再每次 `clear_cache`（可选 `clear_on_evict`）+ `set_cache_limit` 控上限 | ✅ **有效，+37% tok/s** |
| ② | 每层独立 LRU（容量=每层槽数） | ✅ 中性偏正：常驻可预测=槽数×层数，隔离跨层驱逐 |
| ④ | `PersistentSubGLU`：跨 token 复用 SwitchGLU+QSL，仅原地 update | ✅ 无害（I/O 主导下增益小） |
| ③ | 热专家常驻 + 冷专家流式（离线统计激活频率后钉住） | ❌ **实测否决：命中率升但吞吐降** |

数值等价：13 个离线单测全过（含 per-layer 隔离、微型量化模型端到端 logits 等价、record/hot/pin）。

## ① 去 clear_cache 抖动（干净 A/B，同 58.65% 命中 / 768 常驻 / 同内存）

| clear_on_evict | tok/s | gen_s | mlx_peak |
|---|---|---|---|
| true（旧行为，每次驱逐 clear_cache） | 4.03 | 15.9 | 7.06 |
| **false（+ CACHE_GB=2 控上限）** | **5.52** | 11.6 | 7.06 |

低命中率下，每次驱逐 `clear_cache` 造成 malloc/free 抖动；去掉后靠 MLX 缓冲复用，**同等命中与内存下吞吐 +37%**。这是本轮最大的实打实提速；早期最终报告的数据正是带着这个抖动测的。

## ②+ 纯 per-layer LRU 槽位扫描（提速后）

| 每层槽数 | 常驻专家 | RSS_gen (GB) | 命中率 | tok/s |
|---|---|---|---|---|
| 8 | 384 | 2.1 | 33.4% | ~4.1 |
| 16 | 768 | 3.0 | 58.7% | 5.5–6.2 |
| 32 | 1536 | 5.0 | 77.7% | 6.7 |

对比基线最终报告（旧全局 LRU + clear_cache）：旧全局 1024 → 67.5% 命中、5.62 tok/s、RSS 3.96。
提速后每层 32（RSS 5.0、6.7 tok/s）已超过旧最优；每层 16（RSS 3.0）即可达 5.5–6.2 tok/s，内存更省。

## ③ 热专家常驻——实测否决

做法：先用一段校准生成（`CAL_TOK=32`）统计每层专家激活频率，钉住每层最热的 `PIN_HOT` 个专家（永不驱逐），冷专家走小 LRU。

干净同次对比（均 **常驻 768**，`CACHE_GB=2`）：

| 方案 | 常驻(钉住) | 命中率 | tok/s |
|---|---|---|---|
| 纯 LRU16 | 768 (0) | 58.7% | **6.21** |
| 钉住8 + LRU8 | 768 (384) | **63.8%** | 3.85 |

**反直觉但可复现：钉住热专家把命中率从 58.7% 抬到 63.8%，吞吐却从 6.21 掉到 3.85 tok/s。**

原因分析：
1. Qwen3 MoE 训练带负载均衡损失，专家激活**不够偏斜**——没有强主导的「热专家」，钉住的收益本就有限。
2. 每 token 的 gather 计算是**内存带宽受限**的；钉住 384 个常驻活跃数组额外占用统一内存带宽/缓存，拖慢了 GPU kernel。基于「最近使用」的纯 LRU 工作集更紧凑、局部性更好，单位内存的吞吐更高。

→ 在此模型上，**把内存预算花在「更大的 recency LRU」比花在「钉住热专家」更划算**。`PIN_HOT` 功能保留在代码里（默认关），可用于专家激活高度偏斜的其它模型。

## ⑤ 每层单文件 + 单专家切片——实测否决

动机：把每层 8 次文件打开降到 1 次、减少「读放大」。用两个探针实测：

**探针 1（`probe_layerfile.py`）：lazy 加载每层单文件后切 8 个专家**

| 模式 | 文件 | 载入后 RSS | 切完 RSS | MLX 活跃 |
|---|---|---|---|---|
| 整层物化（上界） | 339.7MB | 37MB | 378.5MB | 339.7MB |
| 只切 8 专家 | 339.7MB | 36.9MB | **381.1MB** | 361MB |

→ 切 8 个专家 RSS ≈ 整层。**MLX 不做切片级按需换页：任何 eval 都把整层 340MB 拉进内存**。
即每层单文件会读 16× 多的字节（需 21MB 却读 340MB），要么慢、要么整层常驻=不省内存。

**探针 2（`probe_partial_read.py`）：每层单文件只 pread 选中 8 个专家的字节区间**

| 读法（200 轮×8 专家） | 耗时 |
|---|---|
| per-expert `mx.load`（当前） | **0.286s** |
| 每层单文件 partial `pread` | 0.417s（**慢 1.45×**） |

→ 即便只读选中字节，手写 `seek/read` + `np.frombuffer` + `mx.array`（memcpy）的开销
**比 `mx.load` 的 mmap 零拷贝还慢**，省下的 `open()` 远抵不过。

**结论**：当前 per-expert 布局本就**零读放大**（只读选中专家），`mx.load` 小文件已是
mmap 零拷贝。「减少读放大」的前提在此设计里不成立，`open()` 开销也微不足道。**⑤ 在
两种形态下都更差，否决。** 探针脚本保留作可复现证据。

## 结论与推荐配置

- 实打实提速来自 **①**（去 clear_cache 抖动，+37%）；**②** 让内存可预测；**④** 无害。
- **③ 在 Qwen3-30B-A3B 上得不偿失**，已否决。
- 推荐：纯 per-layer LRU，按可接受内存选槽数。**每层 16（RSS≈3GB、~6 tok/s）** 为平衡档，**每层 32（RSS≈5GB、6.7 tok/s）** 为偏吞吐档。
- 瓶颈在 **per-expert 小文件读 I/O**，但该布局已是**零读放大 + mmap 零拷贝**，⑤（每层单文件）实测更差、已否决。
- 真正还可能有效的后续方向：**预取/重叠**（在算第 L 层时异步预读第 L 层下一批可能用到的专家）、把专家目录**预热进 OS 页缓存**、或换更快的 NVMe；以及 batch>1 场景下专家工作集天然变大、命中率更高。

## 复现命令

```bash
cd hypura/mlx_streaming && . .venv/bin/activate && cd ..
# ① A/B
EXPERT_SLOTS=16 CLEAR_ON_EVICT=1 MODEL=/tmp/qwen3moe EXPERT_DIR=/tmp/mlx_qwen3_experts MAXTOK=64 python3 -m mlx_streaming.run_streaming
EXPERT_SLOTS=16 CACHE_GB=2   MODEL=/tmp/qwen3moe EXPERT_DIR=/tmp/mlx_qwen3_experts MAXTOK=64 python3 -m mlx_streaming.run_streaming
# ③（否决档）
EXPERT_SLOTS=8 PIN_HOT=8 CACHE_GB=2 MODEL=/tmp/qwen3moe EXPERT_DIR=/tmp/mlx_qwen3_experts MAXTOK=64 python3 -m mlx_streaming.run_streaming
```
