# MLX Streaming-MoE 真实模型端到端实测报告

> 日期：2026-06-05　分支：`feature/mlx-streaming-moe`
> 目标：在真实 MoE 模型上验证「能装进统一内存、但**故意不让它全量驻留**，把专家流式到磁盘以腾出内存给其他程序」是否可行，并量化内存/速度/命中率的权衡。
> 前置结论见同目录 `mlx-streaming-moe-phase0-2026-06-05.md`（已判定走**路线 B**：按专家拆文件 + 只加载选中专家 + LRU 驱逐）。

## 环境与模型

| 项 | 值 |
|---|---|
| 机器 | Apple M5，统一内存 32 GiB |
| Python / mlx / mlx-lm | 3.14.5 / 0.31.2 / 0.31.3 |
| 模型 | `mlx-community/Qwen3-30B-A3B-4bit`（本地路径 `/tmp/qwen3moe`） |
| 规格 | 48 层，每层 **128 专家**，top-**8**，hidden=2048，moe_intermediate=768，4-bit group64 |
| 磁盘体积 | 4 个分片共约 **16 GB**（专家约占其中 ~14GB） |
| 生成设置 | prompt 固定，`MAXTOK=64`，decode 速度=64/gen_s |

> 注：模型下载因代理对大文件单连接限速，最终用 **16 路并行分段 curl（按字节范围续传+拼接校验）** 完成，两个 5GB 分片字节数与服务端 `Content-Length` 完全一致。

## 实验结果

度量口径：`rss` = 进程 `ru_maxrss`（进程峰值常驻，含非专家权重+KV+激活+常驻专家）；
`mlx_active/peak` = MLX 自报的工作内存。流式各档均为**独立干净进程**（专家拆分已预先单独完成，不污染 RSS 水位）。

| 模式 | 专家槽位 | 进程 RSS (GB) | MLX 活跃 (GB) | MLX 峰值 (GB) | 专家命中率 | decode tok/s |
|---|---|---|---|---|---|---|
| 全量驻留基线（`run_baseline`，强制 eval 全参） | — | 10.12 | — | **17.21** | — | **24.79** |
| 路线 A：lazy 不强制驻留（`run_mmap`） | — | **17.36** | 17.17 | 17.20 | — | 24.89 |
| **路线 B 流式（`run_streaming`）** | 64 | **1.41** | 1.04 | 1.08 | 0.0% | 3.54 |
| **路线 B 流式** | 384 | 2.26 | 1.89 | 1.93 | 33.4% | 4.19 |
| **路线 B 流式** | 1024 | 3.96 | 3.59 | 3.63 | 67.5% | 5.62 |

补充：路线 B 加载+patch 后（生成前）RSS 仅 **0.36 GB**（lazy 加载，专家未物化）；拆分 6144 个 per-expert 文件耗时 **12.3 s**（一次性）。

## 解读

1. **核心目标达成——「能装下也不驻留」。** 模型能整个放进 32GB 统一内存（基线驻留 ~17GB），但路线 B 把 MoE 专家留在磁盘按需流式后，**进程常驻压到 1.4–4.0 GB**，MLX 工作内存压到 **1.0–3.6 GB**。最省档（槽位 64）相对全量驻留 **约 12× 缩减**，腾出的十几 GB 统一内存可留给本地其他程序。

2. **路线 A 被再次否决。** lazy 加载时 RSS 仅 0.36GB，但 `generate` 触及全部层/专家后整模型被物化，RSS 反而涨到 **17.4GB**——纯 lazy 在生成期完全救不了常驻，与 Phase 0 合成探针结论一致。

3. **内存/速度/命中率是连续可调的前沿。** 槽位越大 → LRU 命中率越高（0% → 33% → 68%）→ decode 越快（3.5 → 4.2 → 5.6 tok/s），代价是常驻线性上升。槽位 64 时命中率为 0，是因为单个 token 前向要穿过 48 层、每层 top-8，理论工作集 ~384 个 (层,专家) 远超 64 槽，token 内即被驱逐；槽位 ≥384 后才开始出现跨层/跨 token 复用。

4. **速度代价明确。** 流式 3.5–5.6 tok/s vs 基线 24.8 tok/s（约 4.4–7× 慢）。开销来自每层每 token 的 per-expert 磁盘读取 + 在 Python 侧重建小 `QuantizedSwitchLinear` + `gather`。这正是「用速度换内存」的预期取舍。

5. **与 hypura(ggml/Metal) 的闭环。** hypura 强制专家流式时因自定义 buffer 报 `is_host=true` 把 MoE 计算回退 CPU、~27× 劣化；MLX 路线 B 在统一内存下 per-expert 加载后 `gather_qmm` **仍在 GPU 上算、不回退 CPU**，所以同样「专家不驻留」的目标下，MLX 的劣化幅度（~4–7×）远小于 hypura 的 CPU 回退。

## 选型建议

- **若首要诉求是“腾出统一内存给别的程序”、能接受较低吞吐**：用路线 B、槽位取能接受的最大值。槽位 1024（RSS ~4GB、命中 68%、5.6 tok/s）是较平衡的一档；要把内存压到极限可用 64（RSS ~1.4GB）。
- **若首要诉求是吞吐**：全量驻留基线（17GB、24.8 tok/s）仍是最优，不要流式。
- **后续可提速方向**（未实现，留作 TODO）：① 把「重建子 linear + gather」从 Python 下沉、复用持久化子模块；② 用 `set_wired_limit` 配合更激进的 `clear_cache` 节流；③ 预取下一层专家与计算重叠；④ 按层独立 LRU（而非全局），使槽位预算更贴近单层 top-8 工作集。

## 复现命令

```bash
cd hypura/mlx_streaming && . .venv/bin/activate && cd ..
# 基线（全量驻留）
MODEL=/tmp/qwen3moe MAXTOK=64 python3 -m mlx_streaming.run_baseline
# 路线 A（lazy 对照）
MODEL=/tmp/qwen3moe MAXTOK=64 python3 -m mlx_streaming.run_mmap
# 路线 B 流式（首次会自动拆分专家到 EXPERT_DIR）
MODEL=/tmp/qwen3moe EXPERT_DIR=/tmp/mlx_qwen3_experts EXPERT_SLOTS=1024 MAXTOK=64 \
  python3 -m mlx_streaming.run_streaming
```
