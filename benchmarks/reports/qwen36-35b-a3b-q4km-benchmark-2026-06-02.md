# Hypura 实测报告：Qwen3.6-35B-A3B Q4_K_M

**日期：** 2026-06-02  
**被测模型：** `Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`（`qwen35moe`，约 34.7B 总参数，256 路 MoE / token 激活 8 路，量化 Q4_K）  
**硬件：** Apple M5，32 GB 统一内存（Hypura 硬件画像缓存）  
**Hypura 版本：** 工作区 `flash-moe/hypura`（含 `--prefer-expert-streaming`）

---

## 1. 摘要

| 模式 | 权重放置（Hypura 规划） | 进程最大 RSS（`time -l`） | 生成吞吐（实测） |
|------|-------------------------|---------------------------|------------------|
| **默认**（稀疏 MoE mmap） | 约 **19.9 GB GPU**，0 NVMe | 约 **13.7 GiB** | 约 **35.3 tok/s**（118 token） |
| **`--prefer-expert-streaming`** | 约 **1.7 GB GPU** + **18.2 GB NVMe**（专家） | 约 **3.4 GiB** | 约 **1.3 tok/s**（128 token） |

在本机上，**默认模式**在吞吐上明显更优；**专家流式**显著降低进程 RSS，但 decode 极慢（专家从 NVMe 经池加载的开销占主导）。两种模式下 **`hypura estimate` 给出的 KV 规划一致**：热点 KV 仍在 **Gpu FP16**（约 320 MB @ 8192 token 热窗），未把 KV 放到 NVMe。

---

## 2. 测试命令

```bash
cd hypura
MODEL="/Users/amos/project/ai/llama.cpp/models/qwen36-35b-a3b/Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf"

# 放置与 KV 预估（不加载整模推理）
./target/release/hypura estimate "$MODEL"
./target/release/hypura estimate --prefer-expert-streaming "$MODEL"

# 实测吞吐 + 进程级内存（/usr/bin/time -l）
/usr/bin/time -l ./target/release/hypura bench --max-tokens 128 --context 2048 "$MODEL"
/usr/bin/time -l ./target/release/hypura bench --max-tokens 128 --context 2048 --prefer-expert-streaming "$MODEL"
```

- **Bench 参数：** `context=2048`，`max_tokens=128`，默认英文 prompt（与 `bench.rs` 内建一致）。  
- **未开** `--baseline`（避免双倍加载与 OOM 风险）。

---

## 3. 模型与放置（`hypura estimate`）

### 3.1 默认（稀疏 MoE mmap）

- **张量总大小：** 约 19.9 GB（GGUF）  
- **Placement：** GPU（Metal）约 **19.9 GB**（40 层），RAM / NVMe 规划为 **0**  
- **KV Cache：** 热窗 **8192 tokens，Gpu，FP16**，约 **320 MB**；温窗 0  

### 3.2 `--prefer-expert-streaming`

- **Placement：** GPU 约 **1.7 GB**，NVMe 约 **18.2 GB**（专家融合权重）  
- **KV Cache：** 与默认相同（**Gpu FP16 热窗**）  
- **预估 Disk I/O：** 约 **20.3 MB/token**（estimate 输出；反映专家流式读盘量级）

---

## 4. 内存说明（权重 vs 进程指标）

### 4.1 Hypura「规划字节数」

Bench 开头打印的 **Placement: X GPU | Y RAM | Z NVMe** 表示调度器把各类张量**标到哪个层级**；不等同于「进程 RSS = 全部权重常驻」。

- **默认：** 全部权重标在 **GPU**；配合 mmap，物理常驻页随访问变化，`time -l` 的 RSS 约为 **13.7 GiB**（低于 19.9 GB 文件体积，与按需页化 / 统计口径有关）。  
- **专家流式：** 专家标在 **NVMe**，运行时经池按需载入；**最大 RSS 约 3.4 GiB**，明显低于全量 mmap 路径。  
- **`peak memory footprint`（`time -l`）：** 默认约 **345 MB**、流式约 **20.5 GiB**——在 Apple 统一内存 + Metal 下，该字段与 RSS **常与用户直觉不一致**，**不宜单独当作「权重占用」**；建议与 **Placement 规划 + RSS** 对照阅读。

### 4.2 进程级原始数据（`maximum resident set size`）

| 模式 | 原始值（字节） | 约 GiB（÷1024³） |
|------|----------------|------------------|
| 默认 | 14764965888 | **13.75** |
| 专家流式 | 3690512384 | **3.44** |

### 4.3 结果 JSON（机器可读）

- 默认：`benchmarks/results/2026-06-02T08-29-29_Qwen_Qwen3.6-35B-A3B-Q4_K_M.json`  
- 流式：`benchmarks/results/2026-06-02T08-31-24_Qwen_Qwen3.6-35B-A3B-Q4_K_M.json`  

---

## 5. 吞吐实测（`hypura bench`）

| 模式 | Prompt 阶段 | 生成阶段 | 实际产出 token | 墙钟 |
|------|-------------|----------|----------------|------|
| 默认 | 约 **0.1 s**（约 **151 tok/s**） | **约 35.3 tok/s** | 118（可能早停） | **约 16.2 s** |
| `--prefer-expert-streaming` | 约 **1.0 s**（约 **7.6 tok/s**） | **约 1.3 tok/s** | 128 | **约 110.7 s** |

> 说明：默认路径在本机已能 **Metal 全层卸载**（`n_gpu_layers=41`），与「稀疏 mmap + 统一内存」组合，decode 较快。专家流式路径在本测中 **decode 约 700–800 ms/token 量级**（日志中可见），故 tok/s 很低。

---

## 6. 结论与建议

1. **若目标是「尽量省物理常驻、把大块专家放 NVMe」**：`--prefer-expert-streaming` 在本测中 **RSS 明显更低**，但 **生成速度大幅下降**，仅适合作为内存压力实验或极端省内存场景。  
2. **若目标是「尽可能快的 tok/s」**：在 **32 GB 统一内存 + 本模型体积** 下，**默认放置更合理**。  
3. **KV**：两种模式 estimate 均显示 **KV 热窗在 GPU FP16**；与「权重尽量 NVMe、KV 留统一内存」的诉求 **一致**（KV 未规划到 NVMe）。  
4. **复现**：请使用本报告 **§2** 命令；若更换机器或模型路径，数值会变化。

---

*本报告由自动化 bench / estimate 输出整理生成。*
