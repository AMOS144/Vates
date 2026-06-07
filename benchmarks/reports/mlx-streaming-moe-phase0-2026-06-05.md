# MLX Streaming-MoE Phase 0 实测发现

> 日期：2026-06-05　分支：`feature/mlx-streaming-moe`
> 目标：验证「让 MoE 专家不全量驻留统一内存」在 MLX 上是否可行，并判定走路线 A 还是 B。

## 环境

| 项 | 值 |
|---|---|
| 机器 | Apple M5，统一内存 **32 GiB**（`memory_size`=34.36e9） |
| GPU 推荐工作集 | `max_recommended_working_set_size` ≈ 26.8 GB（`set_wired_limit` 上限参考） |
| Python | 3.14.5 |
| mlx | 0.31.2 |
| mlx-lm | 0.31.3 |
| Metal | 可用 |

### 环境探针关键结论（`mlx_streaming/env_probe.py`）
- 内存 API 全在 `mx.*`：`set_wired_limit / set_memory_limit / set_cache_limit / clear_cache / get_active_memory / get_peak_memory / get_cache_memory / reset_peak_memory`。
- `mlx_lm.load` **支持 `lazy: bool=False`**，但**不支持 `use_mmap`**。
- `mx.load` 签名为 `load(file, format=None, return_metadata=False, *, stream=None)` —— **不支持 `use_mmap`**（运行时实测 TypeError）。
- 即：文档/社区提到的 mmap 零拷贝路径在 stock MLX 0.31.2 上**不可用**。

## 核心探针：gather_qmm 的内存行为（`mlx_streaming/probe_gather_paging.py`）

用 mlx-lm 真实的 `QuantizedSwitchLinear`（Qwen3 MoE 同款）造一个堆叠专家权重
（E=256 专家，O=4096，I=2048，4-bit affine，约 **1.34 GB**），存盘后用不同方式
重载，只对 K=8 个激活专家做一次 `gather_qmm`，实测每个独立子进程的 RSS（`ru_maxrss`）。

| 模式 | 加载后 RSS (GB) | gather 后 RSS (GB) | MLX 活跃 (GB) | MLX 峰值 (GB) |
|---|---|---|---|---|
| evalall（强制全量物化） | 1.441 | 1.443 | 1.342 | 1.342 |
| lazy（`mx.load` 不 eval） | **0.098** | **1.443** | 1.342 | 1.342 |
| mmap（`use_mmap=True`） | — | — | — | — （API 不支持，TypeError） |
| **perexpert（只载选中 8 个专家文件）** | 0.098 | **0.142** | **0.084** | 0.084 |

### 解读
1. **lazy 在“加载”阶段确实惰性**（RSS 仅 0.098 GB），但一旦 `gather_qmm`，整堆叠
   张量被物化（RSS 升到 1.443 GB = 全量）。原因：`gather_qmm` 的输入是整个堆叠
   权重数组，求值它必须先把整张量从磁盘读入/物化。**纯 lazy 救不了 MoE 常驻。**
2. **mmap 零拷贝路径在 stock MLX 不存在**，所以「靠 OS 按需换页压低常驻」这条捷径不可用。
3. **per-expert 显式加载有效**：只从磁盘读选中的 8 个专家小文件，RSS 仅 0.142 GB、
   MLX 活跃 0.084 GB（≈ 全量的 1/16），且随激活专家数 K 缩放。

## 路线判定

**走路线 B（自定义流式 MoE：按专家拆文件 + 只加载选中专家 + LRU 驱逐）。**

- 路线 A（lazy + `set_wired_limit`/`use_mmap`）被实测否决：`gather_qmm` 物化整张量；
  `use_mmap` 在 stock MLX 不可用。
- 路线 B 被实测证明可把常驻压到正比于「活跃专家数」，正是「能装也不全量驻留」的目标。

这也与 ggml/hypura 的对比形成闭环：MLX 的关键优势不是「自动按需换页」（它没有），
而是**统一内存下 per-expert 加载后 `gather_qmm` 直接在 GPU 上算、不回退 CPU**。

## 当前阻塞

- 真实模型（`mlx-community/Qwen3-30B-A3B-4bit`，约 16GB）**无法下载**：
  - 直连 huggingface.co 连接被重置（被墙）；
  - 本机系统代理 `127.0.0.1:7897` 对 google 通、对 huggingface.co/github SSL 重置；
  - hf-mirror.com 首页通，但 `/resolve/` 请求 308 跳回被墙的 huggingface.co。
- 本机 GGUF（`Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf`）不可用：mlx-lm 不支持加载 GGUF 推理；
  `mx.load` 读 Q4_K_M 会全量反量化成 fp16（35B→约 40GB，超内存）。

因此「真实模型端到端」（Task 3/4 真实跑、Task 9/10）待网络或本地模型就绪后再做；
路线 B 的机制（专家后端 + 流式块 + 等价性）可在无模型下用合成数据单测先行实现。

## 路线 B 机制验证（已完成，无需下载，全部单测通过：9 passed）

| 组件 | 文件 | 验证内容 | 结果 |
|---|---|---|---|
| 内存度量 | `mem.py` / `test_mem.py` | RSS/active/peak 口径 | ✅ 2 测试 |
| LRU 专家后端 | `expert_store.py` / `test_expert_store.py` | 只取选中专家、LRU 命中率、容量驱逐 | ✅ 3 测试 |
| 流式前向 | `streaming_moe.py` / `test_streaming_equiv.py` | 只算选中专家，与 `SwitchGLU` 数值等价（**含量化 QuantizedSwitchLinear 路径**） | ✅ 2 测试 |
| 模型替换 | `streaming_moe.py::patch_model` / `test_patch_model.py` | 在**真实 Qwen3-MoE 架构**（微型合成权重）上替换 MoE 块，patch 前后 logits 等价 | ✅ 2 测试 |

结论：路线 B 的全部核心机制（按需取专家、LRU、只算激活专家、替换进真实 Qwen3-MoE
结构且数值不变）已在本机离线验证通过。唯一未做的是「真实 16GB 模型」上的内存/速度
实测与预算扫描（Task 10）—— 仅因模型下载被网络阻塞。

## 待办（一旦模型可得即可完成）

1. 离线把真实模型的 `switch_mlp.*` 堆叠张量按专家拆成 per-expert 小 safetensors（`split_experts.py`，Task 11）。
2. `patch_model` 接 `LruExpertStore`（per-expert 文件后端），跑 `generate`，实测 RSS/tok-s/命中率（Task 10）。
3. 扫描 LRU 槽数，画「常驻 ↓ / tok-s」前沿，对比全量驻留基线（Task 6 风格报告）。
