# MTP 上限归因 + 并发可行性 de-risk 结论

日期:2026-06-08
模型:`Qwen3-Next-80B-A3B`(2-bit 专家流式,`RESIDENT_POOL=1`,`models/qwen3_next_experts_2bit`)
机器:32GB 统一内存 macOS
背景:回答"I/O 解决后 MTP 应该 24,为什么没有再 ×2 的投机红利",并把"并发能否提速、能提多少"用实测钉死。

---

## 0. TL;DR

- **MTP 在本机有效**(2-bit、256 槽、K=2、batch):`spec 24.18 / baseline 7.78 = 3.11×`,`avg_accept_len=1.778`、`spec_hit_rate=0.986`、`disk_load_ratio=0.15`、峰值 11.1GB。
- **但 MTP 的红利来自摊薄 I/O,不是摊薄计算**。一旦工作集全驻(I/O→0,单路上界 ~28.5),投机顶到"按 token 计算"的墙,`spec(24) < baseline(28.5)`,不会再 ×2。
- 根因:**Qwen3-Next 75% 是 GDN 线性注意力**,verify-K 近线性、不摊薄;25% 注意力层已贡献(verify-2≈1.79× 而非 2.0)。这与 vLLM 实测同因(issue #35387)。
- **并发可提升聚合吞吐(实测 B=8 ~40 tok/s),但每流被算力平摊到 ~5 tok/s**;I/O 被批处理摊掉、命中率不塌、内存不爆(固定槽)。
- **落地优化(已合入,无需 kernel)**:MTP `sync` 去掉 lm_head 白算 + 批量推进。
- **判 NO-GO**:GDN/MoE 的 Metal fused kernel;以及 checkpoint 路径优化(实测零开销,无目标)。

---

## 1. 已落地优化:MTP `sync` 去 lm_head 白算 + 批量推进

`MTPDrafter.sync()` 原先逐 accepted token 调 `mtp_step()`(含 `lm_head` + argmax),但同步 MTP cache 根本不需要 logits。改为新增 `mtp_advance()`(只 embedding+fc+layer+norm、推进 cache、不算 lm_head),并一次性 batch 推进 accepted 序列。

- 单测:新增"sync 不得调用 lm_head""batch advance 与逐 token hidden 等价(<2e-3)";**全套 76 测试绿**。
- 真机分段计时(warm,256 槽,K=2,batch,MAXTOK=96,54 步):

| 段 | 秒 | 占比 |
|---|---:|---:|
| t_verify | 3.481 | **88%** |
| t_draft | 0.389 | 10% |
| t_sync | 0.059 | 1.5% |
| t_snap | 0.031 | — |
| t_commit | 0.006 | — |
| t_finalize | 0.002 | — |

**结论:sync 优化把编排开销归零;剩余 88% 是 GDN+MoE 的不可避免计算。**(冷盘首轮 `t_draft` 看似 3.9s 是首次 kernel 编译/盘预热被记在 draft 段的假象,热盘稳态仅 0.389s。)

---

## 2. 为什么"I/O 解决后没有再 ×2 的投机红利"

热盘实测把每步拆开(54 步、96 token):

```
每步 ≈ 73.5ms:verify 64.5 / draft 7.2 / sync 1.1 / 其它 0.7
每步推进 avg_accept_len = 1.778 → 每 token 41.3ms → 24.2 tok/s
```

verify 的 64.5ms 中,残余读盘仅 ~8.5ms(spec 命中 0.986,~11 次/步 × 0.75ms),其余 ~56ms 是纯计算。即便残余 I/O 清零也只到 ~27 tok/s,逼近全驻上界 28.5。

**投机的红利公式:** 在 GDN + 无 I/O 下 `verify(K) ≈ K×单 token`(线性),则

```
spec ≈ accept_len / verify(K) ≤ accept_len / (K × t_single) ≤ 1/t_single = 单路上界
```

accept_len ≤ K → 投机最多打平单路 I/O-free 上界,超不过。**换任何 drafter(独立小模型/更大 K)都改不了这个上界**,因为墙在 target 的 verify(GDN 不摊薄),与 drafter 无关。

### 为什么 A3B 的"batch K 计算红利"几乎不存在
- dense 共享权重占比极低(A3B 激活仅 3B)→ 没有大块可摊的 dense 权重;
- MoE 专家是 per-token matmul(K 个 token 各算各的 10 专家)→ 算术按 K 线性;
- 唯一次线性的是"专家权重的并集加载",而那已计入 I/O 红利(`disk_load_ratio` 0.15)。
→ **对 A3B,"batch 计算红利"与"摊薄 I/O 红利"是同一份钱;I/O 一旦没了就没了。**

### 25% 注意力层已经在贡献
按层加权 verify-2 成本 ≈ `0.25×1.0 + 0.75×2.0 = 1.75×`,与 vLLM 实测 `verify-2=1.79×` 吻合。即 25% 注意力把 MTP 从"纯 GDN 计算净亏(1.778/2.0=0.89×)"拉到"计算打平(≈1.0×)",净收益全靠 I/O 摊薄。无法在运行时把这 25% 单独"提速"。

---

## 3. 并发可行性(批处理 + 多路共享 LRU)

### 3.1 每 token 读盘随 B 下降(I/O 被摊薄)—— trace 驱动真实 LRU 仿真

| B | 跨领域 读盘/token | 跨领域命中 | 同领域 读盘/token | 同领域命中 |
|--:|--:|--:|--:|--:|
| 1 | 106.1 | 0.779 | 106.1 | 0.779 |
| 2 | 95.1 (0.90×) | 0.794 | 67.7 (0.64×) | 0.835 |
| 4 | 73.7 (0.69×) | 0.834 | 41.9 (0.40×) | 0.878 |
| 8 | 66.3 (0.62×) | 0.829 | 26.0 (0.24×) | 0.904 |

**多路共享 LRU 把每 token 读盘摊薄**(同领域 B=8 降到 0.24×);相关性越高摊得越狠。命中率随 B **上升**(共享热核更快焐热),不塌。

### 3.2 工作集 vs 固定槽(256)—— 命中率塌的临界 B
单流每层 distinct 专家均值 127、最大 225(<256)。组合工作集超 256 的临界:
- **跨领域**:B=2 就 48/48 层超(覆盖损失 15%,上界估计);
- **同领域**:B=2~3 安全(<3%),B=4 起明显(7.3%),B=5 全超。

注:固定槽 → 常驻内存恒定(~11GB,**不随 B 增长**),并发代价是命中率/IO 而非 OOM;且这些 miss 又被 §3.1 的批处理摊薄,实际命中率不降反升。

### 3.3 batch 解码真实计时(每步耗时 → 每流/聚合 tok/s)
512 槽全驻 warm,随机 token,真计时:

| B | 步耗时 ms | 每流 tok/s | 聚合 tok/s | 聚合相对 B=1 |
|--:|--:|--:|--:|--:|
| 1 | 81 | 12.3 | 12.3 | 1.00× |
| 4 | 146 | 6.9 | 27.5 | 2.23× |
| **8** | 197 | **5.0** | **40** | **3.25×** |
| 16 | 331 | 3.0 | 48 | — |
| 24 | 434 | 2.3 | 55 | — |

**聚合亚线性、B=8 是甜点**(每流 5、聚合 40);8 之后边际骤降。**做不到"每流 24"**——M 系列对本模型偏计算受限,批处理主要是把 ~算力上界在 B 流间平摊,不是 ×B。

### 3.4 钉热专家(`store.pin`)治发散并发 —— 跨领域 B=8

| pin/层 | 命中率 | 读盘/token | 额外常驻专家 |
|--:|--:|--:|--:|
| 0 | 0.829 | 66 | 0 |
| 64 | 0.904 | 37 | 3072(~+3GB) |
| 128 | 0.939 | 24 | 6144(~+6GB) |

发散并发命中率 0.83→0.94、读盘减半,代价是常驻内存。`pin 64` 是甜点。

### 3.5 连续批处理 vs 锁步(用 §3.3 实测 step_ms(B) 做调度仿真,BMAX=8)

| 负载 | 锁步 tok/s | 连续 tok/s | 连续/锁步 |
|---|--:|--:|--:|
| 等长 | 40.6 | 40.6 | 1.00× |
| 均匀 | 24.2 | 40.3 | 1.67× |
| 偏斜(多短少长) | 12.1 | 40.1 | **3.32×** |

**连续批处理在请求长短不一时,把持续吞吐稳在聚合上界(~40),锁步会塌到 12。** 越长短不齐收益越大(最高 ~4×)。

---

## 4. checkpoint 路径零开销(判定"无可榨")

同一段 K=2 前向,开/不开 spec checkpoint 捕获对比(warm,2 次复测取干净值):

```
普通前向 59.13ms / 带 checkpoint 59.36ms / 额外开销 0.23ms(0.4% of verify)
```

**verify 的 88% 全是 GDN+MoE 计算,checkpoint 捕获基本免费**(小 K 下逐 token step-ops 甚至不比 chunked 慢——chunked 还要 padding 到 64)。**verify 路径已无 host 同步肥肉**;再快只能降纯计算 = Metal fused kernel(见 §6 NO-GO)。

---

## 5. 社区解法调研(GDN/SSM 上让投机生效)

| 来源 | 解法 | 可移植到 MLX/Metal? |
|---|---|---|
| vLLM #35442(merged) | `num_accepted_tokens` 非阻塞拷贝(消 host 气泡) | 思路同我们 sync 优化,已对齐 |
| vLLM #40172(merged 05-21) | fused Triton kernel 把 Mamba 状态后处理搬上 GPU(连 prefix caching 也修) | CUDA → Metal 要重写 |
| vLLM #29488 / #35777 | `selective_state_update` 加 spec / 融合 gating+delta-rule | CUDA |
| SGLang #18808 SpecV2 | Mamba 专用 verify 状态更新 + `extra_buffer` 调度(**比 V1 快 28%**) | CUDA |
| SGLang #22128 | piecewise CUDA graph + MTP,接受率 **3.46**(Qwen3.5-35B-A3B) | CUDA |
| Mamba Drafters(2506.01206) | Mamba 当 drafter:tree drafting 只需拷 state + batch-wise state cache | 算法可借鉴 |
| SpecMamba(2509.19873,FPGA) | 混合回溯 + FIFO tree 验证 + SSM 串行/线性并行数据流 | 硬件协同 |
| Ring-linear / Flood | 首个支持 tree mask 的线性注意力 kernel(tree 投机) | CUDA |

- vLLM #35387(76% 回归)是 **host 同步 bug**,已被 #35442 + #40172 修复(issue 未正式关闭,陈旧)。**常态下 MTP 在 Qwen3-Next 上 work,官方 1.5–1.8×。**
- **结论:社区已解决,但核心是 CUDA/Triton/FPGA 的 SSM 状态 kernel + 调度。算法骨架(快照/重放/direct-commit)我们已有;缺的是 Metal kernel。** 而 M 系列对本模型偏计算受限,即便移植 kernel 也大概率拿不到 GPU 上的倍数。

---

## 6. 决策

- **GO(已做/可做,无需 kernel)**:
  - MTP `sync` 优化(已合,76 测试绿)。
  - 单用户:吃现有 24 tok/s(MTP I/O 红利满值)。
  - 多用户:batch(B=8 ~40)+ 连续批处理(长短不一 1.5–4×)+ `pin 64`(发散命中率 0.83→0.94)。
- **NO-GO**:
  - GDN/MoE 的 Metal fused kernel(与既往 fused MoE 同级、高风险,且本机计算受限收益有限)。
  - checkpoint 路径优化(实测零开销,无目标)。
  - 换 Mamba/GDN 线性混合模型(Qwen3.6-35B-A3B、Nemotron 3 Super、Qwen3.5)——同墙。

## 7. 模型选型(若要"投机真有效")

判定规则:**能投机 ⟺ 有逐 token 可裁剪 KV(全/GQA/MLA/SWA/稀疏);线性递归(GDN/Mamba)破坏之。**

- 原生 MTP + 非线性 + 更强:**DeepSeek-V3.2 / Kimi-K2 / GLM-5**(MLA+稀疏),投机有效(DeepSeek 实测 1.8×),但 671B–1T、激活 32–37B → 32GB 流式极难。
- 非线性 + 激活小(流式友好)+ 投机友好:**GPT-OSS-120B**(GQA+滑动窗口,激活仅 5.1B)——"投机有效"和"能流式"两头都占,无原生 MTP 用 draft 模型即可。
- 避免:Qwen3-Next/Qwen3.6(GDN)、Nemotron 3 Super(Mamba 主体)、Qwen3.5(GDN)、Ring-linear(线性)。
