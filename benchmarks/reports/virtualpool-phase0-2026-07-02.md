# VirtualPool 统一抽象 —— Phase 0 go/no-go 实测报告

日期：2026-07-02
分支：`feat/virtualpool-unified`
关联 spec：`docs/superpowers/specs/2026-07-02-qwen-virtualpool-unified-design.md`

## 目的

在大改（把 demand 落池搬进惰性 Primitive、消除 per-layer 主线程同步）之前，
用最低成本证伪/证实前提：**「消除每层 GPU→CPU 同步真的能提速吗？」**

## 方法

同一配置（K=3、EXPERT_SLOTS=32、ZEROCOPY_DUAL_SOURCE、SIDEREGION_LFU、POOL_SPEC_SLOTS=32、
MAXTOK=64、WARMUP_TOK=64、REPEAT=2）下跑 `run_mtp_spec`，用 env 门控的 throwaway 探针
在 `acquire_gpu_dual` 里做两个近似「零 per-layer 同步」的短路：

- **P1 `PROBE_ALL_HIT_LAZY`**：`local=mx.take(eff,inds)` 后立即返回，**跳过** `int(mx.sum)`
  同步 + demand 回退（读盘/落池/第二次 `.tolist`）。整前向 48 层惰性搭图、只在前向末尾 eval 一次。
- **P2 `PROBE_NO_DEMAND`**：**保留** 每层 `int(mx.sum)` 同步 barrier，但回退层跳过 demand
  读盘+落池，miss 返回脏 local。

两个探针输出数值都会错（缺失专家读脏字节），**仅用于测速**，量「同步/落池成本为 0」的上界。

## 结果

| 指标 | baseline | P2: 留同步/去 demand | P1: 去同步+去 demand |
|---|---:|---:|---:|
| spec_tok_per_s | **11.07** | **30.38** | **53.92** |
| runs | [11.0, 11.14] | [31.82, 28.93] | [52.24, 55.6] |
| gpu_fastpath / fallback | 493 / 1091 | 48 / 1488 | 1392 / 0 |
| spec_disk_loads | 5493 | 0 | 0 |
| avg_accept_len | 2.286 | 2.37 | 2.667 |
| steps（tokens≈64 固定） | 28 | 27 | 24 |
| t_verify_s（48 层 MoE 前向总时） | 5.454 | 1.956 | 0.906 |
| n_mismatch（探针数值本就错） | 28 | 58 | 40 |
| mlx_peak_gb | 8.4 | — | 8.28 |

**相对基线**：P2 = 2.74x（+174%）；P1 = 4.87x（+387%）。
**每步 verify 前向时间**：0.195 → 0.072 → 0.038 s/step（accept_len 上浮是二阶扰动，
tokens 固定 64，tok/s 主要反映 wall）。

## 成本拆解

- baseline → P2（11→30，t_verify 5.45→1.96s）：去掉 **demand 回退整块**（第二次 `.tolist`
  + pread + 落池 scatter + eff 重建），即使**每层 `int(mx.sum)` 同步仍在**，也已提速 2.7x。
- P2 → P1（30→54，t_verify 1.96→0.91s）：再去掉 **每层 `int(mx.sum)` 同步 barrier**
  （快/慢路径都有的那次同步），又提速 1.8x。

两块成本都很大。

## 与既有「+15% 天花板」证据的调和（关键）

早前证据看似矛盾（cap64 只 +15%、Tier1 无可测提速），实为**每次只动了边角**：
- **cap64**：只把回退**频率** 68%→53%，但**每层仍付 `int(mx.sum)` barrier**、剩余回退层仍走
  主线程落池 → 拿不到 barrier/串行化的收益，故只 +15%。
- **Tier1**：只把回退 `.tolist` 从 ~40 元素缩到 ~3，但 barrier、Python `acquire`、落池 scatter、
  eff 重建**全保留** → 只削了极小一片，故无可测提速。
- **探针**：把**整个机制**（barrier + acquire + 落池）拿掉 → 才看到 2.7x~4.9x。

结论：真正的瓶颈是 **per-layer 同步 barrier + 主线程落池调度打断 GPU 流水线**，
不是 GPU 计算地板、也不是物理 I/O（用户实测暖≈冷 → 物理读已被预取盖住）。

## go/no-go 判定：**GO（强）**

- 上界相对基线 +387%（P1）、去 demand 保 barrier 仍 +174%（P2），**远超 no-go 线（+5%）
  与 cap64 提示（+15%）**。前提「消除 per-layer 同步能提速」被决定性证实。
- **诚实上界修正**：两个探针都把 demand I/O 完全去掉（disk_loads=0），是**乐观上界**。
  Phase 2 仍须把正确字节读进池（I/O 不能省），只能把「读+落池」搬到 eval 线程、并去掉主线程
  barrier。结合「物理 I/O 已被预取盖住（暖≈冷）」→ demand 成本主要是**主线程串行化**（可搬走），
  故 Phase 2 现实收益应能吃下 11→54 这段的**大部分**（保守到 P2 的 30 即 +174%）。

## 下一步（待用户拍板 —— 硬问题 3）

Phase 2（真消同步）在「槽记账归属」上有根本抉择，需用户选定后才落地：
- **方案 C（推荐）**：主线程预留槽 + C++ demand primitive 只填字节，保留一次极小 `.tolist`；
  风险中、可增量验证。
- **方案 B**：C++ 完全接管真实区槽状态（`_slot_of/_free/freq`），零主线程同步但重写状态机；
  收益最高、风险最高。

在用户拍板前不启动 Phase 2 大改。
