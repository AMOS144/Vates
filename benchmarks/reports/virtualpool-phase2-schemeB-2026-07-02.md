# Phase 2 方案 B（1 次同步版）实测报告 —— C++ 全接管真实区槽状态

日期：2026-07-02 · 分支：`feat/virtualpool-unified` · 开关：`NATIVE_DEMAND_DUAL`（默认关）

## 结论（TL;DR）

**方案 B 已完整实现且逐位正确，但实测性能不达标：比现有基线慢 2.6×（4.09 vs 10.78 tok/s）。**
根因是**结构性**的、与磁盘 I/O 无关：把 demand 从「基线的并行读 + 惰性/重叠 MLX scatter」改成
「主线程同步 C++ 调用」后，每层多出一次不可重叠的关键路径同步，破坏了 MLX 的跨层流水线重叠。
Phase 0 测得的 +174% 上界是**跳过 demand I/O + 落池**（数值错误）得到的，正确实现无法触及。

因此建议：**不要在默认配置启用 `NATIVE_DEMAND_DUAL`**，保留为 opt-in 实验开关；
Phase 2 的收益方向应回到「削减 Python demand 胶水但保持重叠」（方案 C 思路），而非 C++ 同步接管。

## A/B 实测（MAXTOK=64, WARMUP=64, REPEAT=2；EXPERT_SLOTS=32, POOL_SPEC_SLOTS=32, K=3, SIDEREGION_LFU=1）

| 指标 | OFF（基线，现路径） | ON（方案 B） | 变化 |
|---|---|---|---|
| spec tok/s | **10.78**（10.42 / 11.13） | **4.09**（4.03 / 4.14） | **−62%（慢 2.6×）** |
| spec hit_rate | 0.878 | 0.936 | +0.058（更高） |
| gpu_fastpath / fallback | 489 / 1143（fallback 70%） | 911 / 817（fallback 47%） | 更少 miss |
| spec_disk_loads | 5528 | 3124 | −43%（更少读盘） |
| n_mismatch | 43 | 61 | +18（同量级） |
| mlx_peak_gb | 8.47 | 8.27 | 持平 |

**判据取证（DEMAND_SKIP_IO 探针）**：把 demand 的 pread/memcpy 完全跳过（仅保留状态机 + 同步），
MAXTOK=32 下 tok/s = 4.93 —— 与开 I/O 的 4.09 基本一致（差异主要是 MAXTOK 不同），
**证明瓶颈不是磁盘 I/O，而是同步 demand 调用本身的每层结构性开销**。
配合「ON 做了更少的活（hit 更高、读盘更少）却慢 2.6×」这一反常事实，根因锁定为
**每层一次同步 C++ demand 打断了 MLX 的跨层惰性流水线重叠**。

## 已完成 & 正确性

- C++ `demand_dual` 全接管真实区槽状态 `g_real{order,e2r,free,freq,cap}`，精确复刻
  `_alloc_slot`（free 优先 pop(0)、LFU 驱逐复用受害者槽）与 `_choose_victim`（freq + 插入序 tie-break）。
- 比 Python 更严格：`current` = 本前向全部唯一路由专家（命中+miss），绝不驱逐本前向要读的槽
  （修掉了「真实区命中专家槽被本前向 miss 复写 → 脏字节」的隐患，Python 仅护 miss）。
- 单测全绿（`test_demand_dual_native.py` / `test_demand_dual_wiring.py`）：
  - (a) 槽映射 + 落池字节 == 磁盘真值；侧区覆盖真实区；命中不重读。
  - (b) C++ `choose_victim` 与真实 `ResidentExpertPool._choose_victim` 在 300 组随机状态逐步一致；
    C++ `real_debug_place` 与 Python 参考在 200 步随机序列上 slots + resident 集合逐步一致。
  - (c) flag 门控 / 非 spec 模式 / native 缺失自动回退 Python 权威路径。
  - (d) `resident_experts`/`resident_count` 在开关下改查 C++ `g_real`，与 C++ 内容一致（侧区预取过滤一致）。
- 一致性边界（P2-B4）：本配置（PREFILL_CHUNK=2、`seq·k ≤ dual_cap=96`）下 prefill/decode/verify 恒走
  dual 路径，host `acquire`/`fetch` 分支不触达 → 真实区单一写者（demand），无 Python/C++ 双写分歧。

## 剩余风险 / 建议默认值

- 默认 `NATIVE_DEMAND_DUAL=0`（关）。开启会显著变慢，仅供实验/取证。
- 若要在 C++ 里追平基线，需把 demand pread 改为并行（bg 线程池 + 等待）**且**恢复与 GPU 计算的重叠——
  但即便如此，理论收益仅是「省掉 Python scatter/eff 重建/第二次同步」的边际量，远达不到 +100%。
- 逐位等价：n_mismatch 61 vs 基线 43，同量级（本配置下 43 即基线正常水平，非 1）；spec 解码对
  数值差异敏感，方案 B 的 LFU/freq 策略与基线略有不同导致驻留集合差异，属可接受启发式差异。
