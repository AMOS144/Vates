# 专家池管理统一迁移到 C++（单一权威）设计

日期：2026-07-04
分支：`perf/async-demand-offload`
关联：
- `2026-07-02-qwen-virtualpool-unified-design.md`（VirtualPool 收口 / Phase 0-2 前序设计）
- `benchmarks/reports/virtualpool-phase0-2026-07-02.md`（瓶颈证据）
- `benchmarks/reports/virtualpool-phase2-spike-2026-07-02.md`（同前向零同步证伪）
- `benchmarks/reports/virtualpool-phase2-schemeB-async-2026-07-02.md`（现有异步方案 B 实测 + 残留 n_mismatch）

---

## 0. 一句话

把现在「Python 管真实区槽 + C++ 管侧区槽 + demand 落池回主线程」的割裂结构，
**统一收口到一套 C++ 池管理器（单一权威）**：真实区 + 侧区 + demand + prefetch 的槽分配、
LFU 驱逐、字节落池全部在 C++ 里完成，Python 只调一次 `acquire(layer, inds) → local`、
不再持有任何槽状态。**目标是把 ~70% 回退层的主线程 demand 工作移出热路径**（decode 提速），
**且迁移后严格保证正确性**（用「容量不变性」验收，见 §3）。

---

## 1. 背景与动机（来自实测证据）

当前 K=3 MTP verify、dual（真实区 cap32 + 侧区 LFU），暖缓存约 **11 tok/s**。探针已定位瓶颈：

- **同步 barrier 无法消除**：`virtualpool-phase2-spike` 用 200 次确定性探针证明——同前向内不
  `inds.eval()` 就拿不到 GPU 算好的路由（读到全 0），回调写入对同前向下游不可见。因此
  **每层 1 次 `inds.eval()` 同步不可避免**（B、C 方案都逃不掉）。
- **真正的大头是「demand 主线程 WORK」**：Phase 0 探针实测——去掉 demand 的主线程工作
  （第二次 `.tolist` + Python 落池 scatter + eff 重建），**即使保留每层同步**，也从 11 → 30 tok/s（+174%）。
  这块主线程停顿打断了 MLX 跨层流水线，是慢的主因。
- **I/O 不是瓶颈**：暖 page cache ≈ 冷（11.365 vs 11.31）→ 物理读盘已被预取盖住。
- **现有异步方案 B（`NATIVE_DEMAND_DUAL`，默认关）已能把 demand 搬到 worker 线程**：
  实测 cap32 +5.9%、cap64 +38.7%，字节校验 0 错。**唯一卡点**：`n_mismatch` 比基线高
  （恒定 61，cap 无关），破坏 bit-exact，故一直默认关。

**结论**：加速的正道是「把 demand 工作移出主线程」；这件事现有方案 B 已做出来大半，
但它把真实区槽权威搬到 C++ 后引入了一处确定性错槽（见 §4），必须先根治正确性，才能默认启用。

---

## 2. 目标与非目标

**目标**
1. **单一权威**：真实区 + 侧区 + demand + prefetch 的槽分配 / LFU 驱逐 / 字节落池全部收进
   一套 C++ 池管理器；Python 侧删除该路径的 `_slot_of / _free / _freq / _slot_table` 死影子。
2. **主线程零 demand 工作**：主线程每层只做那 1 次不可避免的 `inds.eval()`，之后的合并 /
   分配 / pread / 落池全在 C++（worker 线程并行 pread + 惰性 gather）。
3. **严格正确**：迁移后通过「容量不变性 + 字节真值」验收（§3），把现有方案 B 的确定性错槽根治。
4. **decode 提速**：目标吃下 Phase 0 上界（11→30 那档）的大部分；至少不低于现有异步方案 B
   实测（cap 充裕时净收益）。

**非目标**
- 不改预测 / 预取的**候选选择**逻辑、不改侧区 LFU 的**淘汰意图**、不改 cap/spec/预取 budget 的默认配置。
- 不追求消除那 1 次 per-layer `inds.eval()` 同步（已证不可行）。
- 不做与本目标无关的重构。
- 不改 prefill（大 seq、唯一专家可能超 cap）的既有 host/fetch 回退路径——它天然需要一次同步、
  不是热路径瓶颈，统一池对它只做透明转发。

---

## 3. 正确性的定义与验收口径（本设计的地基）

**核心认知：专家池是一个缓存。一个实现正确的流式缓存，对「装了多少 / 淘汰了谁」是输出中性的**
——因为不管某专家是常驻命中还是 miss 后从盘读回，最终喂给 matmul 的都是同一份正确字节。
因此正确性**不需要**「全常驻」（那与 MoE+SSD 流式设计矛盾），而应表述为两条可运行的不变量：

**不变量 A —— 容量不变性（Capacity Invariance）**
> 同一 prompt、greedy 采样，在 cap ∈ {实测最大并集, 32, 64, …} 下跑出的 token 序列**必须完全一致**。
> 正确的流式缓存对容量不变；若换 cap 输出就变，说明存在错槽（某 `local[i]` 指向了不装它字节的槽）。

**不变量 B —— 字节真值（STG_VERIFY 0 错）**
> 每个占用槽装的字节 == 其属主专家在磁盘上的真值。逐槽校验，全程 0 bad。

A + B 共同证明「无损」，无需全常驻。

**关于「浮点噪声地板」的诚实说明**：现有基线在 cap32=45、cap64=30 的 `n_mismatch`（相对 greedy 参照）
**本身就不相等**，说明基线要么也有小的错槽残留、要么混入了 GPU 浮点归约顺序的非确定性噪声。
因此 Phase 0 必须先测出「纯 FP 噪声地板」（基线多种子 / 背靠背自比），
验收标准是**把 C++ 路径高于该地板的确定性 delta 压到 0**，而非追求与某个也不完美的旧实现逐位相同。

---

## 4. 待根治的真正 bug：与容量无关的确定性错槽

**关键数字**（`models/qwen3_next_80b_4bit/config.json` + 实测）：
- `num_experts = 512`，`num_experts_per_tok(top_k) = 10`
- MTP verify seq = K+1 = 4 → **理论最大并集 = 4×10 = 40**（零重叠上界）
- **实测并集 ≈ 19**（相邻 token 路由高度重叠）→ **cap=32 > 19，不溢出**

**推论**：既然 cap=32 下不溢出，`demand_core_locked` 里 `local[i] = … : 0` 那条 slot-0 兜底
（`native_prefetch.cpp:790-793`）**根本不会触发**。所以 `n_mismatch=61` **不是溢出造成的**。

在「无溢出」前提下，mismatch 只可能来自**某个 `local[i]` 解析到了不装它字节的槽**。最可疑的方向：
- **侧区双缓冲 gen 新鲜度**：C++ 合并（side ∪ real → local）把某专家解析到一个侧区槽，
  但该槽在当前 read-gen 尚未被填成该专家的字节（读到上一代 / 其它专家）。这是确定性错槽，
  与容量、LFU 都无关，恰好解释「恒定 61、cap 无关」。
- 次要嫌疑：side / real 优先级或 `e2r` 合并顺序与基线 `acquire_gpu_dual` 的 `eff[keys]=vals` 语义有差。

**这处错槽的 root-cause + 修复，是本设计「确保正确」的核心工作**（走 systematic-debugging，
用不变量 A/B 定位，不靠猜）。注意：`EXPERT_SLOTS` 应设为 **≥ 实测最大并集 + 余量**
（Phase 0 用 `union_prof` 测最大值；大概率 cap=32 已够，无需 40+），
这是防「偶发并集尖峰溢出」的护栏，而**不是**修 61 的手段。

---

## 5. 架构

### 5.1 组件

**C++ `PoolManager`（新的单一权威，扩展现有 `g_real`/`RealLayer` + 侧区 `SideLayer`）**
每层一个分配器，独占以下状态（都在 `g_real_mutex` per-layer 锁下）：
- 槽表 `e2r`（expert → slot），真实区 + 侧区**统一寻址空间**（物理行 = cap + spec_gens×spec_slots）
- free list、LFU 频次 `freq` + decay、驱逐（`choose_victim` 复刻 freq 最小 + 最早插入 tie-break）
- demand：算 local → 分配 miss 槽 → 收集 `(expert, slot)` placements → 锁外派 `BgReader` worker 并行 pread 直写池段
- prefetch：侧区候选的 reserve + 落字节（并入同一分配器，消除「prefetch 与 demand 各自分配槽」的并发歧义）
- 统一的 canonical LFU（每层对全部唯一访问专家 +1），取代 Python 回退层「省略命中 bump」的偷懒版

**Python 侧（`ResidentExpertPool` / `VirtualPool` / `block.py`）**
- `block.py` 计算段只调一次 `vpool.acquire(layer, inds, num_experts) → (pool_arrays, local)`，零分支。
- `acquire` 内部：`inds.eval()`（唯一同步）→ 调 C++ `PoolManager`（统一 demand+侧区+prefetch 落槽）→ 返回 local。
- 删除该路径的 Python 槽状态（`_slot_of/_free/_freq/_slot_table`）；`resident_experts` 等只读查询改查 C++。

### 5.2 数据流（decode / verify 热路径）

```
gate → inds(lazy) ─┐
                   ├─ vpool.acquire(layer, inds):
                   │     inds.eval()                       # 每层唯一同步（不可避免）
                   │     C++ PoolManager.acquire_locked:
                   │       side∪real 合并算 local
                   │       miss → 分配槽(canonical LFU 驱逐) → placements
                   │     锁外: BgReader 并行 pread 直写池段行 → wait(ticket)  # 主线程不参与
                   │     return local(int32)
                   └─→ y = matmul(pool_arrays[local], x)    # 惰性 gather，与后续层重叠
```

主线程每层只付 1 次 `inds.eval()`；「合并 / 分配 / pread / 落池」全在 C++（worker 线程），
不再回主线程 `.tolist` + Python scatter + eff 重建 → 不打断 MLX 跨层流水线。

### 5.3 耦合审计结论：`block.py` 是耦合枢纽

只读审计（2026-07-04）发现，`block.py` 计算段有 **5 条 acquire 分支**（dual / 非 dual GPU-remap /
host / 超 cap fetch / stream_blob），并**直接窥探 10+ 处他人私有内部**：
`store._staging`、`store._resident`、`_stg_mgr.promote`、`_stg_mgr.last_ready`、`stg.src._segs`、
`rp._pools[tgt]`、`store._bg`、`self._blob`、`_slot_of` 等。这些「跨抽象边界摸内部」是耦合的真正来源。

**降耦合总纲**：给 `block.py` 一个干净的单一接口，把内部窥探全收进 C++ `PoolManager` 背后。
目标塌缩形态：

```python
# 目标：block.py 计算段（去掉所有内部窥探与分支）
pool_arrays, local, n_experts = pm.acquire(layer, inds, num_experts, seq_len)   # 命中/miss/侧区/promote 全内聚
y = self._sub.forward(pool_arrays, n_experts, x, local)
# 预取（搭车式，保留 Python 预测 gate 搭图，只把候选交给 C++）：
pm.prefetch(target_layer, pred_inds)                                            # resident 过滤/落槽在 C++ 内
```

### 5.4 `PoolManager` 对外接口（Python 侧零内部窥探）

| 方法 | 语义 | 替代掉的现状 |
|------|------|-------------|
| `acquire(layer, inds, num_experts, seq_len) -> (pool, local, n_experts)` | 呈现「所有专家都在」视角：side∪real 合并 + demand 补齐 + LFU 记账，内部完成 | `_vpool.acquire` / `acquire_gpu(_dual)` / `acquire_host` / `store.acquire` / `fetch` 五条分支 |
| `prefetch(target_layer, pred_inds) -> dummy` | 按候选预取到侧区：resident 过滤 + reserve + 并行 pread 落槽，全在 C++ | `stg.submit` + Python `resident_experts` 快照传参 + `route_used_subset` |
| `begin_forward(layer)` | 前向边界 / gen 推进 | `_vpool.begin_forward`（内聚进 PoolManager） |
| `resident_snapshot(layer)`（只读诊断） | 返回常驻集合，仅供 trace/校验 | `resident_experts` 作为**权威读**的用法取消 |

**关键原则**：`prefetch` / `promote` **不再从 Python 传 resident list**——C++ 内部直接读 `g_real` 过滤。
Python 的 `_slot_of / _free / _freq / _slot_table` 在该路径删除（不再是权威、也不做死影子）。

---

## 6. 分阶段（TDD，每阶段独立可验证、开关可回退）

**Phase 0 —— 前提实测（写实现前必做）**
- `union_prof` 测各层单前向**最大**并集（坐实 cap 下限；预期 ~19，偶发可能更高）。
- 测「FP 噪声地板」：基线固定 prompt greedy 背靠背 / 多种子自比，量 `n_mismatch` 的非确定性基线。
- 在 32GB 上实测能安全跑的 cap（cap64 ≈ 9.4GB 常驻，建议工作集 26.8GB，偏紧要坐实）。
- 判据：确认 cap 下限（大概率 ≤32）、噪声地板量级，作为后续验收基准。

**Phase 1 —— 正确性护栏（不改行为）**
- 搭「容量不变性」回归测试：固定 prompt greedy，断言 cap 扫描下 token 序列一致。
- STG_VERIFY 提为一等公民测试（单测 + 端到端可开）。
- 此阶段只建测试与量具，行为 bit-exact，性能中性。

**Phase 2 —— 根治错槽，让现有方案 B 达到 oracle 干净**
- 用不变量 A/B + systematic-debugging root-cause `n_mismatch=61`（优先查侧区 gen 新鲜度）。
- 修复后：`NATIVE_DEMAND_DUAL` 在 cap ≥ 实测并集下**通过容量不变性 + STG_VERIFY 0 错**。
- 交付物：现有异步方案 B 变成 **exact** → 直接兑现 cap 充裕时的净收益（实测 +38.7%@cap64 / +5.9%@cap32）。

**Phase 3 —— 统一权威 + 降耦合**（按子项收益排序，每子项独立可验证）
把侧区 reserve/落字节 + prefetch 落槽收进同一套 C++ `PoolManager` 分配器（统一锁与 LFU），
并按审计结论逐项消除 `block.py` 的内部窥探。全程保持 Phase 1 的容量不变性 + STG_VERIFY 绿。

- **P3-a promote 落真实区迁 C++**：Python `_place_expert` + `_slice` 逐专家 scatter
  （`native_staging.py:143-177`）→ C++ `promote_to_real(route_inds)` 直写 `g_real`，与侧区预取对称。
- **P3-b `route_used_subset` 并入 C++ promote**：消掉 GPU 路径每层 ≤16 int 的 `.tolist()`
  （`native_staging.py:46`），改为 C++ 内用 GPU membership 现算过滤假阳性。
- **P3-c 侧区∪真实区 overlay 迁 C++**：`acquire_gpu_dual` 的 Python `eff[keys]=vals` 搭图
  （`resident_pool.py:576-578`）→ C++ 单次 gather 合并（`sideregion_kv` 已是 device 数组，基础在）。
- **P3-d 统一并行读（大头）**：`blob_loader` 的 Python `ThreadPoolExecutor(8)` 与 C++ `BgReader`
  是重复的两套并行 pread。→ **BgReader 作为唯一 IO 线程池**，Python demand/host 只提交 ticket + wait。
  风险稍高（涉及物化 numpy→mx.array 的归属），排在本 Phase 末尾、单独开关灰度。
- **P3-e 删 resident 快照跨界**：submit/promote 不再从 Python 传 `resident_experts` list，C++ 内读 `g_real`。
- **P3-f Python 读 C++ 状态降级为只读诊断**：`real_region_contents/_count/real_freq/sideregion_contents/`
  `demand_last_stats/last_ready` 统一为只读 API；删 Python `_slot_of/_free/_freq/_slot_table` 死影子。

> 留在 Python（不迁）：gate + argpartition（已在 GPU、无 host 往返）、`_native_fused_prefetch` 的
> 预测 gate 搭图（要访问下层 gate/norm 权重，迁 C++ 收益有限）、`PersistentSubGLU` 计算、
> MTP `generate.py` 块级调度、配置接线、诊断（route_trace/miss_attrib/STG_VERIFY）。

**Phase 4 —— 收尾**
- oracle 干净后，把统一 C++ 路径设为默认（评估 `NATIVE_DEMAND_DUAL` 默认值翻转）。
- 退役 Python 槽路径（保留 config 开关兜底一段时间）。

---

## 7. 内存约束（32GB）

- 正确性要求 cap ≥ 实测最大并集（Phase 0 测，预期 ~19-25）→ **cap=32 大概率已满足，无需 40/64**。
- cap 在并集之上继续加，只提命中率（速度），不影响正确性；受 32GB 约束（cap64 ≈ 9.4GB 偏紧）。
- 结论：本设计**不依赖**大 cap 来保正确；cap 是可选的速度旋钮，Phase 0 给出安全上限。

---

## 8. 风险与回退

| 风险 | 缓解 |
|------|------|
| 并发：demand 与 prefetch 同改一套 C++ 分配器 | per-layer 单锁 `g_real_mutex`；prefetch/demand 统一走同一分配器（消除抢槽歧义） |
| LFU 驱逐确定性漂移 | canonical LFU + 确定性 tie-break；容量不变性测试兜底 |
| 侧区 gen 新鲜度错槽（即 61 的疑犯） | Phase 2 专门 root-cause；STG_VERIFY 逐槽字节校验 |
| 迁移引入回归 | 全程 config 开关门控（默认 off 直到 oracle 干净），可秒回退 Python 路径 |
| C++ 状态与 Python 只读路径不一致 | Python 只读查询统一改查 C++（单一权威），不维护双份 |
| P3-d 统一并行读：物化（numpy→mx.array）归属与 GIL | BgReader 只统一 pread；物化保留 lazy 切片；单独开关灰度、排 Phase 3 末尾 |

---

## 9. 成功标准（验收）

1. **正确性**：统一 C++ 路径通过**容量不变性**（cap 扫描 token 一致）+ **STG_VERIFY 0 错**；
   相对 greedy 参照的 `n_mismatch` 不超过 Phase 0 测出的 FP 噪声地板（无确定性 delta）。
2. **单一权威**：Python 侧该路径无 `_slot_of/_free/_freq/_slot_table` 活状态；`block.py` 计算段零分支。
3. **提速**：`run_mtp_spec` K=3 REPEAT=2 相对基线净提升；cap 充裕时目标不低于现有异步方案 B（+38.7%@cap64）。
4. **内存**：峰值不显著上升（±0.1GB）。

---

## 10. 测试策略

- **容量不变性回归**：固定 prompt greedy，cap ∈ {并集, 32, 64} token 序列逐位一致（新增端到端测试）。
- **字节真值**：`STG_VERIFY` 逐槽校验单测 + 端到端可开。
- **C++ 状态机单测**：沿用 `test_demand_dual_native.py` / `test_demand_dual_wiring.py`，
  扩展覆盖「侧区+真实区统一分配器」的槽映射 / LFU 驱逐逐步一致 / gen 新鲜度。
- **端到端**：`run_mtp_spec` K=3 REPEAT=2，记录 tok/s、hit、fastpath/fallback、n_mismatch、峰值内存。
- 注意：MLX 0.31.2 不支持布尔索引 `arr[mask]`，一律用 `mx.where` / `mx.take`。

---

## 11. 已定决策（brainstorm 结论）

1. **走全量迁移（方案 B 全量）**：真实区+侧区+demand+prefetch 收进一套 C++ 分配器、单一权威、canonical LFU。
2. **正确性口径 = 容量不变性 + 字节真值**（非「全常驻」）。
3. **顺序**：先修现有方案 B 到 oracle 干净（Phase 2，直接兑现 exact 收益），再统一侧区/prefetch 权威 + 降耦合（Phase 3）。
4. **cap**：不作 40 硬约束；Phase 0 用 `union_prof` 实测下限（预期 cap=32 已够）。
5. **降耦合范围（2026-07-04 审计后并入 Phase 3）**：promote/route_used_subset/overlay 迁 C++、
   统一 BgReader 并行读、删 resident 快照跨界与 Python 死影子；目标是 `block.py` 塌缩为
   「gate → 一次 `PoolManager.acquire` / 一次 `prefetch`」。gate/预测搭图/计算 kernel/MTP 调度留 Python。
