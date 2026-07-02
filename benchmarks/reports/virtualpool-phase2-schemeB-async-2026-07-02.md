# Phase 2 方案 B（异步并行 demand 版）实测报告 —— 复刻基线重叠模型

日期：2026-07-02 · 分支：`feat/virtualpool-unified` · 开关：`NATIVE_DEMAND_DUAL`（默认关）

## 结论（TL;DR）

**把 C++ demand 从「同步内联 pread」改成「派 worker 线程并行 pread + 惰性 gather」后，性能回归被根治：
ON 的 tok/s 从 4.09（慢 2.6×）拉回到 ≥ 基线，甚至净超基线。**

| 配置 | OFF（基线） | ON（方案 B 异步并行） | 结果 |
|---|---|---|---|
| EXPERT_SLOTS=32（默认） | 11.16 tok/s | **11.82 tok/s** | **+5.9%（达标：≥基线）** |
| EXPERT_SLOTS=64（≥seq·k） | 13.86 tok/s | **19.22 tok/s** | **+38.7%（净收益）** |

对比上一版同步串行报告（`virtualpool-phase2-schemeB-2026-07-02.md`：ON 4.09 / OFF 10.78，慢 2.6×），
本次改动**只动 demand 的 I/O 并发模型、未动状态机**，即拿回全部性能，验证了用户判断：
「慢是实现 bug（同步内联 pread），不是 C++ 接管本身的问题」。

**唯一残留**：ON 的 `n_mismatch` 比 OFF 高（61 vs 45@cap32、61 vs 30@cap64），且**与容量无关地恒为 61**——
这与上一版**同步**方案 B 的 61 **逐字节相同**，证明它是方案 B 侧区/真实区合并语义的**既有**路由差异，
**并非本次异步化引入**，且字节落池不变量校验 0 错（见下）。建议默认关、作 opt-in 实验开关。

---

## 1. 基线的重叠机制（复刻目标）

基线 demand（`NATIVE_DEMAND_DUAL=0`）之所以那次 `.tolist()` 同步不拖慢，是因为 **I/O 藏在后台、
落池是惰性图的一部分**，三层配合：

1. **并行 pread（`blob_loader.py`）**：`BlobExpertSource` 持有常驻 `ThreadPoolExecutor(max_workers=8)`。
   - `read_raw()` 命中未 miss 的走预取缓存 `_pf_cache` / 在途 future；真正 miss 的用 `_pf_pool.map(rd, misses)`
     **并行 pread**（`blob_loader.py:141-149`）。即多个 miss 专家的读盘是**多线程并发**，不在调用线程串行阻塞。
   - `prefetch_async()` 跨层把预测专家的字节**提前**在后台读进 `_pf_cache`（`blob_loader.py:87-102`），
     等真正 demand 时多半已就绪 → 命中即返回、零额外读盘。
2. **惰性物化（scatter 进 MLX 图）**：`load_experts*` 用 `np.frombuffer(...)→mx.array` 只建**惰性**数组
   （`blob_loader.py:152-184`）；落池的 `mx.stack`/scatter 都是**惰性图节点**，不立即 eval。
   MLX 把它们挂进计算图、和后续层的 matmul 一起在 GPU 上跑，**I/O/拷贝与 GPU 计算重叠**。
3. **为什么 `.tolist()` 不拖慢**：那次 host 同步只是把已算好的 route ids 取回来（很小），
   而**重活（读盘在后台线程、scatter 在惰性图）都不挂在这次同步上**——同步返回后主线程立刻发下一层的图，
   GPU 不空转。

**上一版同步方案 B 打破的正是第 1、2 点**：它拿到 inds 后在**调用线程原地阻塞 pread + memcpy 落池**，
每层多一段不可重叠的关键路径 I/O，GPU 被迫等它 → 慢 2.6×（探针 `DEMAND_SKIP_IO` 已证瓶颈是同步调用本身、非磁盘）。

## 2. 本次改动：把并发模型复刻进 C++ 接管路径

保留每层 1 次不可避免的 `inds.eval()` 同步（Phase 2 spike 已证零同步不可行），同步之后：

- **不再在调用线程内联 pread**。`demand_core_locked` 重构为**纯 CPU 状态机**（无 I/O、无大缓冲）：
  只算 local、分配 miss 槽、把新放入 `(expert, slot)` 收集进 `placements`（`native_prefetch.cpp:757-798`）。
- **多 miss 专家 pread 派给 worker 线程并行**：`demand_dual` 在锁外对每个 placement 调
  `bg_pread_into_pool(...)` 提交到 `BgReader`（复用后台线程池，`DEMAND_WORKERS` 默认 8，与基线 8 worker 对齐），
  **直接把字节 pread 进池 buffer**（省掉 tmp+memcpy）；随后 `bg_reader_wait(ticket)` 汇合。
- **gather 惰性**：`demand_dual` 返回 local（int32），消费侧对 `pool[local]` 做**惰性** gather，
  和后续层一起 eval，I/O 藏在 GPU 计算背后。
- **落池字节就绪保证**：`bg_reader_wait` 确保 demand_dual 返回前该层字节已落池；draft/verify 两遍之间有
  eval 边界，故不存在「惰性 gather 读到被后一次 demand 覆盖」的竞态。

诊断开关（默认关，供二分定位）：`DEMAND_SKIP_IO`（跳过 pread）、`DEMAND_TIMING`（分段计时）、
`DEMAND_WORKERS`（worker 数）、`STG_VERIFY`（字节落池不变量校验）。

## 3. A/B 实测（MAXTOK=64, WARMUP=64；K=3, SIDEREGION_LFU=1, ZEROCOPY_DUAL_SOURCE=1, STREAM_BLOB_LOADER=1）

### 3.1 EXPERT_SLOTS=32（默认容量，REPEAT=2）

| 指标 | OFF（基线） | ON（方案 B 异步并行） | 变化 |
|---|---|---|---|
| spec tok/s | **11.16**（11.0 / 11.32） | **11.82**（11.69 / 11.95） | **+5.9%（≥基线，达标）** |
| spec hit_rate | 0.874 | 0.887 | +0.013 |
| gpu_fastpath / fallback | 486 / 1146（fb 70.2%） | 569 / 1159（fb 67.1%） | 更少 miss |
| spec_disk_loads | 5703 | 5482 | −3.9% |
| n_mismatch | 45 | 61 | +16（既有差异，见 §4） |
| mlx_peak_gb | 8.50 | 8.27 | −0.23 |

### 3.2 EXPERT_SLOTS=64（≥ seq·top_k=40，REPEAT=1）

| 指标 | OFF（基线） | ON（方案 B 异步并行） | 变化 |
|---|---|---|---|
| spec tok/s | **13.86** | **19.22** | **+38.7%（净收益）** |
| spec hit_rate | 0.946 | 0.953 | +0.007 |
| spec_disk_loads | 2633 | 2308 | −12.3% |
| n_mismatch | 30 | 61 | +31（既有差异，见 §4） |

**关键验收结论**：两档容量下 ON 的 tok/s 均 **≥ 基线**，性能回归根治。容量越充裕（cap=64），
削掉 Python 胶水（第二次同步 + eff 重建 + 逐层 scatter 启动）+ C++ 精确 LFU（更高 hit、更少读盘）的
**净收益越大**（+38.7%）。

## 4. 正确性

- **字节落池不变量（STG_VERIFY）零错**：校验「真实区每个占用槽的池字节 == 该槽当前 C++ 属主专家(g_real)
  的 blob 新鲜真值」，全程 0 bad → **C++ 并行 pread 落池的字节完全正确**，异步化未引入任何字节污染。
- **n_mismatch 恒为 61（与容量无关）**：cap=32 与 cap=64 下 ON 均为 61，而 OFF 随容量增大从 45 降到 30。
  「恒定、cap 无关」说明这**不是超容量别名**，而是方案 B 侧区(read gen)∪真实区合并成 local 时与基线
  `acquire_gpu_dual` 的一处**确定性路由差异**。
- **既有、非本次引入**：上一版**同步**方案 B 报告中 cap=32 ON 的 n_mismatch 同样是 **61**（OFF 43）。
  本次异步改动**只换 I/O 并发模型**，n_mismatch **逐字节不变**，证明该差异是方案 B 状态机的既有属性，
  与并发模型无关。属路由级（用到相邻/次优专家），非崩溃、非发散；对 tok/s 与内存无影响。
- **单测全绿**：`test_demand_dual_native.py` + `test_demand_dual_wiring.py` 共 10 项通过
  （槽映射等价 / LFU 驱逐逐步一致 / native 缺失·超容量回退 / resident_experts 一致）。

## 5. 剩余风险与建议默认值

- **建议 `NATIVE_DEMAND_DUAL` 默认关（opt-in）**：性能已达标且在充裕容量下净超基线，但 n_mismatch 比基线高
  （既有路由差异，见 §4），逐位严格场景不宜默认开。作实验/性能探索开关保留。
- **若启用，务必 `EXPERT_SLOTS ≥ seq·top_k`**（MTP verify 默认 seq·k=40，故 ≥64）：否则真实区超容量、
  多余 miss 落槽 0，逐位进一步劣化（已在 `_acquire_native` 加一次性告警）。
- **待查（后续，非本任务范围）**：把方案 B 的 side∪real→local 合并与基线 `acquire_gpu_dual` 逐位对齐，
  消掉那恒定的 +16~31 mismatch；这是状态机语义细节，与本次「复刻异步重叠模型」目标正交。
