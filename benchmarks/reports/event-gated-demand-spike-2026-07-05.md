# 事件门控 demand spike —— go/no-go 报告

日期：2026-07-05
分支：`perf/eliminate-miss-sync`
关联：
- `benchmarks/reports/virtualpool-phase2-spike-2026-07-02.md`（上一版"同前向零同步不可行"结论）
- `benchmarks/reports/virtualpool-phase0-2026-07-02.md`（去 per-layer 同步 barrier 的收益上界 30→54 tok/s）

---

## 目的

在写正式实现前，用最小 native 探针钉死唯一的机制不确定性：

> **"gate 完成回调里写结果 + MTLSharedEvent signal，下游算子 encodeWait 门控"
> 能否在同一次前向内 bit-exact，且主线程零 inds 同步？**

这正是上一版 spike（`demand_probe`/`demand_probe_handler`）**未测**的第三机制。上一版只测了两种朴素机制并证伪，据此把"消除 per-layer 同步"判为不可行、列为非目标。本 spike 专测第三机制。

## 机制

native 新增两个探针 primitive（`native/ext/spike/event_gate_spike.cpp`，仅测试用）：

- **`spike_signal(inds, pool, event_id, value) -> local`**：主线程**不** eval inds；在 inds 的命令
  缓冲挂 `addCompletedHandler`（此刻 inds 已算完、指针有效），回调里做两件"代表 demand 真实形态"
  的事——写 `local[i]=inds[i]`（代表槽计算）、`memset pool[inds[i]]` 整行为 `inds[i]&0xff`
  （代表 miss 落池字节），然后 CPU 侧 `setSignaledValue(value)`。
  - **强制切命令缓冲**：不能自己 `commit()`（MLX 的 `gpu::eval()` 在 `eval_gpu` 返回后还要往它预先
    抓取的同一 `command_buffer` 挂 handler，我们 commit 会触发
    `addCompletedHandler after commit` 断言崩溃）。改为挂完回调后 dispatch `max_ops+1` 个空 kernel
    把 `buffer_ops_` 顶过 `max_ops`，让 MLX 自己 `needs_commit()` 为真、正确地提交这个含 inds 计算
    与本回调的 buffer → 回调随该 buffer 完成而触发，早于下游 wait-buffer 执行。
- **`spike_wait(local, event_id, value) -> local`**（恒等门控）：`end_encoding()` 后往命令缓冲
  `encodeWait(event, value)`，再 dispatch 恒等 int32 拷贝。下游任何消费其输出的算子都被 GPU 流挡在
  事件之后 → 读到回调已写好的 local / pool。

校验（单次 `mx.eval` 收尾，全程主线程不 eval/tolist inds）：
`inds=(arange(64)*7+3+it)%128`（非平凡 GPU 计算，读脏值/全 0 必露馅）；
`gathered = take(pool, gated_local)`，断言行首字节 == `inds&0xff` 且 `gated_local == inds`。

## 结果（200 次确定性校验）

| 机制 | 做法 | 错误数 |
|---|---|---:|
| **去门控（负对照）** | gather 直接用 `local`（signal 输出），**无** `encodeWait` | **197 / 200 错** |
| **事件门控** | gather 用 `spike_wait(local)`（`encodeWait` 门控） | **0 / 200 错** |

- 去门控 197/200 错，与上一版 spike 的 P-handler（195/200 错）几乎一致 —— **复现**了"完成回调写入
  对同前向下游不可见"的失败模式，确认测试有区分度、非平凡。
- 事件门控 0/200 错 —— **local 正确性 + 池字节可见性**双双 bit-exact，且全程主线程零 inds 同步。

## 判定：**GO（强）**

MTLSharedEvent + `encodeWait` 机制把去门控的 197/200 错**翻成 0/200 错**。上一版判为"不可行"的
"同前向零主线程同步 demand"，在第三机制下**成立**。可进入正式实现（事件门控异步 demand primitive
嵌进 `block.py` 的 acquire→matmul 主路径，复用现有 `demand_core_locked` 精确状态机）。

## 本 spike **未**证明的（正式实现须继续验证，诚实边界）

1. **性能收益**：本 spike 只证"正确性可行"，未测端到端 tok/s。GPU 在回调（槽机+pread）期间的空转、
   强制切命令缓冲的开销，都要在正式实现里用 `run_mtp_spec` 实测，对齐 Phase 0 上界（30→54 那档的多少）。
2. **强制切分的生产形态**：spike 用"dummy 空 kernel 顶满 max_ops"逼提交，是探针取巧；正式实现要找
   更干净的 per-layer 单次切分（评估自定义提交点 / 每层一次而非每 op）。
3. **真实 demand**：spike 的 local=inds 恒等、pool 用 memset 代替真实 `demand_core_locked` +
   `bg_pread_into_pool`；正式实现须接真状态机与真落盘，并过容量不变性 + STG_VERIFY + 同步/异步对拍。
4. **统计/前向边界 drain**：异步下 `demand_last_stats` 时序错位、buffer 生命周期与前向边界排空，
   正式实现需一并处理（见设计 §2.4）。

## 正式实现 + 端到端 A/B（2026-07-05 当天续）

spike GO 后当天即落了正式实现（`demand_dual_async` + `gpu_wait_event`，复用 `demand_core_locked`，
`DEMAND_ASYNC` 开关，默认关），并做了生产配置端到端 A/B（80B，`STREAM_BLOB_LOADER=1
NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32
K=3`）：

| 配置 | tok/s(中位, MAXTOK=128×3) | best | n_mismatch |
|---|---|---|---|
| 同步 demand_dual | 13.20 | 13.28 | 125 |
| 事件门控异步 | 13.14 | 13.45 | 125 |

**结论：异步 bit-exact（n_mismatch 两边同为 125、路由逐槽一致），但性能持平（噪声内，零加速）。**

### 为什么零加速（DEMAND_TIMING 硬数据钉死）

`DEMAND_TIMING=1` 分段计时（累计整段解码），并用 `DEMAND_SKIP_IO=1` 把 I/O 从 "core" 里剥出：

| 成分 | 时间 | 性质 | 异步能救吗 |
|---|---|---|---|
| `inds.eval` | ~21–24s | 主线程等 GPU 算 gate + 上层 matmul，**GPU 真在忙** | 否，墙钟不变 |
| demand 槽机(纯 CPU) | **0.42s**（SKIP_IO 实测） | 可忽略 | 否，本非瓶颈 |
| **demand miss 磁盘 I/O** | **~17s**（`bg_reader_wait`） | **GPU 空转等磁盘（真气泡）** | 否，异步下 GPU 照样 `encodeWait` 等落盘 |

> 注：原 `demand_dual` 的 DEMAND_TIMING "core" 字段（`g_dt[4]=t3-tb`）把 pread I/O 等待也算进去了，
> 故显示 17.4s；SKIP_IO 后掉到 0.42s，证明 17s 全是 miss 落盘 I/O。

**per-layer 同步与 demand CPU 工作都不是瓶颈**：`demand_dual` 早把 CPU 活搬进 C++/后台线程，剩下的
"同步"只是等 GPU 和磁盘。demand 在关键路径上（matmul 必须等其槽位/字节），无论同步异步，GPU 都得空转
等那 ~17s miss I/O。事件门控只是把"主线程阻塞"换成"GPU 侧 encodeWait 挂起"，墙钟不变。

Phase 0 的 +387% 上界不现实：它同时假设 `disk_loads=0`（零 I/O）**且**删掉 demand 工作。

### 真正的提速杠杆（后续方向）
1. 降低 demand miss（提高预取召回，当前 hit=0.87 → 13% miss = 那 ~17s I/O），把 I/O 移出关键路径。
2. 加速 MoE 计算（`t_verify` / `inds.eval` 等的那块 fused MoE kernel）。
3. 更快 I/O（更大块读、page cache 预热）。

## 处置

事件门控机制**技术上可行且 bit-exact**（本报告上半 spike 证明），但对当前架构**提速无效**（本节 profiling
证明，因瓶颈是 miss I/O 而非同步）。据此**回滚全部实现与探针代码**（`demand_dual_async`/`gpu_wait_event`/
`DEMAND_ASYNC`/spike primitives/对拍与 spike 测试），保持代码库干净；**保留本报告为可复现证据**：

- spike 判据：门控 0/200 错、去门控 197/200 错（机制成立）。
- 端到端：同步 13.20 vs 异步 13.14 tok/s，均 n_mismatch=125（bit-exact、零加速）。
- 分段计时：miss 磁盘 I/O ~17s 是真气泡，槽机仅 0.42s，同步非瓶颈。

若未来要重启：机制细节（不能自己 commit、需顶 max_ops 强制切分、encodeWait 前先 end_encoding）见本报告
上半与 git 历史（本分支 `perf/eliminate-miss-sync` 的回滚 commit 之前一版）。
