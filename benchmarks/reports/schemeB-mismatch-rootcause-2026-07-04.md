# 默认双源侧区字节错槽 —— root-cause 报告（2026-07-04）

> 方法：systematic-debugging（先取证再定因）。本报告只定位根因，不含修复实现。

## 1. 复现事实（确定性 + run-to-run 抖动并存）

默认 decode 路径 `ZEROCOPY_DUAL_SOURCE=1`（未开 `NATIVE_DEMAND_DUAL`），开 `DUAL_VERIFY=1` 每次都爆 `[DUAL_VERIFY] BAD`（此前 31/62/54 条/次；短跑 MAXTOK=8 也稳定复现 11+ 条）。锚点 `L1 e453 r32` 跨多次运行复现。

复现命令（短跑，几十秒）：
```
DUAL_VERIFY=1 DUAL_DIAG=1 SIDE_TRACE_LAYER=1 SIDE_TRACE_ROW=32 SIDE_AUDIT=1 \
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=8 WARMUP_TOK=0 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```

BAD 语义（`resident_pool.py:_verify_side_bytes` L488-518，已核实与消费同数组同行）：本次前向**真实路由命中**的专家 E，侧区 e2r 声称 E→物理行 R，但 `_pools[layer][R]` 的字节 ≠ E 的磁盘真值 → 被 gather 消费的就是错专家权重。

## 2. 取证插桩（临时，未提交产品逻辑）

- C++ `native_prefetch.cpp`：`SIDE_TRACE_LAYER/ROW` 追踪某 (layer,row) 的 RESERVE/EVICT/MEMCPY/PUBLISH/CONSUME 全事件（跨线程全局事件序 + 线程 id + `ptrs[0]` 池 buffer 地址）；`SIDE_AUDIT` 查行账本自洽；`array_data_ptr` 取 mx.array 底层 buffer 指针。
- Python `resident_pool.py`：`DUAL_DIAG` 在 BAD 时全专家/跨层扫出行实际占用者 + 消费侧池 buffer 指针。

## 3. 关键证据链（L1 row32，异步默认模式）

```
ev0  RESERVE row32←e149  ptr0=0x1774b4000
ev7  MEMCPY  row32←e414  ptr0=0x1774b4000
ev9  CONSUME row32→e414                      # 此时 OK
ev10 EVICT_REUSE row32 victim=414 newE=453
ev11 MEMCPY  row32←e453  ptr0=0x48bd94000    # ← 同一池段 buffer 地址已变
ev12 PUBLISH row32←e453  ptr0=0x48bd94000
ev13 CONSUME row32→e453
→ [DUAL_VERIFY] BAD L1 e453 row32
→ [DUAL_DIAG] e2r_says=453->row32, 但 row32 实际装 e414 字节；
             消费侧 pool_ptr0=0x423b04000（又是第三个地址）
```
同一池段 `ptr0` 全程漂移：`0x1774b4000 → 0x48bd94000 → 0x56676c000 → 0x4fc0c8000 …`。

## 4. 判别实验：时序竞态 vs buffer 身份（决定性）

开 `SIDEREGION_SYNC=1`（reserve+memcpy+publish 在 Metal 回调线程内**同步**完成，彻底消除异步 bg 线程与 consume 的竞态）后：
- **BAD 依旧出现**（L11/L14/L15/L18/L19…，12 条）；
- `ptr0` **照样每前向漂移**（`0x17b91c000 → 0x518e44000 → 0x59745c000 → 0x5b11f4000 → 0x548610000 → 0x17b91c000`）。

→ **排除「异步时序竞态」为主因**。同步都修不好。

## 5. 根因（确凿）

**侧区池段数组 `_pools[layer][k]` 的底层 buffer 跨前向被 MLX 重分配，而侧区预取用 completion-handler 旁路 `memcpy` 写入的是 fill 时刻捕获的旧 buffer 地址——写入落到孤儿 buffer，消费侧 gather 的活 buffer 从未收到这些字节。**

机制细节：
1. 侧区预取是**预测式**的：forward T 给未来**目标层 `tgt`** 预填（`block.py:325-327` `pool_list=[rp._pools[tgt][...]]`），fill 与 consume 天然分属不同前向。
2. `PrefetchPoolSideRegionPrimitive::eval_gpu` 在 command-buffer 完成回调里 `memcpy` 进 `ptrs[k]=in[k].data()`（`native_prefetch.cpp:404-447, 595-620`）。这是 MLX 图**不可见**的旁路写。
3. `_pools[layer][k]` 虽在创建时 `mx.eval` 钉过一次指针（`resident_pool.py:219-224`），但作为 Primitive 输入反复参与图、且被旁路 mutate，MLX 内存管理并不保证其 buffer 原地保留——实测 `ptrs[0]` 逐前向漂移即证。
4. e2r 映射只记「专家→行」、**不认 buffer**（`native_prefetch.cpp` SideLayer），于是把「row R」交给消费侧的活 buffer，读到的是该行在当前 buffer 里的旧/别专家字节。

对照：真实区 `preallocate`+C++ `g_real` 直写那套字节稳定（同样 eval 钉指针，但走的是 C++ 拥有/直写的稳定 buffer 契约，不是 completion-handler 旁路写会漂移的 MLX 数组）。→ 问题特定于侧区这套写入方式。

### 5.1 「为何 buffer 会漂移」的定点取证（POOL_PTR_TRACE）

在 submit（`block.py:326`）与 consume（`acquire_gpu_dual`）两处对固定层打 `id(池数组对象)` + `array_data_ptr`：

```
obj_id=4720214208  全程不变              → Python 数组对象从未被替换
每前向内 submit.ptr == consume.ptr       → 同一前向内 buffer 稳定
跨前向 ptr 变：0x17b868000(前4个前向稳)→0x552e94000→0x57b504000→0x532bb8000→…
```

判别 **donation/回收 vs 按图重算**：若按 `mx.zeros` 图重算，consume 应读到**全零**；但 `DUAL_DIAG` 实测坏 row 装的是**别的真实专家字节（非零）** → 是 **MLX 把该 buffer 释放回内存池、把对象重新指到一块装着旧数据的回收 buffer**，即 **buffer donation/回收**（非对象重建、非 per-op 拷贝、非重算全零）。

机制定性：`_pools[layer][k]` 虽 concrete，但仍挂着其计算图（`mx.zeros` Full primitive），且每前向被当作 `prefetch_pool_sideregion` 输入 + `mx.take` gather 输入反复入图；MLX 在 eval 的内存优化里认为其 buffer「可重算、非必需」，遂 donate/回收 → 旁路 memcpy 写入的旧 buffer 成孤儿。

## 6. 建议修复方向（只给方向，未实现）

核心：**让侧区字节的「写入 buffer」与「消费 buffer」是同一块、且跨前向稳定不被 MLX 迁移。** 候选：

- **方向 A（对齐真实区、契合既定 plan 的 P3-c/统一权威）**：侧区字节存储改为 **C++ 拥有的稳定 buffer**（与 `g_real` 同构），消费侧从该稳定 buffer gather，不再依赖会漂移的 MLX 池数组做旁路写目标。这直接兑现「侧区∪真实区 overlay 迁 C++」并根治此 bug。
- **方向 B（最小改动验证根因）**：确保传给 prefetch 的池数组与消费数组是同一块**不可迁移**分配（例如显式持有并复用、禁止 MLX 对其 donation/relocation），fill 后消费侧读到同一 buffer。风险：与 MLX 内存管理契约的边界需实测确认，可能仍不稳。
- **方向 C（排除，仅诊断价值）**：`SIDEREGION_SYNC` 只改时序，已证无效。

**推荐 A**：既根治，又与 spec「统一池权威到 C++」方向一致；建议在 Phase 2′ 落地 A 的最小版本（侧区字节 C++ 稳定 buffer + 消费直读），以 `DUAL_VERIFY 0 BAD` + 短复现锚点消失为验收。

## 7. 残留 / 注意

- 约 70% 走慢回退路径的 acquire 当前无 live 字节校验；A 落地后需一并覆盖校验（Phase 1 的字节真值 oracle）。
- 临时插桩（`SIDE_TRACE`/`SIDE_AUDIT`/`array_data_ptr`/`DUAL_DIAG`）取证价值高，建议整理为门控诊断保留或单独提交，勿混入修复 commit。
