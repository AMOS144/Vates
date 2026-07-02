# VirtualPool Phase 2（方案 B）机制探针报告 —— 零同步 demand 可行性

日期：2026-07-02
分支：`feat/virtualpool-unified`
关联：spec `2026-07-02-qwen-virtualpool-unified-design.md`（硬问题 1/3）、plan 同名。

## 背景

用户选定 **方案 B**：C++ 完全接管真实区槽状态、**零主线程同步**、精确复刻 LFU、统一分配器。
方案 B 的**定义性优势**是「零主线程同步」——同一前向内，把 demand（读 miss 字节 + 落池 + 算
local）全部搬进惰性 Primitive，主线程不再每层 `int(mx.sum)` / `.tolist()`。

在写大改代码前，先按 TDD/systematic-debugging 用最小 C++ 探针钉死这条**地基机制**：
**「同前向内，能否不经主线程同步，就拿到 GPU 算出的 inds 值并让下游 matmul 读到落池结果？」**

## 探针与结果

两个候选机制，各跑 200 次确定性校验（`inds=(arange*7+3)%64` 非平凡 GPU 计算，读脏值必露馅）：

| 机制 | 做法 | 结果 |
|---|---|---|
| **P-body** `demand_probe` | 在 `eval_gpu` **函数体**里直接读 `inds.data<uint32_t>()`，写 `local=inds+OFF` | **127/200 错**（读到全 0：inds kernel 尚未执行） |
| **P-handler** `demand_probe_handler` | 在 `addCompletedHandler`（回调）里写 `local`，下游 `mx.take(tbl, local)` 单次 eval | **195/200 错**（回调写入对同前向下游不可见） |

（对照：`materialize_spike` 写**常量**到 output → 下游 gather 200/200 正确，说明「eval_gpu body 写 output → 下游可见」成立；**失败的只是「读 GPU 算出的输入值」这一步**。）

## 结论：**同前向零主线程同步 demand 不可行**（本 MLX 0.31.2）

- `eval_gpu` 被调用时，输入 `inds` 的 GPU kernel 只是**入队、未执行**，body 里读到旧内存（0）。
- 回调（`addCompletedHandler`）虽能拿到算完的 inds，但它在 command buffer **完成之后**触发，
  而同前向的下游 matmul 已在同/后续 buffer 里排好并可能先执行 → 回调写入读不到。
- 这正是侧区预取必须**双缓冲跨前向**的根因：回调机制只对「下一前向」可见，对「本前向 demand」无效。
- 要在本前向拿到 inds 值，只能**同步**（`inds.eval()` / `int(mx.sum)`），即一次主线程 GPU 同步。

**因此「零主线程同步」在同前向 demand 上无法实现**（除非做被 spec 明确排除的投机执行）。

## 对 方案 B vs 方案 C 决策的影响（新的根本性抉择）

「零主线程同步」是方案 B 相对方案 C 的**唯一定义性优势**（见 spec §5 硬问题 3）。该优势现被证伪：

- 无论 B 还是 C，demand 都**必须每层一次同步**拿 inds（不可避免）。
- Phase 0 已证：真正的大头收益（+174%）来自**移除 demand 的主线程 WORK**（第二次 `.tolist`
  + Python 落池 scatter + eff 重建），而**不是**移除那次同步本身（P2 探针保留同步仍 +174%）。
- 移除「主线程 demand WORK」**方案 B 和方案 C 都能做到**：
  - **方案 C**：主线程用**现有** `_alloc_slot` 预留槽（复用 Python LFU/free 分配器，1 次极小
    `.tolist` 拿 unique 路由）→ C++ primitive 只按计划 pread+memcpy 落池。Python 仍是槽状态
    **唯一权威**，无一致性改写。
  - **方案 B**：把 `_slot_of/_free/freq` 全迁 C++、精确复刻 LFU 驱逐、并让 `resident_experts`
    （预取过滤要用）/ prefill host 路径 / pin / prefetch_cpp 全部与 C++ 状态保持一致 —— 大量
    改写与并发一致性风险，**换来的额外收益为 0**（因为零同步已不可得）。

**推荐：从方案 B 改回方案 C。** 同样的 1 次同步、同样吃下「移除主线程 demand WORK」的收益，
但复用 Python 分配器、Python 保持槽状态唯一权威 → 风险大幅下降、无一致性改写。

> 若用户仍坚持「C++ 拥有槽状态」这一形态（即接受 1 次同步版的方案 B），可行但不推荐：
> 需另行处理 `resident_experts` 预取过滤、prefill/pin 与 C++ 状态一致性，风险高而无额外收益。

## 探针去留

`demand_probe` / `demand_probe_handler`（native）+ `test_demand_primitive_spike.py` 保留为
本结论的可复现证据；若最终不采用，随 Phase 2 收尾一并清理。
