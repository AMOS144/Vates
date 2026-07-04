# Route 3 Phase 1：C++ 拥有池 buffer + 全直写（删 pool scatter）验收报告

日期：2026-07-04

## 背景

Route 1（侧区字节写 C++ 稳定缓冲 → 消费侧 MLX scatter 落池）虽拿到 `DUAL_VERIFY 0 BAD`，
但在低 cap（=32）稳态解码引入 **15× 性能回归**（0.89 tok/s，正常应 ~13）：`_publish_side`
每层每前向都做 C++ 段拆分 + per-key MLX scatter，churn 高时每前向数百次碎 scatter，打断惰性流水线。

本阶段目标：**C++ 拥有池 buffer + 侧区/真实区全部 C++ 直写、彻底删掉 pool 上所有 MLX scatter**，
同时恢复 ~13 tok/s 且保持 `DUAL_VERIFY 0 BAD`。

## 根因（systematic-debugging 取证链）

1. 仅让 C++ 拥有 buffer（`mx::allocator::malloc` + no-op deleter）**不足以**达成 0 BAD：仍有 12 个 BAD。
2. `SIDEREGION_SYNC=1`（回调内同步写）不仅没修好、反而更糟（161 BAD）→ **证伪「异步 fill 未完成」假设**。
3. 加 `sideregion_drain()`（前向开头排空上一代 fill）后 BAD 数不变（12→13）→ 再次证伪 fill 未完成。
4. `DUAL_DIAG` 取证：坏行 `row_is_zero=False`、`occ_same_layer=<另一个专家>`、`e2r` 只映射目标专家
   → e2r 说该行是专家 A，物理字节却是被驱逐的旧占用者 B。
5. `SIDE_SELFCHECK`：memcpy 后立刻从 `ptrs` 读回**恒等于刚写的字节**（无 WRITE_MISMATCH），
   但写侧 `ptr0` **每前向都在变**（0x42.../0x5c.../0xbd...），而消费侧 `pool_ptr0` 固定。
6. 最小 spike 复现根因：对**同时被 compute 图引用**的数组做 `pool[k][idx]=v`（MLX in-place scatter），
   MLX 会**重分配底层 buffer 并重绑 `pool[k]`**（owned buffer 每前向都变；mx.zeros 也会变，只是偶尔能 donate 复用）。

**结论**：只要 demand 真实区还用 MLX scatter 写池，pool buffer 每前向就被重分配一次；侧区预取在
submit 时捕获的旧 buffer 指针沦为孤儿，直写字节永远进不了消费端读的规范 buffer。这正是 pre-Route-1
「buffer donation/recycling」bug 的真身，也解释了 Route 1 的消费侧 scatter 为何能掩盖它
（每前向把被重分配的侧区行又补回来）。

## 修复

- **C++ 拥有池 buffer**：`pool_owned_zeros(shape, dtype)`（`mx::allocator::malloc` + no-op deleter，
  进程内持有），spec/dual 模式建池改用它（`_alloc_pool`/`preallocate`/`_bootstrap_dual_pool`）。
- **侧区直写**：`read_publish` 恢复直接把 blob 各段 memcpy 进 owned 池的侧区物理行（删 `g_side_bytes`
  稳定缓冲、删消费侧 scatter、删 `_publish_side`/`sideregion_publish`/`_StagingSide.publish`）。
- **真实区直写**：demand 落池由 MLX scatter 改为 C++ `pool_write_rows` / `pool_write_stacked`
  （已加载专家段 memcpy 进 owned 池行），`_write_slot`/`_write_slots_batch`/`_place_experts_stacked`
  在 owned 池下走 C++ 直写。**至此 pool 上再无任何 MLX scatter → buffer 地址进程内恒定。**
- **消竞态兜底**：`sideregion_drain()` 在前向开头（gen 翻转）排空上一前向在途侧区 fill，
  保证被消费的侧区行字节已写完（inflight 计数在 `eval_gpu` 提交时 +1、后台写完 -1）。

## 验收结果

字节真值（`DUAL_VERIFY`，ZEROCOPY_DUAL_SOURCE 路径逐专家逐 key 对拍磁盘真值）：

| cap | ok | bad |
|----|----|----|
| 32 | 7519 | **0** |
| 64 | 14325 | **0** |

稳态吞吐（K=3, MAXTOK=64, WARMUP=8）：

| cap | spec tok/s | baseline tok/s | mlx_peak_gb | n_mismatch |
|----|----|----|----|----|
| 32 | 13.22 | 6.72 | 8.07 | 37 |
| 48 | 15.02 | 7.91 | 10.77 | 37 |
| 64 | 17.59 | 8.20 | 13.49 | 37 |

- **回归修复**：cap=32 从 0.89 → **13.22 tok/s**（回到目标区间）。
- `n_mismatch=37` **容量不变**（cap 32/48/64 完全一致）→ 属 spec-vs-baseline 的 FP/路由噪声地板，
  非缓存字节问题（容量不变性 oracle 已通过、逐字节一致）。
- 32GB 机上 cap=64 峰值 13.49 GB，安全余量充足。

测试：
- 两 oracle（`test_capacity_invariance.py` / `test_pool_byte_truth.py`）PASS。
- 池/侧区/预取/dual 相关单测全绿（含新增 `test_pool_owned_buffer_spike.py` 地址恒定+直写可见 3/3、
  改写后的 `test_pool_sideregion_native.py` 直读池验证直写真值）。

## 遗留

- Route 3 Phase 2（自定义 fused Metal kernel 进一步提速）仍为后续独立性能项。
- `POOL_OWNED=0` 保留为 A/B 对照开关（回退 mx.zeros + MLX scatter 老路径）。
