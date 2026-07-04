# 侧区字节稳定发布（Route 1）修复验收报告

日期：2026-07-04 ｜ 分支：`perf/async-demand-offload`

## 一句话根因

默认 `ZEROCOPY_DUAL_SOURCE` decode 路径下，侧区异步预取在 C++ `read_publish` 里把磁盘字节**旁路 `memcpy` 进 `_pools[layer][k]` 这块 MLX 数组 buffer**，而该 buffer 会被 MLX 在解码前向之间**重定位**——旁路写对 MLX 不可见，字节被丢在孤儿 buffer，消费侧 matmul 读到别专家字节 → `DUAL_VERIFY BAD`（历史必崩锚点 `L1 e453 r32`）。

## 修法（Route 1：稳定缓冲 + MLX 追踪发布）

侧区字节不再写「会迁移的池数组」，而是走两段式：

1. **C++ 稳定缓冲**（Task 1）：`read_publish` 把整条 blob 记录 `pread` 进 C++ 自己拥有、永不迁移的 `g_side_bytes`（键 `(layer,gen)`，按 `(row-base)*stride` 索引），并用 `SideLayer::dirty` 跟踪「自上次发布起新到达的行」。新增 `sideregion_publish` 绑定按段取出脏行、取出即清脏。缓冲以 `shared_ptr` 持有，bg 线程在途写入不受 `reset`/换代重分配影响（杜绝 UAF）。
2. **Python 拆段适配**（Task 2）：`_StagingSide.publish(layer)` 把 C++ 返回的 per-seg uint8 字节按段元数据（`src._segs`）还原成 per-key 具类型 `mx.array`（uint32 原样 / uint16→bfloat16 / uint8 mxfp4 scales 原样）。dtype 分派收敛为 `_typed_seg` helper，与既有行装配复用同一套逻辑。
3. **消费侧发布落池**（Task 3）：`acquire_gpu_dual` 在 `has_side` 时调 `_publish_side`，用 **MLX 追踪的原地 scatter**（`pool[k][rows]=arr`）把新到侧区行写进池——与真实区同机制，从而随 MLX 的 buffer 迁移一起存活。每行只在字节首次到达时发布一次（脏行跟踪），之后长存，热路径无多余 GPU 同步。

保留侧区异步 I/O 与 MLX 原生 matmul，未引入自定义 kernel。

## 验收证据

| 项目 | 命令要点 | 结果 |
|---|---|---|
| Task 0 spike（前提） | `SPIKE_PUBLISH=1 DUAL_VERIFY=1` 短跑 | `ok=4875 bad=0`，确认 MLX 追踪 scatter 跨前向存活 |
| 短跑（正式代码） | `DUAL_VERIFY=1 MAXTOK=8` | `ok=4906 bad=0` |
| 长跑稳定性 | `DUAL_VERIFY=1 MAXTOK=128 WARMUP_TOK=8` | `ok=63831 bad=0`，锚点 `L1 e453 r32` 未出现 |
| 容量不变性 | `DUMP_IDS=1 EXPERT_SLOTS=32 vs 48, MAXTOK=32` | 两 cap 输出 token 序列**逐字节一致**（此前确定性 `n_mismatch=61` 偏差已消除） |
| 单元测试 | `test_sideregion_publish_native.py` / `test_sideregion_publish_wiring.py` | 全通过（native 字节/脏行、Python 拆段/dtype、消费侧 scatter 落池） |
| 既有回归 | 6 个侧区/双源测试套件 | 无回归；`test_sideregion_segment_scatter` 按新机制对齐断言（改用 `sideregion_publish` 取回字节校验，语义等价） |

## 提交序列

- `0e19085` feat(native)：侧区字节改写 C++ 稳定缓冲 + `sideregion_publish` 脏行取出
- `e5d04b5` fix(native)：稳定缓冲改 `shared_ptr` 保活，杜绝 reset/重分配下的 UAF
- `762dad9` feat(staging)：`_StagingSide.publish` 按段拆出侧区新行
- `2063896` refactor(staging)：dtype 分派收敛为 `_typed_seg` + 段数断言
- `e8bf866` feat(pool)：`acquire_gpu_dual` 发布侧区新行到池（MLX 追踪 scatter）
- `6e53630` test：侧区 scatter 测试对齐新机制

## 遗留项

- **Route 3 真零拷贝**（完全自定义 C++ fused MoE kernel，直接读 C++ 稳定 MTL buffer、绕过 MLX 数组管理）作为**后续独立性能项目**。Route 1 已解耦正确性修复与该性能改造：当前修复用 MLX 追踪 scatter 保正确，Route 3 用于进一步消除发布拷贝、追求真零拷贝性能。
- 门控诊断（`DUAL_VERIFY`/`SIDE_TRACE`/`SIDE_AUDIT`/`DUAL_DIAG`/`POOL_PTR_TRACE`）默认关、零热路径开销，保留供后续排查；无 `SPIKE_PUBLISH` 临时代码残留。
