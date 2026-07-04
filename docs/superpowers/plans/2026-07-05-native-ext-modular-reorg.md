# native/ 目录分级重构（行为保持）

**日期:** 2026-07-05
**分支:** `perf/async-demand-offload`
**目标:** 把 `native/ext/` 的 1200 行大文件 `native_prefetch.cpp` 按职责域拆分为子目录模块,消除"所有函数堆在几个文件里"的混乱。**纯代码搬移 + 头文件按模块拆分,零逻辑改动、Python 侧 API 完全不变。**

---

## 1. 现状

`native/ext/`(生产扩展 `native_moe_ext`)全平铺 4 个源文件:

| 文件 | 行数 | 内容 |
|---|---|---|
| `native_common.h` | 34 | 共用前导(nanobind/Metal/MLX 头 + 命名空间别名) |
| `native_fused.{h,cpp}` | 23/664 | 融合 MoE 计算核(3 工厂) |
| `native_prefetch.{h,cpp}` | 87/1200 | **7 个不相关功能段全塞一起** |
| `native_bindings.cpp` | 145 | nanobind `NB_MODULE` 注册 |

`native_prefetch.cpp` 7 段(注释已标 [0]~[7]):[1] blob 直读、[2] 轻量预取、[3] staging miss→hit + hprof、[4] 段散写侧区、[5] owned 池底座、[6] 真实区 + demand_dual、[7] 后台读线程。

## 2. 跨段耦合(拆 TU 的关键风险点)

1. `bg_submit_task`:定义在 [7] BgReader,被 [3]、[4] 调用 → 由 `io/bg_reader.h` 内部声明暴露(不绑 Python)。
2. `open_blob_nocache`:[4]/[7] pread 都用 → 抽成 `io/blob_io.h` 的 header-only `static inline`。
3. `demand_dual`([6]) 读侧区状态 `g_side`/`g_side_mutex`([4]) → 用 `side_region.h` 暴露内部访问器 `sideregion_snapshot(layer, gen)`(在同一锁下拷贝 e2r,行为与原内联循环一字不差),避免跨 TU 共享全局。
4. 其余 helper(`hprof_steady_now`/`dt_now_us`/`side_tid`/`side_trace_hit`/`side_audit`/`dtype_from_str`)均只在本段用,原地跟随。

## 3. 目标目录树

```
native/ext/
  common.h                 # ← native_common.h（改名）
  bindings.cpp             # ← native_bindings.cpp（改名，NB_MODULE 内容不变，只改 include）
  CMakeLists.txt           # 源列表更新为新路径
  Makefile                 # 不变（只调 cmake）
  compute/
    fused_moe.{h,cpp}      # ← native_fused.{h,cpp}（改名，include 改 ../common.h）
  io/
    blob_io.h              # open_blob_nocache（+ F_NOCACHE 宏，header-only inline）
    blob_load.{h,cpp}      # [1]
    bg_reader.{h,cpp}      # [7] + bg_submit_task（内部声明）
  prefetch/
    prefetch.{h,cpp}       # [2] + [3] + hprof 探针
  pool/
    owned_pool.{h,cpp}     # [5]
    side_region.{h,cpp}    # [4] + sideregion_snapshot（内部访问器）
    demand.{h,cpp}         # [6]
```

最大 .cpp 从 1200 行降到 ~370 行(side_region);目录名即模块边界。

## 4. 搬移映射(源行 → 目标)

| 原 native_prefetch.cpp 行 | 目标文件 |
|---|---|
| 43-47 `open_blob_nocache` + 20-22 `F_NOCACHE` | `io/blob_io.h` |
| 53-105 `BlobLoadPrimitive`+`blob_load` | `io/blob_load.cpp` |
| 107-311 `PrefetchOnComplete`+`PrefetchStaging`+hprof | `prefetch/prefetch.cpp` |
| 313-682 侧区(+`sideregion_snapshot` 新访问器) | `pool/side_region.cpp` |
| 684-770 owned 池 | `pool/owned_pool.cpp` |
| 772-1028 真实区 + demand_dual | `pool/demand.cpp` |
| 1030-1200 BgReader + bg_* | `io/bg_reader.cpp` |

`native_prefetch.h` 的声明按同一映射拆进各模块头。

## 5. 构建

- `CMakeLists.txt` 的 `nanobind_add_module` 源列表改为:`compute/fused_moe.cpp io/blob_load.cpp io/bg_reader.cpp prefetch/prefetch.cpp pool/owned_pool.cpp pool/side_region.cpp pool/demand.cpp bindings.cpp`。
- 删除 `native_common.h / native_fused.{h,cpp} / native_prefetch.{h,cpp} / native_bindings.cpp`。
- `native/bench/`:轻触——`main.cpp`→`pack_loader.cpp`,按 `io/compute/metal/` 分子目录,Makefile 路径更新。bench 各程序彼此独立,零逻辑改动。

## 6. 验收(重构后必须全绿)

1. `make native_moe_ext` 干净编译通过(无 warning 回归)。
2. oracle:`test_capacity_invariance` + `test_pool_byte_truth`(STG_VERIFY 0 BAD)。
3. 单测:`test_demand_dual_wiring` / native / config / sideregion 全绿。
4. tok/s 快测无回归(cap32 ~12.5)。
5. 纯搬移——不引入任何行为变化;Python 侧 `native_moe_ext` 导出符号集不变。

## 7. 完成标准

- 无 1200 行巨型文件;每个 .cpp 单一职责、目录名即边界。
- 全部验收项通过。
- 单条 commit(纯重构,不夹带逻辑改动)。
