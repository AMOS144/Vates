# 设计：连续常驻专家池（消除 per-token `mx.stack`，加速流式 MoE 热路径）

日期：2026-06-07
状态：待评审

## 1. 背景与目标

`Qwen3-Next-80B-A3B` 专家流式方案当前单 token 前向 ≈ 202ms（48 个 MoE 层，2-bit 专家），
物理天花板 ~7.5 tok/s。对热路径做分段 profile（`probe_hotpath.py`，2-bit、warm、N=20）得到：

| 段 | ms/step | 占 MoE | 含义 |
|---|---:|---:|---|
| route | 37.2 | 19% | gate softmax + argpartition（512 选 10） |
| pyremap | 3.6 | 2% | Python uniq/remap |
| **fetch** | **103.9** | **53%** | `store.fetch`：LRU 取专家 + **每次 `mx.stack` 把选中专家堆成连续张量** |
| matmul | 32.4 | 17% | `SwitchGLU` 的 `gather_qmm`（真正的矩阵乘） |
| combine | 18.4 | 9% | 加权合并 + 共享专家 |

**核心发现：瓶颈是 `fetch`（53%），不是矩阵乘（17%）。** 现状是每个 token、每一层都用
`mx.stack`（`expert_store._stack_picked`）把选中的专家权重重新拷贝成一块连续 `(n, O, I)`
张量再喂给 `gather_qmm`。这块内存搬运是大头，且 **baseline 单 token 解码也照付**。

### 1.1 目标

- 用**连续常驻专家池**替代「每次 `mx.stack`」，让命中时的 `fetch` 成本塌缩到接近 0。
- **不增加内存/SSD 占用**（池容量 = 现有 LRU 的每层槽数），不破坏已有流式/投机/缓存结论。
- **数值 bit 级等价**于现有 `SwitchGLU` 输出，保证贪婪 `exact_match` 不变。
- 收益同时作用于 **baseline 单 token** 与 **MTP 批量 verify** 两条路径（用户已确认「两者合一」）。

### 1.2 范围

- **做**：正确的连续 resident pool（避免整块拷贝）+ 与 `gather_qmm` 的零拷贝接口 + 数值/性能验证。
- **不做（推迟到独立 spec）**：完整 fused 反量化+SwiGLU Metal kernel（仅攻 17% matmul，ROI 待
  本 spec 数字确认后再评估）；不改路由、量化、共享专家逻辑；不改磁盘加载/pin/统计机制。
- **替换**：删除现有 `FileExpertStore.fetch_resident` 与 `streaming_moe` 里 `EXPERT_RESIDENT_POOL`
  实验分支（已证退化，原因见 §3）。

## 2. 现状缺陷分析（为什么旧 `fetch_resident` 退化）

旧实验路径 `FileExpertStore.fetch_resident` 在 miss 时执行：

```python
arrays[k][slot] = v     # MLX 数组函数式不可变
```

MLX 的 `array.__setitem__` 是 scatter，**几乎肯定每次都分配一块全新的整层 pool 张量**
（每层 96 专家 × ~0.4MB ≈ 40MB），把「省掉一次小 stack」换成「拷贝整块大 pool」，净亏。
本设计的核心就是**让 miss 写入只触碰单个专家（~0.4MB）而非整块 pool**。

## 3. 设计

### 3.1 数据结构（每层一个池）

新增 `ResidentExpertPool`（独立类，由 `FileExpertStore` 持有并委托）。每层维护：

- `pool: Dict[str, mx.array]`——按**完整参数名**分块，形如 `(capacity, *param_shape)`：
  `gate_proj.weight/scales/biases`、`up_proj.*`、`down_proj.*`。混合精度下各 proj 形状/bit
  不同，按参数名分块天然兼容。
- `slot_of: OrderedDict[expert_id -> slot]`——兼作 LRU 顺序。
- `free: list[int]`——空闲槽位。
- `capacity = EXPERT_SLOTS`（默认 96），与现有 LRU 容量相同 → **内存零增量**。

池**惰性分配**：首次 miss 用首个加载专家的形状/dtype 建 `(capacity, ...)` 全零张量。

### 3.2 命中/缺失数据流

`acquire(layer, expert_ids) -> (pool_arrays, slots)`：

- **命中**（`e in slot_of`）：记录 `slot_of[e]`，`move_to_end`（LRU touch），**不写池**。
- **缺失**：`mx.load` 单个专家 → 取 `free` 槽或淘汰 `slot_of` 头部（LRU 尾）→ **将该专家
  写入 `pool[param][slot]`（§3.4 的零拷贝写）** → 记 `slot_of[e]=slot`。

返回**稳定**的 `pool` 数组引用 + 与 `expert_ids` 对齐的 `slots` 列表。

### 3.3 与 `gather_qmm` 的零拷贝接口

把 `PersistentSubGLU` 的专家数固定为 `capacity`（恒 96），三个 `QuantizedSwitchLinear` 的
weight/scales/biases **直接指向 `pool` 数组**（引用稳定，仅 miss 时原地变更）；routing 的
`local` 张量 = 路由专家的 **slot id**（由 `acquire` 返回的 `slots` 重映射得到）。

`SwitchGLU(x, local)` / `gather_qmm` 按 `local` 只 gather 实际路由的少数专家（≈10），
**计算量仍等于 uniq，不是 capacity**。

结果：命中时既无 `mx.stack` 也无 `QSL.update`（权重引用恒定）；`fetch` 仅在 miss 时付一次
单专家写入。热缓存（~87% 命中）下 `fetch` 段大幅塌缩。

### 3.4 零拷贝写（唯一经验未知数 → 实现第 0 步 = de-risk）

整个方案成立的前提是「miss 写单个专家进池 ≈ 0.4MB，而非整块 ~40MB」。实现**第一步**写一个
~30 行微基准（`probe_pool_write.py`），在真实形状（`(96, 2048, 96)` 量级的 uint32 + scales/biases）
上实测三种写法的**单次 miss 写入**耗时与是否触发整块重分配：

1. `pool[param][slot] = v`（原地索引赋值）
2. `pool[param] = mx.slice_update(pool[param], v, start=(slot,0,0))` + 重新赋值（靠 buffer
   donation 复用旧 buffer）
3. 兜底：极小 `mx.fast.metal_kernel` 原地 scatter（= 方案 B）

**决策门**：谁的单次写 ≈ 单专家量级（且 N 次写后内存不线性膨胀）就用谁。微基准结果与选择
写回本 spec §6。预期 1 或 2 可行（纯 MLX，无自定义 kernel）；若都退化为整块拷贝则落 3。

### 3.5 与 `FileStreamingMoeBlock` 的集成

- `FileStreamingMoeBlock.__call__` 改为：`route → acquire(layer, flat) → local=slots → _sub.forward(pool_arrays, capacity, x, local)`。
- 删除 `EXPERT_RESIDENT_POOL` 开关与旧 `fetch_resident`；resident pool 成为文件后端**默认**路径
  （保留环境变量 `RESIDENT_POOL=0` 可回退到旧 stack 路径，用于对照/排错）。
- 共享专家（Qwen3-Next）仍常驻、逻辑不变。

## 4. 数值等价与测试

- **单测（bit 级等价）**：同一组专家 id，pool 前向输出 == 现有 stack 前向输出（`mx.allclose`
  零容差或 ≤1e-6）。保证 `exact_match` 不变。
- **单测（池语义）**：LRU 淘汰顺序、槽位复用、命中/缺失计数、`capacity >= top_k` 断言、
  惰性分配、混合精度逐 proj 形状正确。
- **集成验证**：
  1. `probe_hotpath`（2-bit）：`fetch` 段应显著下降，`per_token` 下降。
  2. baseline 单 token tok/s 前后对比（同 prompt/MAXTOK）。
  3. MTP K=2 `MTP_VERIFY_MODE=step` 复测 `exact_match` 仍为 true、speedup 变化。
- **回归**：现有 `tests/test_expert_store.py`、`tests/test_mtp_generate.py` 全绿。

## 5. 风险与回退

- **风险 1**：MLX 无便宜原地写 → 落 §3.4 方案 3（小 Metal scatter kernel），范围可控。
- **风险 2**：`gather_qmm` 对「大 capacity 权重 + 稀疏 slot 索引」的访存局部性差于「紧凑 stack」。
  → 集成验证 1 直接观测；若 matmul 段反升且抵消 fetch 收益，则评估把 slot 压紧（仅 gather 命中
  专家到临时紧凑视图）——但这会回到 stack，作为最后手段。
- **回退**：`RESIDENT_POOL=0` 一键回到现有 stack 路径，零风险对照。

## 6. de-risk 结论（已实测）

`probe_pool_write.py`（2-bit 真实形状，weight `(cap,2048,48)` uint32 + scales/biases）实测：

| 写法 | cap=32 | cap=96 | cap=384 | 是否随 cap 增长 |
|---|---:|---:|---:|---|
| **`pool[k][slot]=v`（原地）** | 0.364 | 0.395 | 0.330 | **否（恒定）** |
| `mx.slice_update` 重赋值 | 0.356 | 0.364 | 0.456 | 略增（大容量有拷贝迹象） |

**判定：选用原地索引赋值 `pool[k][slot] = v`。** 单次写 ~0.38ms 且**与池容量无关**，证明它只触碰单个专家槽位、不拷贝整块池（整池 cap=96 约 37.7MB，若整块拷贝应随 cap 线性增长）。**方案 A 成立，无需 Metal scatter（附录 A 不启用）。** 该写仅在 miss（~13%）发生，命中路径零写入。

## 7. 实现收益（已实测，同机 A/B）

`run_mtp_spec`（2-bit，K=2，MAXTOK=96，`MTP_VERIFY_MODE=step`），`RESIDENT_POOL=0` vs `=1`：

| 指标 | 旧 stack | 新池 | 变化 |
|---|---:|---:|---:|
| baseline tok/s | 7.65 | **9.12** | **+19%** |
| 峰值 MLX | 9.50 GB | **7.08 GB** | **-26%** |
| RSS | 6.26 GB | 4.39 GB | -30% |
| exact_match | ✅ | ✅ | 不变 |

`probe_hotpath`：`fetch` 89.4→71.9 ms/step、`per_token` 183→162.6 ms。消除 per-token `mx.stack`
带来 baseline +19% 与峰值内存 -26%，数值精确。MTP K=2 仍 0.90x（线性注意力 step verify 不摊薄，
与 MTP 报告结论一致）。fused matmul kernel（方案 C，仅攻 17% matmul）ROI 有限，暂不启动。
