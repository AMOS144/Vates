# 双源 acquire 回退路径去 host 胶水 设计 (Tier 1 + Tier 2)

日期：2026-07-02
状态：设计已确认（用户选择「降本」+「可容忍略多抖动」），待评审

## 1. 背景与问题

MTP verify 现已走 `acquire_gpu_dual` 双源路径（cap=32 真实区 + 32 侧区）。实测 K=3 时
`spec_hit_rate ≈ 0.90`，但 tok/s 仅 ~11。根因已用探针确认：

- `acquire_gpu_dual` 里 `n_miss == 0` 才走全 GPU 快路径；只要该层**任一路由专家 miss**，
  整层落 host 慢路径（`gpu_fallback`）。
- hit 是按「路由（专家×token）」算，慢路径按「层」触发。每层 unique 路由 ~13，
  `0.9^13 ≈ 0.25` → 约 75% 的层至少一个 miss。**实测 fastpath=463 / fallback=1121 →
  回退占比 70.8%**，与估算吻合。
- 慢路径的额外代价**不是重读全部专家**（回退只 pread 真 miss，通常 1~3 个），而是：
  1. `flat = inds.reshape(-1).tolist()`（整批 ~40 个）——一次 GPU→CPU 同步 + Python list
  2. `sk = set(int(k) for k in keys.tolist())`（侧区 ~32 个）——又一次同步 + Python set
  3. 两个 Python comprehension（成员判断 / 过滤）
  4. `acquire()` 内再 dedup + note_access
  5. 重建 eff、再 take
- 探针佐证：预取 host 开销仅占 0.23% wall（不是瓶颈）；width 扫描里 disk 翻倍 → tok/s 掉 36%
  （读盘在关键路径）。所以要提速的是「70.8% 层 × 主线程 Python 停顿」这块，让 GPU 流水线更连续。

## 2. 核心洞察（决定了改动范围很小）

回退里 `local = mx.take(eff, inds)`，而 **eff 已经把侧区 kv 叠加进去了**
（`eff[keys] = vals`）。因此：

> `local < 0` 的位置**恰好就是真 miss**（既不在真实区、也不在侧区）。

推论：
- 真 miss 的专家 id 可直接在 GPU 上取：`miss_ids = inds.reshape(-1)[local.reshape(-1) < 0]`，
  只把这几个（通常 1~3 个）拉回 host。
- 现有的 `flat`（整批 40）与 `sk`（侧区 32 的 set）**完全不需要**——侧区成员已经编码进 eff。

这是整个降本方案的地基：Tier 1 靠它把两次大同步 + 两个 Python 循环砍成「一次小 miss 同步」。

## 3. 目标与非目标

**目标**
- 消除 `acquire_gpu_dual` 回退分支里的主线程 Python 胶水（整批 `.tolist`、侧区 `keys.tolist`、
  成员/过滤循环），只保留「几个真 miss id」的最小同步。
- 保持数值等价（bit-exact 优先）；LFU 记账允许在「可容忍略多抖动」前提下做近似。
- Tier 2：把 miss 提取 + pread + 落池 + 补表整个塞进一次 C++ 调用，连那几个 miss id 都不回 Python。

**非目标**
- 不追求「零 per-layer 主线程同步」（那需要投机执行 + 容忍读错重补，属激进方案，本次不做）。
- 不改预测/预取逻辑、不改侧区 LFU 淘汰策略、不改 cap/spec 配置。
- 不改快路径（`n_miss == 0`）——它已是零 host 往返 + 一次 `mx.sum` 同步。

## 4. Tier 1 设计（纯 MLX 重写回退，最低风险）

改 `ResidentExpertPool.acquire_gpu_dual`（`mlx_streaming/core/cache/resident_pool.py`）的回退分支
（当前 540-554 行）。

**新回退流程**（`n_miss > 0` 分支）：
1. `flat_local = local.reshape(-1)`；`miss_mask = flat_local < 0`（GPU）。
2. `miss_ids_arr = inds.reshape(-1)[miss_mask]`（GPU 布尔索引）；
   `miss_ids = [int(e) for e in miss_ids_arr.tolist()]`（**唯一同步，仅几个元素**）。
3. `self.acquire(layer, miss_ids)`：复用现有 demand 读盘（`stacked_batch_loader` /
   `native_demand_loader`，本就是 C++ pread）补进真实区、更新 `_slot_table`。
   - `acquire` 内部会 dedup、note_access（对 miss_ids）、pread、落池、`_set_table`。
4. hit 计数：`self.hits += int(inds.size) - int(miss_mask.sum())`（GPU 计数，无额外 host 循环；
   位置口径与快路径 `self.hits += int(inds.size)` 一致）。
5. 重算 local：`base = self._slot_table[layer]`；`eff = mx.array(base) if has_side else base`；
   `if has_side: eff[keys] = vals`；`local = mx.take(eff, inds)`。
6. `return self._pools[layer], local`。

**砍掉**：`flat = inds...tolist()`（40）、`sk = set(keys.tolist())`（32）、两个 Python 成员循环。
**新增**：`miss_ids_arr.tolist()`（~1-3 个）+ 一次 `miss_mask.sum()`（可与步骤 4 合并复用）。

### 4.1 LFU 记账决策（本设计的关键取舍）

现状：回退给「侧区命中」（当前 545/548 行）和「真实区命中」（`acquire` 内 `note_access`）都补 LFU 频次，
避免高频常驻专家被过早驱逐。Tier 1 只把 miss 拉回 host，拿不到「命中的 id」。

决策（用户已选「可容忍略多抖动」）：
- **真实区命中的频次 bump 省略**（不为精确 LFU 把全部命中 id 拉回 host，否则白降本）。
  影响：LFU 频次仅由「被 acquire 的 miss 专家」+「快路径命中」累积；真实区里「本次靠回退层命中但未被 acquire」
  的专家这一步不 bump。属启发式精度损失，不影响正确性，符合已接受的抖动预算。
- **侧区命中的频次 bump 下沉 C++**：侧区 `SideLayer` 已维护 `freq` map（native_prefetch.cpp）。
  Tier 1 阶段可先不动（侧区 freq 在 `reserve/read_publish` 已有累积）；Tier 2 落地时由 C++ 统一维护。

> 备注：`self._note_access` 仅在 `eviction_policy == "lfu"` 时生效；LRU 模式下本决策无影响。

## 5. Tier 2 设计（native 融合，miss id 也不回 Python）

新增一个 native 函数，把「miss 提取 + pread + 落池 + 补表」收进一次 C++ 调用，进一步消除
Tier 1 里那次小 `.tolist()` 与 Python 侧 `acquire` 调度。

**接口（native_moe_ext）**
```
dual_fallback(layer:int, inds:mx.array(uint32), slot_table:mx.array(int32),
              side_keys:mx.array(uint32), side_vals:mx.array(int32),
              pool_tensors:list[mx.array], seg_meta, path:str, stride:int,
              cap:int, num_experts:int) -> mx.array(local int32)
```
C++ 内部：
1. 用 `slot_table` 叠加 `side_keys→side_vals` 得 eff（或直接在 gather 时判两级），对 `inds` 求 `local`。
2. 找 `local < 0` 的 unique miss expert id（C++ set，不回 Python）。
3. 对 miss 批量 `pread`（复用 `blob_load` 的 pread + F_NOCACHE 逻辑），落进真实区空闲/驱逐槽，
   同时原地改 `slot_table[e] = slot`（与 Python `_set_table` 等价）。
4. 维护真实区 LFU（`freq`）与驱逐（复用侧区已有的 LFU 数据结构思路）。
5. 返回修正后的 `local`（int32，形状同 inds）。

Python 侧 `acquire_gpu_dual` 在 `config.native_dual_fallback()`（新开关，默认 off）为真时，
回退分支直接调 `dual_fallback(...)` 拿 local，其余（hit 计数、返回 pool）保持。

**范围控制**：Tier 2 只处理「真实区空闲槽足够」的常见情形；若 miss unique 数 + 现有占用 > cap
（罕见，decode top-k ≤ cap 恒成立，仅极端 verify 批量可能触发），回退到 Tier 1 的 Python 路径
（native 函数返回哨兵，Python 侧检测到即走旧逻辑），保证不破坏正确性。

## 6. 确定性与正确性

- Tier 1 与旧路径**数值等价**：miss 集合相同（`local<0` ≡ 旧 `e not in sk and e not in slot_of`），
  pread/落池/补表复用同一 `acquire`，`local` 由同一 eff 重算 → bit-exact。唯一差异是 LFU 频次
  bump 少了「真实区命中」一项，可能改变后续驱逐顺序 → 极小概率影响哪些专家常驻 → 属已接受的抖动。
- Tier 2 数值等价性由「native pread 字节 == numpy stacked」保证（已有
  `test_native_demand_loader_equiv.py` 佐证 `blob_load` 等价）；`local` 映射与 Python 一致由单测校验。

## 7. 成功标准（验收）

1. **正确性**：新增单测覆盖 Tier 1/Tier 2 的 miss 提取、落池、local 映射，与旧路径逐元素一致。
   现有 `test_resident_sideregion.py` / `test_dual_source_verify_shape.py` 全绿。
2. **降本可见**：K=3 dual+LFU 默认配置下，回退路径主线程 host 时间下降；tok/s 相对当前 ~11
   有净提升（目标 ≥ +5%，以 REPEAT=2 背靠背为准）。若 Tier 1 提升不足再看 Tier 2。
3. **确定性不劣化到不可接受**：`n_mismatch` 保持个位数~低十位数量级（与当前同档）。
4. 内存峰值不显著上升（±0.1GB 内）。

## 8. 测试策略

- **单元（Tier 1）**：构造池 + `_Side.kv` mock（沿用 `test_resident_sideregion.py` 风格），
  断言回退后 `local` 与旧实现逐元素相等、`gpu_fallback` 计数、hit 计数、真 miss 落 `[0,cap)`。
- **单元（Tier 2）**：native 编译存在时测 `dual_fallback` 返回的 local 与 Python 路径一致；
  未编译则 skip（沿用 `test_native_demand_loader_equiv.py` 的 skip 约定）。
- **等价性回归**：一段固定 prompt greedy 解码，Tier1 on/off 的 token 序列一致（bit-exact）。
- **端到端**：`run_mtp_spec` K=3 REPEAT=2，记录 tok/s、hit、fallback 占比、n_mismatch、峰值内存。

## 9. 风险与回退

- 布尔索引 `inds[mask]` 变长输出 → 一次 size 同步（不可避免，正是那唯一小同步）。
- Tier 2 C++ 落池/驱逐要与 Python `_place_expert` 语义完全对齐（槽分配、驱逐选择），否则字节错位。
  用 `STG_VERIFY` 风格的逐段字节校验兜底；有哨兵回退到 Tier 1 保正确。
- 所有改动经 `config` 开关门控（Tier 2 默认 off），可一键回退旧路径。

## 10. 开关

- Tier 1：直接替换回退实现（等价，无需开关）；若保守可加 `DUAL_FALLBACK_GPU_MISS`（默认 on）。
- Tier 2：`NATIVE_DUAL_FALLBACK`（默认 off，opt-in）。
