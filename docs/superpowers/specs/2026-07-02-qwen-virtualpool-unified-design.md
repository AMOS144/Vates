# VirtualPool 统一专家管理抽象 设计

日期：2026-07-02
状态：设计草案（待 Phase 0 go/no-go 实测 + 用户对「硬问题 3 槽记账归属」拍板）
关联：`2026-07-02-qwen-dual-fallback-native-design.md`（Tier1/Tier2 降本）、
`2026-06-25-qwen-virtual-pool-double-buffer.md`（双源双缓冲）

---

## 0. 一句话

把现在散落在 `block.py`（fastpath / fallback / host 三分支）+ `ResidentExpertPool`
（Python 槽分配/驱逐/记账）+ `native_prefetch.cpp`（侧区读盘）里的专家管理，
统一收口进 `VirtualPool`：**对计算侧只暴露一个 `acquire(layer, inds, num_experts)`，
呈现「所有专家都在」的视角**，内部自己完成命中直取 + 未命中 demand 补齐，
最终目标是**从根本上消除每层主线程的 GPU→CPU 同步**。

本设计诚实结论先行：**收益上限约 +15%，风险高**，因此本 spec 把改造切成三段，
每段独立可验证、可回退，并在真正高风险处（把 demand 落池搬进惰性 Primitive）
明确停下等用户拍板。

---

## 1. 背景与问题（动机，来自实测证据）

当前配置 `K=3` MTP verify、dual（真实区 cap32 + 侧区 LFU cap32），暖缓存约 **11 tok/s**。
探针已定位瓶颈构成：

- **I/O 不是瓶颈**：暖 page cache 11.365 vs 冷 11.31，几乎相同 → 预取已盖住物理读盘。
- **预取 host 开销仅 0.23% wall**（`PREFETCH_TPROF` 实测）。
- **真瓶颈之一：~70% 的层走 host 回退**。hit 按「路由（专家×token）」算 ≈0.90，
  但慢路径按「层」触发：每层 unique 路由 ~13，`0.9^13 ≈ 0.25` → 约 70% 层至少一个 miss
  （实测 `gpu_fastpath=520 / gpu_fallback=1064`，回退占比 ≈67%）。
- 每个回退层付出：
  1. `acquire_gpu_dual` 里 `n_miss = int(mx.sum(...))` —— **一次 GPU→CPU 同步（每层都有，含快路径）**；
  2. 回退分支 `miss_or_neg.tolist()` —— **又一次同步**（Tier1 后已缩到只取几个 miss id）；
  3. Python `acquire()`：dedup + note_access + 批量 pread + 落池 scatter + `_set_table` 记账；
  4. eff 重建 + 再 `mx.take`。
  这些主线程停顿**打断 GPU 流水线**，是慢的主因。
- **该 CPU 开销天花板约 15%**：`cap64` 把回退 68%→53%，暖缓存 tok/s 11.37→13.05（+15%）。
  其余是 **GPU 计算硬地板**（3B 激活 × K+1=4 token × 48 层），软件改不动。
- 已试过一个 Tier1（`mx.where` 只取 miss id、砍侧区 `keys.tolist`+set）：
  **无可测提速** → 成本不在小 Python 胶水，而在「每层那次同步 + 落池调度打断流水线」。

> **ROI 诚实判断**：收益上限约 +15%（把 67% 回退层的 per-layer 同步/落池全消掉，
> 逼近 GPU 计算地板），风险高（要把 demand 落池搬进 eval/回调线程，涉及排序依赖与
> 槽状态并发）。因此**必须先做 Phase 0 证伪**，再决定是否值得。

---

## 2. 现状代码地图（改造对象）

| 组件 | 文件 | 现职责 | 痛点 |
|---|---|---|---|
| 计算侧调用点 | `core/moe/block.py:147-236` | dual / GPU-remap / host 三条分支各自算 `local`、各自 promote/acquire | 三分支重复、分支判据散、每条都含 `.tolist`/`int(mx.sum)` |
| 双源协调 | `core/cache/virtual_pool.py` | `begin_forward`/`read_gen`/`fill_gen`/`acquire`/`prefetch`/`target_for` | `acquire` 只是薄转发到 `acquire_gpu_dual`，未真正「收口」 |
| 常驻池 | `core/cache/resident_pool.py` | `acquire_gpu_dual`（快/慢路径）、`acquire`（host demand）、`_alloc_slot`/`_choose_victim`/`_set_table`/`_note_access` | **槽分配/驱逐/LFU 记账全在 Python**，是 per-layer 同步后必跑的主线程活 |
| 侧区读盘 | `native/ext/native_prefetch.cpp` | `PrefetchPoolSideRegionPrimitive`：GPU 完成回调里读惰性 id + reserve 侧区行 + pread + memcpy + publish `e2r`；`blob_load`；`sideregion_kv` | 已证明「不回主线程同步就能读 id + pread」可行，但仅用于**预取**（fire-and-forget），未用于 **demand**（关键路径、matmul 立刻要字节） |

关键既有事实（决定可行性）：
- `PrefetchPoolSideRegionPrimitive::eval_gpu` 已经在 `addCompletedHandler`（**回调线程**）里
  读惰性 `idp`（此刻 command buffer 已完成、id 已算好）、`reserve` 侧区行、`read_publish`
  （pread + memcpy + 发布）。这是「零主线程同步读 id + 落盘字节」的**现成机制**。
- `MaterializeSpikePrimitive`（Task1 spike）已验证「Primitive 输出别名/依赖 → 下游 `take` 建立
  确定性依赖边」这一 MLX 机制成立（`test_ingraph_alias_spike.py`）。
- `preallocate(layer, sample, cap)` 已支持「满 cap 预分配 + eval 固定 data 指针」，
  之后「槽由调用方 mutate `_slot_of`+`_set_table`、字节由 C++ pread 写入」——
  这正是 demand-into-primitive 需要的池布局前提。

---

## 3. 用户设计想法（要评估并实现的对象）

> 每层计算时向 `VirtualPool` 请求「这层要的 N 个专家」，`VirtualPool` 对外呈现「所有专家都在」
> 的视角（计算侧看不到 miss/回退）。内部：命中的（真实区 + 侧区 dual pool）直接给，
> 未命中的（如 13 个里 3 个不在）由 `VirtualPool` 内部调度 demand 读盘补齐。
> 目标：从根本上消除每层主线程的 GPU→CPU 同步；把 `block.py` 三分支收敛成一次 `VirtualPool.acquire`。

这个想法拆成**两个可分离的诉求**，它们的风险/收益完全不同，spec 必须分开对待：

- **诉求 A（API 抽象 / 收口）**：计算侧零分支、只调一次 `acquire`。
  —— **低风险、0 性能变化**，是纯重构。可独立交付。
- **诉求 B（真正消除 per-layer 同步）**：把 demand 落池搬进惰性 Primitive，
  在 `mx.eval`/回调线程执行，主线程只搭图。
  —— **高风险、收益上限 +15%**，涉及排序依赖 + 槽状态并发（硬问题 2/3）。

**结论：A 先做（Phase 1 MVP），B 待 Phase 0 证实 + 用户对硬问题 3 拍板后再做（Phase 2）。**

---

## 4. 目标与非目标

**目标**
- 计算侧（`block.py`）对专家管理**零分支**：一次 `VirtualPool.acquire(layer, inds, num_experts)
  -> (pool_arrays, local)`，内部完成 dual gather + demand 补齐 + 记账。
- 数值等价：与现 dual 路径逐元素一致（Phase 1 bit-exact；Phase 2 允许已接受的 `n_mismatch` 抖动，不显著劣化）。
- （Phase 2）消除回退层的 per-layer GPU→CPU 同步与主线程落池调度，逼近 GPU 计算地板。

**非目标**
- 不改预测/预取逻辑、不改侧区 LFU 淘汰策略、不改 cap/spec/预取 budget 配置。
- 不改 prefill（大 seq、唯一专家可能超 cap）的既有 host/fetch 回退路径 —— 它天然需要一次同步且不是热路径瓶颈；VirtualPool 对 prefill 只做透明转发。
- 不追求「投机执行 + 容忍读错重补」这类激进方案。
- 不做与本目标无关的重构（如重排 `ResidentExpertPool` 其它方法）。

---

## 5. 六个硬问题逐条分析

### 硬问题 1：真正消同步的途径是否可行？

**诉求**：把 dual gather + demand 补齐塞进 MLX 惰性 Primitive，`mx.eval` 时于 eval/回调线程执行，
主线程只搭图。对 **demand**（关键路径，matmul 立刻要字节）是否可行？

**分析**：
- 预取侧已证成立：`PrefetchPoolSideRegionPrimitive` 在完成回调里读惰性 id + pread + memcpy。
- demand 与预取的**本质差别**：预取是 fire-and-forget（这一前向没读到就下一前向再说）；
  demand 的字节**必须在同一前向内、被本层 matmul 消费之前**写进池槽。
- MLX 惰性图机制支持这点：让 demand primitive **输出**一个「已补齐的 `local`（或池数组别名）」，
  matmul 消费该输出 → 建立依赖边 → MLX 保证 primitive 的 `eval_gpu` 先于 matmul 执行
  （`MaterializeSpikePrimitive` spike 已验证「fill primitive 输出 → 下游 gather」这条边）。
- **关键约束**：demand 的 pread + memcpy 必须发生在 primitive 的 `eval_gpu` **主体**里（同步、
  在该 command buffer 依赖链上），**不能**像侧区那样丢进 `addCompletedHandler` 后台线程再返回
  —— 后台是 fire-and-forget，matmul 不会等它。也就是说 demand 的 I/O 落在 GPU 提交线程上
  同步跑（一次 primitive 内），但**不回主 Python 线程**，从而消除 per-layer 主线程停顿。

**可行性结论**：可行，但 demand primitive 必须**同步**完成 pread+落池（在 eval 线程，非主线程），
并把结果接入图。**这不消除 I/O 时间本身，只把 I/O + 落池从主线程搬到 eval 线程**，
使主线程不再每层停顿去 `.tolist()` + Python 落池。由于实测 I/O 不是瓶颈、且回退只读 1~3 个 miss，
搬走的正是「主线程停顿」这块。

### 硬问题 2：排序风险（matmul 必须排在 demand 落池之后）

- demand primitive 的**输出必须被 matmul 消费**，才能建立「先落池、后 matmul」的依赖边。
  两种建边方式：
  - **(推荐) 输出修正后的 `local`**：demand primitive 输入 `(inds, slot_table, side_kv, pool_arrays...)`，
    输出 `local`（int32，形同 inds）。matmul 用 `local` gather 池 → 天然依赖 demand 完成。
    池数组作为 primitive 的**输入**（被原地写），MLX 会把它标记为该 primitive 的依赖；
    matmul 读池数组时，因 `local` 依赖 demand、池被 demand 写过，排序成立。
  - (备选) 输出「池数组别名」：demand primitive `copy_shared_buffer` 池 buffer 后原地写目标行，
    输出别名池数组，matmul 直接用别名 → 更强的 buffer 级依赖。复杂度更高（多输出别名），暂不选。
- **绝不 fire-and-forget**：demand 与侧区预取不同，pread 必须在 primitive eval 主体内同步完成，
  返回前池槽已就绪。
- 风险点：MLX 对「primitive 原地写输入 buffer」的别名/依赖语义需实测确认（spike 已初步验证，
  Phase 2 第一步要专门写一个 demand-spike 单测钉死这条边）。

### 硬问题 3：槽记账归属（**本设计最大待拍板点**）

落池要：分配槽 + 可能驱逐 + 更新 `_slot_of`/`_free`/`freq`/`_slot_table`（现全在 Python）。
Phase 2 要把落池搬进 eval 线程，槽状态归属有三个方案：

- **方案 A：主线程预留槽，C++ 只填字节（推荐）**
  - 主线程在搭图时（**不同步**、纯 Python dict 操作，不碰 GPU）先为「本层可能 miss 的专家」
    预留槽：复用 `_alloc_slot`（free 优先 / LFU 驱逐），得到 `(expert, slot)` 计划表 + 更新
    `_slot_of`/`_slot_table`。C++ demand primitive 只按计划表 pread+memcpy 进指定槽。
  - **难点**：搭图时主线程还不知道哪些是 miss（那正是要消除的同步！）。
    → 破解：**主线程对「本层全部 unique 路由专家」都预留/确认槽**（命中的已在池、直接拿槽；
    未命中的分配空槽）。这不需要 GPU 同步——unique 路由专家 id 主线程本来就有（`inds` 是路由结果，
    但 `inds` 是 GPU lazy…）。**这里有个真问题**：`inds` 是 GPU 上算出的 lazy 数组，
    主线程要拿到 unique id 就得 `.tolist()`（即那次同步）。
    → 只有在**主线程已持有路由 id（host 侧）**时方案 A 才零同步。当前 decode/verify 路径 `inds`
    是 GPU lazy 的。除非上游把路由 argpartition 的结果以别的方式提供（超出本次范围）。
  - **权衡**：正确性最稳（槽状态仍单线程 Python 管理，无并发），但**无法零同步**——
    仍需一次 `.tolist()` 拿 unique 路由。它消除的是「落池 scatter/记账」那部分主线程活，
    保留「一次小同步取 unique id」。**收益打折，但风险最低。**

- **方案 B：C++ 接管槽状态（最彻底、最高风险）**
  - 把 `_slot_of`/`_free`/`freq`/`e2r` 的真实区版本搬进 C++（类似侧区 `SideLayer`）。
    demand primitive 在 eval/回调线程里：读惰性 `inds` → 算 `local`（叠加 C++ 真实区表 + 侧区表）→
    找 miss → C++ 内分配槽/驱逐/pread/memcpy/更新表 → 输出修正 `local`。
    主线程**完全不 `.tolist()`、完全不碰槽状态**。
  - **收益**：真正零 per-layer 主线程同步，逼近上限。
  - **风险**：
    1. 真实区表要与 Python 侧任何仍读它的路径（stats、trace、prefetch_cpp 槽分配）保持一致 →
       要么 Python 侧全部改走 C++ 查询，要么维护双份（易漂移）。
    2. LFU 驱逐语义要在 C++ 精确复刻 `_choose_victim`（freq + LRU tie-break + pinned/current 保护），
       否则字节错位 / 驱逐了本次要用的专家。
    3. demand 与预取（也写真实区？不，预取写侧区）、与 `prefetch_cpp`（写真实区槽）并发 →
       需要 C++ 侧锁 + 与主线程分配器互斥。当前 `prefetch_cpp` 在主线程分配真实区槽，
       若 demand 也在 C++ 分配真实区槽，两者抢槽需统一到一套分配器。
  - 这是「把 `ResidentExpertPool` 的核心状态机迁到 C++」，是最大的工程与正确性风险。

- **方案 C：混合（Phase 2 先做「主线程预留 + C++ 填字节」，把同步缩到一次极小 `.tolist`）**
  - 即方案 A 的务实版：接受「一次 unique 路由 `.tolist()`」（几十个元素，Tier1 已证其本身不贵），
    但把**落池 pread+memcpy+scatter 全搬进 C++ demand primitive**（在 eval 线程），
    并把 LFU 记账降级为「只对 miss 记账」（Tier1 已接受的近似）。
  - 消除的是「主线程落池 scatter + 记账 + eff 重建」，保留「一次小同步」。
  - **推荐作为 Phase 2 的第一步**：风险中等、收益部分兑现、不动 C++ 槽状态机。
    若实测仍不够（那次 `.tolist` + 图依赖开销吃掉收益），再评估方案 B。

> **待用户拍板**：Phase 2 走 **方案 C（主线程预留槽 + C++ 填字节，保留一次极小同步）**
> 还是 **方案 B（C++ 完全接管槽状态，零主线程同步但重写状态机）**？
> spec 推荐 **C 优先**（风险可控、可增量验证），B 仅当 C 实测收益不足再上。
> 这是根本性抉择，需用户明确选择后才写 Phase 2 plan。

### 硬问题 4：确定性

- 当前 MTP 已接受个位~低十位 `n_mismatch` 抖动（dual 有良性时序噪声，字节校验 0 BAD）。
- Phase 1（纯 API 收口）：**bit-exact**，`n_mismatch` 不变。
- Phase 2（方案 C）：LFU 记账近似（只对 miss bump）与 Tier1 已落地的取舍一致，
  抖动预算不变；C++ demand 落池的字节等价由「pread 字节 == numpy stacked」保证
  （已有 `test_native_demand_loader_equiv.py` 佐证 `blob_load`）。
- 验收线：`n_mismatch` 保持与当前同档（个位~低十位），不显著劣化；字节校验（`STG_VERIFY` 风格）0 BAD。

### 硬问题 5：API 抽象（VirtualPool 对外接口 + block.py 收敛）

**新对外接口**（`VirtualPool`）：

```python
def acquire(self, layer, inds, num_experts) -> tuple[dict, mx.array]:
    """呈现「所有专家都在」的视角：返回 (pool_arrays, local)。
    内部：真实区表 ∪ 侧区(读代) → gather；miss 由内部 demand 补齐（Phase 2 惰性化）。
    计算侧零分支：命中/未命中、快/慢路径都在此收口，调用方只拿 (pool, local) 喂 matmul。"""
```

`block.py` 现状三条分支（decode/verify GPU-remap dual、host、超容量 fetch）如何收敛：

- **dual 路径**（`zerocopy_dual_source` 且 verify/decode 可 GPU 重映射）：
  当前 `self._vpool.acquire(...)` 已是入口 → Phase 1 把「读代 side、调 `acquire_gpu_dual`」
  这段逻辑保持在 `VirtualPool.acquire` 内即可（现已如此），**block.py 侧无变化**。
- **非 dual 的 GPU-remap 路径**（`store.acquire_gpu`）：Phase 1 让 `VirtualPool.acquire`
  在「无 staging/非 dual」时内部转发到 `acquire_gpu`，block.py 只调 `_vpool.acquire`。
- **host 路径 + 超容量 fetch**：prefill/大 seq。VirtualPool.acquire 内部判 `seq/uniq>cap`
  → 转发到现有 host `acquire` 或 `fetch`（透明），block.py 不再自己分支。
- 收敛后 block.py 计算段伪代码：

```python
# 收敛后：计算侧零分支
pool_arrays, local, n_experts = self._vpool.acquire(self.layer_idx, inds, gates.shape[-1], x_seq=x.shape[1])
y = self._sub.forward(pool_arrays, n_experts, x, local)
```

（`n_experts` 由 VirtualPool 依路径返回 `cap+侧区` 或 `layer_cap` 或 `len(uniq)`；
promote 调用也一并收进 VirtualPool，block.py 不再直接碰 `_stg_mgr`。）

> **抽象边界检验**：收敛后 block.py 不需要知道 dual/host/fetch 的存在，也不需要 `_slot_table`/
> `side.kv`/promote 细节；VirtualPool 内部可自由改（Phase 2 惰性化）而不动 block.py → 边界干净。

### 硬问题 6：最小可行切法（MVP）+ 分阶段

- **Phase 0（go/no-go，写 plan 前必做）**：低成本证伪「消除 per-layer 同步真能提速」。
  见 §6。
- **Phase 1（MVP，低风险、可独立交付）**：**纯 API 收口**。把 block.py 三分支收敛进
  `VirtualPool.acquire`，行为 bit-exact、性能中性（±噪声）。价值：抽象边界成型、
  为 Phase 2 铺路、代码可维护性提升。**不消除同步，不改槽记账。** 走 TDD。
- **Phase 2（真消同步，需用户对硬问题 3 拍板）**：按选定方案（推荐 C）把 demand 落池
  搬进惰性 Primitive。走 TDD，先 demand-spike 钉死排序依赖边，再落地、A/B 实测。

---

## 6. Phase 0：go/no-go 实测设计

**目的**：在大改前，用最低成本证伪/证实「消除 per-layer 同步真能提速」这个前提。

**方法**：构造一条**近似「零 per-layer 同步」上界**的原型路径，量 tok/s 上界。

**具体探针**（throwaway 原型，Phase 0 用完即弃，不进主分支）：
在 `acquire_gpu_dual` 里 `local = mx.take(eff, inds)` 之后、`n_miss = int(mx.sum(...))` 之前，
加一个 env 门控的短路分支：

```python
if config.probe_all_hit_lazy():   # PROBE_ALL_HIT_LAZY=1
    # Phase 0 上界探针：跳过 n_miss 同步与 demand 回退，全当命中，整前向惰性搭图。
    # 输出数值会错（缺失专家读到脏字节/零），仅用于量「零 per-layer 同步」的 tok/s 上界。
    self.gpu_fastpath += 1
    return self._pools[layer], local
```

这消除了**每层**的两处主线程同步（`int(mx.sum)` + 回退 `.tolist()`）与落池调度，
使整前向 48 层惰性搭图、只在前向末尾（采样/verify 比较）eval 一次。

**为何是合理上界**：
- 缺失专家的 matmul 仍在跑（同 shape、同 kernel）→ GPU 计算量不变；只去掉了「同步 + 落池」。
- 实测 I/O 不是瓶颈（暖=冷）、回退只读 1~3 miss → 去掉 demand 读盘不虚高上界太多。
- 因此该路径 tok/s ≈「若 per-layer 同步/落池成本为 0」的理论上界。

**判据**：
- 跑 `run_mtp_spec` K=3 REPEAT=2（与基线同配置），对比 `spec_tok_per_s`。
- **上界相对基线 ≥ ~15%（接近 cap64 的 +15%）→ go**（值得为 Phase 2 投入）。
- **上界相对基线 5%~15% → 边界**（低风险 Phase 1 照做；Phase 2 谨慎、先方案 C 小步试）。
- **上界相对基线 < ~5% → no-go**：per-layer 同步不是瓶颈，成本在 GPU 计算地板，
  **停下报告用户，不硬做大重写**（Phase 1 API 收口仍可作为可维护性收益单独评估）。

**基线命令**：
```
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```
上界跑同命令加 `PROBE_ALL_HIT_LAZY=1`。看 `spec_tok_per_s / spec_hit_rate /
gpu_fastpath / gpu_fallback / n_mismatch / mlx_peak_gb`。

---

## 7. 成功标准（验收）

**Phase 1（MVP，API 收口）**
1. 正确性：新增/现有单测（`test_virtual_pool.py`、`test_resident_sideregion.py`、
   `test_dual_source_verify_shape.py`、`test_dual_fallback_gpu_miss.py`）全绿。
   新增单测断言「VirtualPool.acquire 在 dual / gpu-remap / host 三种输入下返回的
   `(pool, local)` 与旧 block.py 分支逐元素一致」。
2. 等价性：一段固定 prompt greedy，收口前后 token 序列 bit-exact（`exact_match=true`）。
3. 性能中性：`run_mtp_spec` tok/s 相对基线在噪声带内（不回退）。
4. 抽象边界：block.py 计算段无 dual/host/fetch 分支、不直接引用 `_slot_table`/`side.kv`/`_stg_mgr`。

**Phase 2（真消同步，条件性）**
1. 正确性：demand primitive 单测证「落池后 matmul 读到正确字节」（排序依赖成立），
   与 Python demand 路径逐元素一致。
2. 降本可见：tok/s 相对基线净提升（目标 ≥ Phase 0 上界的一半，REPEAT=2 背靠背）。
3. 确定性：`n_mismatch` 保持个位~低十位；字节校验 0 BAD。
4. 内存峰值不显著上升（±0.1GB）。

---

## 8. 测试策略

- **单元（Phase 1）**：沿用 `test_resident_sideregion.py` 的 `_Side.kv` mock 风格，
  构造 `ResidentExpertPool` + fake side，断言 `VirtualPool.acquire` 三路径返回值与
  直接调 `acquire_gpu_dual`/`acquire_gpu`/`acquire` 一致。
- **单元（Phase 2）**：demand-spike——构造预分配池 + 惰性 demand primitive，
  断言 matmul（或 take）读到 pread 后的正确字节（排序边成立）；native 未编译则 skip
  （沿用 `test_native_demand_loader_equiv.py` skip 约定）。
- **等价性回归**：固定 prompt greedy，改动 on/off token 序列一致。
- **端到端**：`run_mtp_spec` K=3 REPEAT=2，记录 tok/s、hit、fastpath/fallback、n_mismatch、峰值内存。
- 注意：MLX 0.31.2 **不支持布尔索引** `arr[mask]`，一律用 `mx.where` / `mx.take` 变通。

---

## 9. 风险与回退

- Phase 1 是纯重构，风险低；用「收口前后逐元素一致」单测 + bit-exact 端到端兜底。
- Phase 2 demand-into-primitive 的**排序依赖**若不成立（matmul 读到未落池的脏字节）→
  先用 demand-spike 单测证伪；不成立则 Phase 2 不可行，回退到「主线程一次小同步 + Python 落池」
  （即今天的 Tier1 路径），API 抽象仍保留。
- 所有 Phase 2 改动经 config 开关门控（默认 off），可一键回退旧路径。
- 硬问题 3 若选方案 B（C++ 接管槽状态），额外风险：LFU 驱逐语义漂移、与 `prefetch_cpp`
  抢槽、双份状态不一致 → 必须逐段字节校验（`STG_VERIFY` 风格）+ 与 Python 分配器统一。

---

## 10. 开关

- Phase 0：`PROBE_ALL_HIT_LAZY`（throwaway，实测后删除，不进主分支）。
- Phase 1：无需开关（等价重构直接替换 block.py 调用点）；保守可加 `VPOOL_UNIFIED`（默认 on）。
- Phase 2：`NATIVE_DEMAND_PRIMITIVE`（默认 off，opt-in）。

---

## 11. 待用户拍板点（汇总）

1. **Phase 0 go/no-go 结果**：若 < ~5% 上界 → 是否仍要做 Phase 1（纯可维护性收益）？
2. **硬问题 3 槽记账归属**：Phase 2 走 **方案 C（主线程预留槽 + C++ 填字节，保留一次极小同步，
   推荐）** 还是 **方案 B（C++ 完全接管槽状态，零同步但重写状态机，高风险）**？
