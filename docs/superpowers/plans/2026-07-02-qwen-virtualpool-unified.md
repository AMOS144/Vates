# VirtualPool 统一专家管理抽象 实施计划

> **给执行者：** 必需子技能：用 superpowers:subagent-driven-development（推荐）或
> superpowers:executing-plans 逐任务实施。步骤用 checkbox（`- [ ]`）跟踪。

**目标：** 把 `block.py` 里 fastpath/fallback/host/fetch 四条专家取用分支收敛成一次
`VirtualPool.acquire`，让 `VirtualPool` 对外呈现「所有专家都在」的视角；为 Phase 2
（消除 per-layer 同步）铺好抽象边界。

**架构：** Phase 1（本计划主体，MVP）纯 API 收口，行为 **bit-exact**、性能中性；
Phase 2（消同步）在用户对「硬问题 3 槽记账归属」拍板后再落地。

**技术栈：** Python + MLX 0.31.2（**无布尔索引，用 `mx.where`/`mx.take`**）；
native_moe_ext（C++/Metal Primitive）；pytest。

**前置结论（Phase 0，已完成）：** go/no-go = **强 GO**（去 per-layer 同步 + demand 落池的
tok/s 上界 11→54，+387%；保 barrier 去 demand 仍 +174%）。见
`benchmarks/reports/virtualpool-phase0-2026-07-02.md`。

---

## 决策门（写 Phase 2 具体代码前必须通过）

**硬问题 3：Phase 2 的槽记账归属选哪个？**
- **方案 C（推荐）**：主线程预留槽 + C++ demand primitive 只填字节，保留一次极小 `.tolist`。
- **方案 B**：C++ 完全接管真实区 `_slot_of/_free/freq`，零主线程同步但重写状态机。

**Phase 1 不依赖此决策，可先行执行。Phase 2 的任务代码待用户选定 B/C 后补全。**

---

## 文件结构

- 修改：`mlx_streaming/core/cache/virtual_pool.py` —— 新增/扩展 `acquire`，收口路径选择。
- 修改：`mlx_streaming/core/moe/block.py:147-236` —— 计算段用一次 `_vpool.acquire` 替换四分支。
- 新增测试：`mlx_streaming/tests/test_virtual_pool_unified.py` —— 收口等价性单测。
- 复用测试：`test_virtual_pool.py`、`test_resident_sideregion.py`、
  `test_dual_source_verify_shape.py`、`test_dual_fallback_gpu_miss.py`。
- 清理：`config.py` / `resident_pool.py` 的 Phase 0 探针（`PROBE_ALL_HIT_LAZY` /
  `PROBE_NO_DEMAND`）在 Phase 1 合并前删除（见 Task 9）。

---

## Phase 1：API 收口（MVP，bite-sized，TDD）

### 设计：`VirtualPool.acquire` 统一签名

现状 `VirtualPool.acquire(layer, inds, num_experts)` 只覆盖 dual 路径。扩展为：

```python
def acquire(self, layer, inds, num_experts, *, seq_len, layer_cap):
    """呈现「所有专家都在」的视角，返回 (pool_arrays, local, n_experts)。

    内部按现状判据选择路径（Phase 1 保持与 block.py 逐分支等价）：
      - dual（zerocopy_dual_source 且 verify/decode 装得下）：真实区表 ∪ 侧区(读代) → acquire_gpu_dual；
        n_experts = layer_cap + rp.spec_gens*rp.spec_slots。
      - gpu-remap（非 dual、seq==1 或 verify_gpu）：acquire_gpu；n_experts = layer_cap。
      - host（seq>1 或关 remap，且 uniq<=cap）：acquire(flat)；n_experts = layer_cap。
      - fetch（uniq>cap）：fetch(uniq_sorted)；n_experts = len(uniq_sorted)。
    路径选择所需的 flat/uniq 计算与 block.py 原逻辑逐行等价（含 .tolist 时机）。
    """
```

> 说明：Phase 1 **只搬「取用分支」**，不动 promote / miss_attrib / route_trace / stg_verify
> 诊断（它们保留在 block.py，靠已算好的 inds/uniq 传参）。这样收口面最小、最易证 bit-exact。

---

### Task 1：给 VirtualPool.acquire 加 dual 路径的 n_experts 返回

**Files:**
- Modify: `mlx_streaming/core/cache/virtual_pool.py:73-76`
- Test: `mlx_streaming/tests/test_virtual_pool_unified.py`

- [ ] **Step 1: 写失败测试** —— dual 路径返回三元组且 n_experts 正确

```python
# mlx_streaming/tests/test_virtual_pool_unified.py
import mlx.core as mx
from mlx_streaming.core.cache.virtual_pool import VirtualPool


class _RP:
    spec_gens = 2
    spec_slots = 16
    def cap_for(self, layer): return 32
    def acquire_gpu_dual(self, layer, inds, num_experts, side):
        return ("POOL_DUAL", "LOCAL_DUAL")


class _Stg:
    def sideregion_kv(self, layer, gen): return (mx.array([], dtype=mx.uint32),
                                                 mx.array([], dtype=mx.int32))


def test_acquire_dual_returns_n_experts():
    vp = VirtualPool(_RP(), _Stg(), spec_slots=16)
    vp.begin_forward(0)
    pool, local, n_exp = vp.acquire(0, "INDS", 128, seq_len=4, layer_cap=32)
    assert pool == "POOL_DUAL"
    assert n_exp == 32 + 2 * 16      # layer_cap + spec_gens*spec_slots
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py::test_acquire_dual_returns_n_experts -v`
Expected: FAIL（`acquire` 返回二元组 / 缺 seq_len 参数）

- [ ] **Step 3: 最小实现** —— 扩展 `acquire`（dual 分支先行）

```python
# virtual_pool.py，替换现有 acquire
def acquire(self, layer, inds, num_experts, *, seq_len=None, layer_cap=None):
    """真实区表 ∪ 侧区(读代) → 单次 gather。返回 (pool_arrays, local, n_experts)。"""
    from mlx_streaming.core.prefetch.native_staging import _StagingSide
    side = _StagingSide(self._stg, self.read_gen())
    pool, local = self._rp.acquire_gpu_dual(layer, inds, num_experts, side)
    n_exp = int(layer_cap) + self._rp.spec_gens * self._rp.spec_slots
    return pool, local, n_exp
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/core/cache/virtual_pool.py mlx_streaming/tests/test_virtual_pool_unified.py
git commit -m "feat(vpool): acquire 返回 n_experts(dual 分支收口第一步)"
```

---

### Task 2：VirtualPool.acquire 收口 gpu-remap（非 dual）分支

**Files:**
- Modify: `mlx_streaming/core/cache/virtual_pool.py`（`acquire` 增加非 dual 分支）
- Test: `mlx_streaming/tests/test_virtual_pool_unified.py`

- [ ] **Step 1: 写失败测试** —— 无 staging（非 dual）时走 `acquire_gpu`，n_experts=layer_cap

```python
class _RPRemap:
    spec_gens = 1
    spec_slots = 0
    def cap_for(self, layer): return 32
    def acquire_gpu(self, layer, inds, num_experts):
        return ("POOL_REMAP", "LOCAL_REMAP")


def test_acquire_nondual_gpu_remap():
    vp = VirtualPool(_RPRemap(), staging=None, spec_slots=0)
    vp.begin_forward(0)
    pool, local, n_exp = vp.acquire(0, "INDS", 128, seq_len=1, layer_cap=32)
    assert pool == "POOL_REMAP"
    assert n_exp == 32
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py::test_acquire_nondual_gpu_remap -v`
Expected: FAIL（`acquire` 无条件走 dual → `_stg` 为 None 崩 / n_experts 错）

- [ ] **Step 3: 最小实现** —— `acquire` 增加 dual/非 dual 判据

```python
def acquire(self, layer, inds, num_experts, *, seq_len=None, layer_cap=None):
    if self._stg is not None and self._spec > 0:      # dual：真实区 ∪ 侧区
        from mlx_streaming.core.prefetch.native_staging import _StagingSide
        side = _StagingSide(self._stg, self.read_gen())
        pool, local = self._rp.acquire_gpu_dual(layer, inds, num_experts, side)
        n_exp = int(layer_cap) + self._rp.spec_gens * self._rp.spec_slots
        return pool, local, n_exp
    pool, local = self._rp.acquire_gpu(layer, inds, num_experts)   # 非 dual GPU-remap
    return pool, local, int(layer_cap)
```

- [ ] **Step 4: 跑测试确认通过**（两个测试都过）

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(vpool): acquire 收口非 dual GPU-remap 分支"
```

---

### Task 3：VirtualPool.acquire_host 收口 host + fetch 分支

**Files:**
- Modify: `mlx_streaming/core/cache/virtual_pool.py`
- Test: `mlx_streaming/tests/test_virtual_pool_unified.py`

> host/fetch 分支需要 host 侧 `flat`/`uniq`（block.py 现有 `.tolist()`），与 GPU-remap 路径
> 输入语义不同（后者吃 lazy inds）。为保持 bit-exact 且不引入额外同步，用**单独方法**
> `acquire_host(layer, flat, inds_shape, inds_dtype, layer_cap)`，由 block.py 在已做 `.tolist()`
> 后调用（时机与现状一致）。

- [ ] **Step 1: 写失败测试** —— host 分支（uniq<=cap 走 acquire、uniq>cap 走 fetch）

```python
class _RPHost:
    def __init__(self): self.calls = []
    def cap_for(self, layer): return 4
    def acquire(self, layer, flat):
        self.calls.append(("acquire", tuple(flat)))
        return ("POOL_HOST", [0, 1, 0])
    def fetch(self, layer, uniq_sorted):
        self.calls.append(("fetch", tuple(uniq_sorted)))
        return "POOL_FETCH"


def test_acquire_host_under_cap_uses_acquire():
    vp = VirtualPool(_RPHost(), staging=None, spec_slots=0)
    pool, local, n_exp = vp.acquire_host(0, [10, 11, 10], (1, 3), mx.uint32, layer_cap=4)
    assert pool == "POOL_HOST" and n_exp == 4
    assert [int(v) for v in local.reshape(-1).tolist()] == [0, 1, 0]


def test_acquire_host_over_cap_uses_fetch():
    vp = VirtualPool(_RPHost(), staging=None, spec_slots=0)
    pool, local, n_exp = vp.acquire_host(0, [10, 11, 12, 13, 14], (1, 5), mx.uint32, layer_cap=4)
    assert pool == "POOL_FETCH" and n_exp == 5      # 5 uniq > cap 4 → fetch
    # local 为 remap 到 [0,5) 的连续索引
    assert sorted(set(int(v) for v in local.reshape(-1).tolist())) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py -k host -v`
Expected: FAIL（`acquire_host` 未定义）

- [ ] **Step 3: 最小实现**

```python
def acquire_host(self, layer, flat, inds_shape, inds_dtype, layer_cap):
    """host/fetch 路径收口（prefill/大 seq 或关 remap）。flat 为 host 侧路由 id 列表。
    返回 (pool_arrays, local, n_experts)，与 block.py 原 host/fetch 分支逐元素等价。"""
    import mlx.core as mx
    uniq_set = set(flat)
    cap = int(layer_cap)
    if len(uniq_set) <= cap:
        pool, slots = self._rp.acquire(layer, flat)
        local = mx.array(slots, dtype=inds_dtype).reshape(inds_shape)
        return pool, local, cap
    uniq_sorted = sorted(uniq_set)
    remap = {g: i for i, g in enumerate(uniq_sorted)}
    local = mx.array([remap[i] for i in flat], dtype=inds_dtype).reshape(inds_shape)
    fetched = self._rp.fetch(layer, uniq_sorted)
    return fetched, local, len(uniq_sorted)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py -k host -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat(vpool): acquire_host 收口 host+fetch 分支"
```

---

### Task 4：block.py 计算段改用 VirtualPool.acquire（GPU-remap 段）

**Files:**
- Modify: `mlx_streaming/core/moe/block.py:192-205`
- Test: 端到端 bit-exact（Task 8）

> 注意：`_vpool` 仅在 dual 模式下由 model_builder 注入。非 dual GPU-remap 路径当前直接调
> `store.acquire_gpu`。为让 block.py 单一入口，需确认非 dual 时也有 `_vpool`（见 Task 6 前置）。
> 若非 dual 无 `_vpool`，Task 4 只收口 dual 段，非 dual 段 Task 6 处理。

- [ ] **Step 1: 端到端基线快照**（收口前 token 序列，供 bit-exact 对比）

Run:
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=32 WARMUP_TOK=32 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec > /tmp/vp/before.json
```
记录 `n_mismatch` 与（若加）token id 序列。

- [ ] **Step 2: 替换 dual 段**（block.py 192-197）

```python
            if config.zerocopy_dual_source():
                rp = self.store._resident
                pool_arrays, local, n_experts = self._vpool.acquire(
                    self.layer_idx, inds, gates.shape[-1],
                    seq_len=x.shape[1], layer_cap=layer_cap)
                y = self._sub.forward(pool_arrays, n_experts, x, local)
            else:
                pool_arrays, local = self.store.acquire_gpu(
                    self.layer_idx, inds, gates.shape[-1])
                if config.stg_verify():
                    self.store._resident.verify_acquire_bytes(
                        self.layer_idx, inds, _stg_mgr)
                y = self._sub.forward(pool_arrays, layer_cap, x, local)
```

- [ ] **Step 3: 跑 vpool 单测 + dual 相关单测**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool_unified.py mlx_streaming/tests/test_virtual_pool.py mlx_streaming/tests/test_dual_source_verify_shape.py -v`
Expected: PASS

- [ ] **Step 4: 端到端 bit-exact 校验**

Run: 同 Step 1 命令输出到 `/tmp/vp/after.json`；比对 `n_mismatch` 与 token 序列一致（dual 有良性时序噪声，允许 ±小抖动，但 `spec_hit_rate`/`gpu_fastpath`/`gpu_fallback` 应同档）。
Expected: token 序列一致或抖动在当前基线同档（n_mismatch 个位~低十位）。

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/core/moe/block.py
git commit -m "refactor(block): dual 取用改走 VirtualPool.acquire(三元组)"
```

---

### Task 5：block.py host 段改用 VirtualPool.acquire_host

**Files:**
- Modify: `mlx_streaming/core/moe/block.py:225-236`

- [ ] **Step 1: 替换 host+fetch 段**（保留 promote/miss_attrib 在其上方不动）

```python
            # 池只能同时容纳 ≤该层容量 个唯一专家；超容量 acquire_host 内部走 fetch
            pool_arrays, local, n_experts = self._vpool.acquire_host(
                self.layer_idx, flat, inds.shape, inds.dtype, layer_cap)
            y = self._sub.forward(pool_arrays, n_experts, x, local)
```

- [ ] **Step 2: 跑 host 路径单测（prefill）**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_prefill_chunked.py mlx_streaming/tests/test_file_streaming.py -v`
Expected: PASS

- [ ] **Step 3: 端到端 bit-exact**（含 prefill 分块，同 Task 4 Step 4 命令）
Expected: 同档一致。

- [ ] **Step 4: 提交**

```bash
git add mlx_streaming/core/moe/block.py
git commit -m "refactor(block): host+fetch 取用改走 VirtualPool.acquire_host"
```

---

### Task 6：非 dual GPU-remap 路径也走 VirtualPool（可选，视 `_vpool` 注入）

**Files:**
- Modify: `mlx_streaming/model_builder.py`（非 dual 也注入 `_vpool`）或 `block.py`（保留 else 分支）

- [ ] **Step 1: 探明 `_vpool` 注入条件**

Run: `.venv/bin/python -c "from mlx_streaming import model_builder"` 后用 Grep 找 `_vpool =` 注入点。

- [ ] **Step 2: 决策**
  - 若非 dual 也有 `_vpool` → 把 Task 4 的 `else` 分支也改走 `self._vpool.acquire`（内部转 `acquire_gpu`）。
  - 若非 dual 无 `_vpool` → **保留 else 分支现状**（YAGNI：非 dual 非默认路径），只在注释注明。

- [ ] **Step 3: 单测 + 提交**（若改动）

```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_resident_pool.py -v
git add -A && git commit -m "refactor(block): 非 dual GPU-remap 收口(或注明保留)"
```

---

### Task 7：block.py 收敛后可读性清理

**Files:**
- Modify: `mlx_streaming/core/moe/block.py`

- [ ] **Step 1: 更新 block.py 顶部 docstring / 分支注释**，说明「取用统一走 `_vpool.acquire*`，
  VirtualPool 呈现『所有专家都在』视角」。（仅注释，无逻辑变化。）

- [ ] **Step 2: 跑全量相关单测**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/ -k "virtual or resident or dual or block or file_streaming or prefill" -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add mlx_streaming/core/moe/block.py && git commit -m "docs(block): 更新收口后取用路径注释"
```

---

### Task 8：Phase 1 端到端验证（性能中性 + bit-exact）

- [ ] **Step 1: 跑完整基线对比**

Run:
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```

- [ ] **Step 2: 验收**：`spec_tok_per_s` 相对 11.07 在噪声带内（±0.3）；`spec_hit_rate`≈0.875、
  `gpu_fastpath/fallback`≈493/1091、`n_mismatch` 个位~低十位、`mlx_peak_gb`≈8.4。

- [ ] **Step 3: 记录到报告**（追加到 `benchmarks/reports/virtualpool-phase0-2026-07-02.md`
  的「Phase 1 验证」小节）。

---

### Task 9：清理 Phase 0 探针

**Files:**
- Modify: `mlx_streaming/config.py`（删 `probe_all_hit_lazy` / `probe_no_demand`）
- Modify: `mlx_streaming/core/cache/resident_pool.py`（删两处探针短路）

- [ ] **Step 1: 删除探针代码**（Phase 0 结论已记入报告，探针使命完成）
- [ ] **Step 2: 跑单测确认无引用残留**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_resident_sideregion.py -v`
Expected: PASS

- [ ] **Step 3: 提交**

```bash
git add -A && git commit -m "chore: 移除 Phase 0 go/no-go throwaway 探针"
```

---

## Phase 2：真正消除 per-layer 同步（**待用户对硬问题 3 拍板后补全代码**）

> 下述为设计级任务骨架；**具体代码在用户选定方案 B/C 后写入本节**。两方案共用「先写
> demand-spike 单测钉死排序依赖边」这一起点。

### Task P2-0（两方案共用）：demand-into-primitive 排序依赖 spike

- [ ] 写单测：预分配池（`preallocate`）+ 一个惰性 demand primitive（输入 inds/pool、输出修正
  local），断言 `mx.take`/matmul 消费该 local 时读到的是 primitive 在 eval 内 pread 落池后的
  **正确字节**（不是脏字节）。证「Primitive 输出 → 下游 gather」依赖边成立
  （参考 `test_ingraph_alias_spike.py` + `MaterializeSpikePrimitive`）。native 未编译则 skip。
- [ ] 若依赖边不成立 → **Phase 2 不可行，停下报告**，回退到 Tier1 路径（Phase 1 抽象保留）。

### 方案 C（推荐）：主线程预留槽 + C++ demand primitive 只填字节

- [ ] Task P2-C1：新增 native `demand_fill(inds, slot_table, side_kv, pool_list, seg_meta, path,
  stride, plan) -> local`：C++ 在 eval_gpu 主体（非后台）里按主线程给的 `(expert→slot)` 计划
  pread + memcpy，输出修正 local。plan 由主线程用现有 `_alloc_slot` 预留（一次极小 `.tolist`
  拿 unique 路由）。
- [ ] Task P2-C2：`ResidentExpertPool.acquire_gpu_dual` 在 `NATIVE_DEMAND_PRIMITIVE=1` 时，
  回退分支改为「主线程预留槽 → 调 `demand_fill` → 返回 local」，去掉主线程落池 scatter/eff 重建。
- [ ] Task P2-C3：LFU 记账降级为「只对 miss bump」（与 Tier1 已接受取舍一致）。
- [ ] Task P2-C4：字节等价单测（`STG_VERIFY` 风格）+ A/B 实测（目标吃下 11→54 大部分，
  保守 ≥ +100%）。

### 方案 B：C++ 完全接管真实区槽状态

- [ ] Task P2-B1：把 `_slot_of/_free/freq` 真实区版本迁入 C++（仿 `SideLayer`），暴露查询接口
  给 Python 侧 stats/trace/prefetch_cpp。
- [ ] Task P2-B2：C++ demand primitive 在 eval/回调线程内完成 membership + 分配/驱逐/pread/
  memcpy/更新表，输出修正 local，主线程完全不 `.tolist`、不碰槽。
- [ ] Task P2-B3：C++ 精确复刻 `_choose_victim`（freq + LRU tie-break + pinned/current 保护）；
  与 `prefetch_cpp` 真实区槽分配统一到一套 C++ 分配器（互斥）。
- [ ] Task P2-B4：逐段字节校验 + 与 Python 分配器一致性回归 + A/B 实测。

---

## 自检（Phase 1）

1. **spec 覆盖**：dual/gpu-remap/host/fetch 四分支 → Task 1/2/3；block.py 收敛 → Task 4/5/6；
   抽象边界 → Task 7；bit-exact/性能中性 → Task 8；探针清理 → Task 9。✓
2. **占位符扫描**：Phase 1 各步含完整代码与命令。Phase 2 明确标注「待拍板后补全」。✓
3. **类型一致**：`acquire` 全程返回三元组 `(pool, local, n_experts)`；`acquire_host` 同。✓
