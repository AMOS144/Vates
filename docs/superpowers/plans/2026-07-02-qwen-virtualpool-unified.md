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

## Phase 2：真正消除 per-layer 主线程 demand WORK

### Task P2-0（已执行）：demand-into-primitive 机制 spike —— **结论：零同步不可行**

用户选定方案 B（零主线程同步）后，先用最小 C++ 探针钉死其地基机制。**已执行，结论如下**
（详见 `benchmarks/reports/virtualpool-phase2-spike-2026-07-02.md`、单测
`mlx_streaming/tests/test_demand_primitive_spike.py`）：

- [x] `demand_probe`（eval_gpu body 读 GPU 算出的 inds）：**127/200 错**——inds kernel 未执行，读到脏值。
- [x] `demand_probe_handler`（完成回调写 local，同前向下游 `mx.take` 读）：**195/200 错**——回调写入对同前向不可见。
- [x] 正对照 `materialize_spike`（body 写常量 → 下游可见）：**成立**。
- [x] `demand_probe` 在 **inds 已 eval** 后读值：**正确**——「1 次同步 + C++ 落池」地基成立。

**⇒ 同前向「零主线程同步」demand 不可行**（本 MLX 版本；除非做 spec 明确排除的投机执行）。
拿 inds 值必须每层一次同步。**这是新的根本性发现，已停下报告，等待用户对 B/C 重新拍板。**

### 影响：方案 B 的定义性优势（零同步）被证伪 → 推荐改回方案 C

- Phase 0 已证：大头收益（+174%）来自**移除主线程 demand WORK**（第二次 `.tolist` + Python
  落池 scatter + eff 重建），**而非**移除那次同步本身（P2 探针保留同步仍 +174%）。
- 移除「主线程 demand WORK」B、C 都能做到，且都各需**每层 1 次同步**（不可避免）。方案 B 额外
  的「C++ 接管槽状态 + 一致性改写」现在**换不到任何额外收益**，只剩风险。
- **推荐方案 C**：复用 Python `_alloc_slot`（Python 保持槽状态唯一权威，无一致性改写）。

### 方案 C（推荐，待用户确认后落地）：主线程预留槽 + C++ demand primitive 只填字节

- [ ] Task P2-C1：新增 native `demand_fill(inds_evaluated, side_kv, pool_list, seg_meta, path,
  stride, plan) -> local`：C++ 在 eval_gpu 主体按主线程给的 `(expert→slot)` 计划 pread + memcpy，
  输出修正 local。plan 由主线程用现有 `_alloc_slot` 预留（一次极小 `.tolist` 拿 unique 路由）。
- [ ] Task P2-C2：`acquire_gpu_dual` 在 `NATIVE_DEMAND_PRIMITIVE=1` 时，回退分支改为「主线程预留槽
  → 调 `demand_fill` → 返回 local」，去掉主线程落池 scatter/eff 重建。native 缺失自动回退现路径。
- [ ] Task P2-C3：LFU 记账保持 Python 侧现语义（`_note_access` 复用）。
- [ ] Task P2-C4：字节等价单测（`STG_VERIFY` 风格）+ A/B 实测（保守目标 ≥ +100%）。

> **【实测结论 2026-07-02，方案 B 已落地并证伪】** 方案 B 全部实现完成、单测全绿、逐位正确，
> 但实测比基线**慢 2.6×**（4.09 vs 10.78 tok/s）。DEMAND_SKIP_IO 探针证明瓶颈是**同步 demand 调用
> 本身的结构性开销**（每层一次同步打断 MLX 跨层惰性流水线重叠），与磁盘 I/O 无关；且「ON 做更少的活
> 却更慢」佐证根因。Phase 0 的 +174% 上界因「跳过 demand I/O+落池（数值错误）」而不可达。
> 详见 `benchmarks/reports/virtualpool-phase2-schemeB-2026-07-02.md`。
> **建议默认 `NATIVE_DEMAND_DUAL=0`；收益方向应回到「削减 Python 胶水但保持重叠」（方案 C）。**

### 方案 B（1 次同步版，用户拍板选定）：C++ 完全接管真实区槽状态

> 用户已知悉「零同步不可行」结论，仍选方案 B：每层 1 次 inds 同步（不可避免，替换现有
> `int(mx.sum)`+`.tolist` 两次同步为一次），但真实区 `_slot_of/_free/freq` 全由 C++ 拥有。

**语义基线（必须复刻，取自 `acquire_gpu_dual` + `acquire`/`_alloc_slot`/`_choose_victim`）：**
- 成员优先级：`eff=base∪side` 且 `eff[keys]=vals` → **侧区覆盖真实区**（同 expert 两处都有时用侧区行）。
- 槽分配：free 优先（`free.pop(0)`，spec 模式 free 初始 `[0..cap)`）；free 空 → LFU 驱逐，
  受害者槽**直接复用**（不回 free）。
- `_choose_victim`：candidates = `slot_of` 插入序中「非 pinned 且非 current(本步 miss 集)」者；
  victim = `min(freq, 候选下标)` → freq 最小、并列取插入序最早。驱逐**不删 freq**（与 Python 一致）。
- LRU 序：dual 模式真实区命中**不** move_to_end（快路径不过 `acquire`）；只有新放入的 miss 追加到序尾。
- freq bump（LFU）：canonical 策略统一为「本次 inds 全部唯一专家各 +1」（对齐快路径 `_note_access(all)`，
  比基线「快路径 bump 全部 / 回退仅 bump miss uniq」的不一致更正确；由 n_mismatch/hit 不劣化验证）。
- decay：累计访问达 `lfu_decay_interval` 则 `freq//=2`、去 0（同 `_note_access`）。
- 统计口径：`hits += 命中位置数`、`misses += 本次新读入唯一专家数(=disk_loads)`；
  无 miss 位 → `gpu_fastpath+1`，否则 `gpu_fallback+1`。

**一致性边界（P2-B4，方案 B 强制）：** 开 `NATIVE_DEMAND_DUAL=1` 后，dual 路径真实区状态由 C++
`g_real` 唯一拥有，Python `_slot_of/_free/_freq/_slot_table` 在该路径**不再维护**（成为死影子）。
- `resident_experts(layer)`（侧区预取 `submit_pool_sideregion` 的常驻过滤要用）→ 改查 C++
  `real_region_contents`；`resident_count` 同。否则预取会把已在真实区的专家重复填侧区。
- 本仓当前配置（PREFILL_CHUNK=2、top_k≈10、dual_cap=cap+spec_gens·spec=96）下 prefill(seq2)/
  decode(seq1)/verify(seq4) 的 `seq·k ≤ 96` **恒走 dual 路径**，host `acquire`/`fetch` 分支不触达 →
  真实区单一写者（demand），无 Python/C++ 双写分歧。
- **失效自动回退（默认关，opt-in）**：native 未编译、`pinned` 非空、或某层被判定走 host 路径
  （`seq·k>dual_cap`）→ 该层整层回退现有 Python 权威 `acquire_gpu_dual`（Python 状态仍在，安全）。
  一旦回退发生即视为「方案 B 前提被打破」，日志告警。

**Bite-sized 步骤（TDD，每步先红后绿 + 跑单测）：**
- [ ] Task P2-B1（native 状态 + 只读接口）：C++ 加 `RealLayer{order,e2r,free_rows,freq,cap}` +
  `g_real`；`real_init(layer,cap)` / `real_region_contents(layer)` / `real_region_count(layer)` /
  `real_reset()` + 绑定。单测：init→contents 空、越界安全。
- [ ] Task P2-B2（`_choose_victim` 端口）：C++ `choose_victim(layer,current)` 复刻 freq+插入序 tie-break；
  暴露测试壳 `real_debug_place(layer,experts,cap,lfu,decay)`（纯状态推进，不 pread）+ `real_freq(layer,e)`。
  单测(b)：对同一 (order,freq,current) 序列，C++ 选的 victim 与 Python `_choose_victim` 逐步一致。
- [ ] Task P2-B3（demand 全接管 primitive）：`demand_dual(inds,pool_list,seg_nbytes,layer,side_gen,
  path,stride,cap,lfu,decay)->local`：`inds.eval()`（1 次同步）→ 读 inds → 侧区(g_side[side_gen])∪
  真实区算 local、收集 miss → 分配槽(free/LFU)+pread+按段 memcpy 落真实区行 → 补 e2r/order/freq →
  返回 int32 local。计数进 `g_demand_*`，`demand_last_stats()` 取本次 [hitpos,misspos,loads,fallback]。
  单测(a)：小 blob 造池，demand 后 local 指向的池行字节 == 磁盘真值（复用 sideregion 单测风格）。
- [ ] Task P2-B4（Python 接线 + 一致性 + 回退）：`config.native_demand_dual()`；
  `NativeStagingManager.demand_pool_dual(...)` 包 `demand_dual`；`ResidentExpertPool` 加 `_native_demand`
  开关、`_bootstrap_dual_pool(layer,sample)`（预分配 cap+spec 池并 eval 固定指针 + `real_init`）、
  `resident_experts/`_count` 在开关下改查 C++；`acquire_gpu_dual` 开关命中则委派 native、
  否则/回退条件走原路径并告警。单测(c/d)：native 缺失自动回退、pinned 非空回退、
  `resident_experts` 与 C++ 内容一致、超容量层不启用。
- [ ] Task P2-B5（字节等价 + A/B 实测）：开 `STG_VERIFY=1` 跑一轮 `run_mtp_spec` 验证池槽字节 == 磁盘真值；
  记录前后 tok/s / hit / fallback% / disk_loads / n_mismatch / peak_gb 到 `benchmarks/reports/`。
  收益目标 ≥ +100%。

---

## 自检（Phase 1）

1. **spec 覆盖**：dual/gpu-remap/host/fetch 四分支 → Task 1/2/3；block.py 收敛 → Task 4/5/6；
   抽象边界 → Task 7；bit-exact/性能中性 → Task 8；探针清理 → Task 9。✓
2. **占位符扫描**：Phase 1 各步含完整代码与命令。Phase 2 明确标注「待拍板后补全」。✓
3. **类型一致**：`acquire` 全程返回三元组 `(pool, local, n_experts)`；`acquire_host` 同。✓
