# 连续常驻专家池 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用每层连续常驻专家池替代流式 MoE 热路径里每 token 的 `mx.stack`，命中零拷贝、miss 只写单个专家，吃掉 profile 中占 53% 的 `fetch` 段。

**Architecture:** 新增 `ResidentExpertPool`（每层维护 `(capacity,*shape)` 连续张量 + slot LRU），由 `FileExpertStore.acquire()` 委托。`FileStreamingMoeBlock` 把 `PersistentSubGLU` 的专家数固定为 capacity、三个 `QuantizedSwitchLinear` 的权重直接指向池数组，routing 用 slot id 喂 `gather_qmm`，只 gather 实际路由的专家。

**Tech Stack:** Python, MLX 0.31.2 (`mx.array` 索引赋值 / `mx.slice_update` / 可选 `mx.fast.metal_kernel`), `mlx_lm.models.switch_layers`(`QuantizedSwitchLinear`/`SwitchGLU`), pytest。

参考 spec：`docs/superpowers/specs/2026-06-07-resident-expert-pool-design.md`

---

## File Structure

- `mlx_streaming/probe_pool_write.py`（新建）：de-risk 微基准，测三种单专家写法成本，决定 `_write_slot` 机制。
- `mlx_streaming/expert_store.py`（修改）：新增 `ResidentExpertPool` 类；`FileExpertStore` 增加 `acquire()` 委托；删除旧 `fetch_resident`/`_pools`/`_pool_hits`/`_pool_misses`。
- `mlx_streaming/streaming_moe.py`（修改）：`FileStreamingMoeBlock.__call__`/`_call_prof` 改走 `acquire()`；删除 `EXPERT_RESIDENT_POOL` 分支；新增 `RESIDENT_POOL=0` 回退到旧 stack 路径。
- `mlx_streaming/tests/test_resident_pool.py`（新建）：池语义 + 数值等价单测。

---

## Task 0: de-risk 微基准（决定单专家写法）

**Files:**
- Create: `mlx_streaming/probe_pool_write.py`

- [ ] **Step 1: 写微基准脚本**

```python
"""de-risk：测「把单个专家写进连续 pool 的某个槽位」三种写法的成本。

判定：单次写应 ≈ 单专家字节量级、且 N 次写后内存不线性膨胀。
谁最便宜就用谁作为 ResidentExpertPool._write_slot。
"""
import time
import mlx.core as mx

CAP, O, I, GROUP, BITS = 96, 2048, 768, 64, 2
# 2-bit 打包：weight 是 uint32，列数 = I*BITS/32
W_COLS = I * BITS // 32


def _new_pool():
    return {
        "weight": mx.zeros((CAP, O, W_COLS), dtype=mx.uint32),
        "scales": mx.zeros((CAP, O, I // GROUP), dtype=mx.float16),
        "biases": mx.zeros((CAP, O, I // GROUP), dtype=mx.float16),
    }


def _new_expert():
    return {
        "weight": mx.random.randint(0, 2**31, (O, W_COLS)).astype(mx.uint32),
        "scales": mx.random.normal((O, I // GROUP)).astype(mx.float16),
        "biases": mx.random.normal((O, I // GROUP)).astype(mx.float16),
    }


def bench(name, write_fn, n=200):
    pool, ex = _new_pool(), _new_expert()
    mx.eval(list(pool.values()) + list(ex.values()))
    t0 = time.perf_counter()
    for s in range(n):
        write_fn(pool, s % CAP, ex)
        mx.eval(list(pool.values()))
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"{name:>16}: {dt:.3f} ms/write")


def w_inplace(pool, slot, ex):
    for k, v in ex.items():
        pool[k][slot] = v


def w_slice_update(pool, slot, ex):
    for k, v in ex.items():
        pool[k] = mx.slice_update(pool[k], v[None], mx.array([slot, 0, 0]), axes=(0, 1, 2))


def main():
    print(f"单专家字节量 weight={O*W_COLS*4/1e6:.2f}MB  整池≈{CAP*O*W_COLS*4/1e6:.1f}MB")
    bench("inplace", w_inplace)
    bench("slice_update", w_slice_update)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行并记录结果**

Run: `source mlx_streaming/.venv/bin/activate && python -m mlx_streaming.probe_pool_write`
Expected: 打印两种写法 ms/write。判定标准：哪种 ≈ 单专家 weight（~0.6MB）量级而非整池（~57MB）量级。

- [ ] **Step 3: 把结论回填 spec §6**

修改 `docs/superpowers/specs/2026-06-07-resident-expert-pool-design.md` §6：写明选定写法、单次写 ms、是否需要 Metal scatter（若两种都退化为整池拷贝，则在本 plan 末尾启用附录 A 的 Metal 方案）。

- [ ] **Step 4: Commit**

```bash
git add mlx_streaming/probe_pool_write.py docs/superpowers/specs/2026-06-07-resident-expert-pool-design.md
git commit -m "spike: 专家池单槽写入微基准 + 回填 de-risk 结论"
```

> 后续 Task 1 的 `_write_slot` 用本任务选定的写法。默认假设 `inplace` 胜出；若 `slice_update` 胜出，把 Task 1 Step 3 中 `_write_slot` 的方法体替换为 `w_slice_update` 的逻辑（代码已在上方给全）。

---

## Task 1: `ResidentExpertPool` 类（池语义）

**Files:**
- Modify: `mlx_streaming/expert_store.py`
- Test: `mlx_streaming/tests/test_resident_pool.py`

- [ ] **Step 1: 写失败测试（acquire 命中/缺失/槽位/LRU/惰性分配/容量断言）**

```python
import mlx.core as mx
import pytest

from mlx_streaming.expert_store import ResidentExpertPool


def _loader_factory():
    # 每个专家是可区分的小张量：weight[e] 全 e
    def load(layer, e):
        return {"weight": mx.full((4, 3), float(e))}
    return load


def test_acquire_miss_then_hit_slots_stable():
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    arrs1, slots1 = pool.acquire(0, [2, 5])      # 2 miss
    arrs2, slots2 = pool.acquire(0, [2, 5])      # 2 hit，槽位不变
    assert slots1 == slots2
    assert pool.misses == 2 and pool.hits == 2
    # 池里对应槽位内容 == 专家值
    assert mx.array_equal(arrs1["weight"][slots1[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs1["weight"][slots1[1]], mx.full((4, 3), 5.0)).item()


def test_acquire_lru_evicts_least_recent():
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.acquire(0, [0])          # slot for 0
    pool.acquire(0, [1])          # slot for 1
    pool.acquire(0, [0])          # touch 0 -> 1 是最久未用
    _, slots = pool.acquire(0, [2])   # 淘汰 1，复用其槽
    assert pool.resident_count(0) == 2
    # 专家 1 应已被淘汰：再取触发 miss
    m0 = pool.misses
    pool.acquire(0, [1])
    assert pool.misses == m0 + 1


def test_per_layer_isolation():
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.acquire(0, [0, 1])
    pool.acquire(1, [0, 1])       # 不挤掉 layer0
    assert pool.resident_count(0) == 2 and pool.resident_count(1) == 2


def test_capacity_must_cover_topk():
    pool = ResidentExpertPool(capacity=1, loader=_loader_factory())
    with pytest.raises(ValueError):
        pool.acquire(0, [0, 1])   # 一次请求 2 个专家但容量 1
```

- [ ] **Step 2: 运行确认失败**

Run: `source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_resident_pool.py -v`
Expected: FAIL，`ImportError: cannot import name 'ResidentExpertPool'`。

- [ ] **Step 3: 实现 `ResidentExpertPool`**

在 `mlx_streaming/expert_store.py` 顶部 import 处保持不变，在 `LruExpertStore` 之前插入：

```python
class ResidentExpertPool:
    """每层一个连续常驻池：(capacity,*shape) 张量 + slot LRU。

    命中只返回槽位、不写池；miss 只把单个专家写进它的槽位（_write_slot）。
    loader(layer, e) -> Dict[str, mx.array]，单个专家的参数（未堆叠）。
    """

    def __init__(self, capacity: int, loader):
        self.capacity = capacity
        self.loader = loader
        # 每层：pool 参数块、slot_of(LRU)、free 槽
        self._pools: "Dict[int, Dict[str, mx.array]]" = {}
        self._slot_of: "Dict[int, OrderedDict[int, int]]" = {}
        self._free: "Dict[int, list]" = {}
        self.hits = 0
        self.misses = 0

    def _ensure_layer(self, layer: int):
        if layer not in self._slot_of:
            self._slot_of[layer] = OrderedDict()
            self._free[layer] = list(range(self.capacity))

    def _alloc_pool(self, layer: int, sample: "Dict[str, mx.array]"):
        self._pools[layer] = {
            k: mx.zeros((self.capacity,) + v.shape, dtype=v.dtype)
            for k, v in sample.items()
        }

    def _write_slot(self, layer: int, slot: int, expert: "Dict[str, mx.array]"):
        pool = self._pools[layer]
        for k, v in expert.items():
            pool[k][slot] = v        # Task 0 选定写法（默认原地赋值）

    def resident_count(self, layer: int) -> int:
        return len(self._slot_of.get(layer, ()))

    def acquire(self, layer: int, expert_ids: List[int]):
        if len(expert_ids) > self.capacity:
            raise ValueError(
                f"请求 {len(expert_ids)} 个专家 > 池容量 {self.capacity}")
        self._ensure_layer(layer)
        slot_of, free = self._slot_of[layer], self._free[layer]
        slots = []
        for e in expert_ids:
            e = int(e)
            if e in slot_of:
                self.hits += 1
                slot_of.move_to_end(e)
                slots.append(slot_of[e])
                continue
            self.misses += 1
            expert = self.loader(layer, e)
            if layer not in self._pools:
                self._alloc_pool(layer, expert)
            slot = free.pop(0) if free else slot_of.popitem(last=False)[1]
            self._write_slot(layer, slot, expert)
            slot_of[e] = slot
            slot_of.move_to_end(e)
            slots.append(slot)
        return self._pools[layer], slots

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
```

确保文件顶部已有 `from collections import OrderedDict, Counter` 与 `from typing import Dict, List`（现状已有）。

- [ ] **Step 4: 运行确认通过**

Run: `source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_resident_pool.py -v`
Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add mlx_streaming/expert_store.py mlx_streaming/tests/test_resident_pool.py
git commit -m "feat: ResidentExpertPool 连续常驻池(命中零拷贝 miss 单槽写)"
```

---

## Task 2: `FileExpertStore.acquire()` 委托 + 删除旧实验路径

**Files:**
- Modify: `mlx_streaming/expert_store.py`
- Test: `mlx_streaming/tests/test_resident_pool.py`

- [ ] **Step 1: 写失败测试（FileExpertStore.acquire 从磁盘加载、命中计数、与 fetch 同源）**

在 `test_resident_pool.py` 追加：

```python
from mlx_streaming.expert_store import FileExpertStore


def test_file_store_acquire_matches_disk(tmp_path):
    d = str(tmp_path)
    for e in range(4):
        mx.save_safetensors(f"{d}/layer00_expert{e:03d}.safetensors",
                            {"weight": mx.full((4, 3), float(e))})
    store = FileExpertStore(d, capacity=4)
    arrs, slots = store.acquire(0, [1, 3])
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 1.0)).item()
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 3.0)).item()
    arrs2, slots2 = store.acquire(0, [1, 3])   # 命中
    assert slots == slots2
    assert store.hits >= 2
```

- [ ] **Step 2: 运行确认失败**

Run: `source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_resident_pool.py::test_file_store_acquire_matches_disk -v`
Expected: FAIL，`AttributeError: 'FileExpertStore' object has no attribute 'acquire'`。

- [ ] **Step 3: 给 `FileExpertStore` 加 `acquire`，删除旧 `fetch_resident`/`_pools`/`_pool_hits`/`_pool_misses`**

在 `FileExpertStore.__init__` 中，把这三行删除：

```python
        self._pool_hits = 0
        self._pool_misses = 0
        self._pools = {}
```

替换为：

```python
        self._resident = ResidentExpertPool(capacity, loader=self._load_one)
```

把 `hits`/`misses` 属性改为合并 resident 计数：

```python
    @property
    def hits(self) -> int:
        return self._lru.hits + self.pinned_hits + self._resident.hits

    @property
    def misses(self) -> int:
        return self._lru.misses + self._resident.misses
```

`reset_stats` 中把 `self._pool_hits = 0` / `self._pool_misses = 0` 两行替换为：

```python
        self._resident.hits = 0
        self._resident.misses = 0
```

删除整个 `def fetch_resident(self, layer, expert_ids):` 方法（约第 226-265 行），新增：

```python
    def acquire(self, layer: int, expert_ids: List[int]):
        """连续常驻池取专家：返回 (pool_arrays, slots)。命中零拷贝，miss 单槽写。"""
        if self.record:
            self.note(layer, expert_ids)
        return self._resident.acquire(layer, expert_ids)
```

- [ ] **Step 4: 运行确认通过 + 回归 expert_store 旧测试**

Run: `source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_resident_pool.py mlx_streaming/tests/test_expert_store.py -v`
Expected: 全 passed（旧 `fetch` 路径未动，仍绿）。

- [ ] **Step 5: Commit**

```bash
git add mlx_streaming/expert_store.py mlx_streaming/tests/test_resident_pool.py
git commit -m "feat: FileExpertStore.acquire 委托 ResidentExpertPool, 删除旧 fetch_resident"
```

---

## Task 3: 数值等价（池前向 == stack 前向，bit 级）

**Files:**
- Test: `mlx_streaming/tests/test_resident_pool.py`

- [ ] **Step 1: 写等价测试**

构造一个量化小 `SwitchGLU`，分别用「stack 路径」和「pool+slot 路径」算同一组路由，断言输出一致。在 `test_resident_pool.py` 追加：

```python
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU


def _quant_glu(num_experts, hidden, inter, group=64, bits=4):
    glu = SwitchGLU(hidden, inter, num_experts)
    glu.gate_proj = QuantizedSwitchLinear(hidden, inter, num_experts, bias=False, group_size=group, bits=bits)
    glu.up_proj = QuantizedSwitchLinear(hidden, inter, num_experts, bias=False, group_size=group, bits=bits)
    glu.down_proj = QuantizedSwitchLinear(inter, hidden, num_experts, bias=False, group_size=group, bits=bits)
    return glu


def test_pool_forward_bit_equiv_to_stack():
    hidden, inter, E = 64, 128, 8
    full = _quant_glu(E, hidden, inter)
    mx.eval(full.parameters())
    x = mx.random.normal((1, 1, hidden))
    routed = [2, 5]                      # 该 token 路由的专家

    # stack 路径：把专家 2,5 切出来，local=[0,1]
    def slice_qsl(lin, ids):
        new = QuantizedSwitchLinear(lin.input_dims, lin.output_dims, len(ids),
                                    bias=False, group_size=lin.group_size, bits=lin.bits)
        new.update({k: v[mx.array(ids)] for k, v in lin.parameters().items()})
        return new
    stack = SwitchGLU(hidden, inter, len(routed))
    stack.gate_proj = slice_qsl(full.gate_proj, routed)
    stack.up_proj = slice_qsl(full.up_proj, routed)
    stack.down_proj = slice_qsl(full.down_proj, routed)
    y_stack = stack(mx.expand_dims(x, -2), mx.array([[[0, 1]]]))

    # pool 路径：直接用 full 作为「池」(num_experts=E)，slot=路由 id
    y_pool = full(mx.expand_dims(x, -2), mx.array([[routed]]))

    assert mx.allclose(y_stack, y_pool, atol=1e-6).item()
```

- [ ] **Step 2: 运行确认通过**

Run: `source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_resident_pool.py::test_pool_forward_bit_equiv_to_stack -v`
Expected: PASS（证明「池+slot 索引」与「切片+紧凑 local」数值等价，因为 `gather_qmm` 只取索引行）。

- [ ] **Step 3: Commit**

```bash
git add mlx_streaming/tests/test_resident_pool.py
git commit -m "test: 证明 pool+slot 前向与 stack 前向 bit 级等价"
```

---

## Task 4: 接入 `FileStreamingMoeBlock`（默认走池 + `RESIDENT_POOL=0` 回退）

**Files:**
- Modify: `mlx_streaming/streaming_moe.py:252-323`

- [ ] **Step 1: 改 `__call__` 主路径**

把 `FileStreamingMoeBlock.__call__`（约 252-278 行）中从 `flat = ...` 到 `y = (y * scores...)` 之间的分支替换为：

```python
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        if os.environ.get("RESIDENT_POOL", "1") == "1":
            # slots 与 flat 一一对应(含重复专家)，直接 reshape 成 routing 的 slot 索引
            pool_arrays, slots = self.store.acquire(self.layer_idx, flat)
            local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
            y = self._sub.forward(pool_arrays, self.store.capacity, x, local)
        else:
            uniq_sorted = sorted(set(flat))
            remap = {g: i for i, g in enumerate(uniq_sorted)}
            local = mx.array([remap[i] for i in flat], dtype=inds.dtype).reshape(inds.shape)
            fetched = self.store.fetch(self.layer_idx, uniq_sorted)
            y = self._sub.forward(fetched, len(uniq_sorted), x, local)
        y = (y * scores[..., None]).sum(axis=-2)
```

> 说明：`acquire` 返回的 `slots` 与 `flat` 一一对应，但 `flat` 可能含重复专家；用 `remap[e]=s` 去重后按 `flat` 重建 `local`（slot id），喂给 `_sub.forward`。`n=self.store.capacity`（QSL 专家数=池容量，权重指向池数组）。

- [ ] **Step 2: 改 `_call_prof`（保持 profile 口径一致）**

把 `_call_prof`（约 291-311 行）的 fetch 段替换为对应逻辑：`pyremap` 段计算 `flat`，`fetch` 段调用 `acquire` 并构造 `local`：

```python
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        use_pool = os.environ.get("RESIDENT_POOL", "1") == "1"
        _tick("pyremap", t); t = time.perf_counter()

        if use_pool:
            pool_arrays, slots = self.store.acquire(self.layer_idx, flat)
            local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
            mx.eval(local)
            fetched, n_experts = pool_arrays, self.store.capacity
        else:
            uniq_sorted = sorted(set(flat))
            remap = {g: i for i, g in enumerate(uniq_sorted)}
            local = mx.array([remap[i] for i in flat], dtype=inds.dtype).reshape(inds.shape)
            fetched = self.store.fetch(self.layer_idx, uniq_sorted)
            n_experts = len(uniq_sorted)
        mx.eval(list(fetched.values()))
        _tick("fetch", t); t = time.perf_counter()
```

删除 `_call_prof` 中原来的 `use_pool = (os.environ.get("EXPERT_RESIDENT_POOL"...` 块与 `fetch_resident` 调用。

- [ ] **Step 3: 运行回归测试**

Run: `source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/ -v`
Expected: 全 passed（含 test_mtp_generate / test_expert_store / test_resident_pool）。

- [ ] **Step 4: Commit**

```bash
git add mlx_streaming/streaming_moe.py
git commit -m "feat: FileStreamingMoeBlock 默认走常驻池(RESIDENT_POOL=0 回退)"
```

---

## Task 5: 集成验证（真机）+ 回填收益

**Files:**
- 仅运行，不改代码（除回填 spec/报告）

- [ ] **Step 1: 热路径分段对比（池 vs 旧 stack）**

```bash
source mlx_streaming/.venv/bin/activate
EXPERT_DIR=/tmp/qwen3_next_experts_2bit EXPERT_SLOTS=96 STREAM_PROF=1 N=20 RESIDENT_POOL=1 python -m mlx_streaming.probe_hotpath
EXPERT_DIR=/tmp/qwen3_next_experts_2bit EXPERT_SLOTS=96 STREAM_PROF=1 N=20 RESIDENT_POOL=0 python -m mlx_streaming.probe_hotpath
```
Expected: `RESIDENT_POOL=1` 的 `fetch` ms/step 显著低于 `=0`（基线 103.9ms），`per_token` 下降。

- [ ] **Step 2: baseline tok/s 对比**

```bash
EXPERT_DIR=/tmp/qwen3_next_experts_2bit EXPERT_SLOTS=96 K=2 MAXTOK=96 MTP_VERIFY_MODE=step RESIDENT_POOL=1 python -m mlx_streaming.run_mtp_spec
```
Expected: `exact_match=true` 不变；`baseline_tok_per_s` 高于池前的 7.49；记录 `spec` 与 `speedup`。

- [ ] **Step 3: 回填报告与 spec §6**

把 Step 1/2 数字写入 `benchmarks/reports/qwen3next-mtp-selfspec-2026-06-07.md` 新增「常驻池热路径优化」一节，并在 spec §6 标注最终选用的写法与实测 fetch 降幅。

- [ ] **Step 4: Commit**

```bash
git add benchmarks/reports/qwen3next-mtp-selfspec-2026-06-07.md docs/superpowers/specs/2026-06-07-resident-expert-pool-design.md
git commit -m "docs: 回填常驻池热路径优化真机收益"
```

---

## 附录 A：若 Task 0 判定两种纯 MLX 写法都退化为整池拷贝

启用 Metal scatter（方案 B），把 `ResidentExpertPool._write_slot` 替换为：

```python
import mlx.core as mx

_SCATTER_KERNEL = mx.fast.metal_kernel(
    name="pool_scatter_row",
    input_names=["pool", "row", "slot"],
    output_names=["out"],
    source=r"""
        uint gid = thread_position_in_grid.x;
        uint cols = pool_shape[1] * pool_shape[2];
        uint base = slot[0] * cols;
        if (gid < cols) out[base + gid] = row[gid];
    """,
)

def _write_slot(self, layer, slot, expert):
    pool = self._pools[layer]
    for k, v in expert.items():
        flat = pool[k]                       # (cap, O, C)
        out = _SCATTER_KERNEL(
            inputs=[flat, v.reshape(-1), mx.array([slot], dtype=mx.uint32)],
            output_shapes=[flat.shape], output_dtypes=[flat.dtype],
            grid=(v.size, 1, 1), threadgroup=(256, 1, 1),
        )[0]
        pool[k] = out
```

> 注意：Metal kernel 仍产出新 `out`；真正的零拷贝需配合 `donation`（把 `flat` buffer 让渡给 `out`）。若 MLX 此版本不支持 buffer 让渡，则该附录方案收益有限，应回到「降低 miss 率」（pin 热专家、加大 capacity）作为替代提速手段，并在 spec 记录该负面结论。
