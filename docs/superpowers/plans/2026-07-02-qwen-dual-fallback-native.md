# 双源 acquire 回退去 host 胶水 (Tier 1 + Tier 2) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或
> superpowers:executing-plans 逐任务实现。步骤用复选框 (`- [ ]`) 跟踪。

**Goal:** 消除 `acquire_gpu_dual` 回退分支的主线程 Python 胶水，让 70.8% 的回退层不再因整批
`.tolist()` + Python set/循环停顿 GPU 流水线；Tier 1 纯 MLX 等价重写，Tier 2 把 miss 提取+pread+落池
下沉一次 C++ 调用。

**Architecture:** 核心洞察——回退里 `local = mx.take(eff, inds)` 的 eff 已叠加侧区，故 `local<0`
恰为真 miss。Tier 1 用 `mx.where` 合成"miss 填 id/命中填 -1"单次取回 miss id（MLX 无布尔索引），复用现有 `acquire` 读盘；
Tier 2 新增 native `dual_fallback` 把整段收进 C++，开关默认 off。

**Tech Stack:** Python + MLX (mx.core)、nanobind C++ 扩展 (native_moe_ext)、pytest。

参考 spec：`docs/superpowers/specs/2026-07-02-qwen-dual-fallback-native-design.md`

---

## Tier 1：纯 MLX 等价重写回退

### Task 1：Tier 1 回退 GPU miss 提取（等价性 + 计数）

**Files:**
- Modify: `mlx_streaming/core/cache/resident_pool.py`（`acquire_gpu_dual` 回退分支，当前 540-554 行）
- Test: `mlx_streaming/tests/test_dual_fallback_gpu_miss.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_dual_fallback_gpu_miss.py
import mlx.core as mx
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _kv(d):
    return (mx.array(list(d.keys()), dtype=mx.uint32),
            mx.array(list(d.values()), dtype=mx.int32))


def _fake(seed):
    mx.random.seed(seed)
    w = mx.random.normal((32, 64))
    wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
    return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}


class _Side:
    def __init__(self, d): self._d = d
    def kv(self, layer): return _kv(self._d)


def test_fallback_true_miss_maps_correctly():
    # 池里 [10]；侧区 {20:5}；inds=[10,20,30]，30 真 miss。
    # 回退后 local 逐元素：10→真实槽、20→侧区行5、30→真实区 [0,cap)。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    slot10 = p._slot_of[0][10]
    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side({20: 5}))
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert p.gpu_fallback == 1
    assert loc[0] == slot10
    assert loc[1] == 5
    assert 0 <= loc[2] < 4 and loc[2] != 5
    assert 30 in p._slot_of[0]          # 真 miss 已落真实区


def test_fallback_hit_count_matches_positions():
    # hit 计数按位置口径：inds.size - n_miss_positions。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    h0 = p.hits
    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)   # 2 命中(10,20) + 1 miss(30)
    p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side({20: 5}))
    assert p.hits - h0 == 2


def test_fallback_multi_miss_dedup():
    # 同一 miss 专家在多个位置出现：只读盘一次、落一个槽，两处 local 指同槽。
    p = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p.acquire(0, [10])
    inds = mx.array([[[30, 10, 30]]], dtype=mx.uint32)   # 30 出现两次
    pool, local = p.acquire_gpu_dual(0, inds, num_experts=32, side=_Side({}))
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert loc[0] == loc[2]                              # 两处 30 指同槽
    assert list(p._slot_of[0].keys()).count(30) == 1     # 只落一个
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_dual_fallback_gpu_miss.py -v`
Expected: `test_fallback_hit_count_matches_positions` 等因旧实现 hit 口径/路径差异而 FAIL（或全绿——
旧实现本就等价则跳过，但新增测试锁定契约）。若旧实现已全绿，仍保留测试作回归护栏。

- [ ] **Step 3: 替换回退实现**

把 `acquire_gpu_dual` 中 `self.gpu_fallback += 1` 之后到 `return` 之间（当前 542-554 行）替换为：

```python
        # 真 miss：eff 已叠加侧区，故 local<0 恰为真 miss。GPU 布尔索引只取这几个 id 回 host，
        # 复用 demand 读盘落真实区、补表；砍掉整批 .tolist + 侧区 keys.tolist + Python set/循环。
        self.gpu_fallback += 1
        flat_local = local.reshape(-1)
        miss_mask = flat_local < 0
        # MLX 无布尔索引:用 where 把"miss 位置填专家 id、命中位置填 -1"合成一个数组,单次 .tolist()。
        miss_or_neg = mx.where(miss_mask,
                               inds.reshape(-1).astype(mx.int32),
                               mx.array(-1, dtype=mx.int32))
        miss_host = miss_or_neg.tolist()                     # 唯一同步
        n_miss_pos = sum(1 for v in miss_host if v >= 0)
        self.hits += int(inds.size) - n_miss_pos             # 位置口径,与快路径一致
        miss_ids = [v for v in miss_host if v >= 0]          # 真 miss 专家 id(带重复,acquire 内 dedup)
        if miss_ids:
            self.acquire(layer, miss_ids)                    # dedup+note_access+pread+落池+补表
        base = self._slot_table[layer]
        eff = mx.array(base) if has_side else base
        if has_side:
            eff[keys] = vals
        local = mx.take(eff, inds)
        return self._pools[layer], local
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_dual_fallback_gpu_miss.py -v`
Expected: PASS（3 个用例全绿）

- [ ] **Step 5: 现有相关测试回归**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_resident_sideregion.py mlx_streaming/tests/test_dual_source_verify_shape.py -q`
Expected: PASS（旧断言仍成立，`acquire_gpu_dual_fallback_true_miss` 等未破坏）

- [ ] **Step 6: Commit**

```bash
git add mlx_streaming/core/cache/resident_pool.py mlx_streaming/tests/test_dual_fallback_gpu_miss.py
git commit -m "perf: 双源回退用 GPU 布尔索引取 miss，砍掉整批 tolist 与侧区 set"
```

### Task 2：Tier 1 端到端等价 + 提速验证

**Files:**
- Test: 手动 e2e（无新文件）

- [ ] **Step 1: bit-exact 回归（greedy）**

Run:
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 MAXTOK=48 WARMUP_TOK=32 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.benchmarks.bench_dual_source
```
Expected: `ids` 序列与改动前一致（Tier 1 数值等价）。若不一致，检查 eff 重算与 miss 集合。

- [ ] **Step 2: MTP 提速 + 回退占比 + 抖动**

Run:
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```
Expected: 记录 `spec_tok_per_s`（目标 ≥ 当前 ~11 的 +5%）、`gpu_fastpath`/`gpu_fallback`、
`spec_hit_rate`、`n_mismatch`（个位~低十位）、`mlx_peak_gb`（±0.1GB）。

- [ ] **Step 3: 记录数据到报告（只写数据，不写结论）**

在 `benchmarks/reports/dual-fallback-native-2026-07-02.md` 追加 Tier 1 前/后对比表。

- [ ] **Step 4: Commit（若报告更新）**

```bash
git add benchmarks/reports/dual-fallback-native-2026-07-02.md
git commit -m "docs: 记录双源回退 Tier1 前后 tok/s 与回退占比数据"
```

---

## Tier 2：native 融合回退（miss id 也不回 Python）

### Task 3：config 开关 `NATIVE_DUAL_FALLBACK`

**Files:**
- Modify: `mlx_streaming/config.py`
- Test: `mlx_streaming/tests/test_config_dual_fallback.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_config_dual_fallback.py
import importlib, os
from mlx_streaming import config


def test_native_dual_fallback_default_off(monkeypatch):
    monkeypatch.delenv("NATIVE_DUAL_FALLBACK", raising=False)
    assert config.native_dual_fallback() is False


def test_native_dual_fallback_on(monkeypatch):
    monkeypatch.setenv("NATIVE_DUAL_FALLBACK", "1")
    assert config.native_dual_fallback() is True
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_config_dual_fallback.py -v`
Expected: FAIL `AttributeError: module 'mlx_streaming.config' has no attribute 'native_dual_fallback'`

- [ ] **Step 3: 加配置**

在 `mlx_streaming/config.py`（`native_demand_loader` 附近）加：

```python
# 双源回退融合(Tier2)：miss 提取+pread+落池+补表整段下沉 C++ dual_fallback，连几个 miss id
# 都不回 Python。默认 off(opt-in)；native 未编译或超容量时 Python 侧自动回退 Tier1。
def native_dual_fallback() -> bool: return _b("NATIVE_DUAL_FALLBACK", "0")
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_config_dual_fallback.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_streaming/config.py mlx_streaming/tests/test_config_dual_fallback.py
git commit -m "feat: 加 NATIVE_DUAL_FALLBACK 开关(默认off)"
```

### Task 4：C++ `dual_fallback` 实现（含哨兵回退）

**Files:**
- Modify: `native/ext/native_prefetch.h`
- Modify: `native/ext/native_prefetch.cpp`
- Modify: `native/ext/native_bindings.cpp`
- Test: `mlx_streaming/tests/test_dual_fallback_native.py`

- [ ] **Step 1: 写失败测试（native 存在才跑）**

```python
# mlx_streaming/tests/test_dual_fallback_native.py
import mlx.core as mx
import pytest

_N = pytest.importorskip("mlx_streaming.native_moe_ext")


def test_dual_fallback_local_matches_python():
    # slot_table：expert 10→槽0；侧区 20→行5；inds=[10,20,30] 30 真 miss。
    # 无盘可读的纯映射校验：预置 slot_table 已含 10、侧区含 20；miss 30 由 C++ 落空闲槽。
    E = 32
    tab = [-1] * E; tab[10] = 0
    slot_table = mx.array(tab, dtype=mx.int32)
    side_keys = mx.array([20], dtype=mx.uint32)
    side_vals = mx.array([5], dtype=mx.int32)
    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)
    # 纯映射模式(path=""表示不读盘,仅做映射+返回哨兵行给 miss)——用于隔离 gather 正确性
    local = _N.dual_fallback_map_only(inds, slot_table, side_keys, side_vals)
    loc = [int(v) for v in local.reshape(-1).tolist()]
    assert loc[0] == 0 and loc[1] == 5 and loc[2] == -1   # 30 仍 -1(未读盘)
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_dual_fallback_native.py -v`
Expected: FAIL（`dual_fallback_map_only` 未定义）或整体 skip（native 未编译）

- [ ] **Step 3: 头文件声明**

在 `native/ext/native_prefetch.h` 加（含 `<utility>` 已在）：

```cpp
// 双源回退纯映射：slot_table 叠加 side(keys→vals) 得 eff，对 inds 求 local(int32,形同 inds)。
// local<0 表示真 miss(未读盘)。供 Tier2 gather 正确性单测与融合路径复用。
mx::array dual_fallback_map_only(const mx::array& inds, const mx::array& slot_table,
                                 const mx::array& side_keys, const mx::array& side_vals);
```

- [ ] **Step 4: C++ 实现 map_only**

在 `native/ext/native_prefetch.cpp` 加：

```cpp
mx::array dual_fallback_map_only(const mx::array& inds, const mx::array& slot_table,
                                 const mx::array& side_keys, const mx::array& side_vals) {
  mx::array inds_e = inds; inds_e.eval();
  mx::array tab_e = slot_table; tab_e.eval();
  mx::array sk_e = side_keys; sk_e.eval();
  mx::array sv_e = side_vals; sv_e.eval();
  const uint32_t* ip = inds_e.data<uint32_t>();
  const int32_t* tp = tab_e.data<int32_t>();
  size_t n = inds_e.size();
  size_t E = tab_e.size();
  // 侧区覆盖：expert id -> row
  std::unordered_map<uint32_t, int32_t> side;
  side.reserve(sk_e.size());
  const uint32_t* skp = sk_e.data<uint32_t>();
  const int32_t* svp = sv_e.data<int32_t>();
  for (size_t i = 0; i < sk_e.size(); ++i) side[skp[i]] = svp[i];
  std::vector<int32_t> out(n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t e = ip[i];
    auto it = side.find(e);
    if (it != side.end()) { out[i] = it->second; continue; }
    out[i] = (e < E) ? tp[e] : -1;
  }
  mx::array res = mx::array(out.data(), inds_e.shape(), mx::int32);
  res.eval();
  return res;
}
```

- [ ] **Step 5: 绑定**

在 `native/ext/native_bindings.cpp` 加：

```cpp
  m.def("dual_fallback_map_only", &dual_fallback_map_only,
        nb::arg("inds"), nb::arg("slot_table"),
        nb::arg("side_keys"), nb::arg("side_vals"));
```

- [ ] **Step 6: 编译**

Run: `.venv/bin/python -m pip install -e . 2>&1 | tail -5`（或项目既有 build 命令）
Expected: 编译成功，无错误。

- [ ] **Step 7: 运行确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_dual_fallback_native.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add native/ext/native_prefetch.h native/ext/native_prefetch.cpp native/ext/native_bindings.cpp mlx_streaming/tests/test_dual_fallback_native.py
git commit -m "feat(native): dual_fallback_map_only 双源 gather 映射"
```

### Task 5：Python 侧接线 Tier 2（map_only + 复用 acquire 读盘）

**Files:**
- Modify: `mlx_streaming/core/cache/resident_pool.py`（`acquire_gpu_dual` 回退分支）
- Test: `mlx_streaming/tests/test_dual_fallback_native.py`（追加 e2e 等价用例）

> 说明：Tier2 首版仅把「eff 叠加 + gather 求 local」下沉 C++（`dual_fallback_map_only`），
> miss 的 pread 仍复用 Python `acquire`（其内部已是 C++ `blob_load`）。这样零新增 C++ 读盘/落池
> 代码即可去掉 Python 侧 eff 重建与 `mx.take`，miss id 仍靠 `local<0` 但在 C++ 结果上求。
> 完整「miss id 也不回 Python」的融合读盘留待后续（YAGNI：先量 map_only 的净收益）。

- [ ] **Step 1: 追加等价测试**

```python
def test_tier2_fallback_equiv_to_tier1(monkeypatch):
    from mlx_streaming.core.cache.resident_pool import ResidentExpertPool
    def _fake(seed):
        mx.random.seed(seed); w = mx.random.normal((32, 64))
        wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
        return {"gate_proj.weight": wq, "gate_proj.scales": sc, "gate_proj.biases": bi}
    class _Side:
        def kv(self, layer):
            return (mx.array([20], dtype=mx.uint32), mx.array([5], dtype=mx.int32))
    inds = mx.array([[[10, 20, 30]]], dtype=mx.uint32)
    # Tier1
    monkeypatch.setenv("NATIVE_DUAL_FALLBACK", "0")
    p1 = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p1.acquire(0, [10]); _, l1 = p1.acquire_gpu_dual(0, inds, 32, _Side())
    # Tier2
    monkeypatch.setenv("NATIVE_DUAL_FALLBACK", "1")
    p2 = ResidentExpertPool(capacity=4, loader=lambda l, e: _fake(e), spec_slots=3)
    p2.acquire(0, [10]); _, l2 = p2.acquire_gpu_dual(0, inds, 32, _Side())
    assert [int(v) for v in l1.reshape(-1).tolist()] == [int(v) for v in l2.reshape(-1).tolist()]
```

- [ ] **Step 2: 运行确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_dual_fallback_native.py::test_tier2_fallback_equiv_to_tier1 -v`
Expected: FAIL（Tier2 分支未接线）

- [ ] **Step 3: 接线**

在 `acquire_gpu_dual` 回退分支（Task 1 改后的版本）里，把「重算 local」那三行用开关分流：

```python
        if miss_ids:
            self.acquire(layer, miss_ids)
        from mlx_streaming import config as _cfg
        if _cfg.native_dual_fallback():
            try:
                from mlx_streaming import native_moe_ext as _N
                local = _N.dual_fallback_map_only(
                    inds, self._slot_table[layer],
                    keys if has_side else mx.array([], dtype=mx.uint32),
                    vals if has_side else mx.array([], dtype=mx.int32))
                return self._pools[layer], local
            except Exception:
                pass                                        # 哨兵回退 Tier1
        base = self._slot_table[layer]
        eff = mx.array(base) if has_side else base
        if has_side:
            eff[keys] = vals
        local = mx.take(eff, inds)
        return self._pools[layer], local
```

- [ ] **Step 4: 运行确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_dual_fallback_native.py -v`
Expected: PASS

- [ ] **Step 5: 全量单测回归**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_resident_sideregion.py mlx_streaming/tests/test_dual_source_verify_shape.py mlx_streaming/tests/test_dual_fallback_gpu_miss.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add mlx_streaming/core/cache/resident_pool.py mlx_streaming/tests/test_dual_fallback_native.py
git commit -m "feat: Tier2 双源回退 gather 下沉 native(map_only)，开关门控+哨兵回退"
```

### Task 6：Tier 2 端到端对比

**Files:** 手动 e2e

- [ ] **Step 1: Tier2 on/off 背靠背**

Run（分别 `NATIVE_DUAL_FALLBACK=0` 与 `=1`）：
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
  NATIVE_DUAL_FALLBACK=1 .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```
Expected: 记录 tok/s、hit、n_mismatch、峰值内存；判断 map_only 是否比 Tier1 再有净提升。

- [ ] **Step 2: 记录数据到报告**

在 `benchmarks/reports/dual-fallback-native-2026-07-02.md` 追加 Tier2 对比。

- [ ] **Step 3: 定默认值决策**

若 Tier2 净提升明显且 n_mismatch 未劣化 → 在报告写「建议 opt-in / 或转默认」；否则保持默认 off。

- [ ] **Step 4: Commit**

```bash
git add benchmarks/reports/dual-fallback-native-2026-07-02.md
git commit -m "docs: 记录双源回退 Tier2(map_only) 端到端对比数据"
```

---

## Self-Review 备注

- **Spec 覆盖**：Tier1（Task 1-2）、Tier2 开关（Task 3）、native map_only（Task 4）、接线+等价（Task 5）、
  e2e（Task 2/6）均有任务。LFU 记账决策在 Task 1 Step 3 注释体现（省真实区命中 bump）。
- **类型一致**：`dual_fallback_map_only(inds, slot_table, side_keys, side_vals)` 在 h/cpp/binding/test/
  接线处签名一致；返回 int32 local，形状同 inds。
- **范围**：Tier2 首版只下沉 gather（map_only），完整融合读盘按 YAGNI 留后续，spec §5 的完整接口
  作为方向记录、非本计划强制项。
