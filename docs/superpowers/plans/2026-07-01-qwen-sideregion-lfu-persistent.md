# 侧区持久 LFU 二级缓存 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development(推荐)或 superpowers:executing-plans 逐 task 执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 把 dual-source 侧区从"每步全清重填的一次性预取批"改成"跨步持久的 LFU 二级缓存",在接近 cap=32 内存下把命中推向 cap=64(~0.87),并消除 spec 放大后的输出不确定。

**Architecture:** 复用现有零拷贝侧区(字节只存一份)+ `acquire_gpu_dual` 单次 gather。第一期只改 C++ `reserve()` 的淘汰策略为 LFU 持久(∉P 不再清、free 空才淘 freq 最小)、freq 在 reserve 内计(零 host 同步);侧区用单代(`spec_gens=1`)省内存。门控 `SIDEREGION_LFU`(默认 0,零风险回退)。第二期(仅第一期确定性不达标时)把 e2r 发布/淘汰上移主线程。

**Tech Stack:** C++(MLX Primitive + nanobind)、Python(mlx)、pytest。构建:`cd native/ext && make native_moe_ext`。

**Spec:** `docs/superpowers/specs/2026-07-01-qwen-sideregion-lfu-persistent-design.md`

---

## Task 1: config 开关 `sideregion_lfu`

**Files:**
- Modify: `mlx_streaming/config.py`(约 141 行,`zerocopy_dual_source` 附近)
- Test: `mlx_streaming/tests/test_config_sideregion_lfu.py`(新增)

- [ ] **Step 1: 写失败测试**

Create `mlx_streaming/tests/test_config_sideregion_lfu.py`:

```python
import importlib
import mlx_streaming.config as config


def test_sideregion_lfu_default_off(monkeypatch):
    monkeypatch.delenv("SIDEREGION_LFU", raising=False)
    importlib.reload(config)
    assert config.sideregion_lfu() is False


def test_sideregion_lfu_on(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    importlib.reload(config)
    assert config.sideregion_lfu() is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_config_sideregion_lfu.py -v`
Expected: FAIL(`AttributeError: module ... has no attribute 'sideregion_lfu'`)

- [ ] **Step 3: 加 config**

在 `mlx_streaming/config.py` 的 `def pool_spec_slots()` 那一行**之后**加:

```python
def sideregion_lfu() -> bool: return _b("SIDEREGION_LFU")  # 侧区持久 LFU 二级缓存(默认 off);见 spec 2026-07-01
```

- [ ] **Step 4: 跑测试**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_config_sideregion_lfu.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/config.py mlx_streaming/tests/test_config_sideregion_lfu.py
git commit -m "feat(config): 加 SIDEREGION_LFU 侧区持久 LFU 开关(默认 off)"
```

---

## Task 2: C++ 侧区 LFU 持久淘汰(第一期核心)

**Files:**
- Modify: `native/ext/native_prefetch.cpp`
  - `struct SideLayer`(约 366-370 行):加 `freq` 字段
  - `PrefetchPoolSideRegionPrimitive::reserve`(约 451-484 行):LFU 分支
  - `PrefetchPoolSideRegionPrimitive::read_publish`(约 516-520 行发布处):新专家初始化 freq
- Test: `mlx_streaming/tests/test_sideregion_lfu_native.py`(新增)

> 改前先 Read 这三处核对行号。构建命令见 Step 5。

- [ ] **Step 1: 写失败测试**

Create `mlx_streaming/tests/test_sideregion_lfu_native.py`:

```python
"""侧区持久 LFU:∉P 不清、free 空才淘 freq 最小;off 时回退旧'∉P 全弃'。"""
import os, struct, tempfile, time
import mlx.core as mx
import mlx_streaming.native_moe_ext as N

W, S = 16, 8
SEG = [W * 4, S * 1]
STRIDE = sum(SEG)
NE = 16


def _blob(path):
    with open(path, "wb") as f:
        for e in range(NE):
            f.write(struct.pack(f"<{W}I", *([e + 1] * W)))
            f.write(bytes([(e + 100) & 0xFF] * S))


def _pool(cap, spec):
    w = mx.zeros((cap + spec, W), dtype=mx.uint32)
    sc = mx.zeros((cap + spec, S), dtype=mx.uint8)
    mx.eval(w, sc)
    return [w, sc]


def _fill(pool, experts, layer, cap, spec):
    d = N.prefetch_pool_sideregion(
        pool, SEG, mx.array(experts, dtype=mx.uint32), layer, _PATH, STRIDE, [], spec, cap, gen=0)
    mx.eval(d)


def _wait(layer, want, timeout=2.0):
    t = time.time() + timeout
    while time.time() < t:
        c = N.sideregion_contents(layer, 0)
        if len(c) // 2 >= want:
            break
        time.sleep(0.01)
    flat = N.sideregion_contents(layer, 0)
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


_PATH = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
_blob(_PATH)


def test_lfu_persist_keeps_old(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    cap, spec = 4, 4          # 4 侧区行,留空
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 0, cap, spec); _wait(0, 2)
    _fill(pool, [7, 8], 0, cap, spec)       # 与 fill1 完全不重叠
    m = _wait(0, 4)
    assert set(m.keys()) == {5, 6, 7, 8}    # 旧专家 5,6 未被清(持久)


def test_lfu_evicts_min_freq(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    cap, spec = 4, 2          # 只有 2 侧区行 → 满
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 1, cap, spec); _wait(1, 2)
    _fill(pool, [5], 1, cap, spec)          # 再预测 5 → freq[5] 升(reserve 内 +1)
    _fill(pool, [5], 1, cap, spec)
    _fill(pool, [7], 1, cap, spec)          # free 空 → 淘 freq 最小且 ∉P 的 6
    m = _wait(1, 2)
    assert set(m.keys()) == {5, 7}          # 保留高频 5,淘汰低频 6


def test_lfu_off_is_legacy(monkeypatch):
    monkeypatch.delenv("SIDEREGION_LFU", raising=False)
    N.sideregion_reset()
    cap, spec = 4, 4
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 2, cap, spec); _wait(2, 2)
    _fill(pool, [7, 8], 2, cap, spec)       # 旧行为:∉P 全弃
    m = _wait(2, 2)
    assert set(m.keys()) == {7, 8}          # 5,6 被清(回退旧语义)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_lfu_native.py -v`
Expected: `test_lfu_persist_keeps_old` / `test_lfu_evicts_min_freq` FAIL(当前 ∉P 全弃 → 5,6 被清);`test_lfu_off_is_legacy` PASS。

- [ ] **Step 3: `SideLayer` 加 freq 字段**

在 `native/ext/native_prefetch.cpp` 的 `struct SideLayer`(约 366 行)改为:

```cpp
struct SideLayer {
  std::map<int, int> e2r;              // expert -> 物理侧区行 [base_row, base_row+spec)
  std::vector<int> free_rows;
  std::map<int, uint32_t> freq;        // expert -> 预测频次(LFU 分数;仅 SIDEREGION_LFU 用)
  bool inited = false;
};
```

- [ ] **Step 4: `reserve()` 加 LFU 分支**

把 `reserve()`(约 451-484 行)整体替换为:

```cpp
  static std::vector<std::pair<int, int>> reserve(
      const uint32_t* idp, size_t n, int layer, int gen, const std::vector<int>& resident,
      int spec, int base) {
    const char* lfu_env = std::getenv("SIDEREGION_LFU");   // 每次读,便于测试切换
    bool lfu = lfu_env && lfu_env[0] == '1';
    std::unordered_set<int> res(resident.begin(), resident.end());
    std::vector<int> P;
    std::unordered_set<int> Pset, seen;
    for (size_t i = 0; i < n; ++i) {
      int e = static_cast<int>(idp[i]);
      if (res.count(e) || !seen.insert(e).second) continue;
      P.push_back(e);
      Pset.insert(e);
    }
    std::vector<std::pair<int, int>> to_read;
    std::lock_guard<std::mutex> lk(g_side_mutex);
    SideLayer& c = g_side[{layer, gen}];
    if (!c.inited) {
      for (int r = 0; r < spec; ++r) c.free_rows.push_back(base + r);
      c.inited = true;
    }
    if (!lfu) {
      // 旧行为:∉P 全弃(一次性预取批)。
      for (auto it = c.e2r.begin(); it != c.e2r.end();) {
        if (!Pset.count(it->first)) {
          c.free_rows.push_back(it->second);
          it = c.e2r.erase(it);
        } else {
          ++it;
        }
      }
      for (int e : P) {
        if (c.e2r.count(e) || c.free_rows.empty()) continue;
        to_read.emplace_back(e, c.free_rows.back());
        c.free_rows.pop_back();
      }
      return to_read;
    }
    // LFU 持久:∉P 不清;再预测命中已驻专家 freq+1(越常预测越热)。
    for (int e : P) {
      if (c.e2r.count(e)) c.freq[e] += 1;
    }
    for (int e : P) {
      if (c.e2r.count(e)) continue;               // 已驻,跳过(不重读)
      int row;
      if (!c.free_rows.empty()) {
        row = c.free_rows.back();
        c.free_rows.pop_back();
      } else {
        // free 空:LFU 淘汰 e2r 中 freq 最小且 ∉P 者(tie-break:最小 expert id)。
        int victim = -1;
        uint32_t best = 0;
        for (auto& kv : c.e2r) {
          if (Pset.count(kv.first)) continue;     // 不淘本步要用的
          uint32_t f = c.freq.count(kv.first) ? c.freq[kv.first] : 0;
          if (victim < 0 || f < best || (f == best && kv.first < victim)) {
            victim = kv.first;
            best = f;
          }
        }
        if (victim < 0) continue;                 // 全是 P 热,无可淘 → 本步不读
        row = c.e2r[victim];
        c.e2r.erase(victim);
        c.freq.erase(victim);
      }
      to_read.emplace_back(e, row);
    }
    return to_read;
  }
```

- [ ] **Step 5: `read_publish()` 发布时初始化 freq**

把 `read_publish()` 末尾发布 e2r 的块(约 516-520 行)改为:

```cpp
    {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : done) {                      // 字节就绪后才发布 e2r
        c.e2r[pr.first] = pr.second;
        if (!c.freq.count(pr.first)) c.freq[pr.first] = 1;   // 新专家初始 freq
      }
    }
```

确认文件顶部已 include `<cstdlib>`(getenv);若无则加。

- [ ] **Step 6: 重编 native 扩展**

Run: `cd native/ext && make native_moe_ext`
Expected: 编译成功,产出 `mlx_streaming/native_moe_ext*.so`。

- [ ] **Step 7: 跑测试**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_lfu_native.py mlx_streaming/tests/test_sideregion_gen_native.py mlx_streaming/tests/test_pool_sideregion_native.py -v`
Expected: 全 PASS(新 LFU 测试过;旧侧区测试因 `SIDEREGION_LFU` 未设走旧路径,不破)。

- [ ] **Step 8: 提交**

```bash
git add native/ext/native_prefetch.cpp mlx_streaming/tests/test_sideregion_lfu_native.py mlx_streaming/native_moe_ext*.so
git commit -m "feat(native): 侧区持久 LFU 淘汰(∉P 不清、free 空淘 freq 最小),门控 SIDEREGION_LFU"
```

---

## Task 3: VirtualPool 支持单代(`spec_gens=1`)

**Files:**
- Modify: `mlx_streaming/core/cache/virtual_pool.py`(`read_gen`/`fill_gen`)
- Test: `mlx_streaming/tests/test_virtual_pool.py`(追加)

> 现 `read_gen`/`fill_gen` 用 `&1`(硬编码 2 代)。改成 `% spec_gens`,使 `spec_gens=1` 时读=填=0(单代持久),`spec_gens=2` 时行为不变(`%2==&1`)。

- [ ] **Step 1: 追加失败测试**

在 `mlx_streaming/tests/test_virtual_pool.py` 末尾追加:

```python
def test_single_gen_read_equals_fill():
    class _RP1:
        spec_gens = 1
        def cap_for(self, layer): return 32
        def acquire_gpu_dual(self, layer, inds, num_experts, side): return None
    class _Stg:
        def submit_pool_sideregion(self, *a, **k): return None
    vp = VirtualPool(_RP1(), _Stg(), spec_slots=16)
    vp.begin_forward(0)
    assert vp.read_gen() == 0 and vp.fill_gen() == 0      # 单代:读=填=0
    vp.begin_forward(0)                                   # 下一前向(层号回绕)推进 gen
    assert vp.read_gen() == 0 and vp.fill_gen() == 0      # 单代恒 0


def test_double_gen_still_alternates():
    class _RP2:
        spec_gens = 2
        def cap_for(self, layer): return 32
        def acquire_gpu_dual(self, layer, inds, num_experts, side): return None
    class _Stg:
        def submit_pool_sideregion(self, *a, **k): return None
    vp = VirtualPool(_RP2(), _Stg(), spec_slots=16)
    vp.begin_forward(0)
    assert vp.read_gen() != vp.fill_gen()                 # 双代:读填不同
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool.py::test_single_gen_read_equals_fill -v`
Expected: FAIL(`&1` 下 `_RP1` 无 spec_gens 用,且 fill=gen&1、read=(gen-1)&1 → 首前向 read=1≠0)。

- [ ] **Step 3: 改 `read_gen`/`fill_gen` 用 `% spec_gens`**

在 `mlx_streaming/core/cache/virtual_pool.py` 把:

```python
    def read_gen(self) -> int:
        return (self._gen - 1) & 1   # 读上一前向填好的代

    def fill_gen(self) -> int:
        return self._gen & 1         # fill 写本代（与读代恒不同）
```

改为:

```python
    def _gens(self) -> int:
        # 代数取自常驻池;单代(=1)时读=填=0(持久 LFU 单区),双代(=2)时交替(%2==&1)。
        g = getattr(self._rp, "spec_gens", 2) if self._rp is not None else 2
        return max(1, int(g))

    def read_gen(self) -> int:
        return (self._gen - 1) % self._gens()   # 读上一前向填好的代;单代恒 0

    def fill_gen(self) -> int:
        return self._gen % self._gens()         # fill 写本代;单代恒 0
```

- [ ] **Step 4: 跑测试**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_virtual_pool.py -v`
Expected: 全 PASS(新增两测过;原有协调器/调度器测试因 `_FakeRP` 无 `spec_gens` → 默认 2,`%2==&1` 行为不变)。

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/core/cache/virtual_pool.py mlx_streaming/tests/test_virtual_pool.py
git commit -m "feat(pool): VirtualPool 支持单代 spec_gens=1(读=填=0,持久 LFU 用)"
```

---

## Task 4: model_builder LFU 模式接线(单代 + 断言)

**Files:**
- Modify: `mlx_streaming/model_builder.py`(约 84-113 行:dual-source 常驻池构造 + 不变量断言)
- Test: 静态自检(import + 构造),无独立单测(端到端在 Task 5)

> LFU 模式用单代:`spec_gens=1`(物理行 `cap+spec_slots`,省一半侧区内存)。非 LFU 保持 `max(2, SPEC_GENS)`。

- [ ] **Step 1: 改常驻池 spec_gens**

在 `mlx_streaming/model_builder.py` 把(约 88-91 行):

```python
        store._resident = ResidentExpertPool(
            _old.capacity, loader=_old.loader, layer_caps=_old.layer_caps,
            spec_slots=config.pool_spec_slots(),
            spec_gens=max(2, int(os.environ.get("SPEC_GENS", "2"))))
```

改为:

```python
        # 侧区持久 LFU 用单代(一份工作集,省一半侧区内存);非 LFU 保持双缓冲。
        _spec_gens = 1 if config.sideregion_lfu() else max(2, int(os.environ.get("SPEC_GENS", "2")))
        store._resident = ResidentExpertPool(
            _old.capacity, loader=_old.loader, layer_caps=_old.layer_caps,
            spec_slots=config.pool_spec_slots(),
            spec_gens=_spec_gens)
```

- [ ] **Step 2: 静态自检(import 不崩)**

Run: `.venv/bin/python -c "import mlx_streaming.model_builder; import mlx_streaming.config as c; print('ok', c.sideregion_lfu())"`
Expected: 输出 `ok False`,无异常。

- [ ] **Step 3: 提交**

```bash
git add mlx_streaming/model_builder.py
git commit -m "feat(builder): SIDEREGION_LFU 时侧区用单代 spec_gens=1(省一半侧区内存)"
```

---

## Task 5: 端到端 go/no-go(确定性 + 命中 + 内存)

**Files:**
- 用现有 `benchmarks/bench_dual_source.py`(无需改)
- 产出:`benchmarks/reports/sideregion-lfu-2026-07-01.md`(仅数据)

> 这是第一期的验收门。达标则完成;确定性不达标才进 Task 6。命令里 `POOL_SPEC_SLOTS=32` 让第二级容量≈32(cap32+spec32≈cap64 的有效常驻),`spec_gens=1` 故物理行 = 32+32 = 64(≈cap=64 但省掉独立 staging)。

- [ ] **Step 1: 基线 ref(dual off)**

```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=0 \
  MAXTOK=64 WARMUP_TOK=64 .venv/bin/python -m benchmarks.bench_dual_source > /tmp/lfu_off.json 2>/dev/null
```

- [ ] **Step 2: LFU dual-on 跑两次(测自身确定性)**

```bash
for i in 1 2; do
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 MAXTOK=64 WARMUP_TOK=64 \
  .venv/bin/python -m benchmarks.bench_dual_source > /tmp/lfu_on_$i.json 2>/dev/null
done
```

- [ ] **Step 3: 判定**

```bash
echo "=== 自身确定性(on1 vs on2) ==="
.venv/bin/python -m benchmarks.bench_dual_source --diff /tmp/lfu_on_1.json /tmp/lfu_on_2.json
echo "=== 正确性 + 命中/内存(off vs on1) ==="
.venv/bin/python -m benchmarks.bench_dual_source --diff /tmp/lfu_off.json /tmp/lfu_on_1.json
```

判定标准(全部满足 = 第一期成功,跳过 Task 6):
- **确定性**:on1-vs-on2 `n_mismatch ≤ 1`。
- **正确性**:off-vs-on1 `n_mismatch ≤ 2`(与基线良性本底同级)。
- **命中**:on1 `hit_rate ≥ 0.85`。
- **内存**:on1 `active_gb ≤ 7.5`(明显低于 cap=64 的 9.38)。

- [ ] **Step 4: 记录数据(不写结论)**

把 Step 3 两段 JSON 与 cap=32/cap=64 基线一并写入 `benchmarks/reports/sideregion-lfu-2026-07-01.md`(表格,仅数据)。

- [ ] **Step 5: 提交**

```bash
git add benchmarks/reports/sideregion-lfu-2026-07-01.md
git commit -m "bench: 侧区持久 LFU 端到端数据(确定性/命中/内存 vs cap32/cap64)"
```

- [ ] **Step 6: go/no-go**

- 四项全达标 → 第一期完成,**结束**(不做 Task 6)。
- 仅"确定性"不达标(命中/内存达标)→ 进 Task 6(主线程发布)。
- 命中不达标 → 停,回 spec §9 风险2 复盘(调 `POOL_SPEC_SLOTS`/预测宽度,或降级为"仅省内存+race-free"),与人确认再继续。

---

## Task 6(条件性,仅 Task 5 确定性不达标):e2r 发布/淘汰上移主线程

**前置:** 只有 Task 5 Step 6 判为"确定性不达标"才做。否则跳过。

**Files:**
- Modify: `native/ext/native_prefetch.cpp`(`SideLayer` 加 `pending`/`done_rows`;`read_publish` 改为写 pending;新增 `sideregion_commit`)
- Modify: `native/ext/native_bindings.cpp`(导出 `sideregion_commit`)
- Modify: `mlx_streaming/core/prefetch/native_staging.py`(封装 `sideregion_commit`)
- Modify: `mlx_streaming/core/cache/virtual_pool.py`(`prefetch` 前先 `commit`)
- Test: `mlx_streaming/tests/test_sideregion_commit_native.py`(新增)

- [ ] **Step 1: 写失败测试**

Create `mlx_streaming/tests/test_sideregion_commit_native.py`:

```python
"""主线程发布:pread 完只置 pending;commit 后才进 e2r(contents)。"""
import os, struct, tempfile, time
import mlx.core as mx
import mlx_streaming.native_moe_ext as N

W, S = 16, 8
SEG = [W * 4, S * 1]
STRIDE = sum(SEG)


def _blob(path):
    with open(path, "wb") as f:
        for e in range(16):
            f.write(struct.pack(f"<{W}I", *([e + 1] * W)))
            f.write(bytes([(e + 100) & 0xFF] * S))


_PATH = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
_blob(_PATH)


def test_commit_publishes(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    cap, spec = 4, 4
    pool = [mx.zeros((cap + spec, W), dtype=mx.uint32), mx.zeros((cap + spec, S), dtype=mx.uint8)]
    mx.eval(pool)
    d = N.prefetch_pool_sideregion(pool, SEG, mx.array([5, 6], dtype=mx.uint32),
                                   0, _PATH, STRIDE, [], spec, cap, gen=0)
    mx.eval(d)
    time.sleep(0.3)                                   # pread 完成(进 pending)
    assert N.sideregion_contents(0, 0) == []          # commit 前 e2r 空
    N.sideregion_commit(0, 0)
    flat = N.sideregion_contents(0, 0)
    assert set(flat[i] for i in range(0, len(flat), 2)) == {5, 6}   # commit 后可见
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_commit_native.py -v`
Expected: FAIL(`sideregion_commit` 未定义;且当前 pread 完直接进 e2r)。

- [ ] **Step 3: `SideLayer` 加 pending**

`struct SideLayer` 加字段:

```cpp
  std::map<int, int> pending;          // pread 完待主线程发布 (expert->row)
```

- [ ] **Step 4: `read_publish` 写 pending 而非 e2r**

把 `read_publish` 末尾发布块改为:

```cpp
    {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : done) c.pending[pr.first] = pr.second;   // 只置 pending,等主线程 commit
    }
```

- [ ] **Step 5: 新增 `sideregion_commit`**

在 `sideregion_contents` 之前加:

```cpp
void sideregion_commit(int layer, int gen) {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  auto it = g_side.find({layer, gen});
  if (it == g_side.end()) return;
  SideLayer& c = it->second;
  for (auto& pr : c.pending) {                 // pending -> e2r,主线程发布
    c.e2r[pr.first] = pr.second;
    if (!c.freq.count(pr.first)) c.freq[pr.first] = 1;
  }
  c.pending.clear();
}
```

并在 `native_prefetch.h` 声明 `void sideregion_commit(int layer, int gen);`。

- [ ] **Step 6: 绑定导出**

在 `native/ext/native_bindings.cpp` 的 `sideregion_contents` 绑定行后加:

```cpp
  m.def("sideregion_commit", &sideregion_commit, "layer"_a, "gen"_a = 0);
```

- [ ] **Step 7: reserve 的"已驻"判断纳入 pending**

在 `reserve()` LFU 分支里,把"已驻,跳过"判断从 `c.e2r.count(e)` 改为同时看 pending(避免对已 pread 未 commit 的专家重复预留):在 LFU 分支两处 `c.e2r.count(e)` 改为 `(c.e2r.count(e) || c.pending.count(e))`。

- [ ] **Step 8: native_staging 封装**

在 `mlx_streaming/core/prefetch/native_staging.py` 的 `NativeStagingManager` 加方法:

```python
    def sideregion_commit(self, layer, gen=0):
        """主线程发布:pending -> e2r(双源 LFU 单代 race-free)。"""
        import mlx_streaming.native_moe_ext as _N
        _N.sideregion_commit(int(layer), int(gen))
```

- [ ] **Step 9: VirtualPool.prefetch 前先 commit**

在 `mlx_streaming/core/cache/virtual_pool.py` 的 `prefetch` 开头加(commit 本层填代已就绪字节):

```python
    def prefetch(self, layer, pred, resident, pool_list):
        """向 fill 代 submit 预读:base_row = cap_for(layer) + fill_gen*spec。"""
        g = self.fill_gen()
        if getattr(self._stg, "sideregion_commit", None) is not None:
            self._stg.sideregion_commit(layer, g)   # 主线程发布上一轮已就绪字节
        base = self._rp.cap_for(layer) + g * self._spec
        return self._stg.submit_pool_sideregion(layer, pred, resident, pool_list, base, gen=g)
```

- [ ] **Step 10: 重编 + 跑单测**

Run:
```bash
cd native/ext && make native_moe_ext && cd ../..
.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_commit_native.py mlx_streaming/tests/test_sideregion_lfu_native.py mlx_streaming/tests/test_virtual_pool.py -v
```
Expected: 全 PASS。

- [ ] **Step 11: 重跑 Task 5 端到端判定**

重复 Task 5 Step 1-3(SIDEREGION_LFU=1),确认确定性达标(on-vs-on `n_mismatch ≤ 1`、off-vs-on `≤ 2`),命中/内存不回退。

- [ ] **Step 12: 提交**

```bash
git add native/ext/native_prefetch.cpp native/ext/native_prefetch.h native/ext/native_bindings.cpp \
        mlx_streaming/core/prefetch/native_staging.py mlx_streaming/core/cache/virtual_pool.py \
        mlx_streaming/tests/test_sideregion_commit_native.py mlx_streaming/native_moe_ext*.so
git commit -m "feat(native): 侧区 e2r 发布上移主线程 sideregion_commit(单代 race-free)"
```

---

## Self-Review 检查(已过)

- **Spec 覆盖**:§5.5 第一期→Task 1-5;第二期→Task 6;§6 各组件均有对应 Task;§8 测试→各 Task 的 test。
- **类型/命名一致**:`sideregion_lfu`、`SIDEREGION_LFU`、`freq`、`pending`、`sideregion_commit`、`spec_gens` 全程一致。
- **无占位符**:每步含实际代码/命令/期望输出。
- **门控**:Task 5 明确 go/no-go 阈值,避免硬推。
