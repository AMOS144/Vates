# 侧区字节稳定发布（Route 1）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现本计划。步骤用 checkbox（`- [ ]`）跟踪。

**Goal:** 修复默认 `ZEROCOPY_DUAL_SOURCE` 路径侧区「装错字节」bug，使 `DUAL_VERIFY` 稳定 0 BAD，同时保留侧区异步 I/O 与 MLX 原生 matmul。

**Architecture:** 侧区异步预取的字节不再旁路 memcpy 进「会被 MLX 跨前向重定位」的池数组，而是写进 **C++ 自己拥有、永不迁移的稳定缓冲区**；消费时用 **MLX 追踪的原地 scatter**（`pool[k][rows]=typed`）把新到的侧区行发布进池——与真实区同机制，从而随 MLX 的 buffer 迁移一起存活。每行只在字节首次到达时发布一次（脏行跟踪），之后像真实区一样长存。

**Tech Stack:** C++17 + MLX 0.31（nanobind 绑定，`native_moe_ext`）、Python 3.12、pytest。

---

## 背景与根因（必读）

已用 systematic-debugging 定位（详见 `benchmarks/reports/schemeB-mismatch-rootcause-2026-07-04.md`）：

- 侧区预取在 `PrefetchPoolSideRegionPrimitive::read_publish`（`native/ext/native_prefetch.cpp:573-627`）里，把磁盘字节 `memcpy` 进 `ptrs[k] = in[k].data()`，即 Python `_pools[layer][k]` 这块 MLX 数组的 buffer。
- 这块 MLX 池数组的**底层 buffer 会被 MLX 在解码前向之间重定位**（实测 `obj_id` 不变、`array_data_ptr` 逐前向漂移；补 `mx.eval` 无效；`PTR_PROBE` 证明与真实区 scatter 无关）。
- 真实区字节走 `pool[k][idx]=v`（MLX 追踪的 scatter），能随迁移保留；侧区旁路 `memcpy` 对 MLX 不可见，被丢在孤儿 buffer → 消费侧 matmul 读到旧/别专家字节 → `DUAL_VERIFY BAD`。

**结论**：修法只能是「让侧区字节走 MLX 追踪的写」。本计划即 Route 1。

**关键不变量**（贯穿全计划）：
- 物理行布局：真实区 `[0, cap)`；侧区 gen g 行 `[cap+g*spec, cap+(g+1)*spec)`，`base_row = cap+g*spec`。
- `local[i]` = 物理行号；`sideregion_kv` 只在 `read_publish` 把 e2r 发布后才暴露 `expert→row`，即「kv 里出现 E→R」⇒「R 的字节已在稳定缓冲区就绪」。
- blob 一条记录 = 各段字节拼接，总长 `stride`；段顺序 = `src._segs` 顺序 = 池 dict key 顺序 = `seg_nbytes` 顺序。

---

## 文件结构（改动清单）

- Modify: `native/ext/native_prefetch.cpp` — `SideLayer` 加稳定字节缓冲元数据；`read_publish` 把字节写进 C++ 稳定缓冲 + 标脏；新增 `sideregion_publish`；`sideregion_reset` 一并清缓冲。
- Modify: `native/ext/native_prefetch.h`（若有声明）/ `native/ext/native_bindings.cpp` — 导出 `sideregion_publish`。
- Modify: `mlx_streaming/core/prefetch/native_staging.py` — `NativeStagingManager` 加 `sideregion_publish` 委托；`_StagingSide` 加 `publish(layer)`（按段拆成 per-key 具类型数组）。
- Modify: `mlx_streaming/core/cache/resident_pool.py` — `acquire_gpu_dual` 在 `has_side` 时调用发布，把新到侧区行以 MLX scatter 落池。
- Test:
  - `mlx_streaming/tests/test_sideregion_publish_native.py`（新建）— C++ 稳定缓冲 + `sideregion_publish` 字节/脏行正确性。
  - `mlx_streaming/tests/test_sideregion_publish_wiring.py`（新建）— Python `_StagingSide.publish` 拆段/dtype/scatter 语义。
  - 复用 `mlx_streaming/runtime/run_mtp_spec.py` 的 `DUAL_VERIFY` 做集成验收。

---

## Task 0：核心前提确认（一次性 spike，验后即弃）

**目的**：在动 C++ 前，先用最便宜方式确认「MLX 追踪的 scatter 写入侧区行，能跨前向随 buffer 迁移存活」这一前提（真实区已隐含证明，但显式验证使后续任务可信）。

**Files:**
- Modify（临时，Task 4 前回退）: `mlx_streaming/core/cache/resident_pool.py`

- [ ] **Step 1：在 `acquire_gpu_dual` 的 `has_side` 分支加临时发布（re-read 版）**

在 `resident_pool.py` 的 `acquire_gpu_dual` 里，`keys, vals = side.kv(layer)` 与 `has_side = ...` 之后插入（临时、门控 `SPIKE_PUBLISH=1`）：

```python
        keys, vals = side.kv(layer)
        has_side = int(keys.size) > 0
        if has_side and os.environ.get("SPIKE_PUBLISH") == "1":
            # 临时 spike：用 loader 重读侧区命中专家真值，MLX scatter 落池，验证「追踪写跨前向存活」。
            pool = self._pools[layer]
            sc = {int(k): int(v) for k, v in zip(keys.tolist(), vals.tolist())}
            for e, row in sc.items():
                truth = self.loader(layer, e)
                for k in pool:
                    if k in truth:
                        pool[k][row] = truth[k]
```

- [ ] **Step 2：跑短复现，看 DUAL_VERIFY 是否清零**

Run:
```bash
SPIKE_PUBLISH=1 DUAL_VERIFY=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 \
  EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 \
  K=3 MAXTOK=8 WARMUP_TOK=0 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | grep -E "BAD|VERIFY_SUMMARY"
```
Expected: 无 `[DUAL_VERIFY] BAD`；`VERIFY_SUMMARY` 中 `_verify_side_bytes` 的 `bad=0`。
- 若 0 BAD → 前提成立，进入 Task 1（正式实现，避免每前向 re-read）。
- 若仍有 BAD → **停止**，前提不成立，回到 systematic-debugging 复核（可能 scatter 也被迁移影响，需重新评估 Route 1 可行性）。

- [ ] **Step 3：回退 spike 代码**

删除 Step 1 加入的 `SPIKE_PUBLISH` 临时块（不提交）。

- [ ] **Step 4：提交（仅记录结论，无代码）**

不产生代码提交；把「0 BAD 已确认」记入 Task 4 的验收备注。

---

## Task 1：C++ 侧区稳定字节缓冲 + read_publish 改写 + 脏行跟踪

**Files:**
- Modify: `native/ext/native_prefetch.cpp:366-373`（`SideLayer` 定义）
- Modify: `native/ext/native_prefetch.cpp:573-627`（`read_publish`）
- Modify: `native/ext/native_prefetch.cpp:702-705`（`sideregion_reset`）
- Test: `mlx_streaming/tests/test_sideregion_publish_native.py`

- [ ] **Step 1：写失败测试（字节进稳定缓冲，不再依赖池数组）**

新建 `mlx_streaming/tests/test_sideregion_publish_native.py`：

```python
import os
import struct
import tempfile
import numpy as np
import mlx.core as mx
import pytest

nmoe = pytest.importorskip("mlx_streaming.native_moe_ext")


def _make_blob(path, n_experts, stride):
    # 每个专家一条 stride 字节记录，内容 = 专家 id 的可辨识填充。
    with open(path, "wb") as f:
        for e in range(n_experts):
            f.write(bytes([(e * 7 + i) & 0xFF for i in range(stride)]))


def test_sideregion_publish_returns_written_bytes():
    nmoe.sideregion_reset()
    layer, gen, cap, spec = 0, 0, 4, 3
    stride = 16
    seg_nbytes = [8, 8]           # 两段，各 8 字节，和 = stride
    n_experts = 10
    with tempfile.TemporaryDirectory() as d:
        blob = os.path.join(d, "layer0.blob")
        _make_blob(blob, n_experts, stride)
        base = cap                # 侧区 gen0 起始物理行
        # 用一个 dummy 池（本实现不再写它，仅为兼容 prefetch_pool_sideregion 签名）。
        pool = [mx.zeros((cap + spec, s), dtype=mx.uint8) for s in seg_nbytes]
        mx.eval(pool)
        ids = mx.array([5, 7], dtype=mx.uint32)   # 预取专家 5、7 → 侧区
        tok = nmoe.prefetch_pool_sideregion(
            pool, seg_nbytes, ids, layer, blob, stride,
            [], spec, base, gen)                   # resident=[]（都 miss）
        mx.eval(tok)
        # 等异步 read_publish 完成（同步模式或轮询 e2r）。
        import time
        for _ in range(200):
            if len(nmoe.sideregion_contents(layer, gen)) >= 4:
                break
            time.sleep(0.01)
        rows, seg_arrays = nmoe.sideregion_publish(layer, gen, seg_nbytes)
        assert int(rows.shape[0]) == 2
        assert len(seg_arrays) == 2
        # 校验字节 == 磁盘真值：把 rows 对应的专家真值拼出来逐段比对。
        kv = dict(zip(*[a.tolist() for a in nmoe.sideregion_kv(layer, gen)]))  # expert->row
        row2exp = {int(r): int(e) for e, r in kv.items()}
        for i, r in enumerate(rows.tolist()):
            e = row2exp[int(r)]
            truth = bytes([(e * 7 + j) & 0xFF for j in range(stride)])
            off = 0
            for k, nb in enumerate(seg_nbytes):
                got = bytes(np.array(seg_arrays[k][i]).tobytes())
                assert got == truth[off:off + nb], (e, k, got, truth[off:off+nb])
                off += nb


def test_sideregion_publish_clears_dirty():
    nmoe.sideregion_reset()
    layer, gen, cap, spec, stride = 0, 0, 4, 3, 16
    seg_nbytes = [8, 8]
    with tempfile.TemporaryDirectory() as d:
        blob = os.path.join(d, "layer0.blob")
        _make_blob(blob, 10, stride)
        pool = [mx.zeros((cap + spec, s), dtype=mx.uint8) for s in seg_nbytes]
        mx.eval(pool)
        nmoe.prefetch_pool_sideregion(pool, seg_nbytes, mx.array([5], dtype=mx.uint32),
                                      layer, blob, stride, [], spec, cap, gen)
        import time
        for _ in range(200):
            if len(nmoe.sideregion_contents(layer, gen)) >= 2:
                break
            time.sleep(0.01)
        rows1, _ = nmoe.sideregion_publish(layer, gen, seg_nbytes)
        assert int(rows1.shape[0]) == 1          # 首次发布：1 行脏
        rows2, _ = nmoe.sideregion_publish(layer, gen, seg_nbytes)
        assert int(rows2.shape[0]) == 0          # 再次发布：脏已清，0 行
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
cd native/ext && make native_moe_ext && cd ../.. && \
  .venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_publish_native.py -v
```
Expected: FAIL，`AttributeError: module 'mlx_streaming.native_moe_ext' has no attribute 'sideregion_publish'`。

- [ ] **Step 3：`SideLayer` 加稳定缓冲元数据 + 全局字节缓冲**

在 `native/ext/native_prefetch.cpp` 把 `SideLayer`（366-373）改为：

```cpp
struct SideLayer {
  std::map<int, int> e2r;          // expert -> 物理侧区行 [base, base+spec)
  std::vector<int> free_rows;
  std::map<int, uint32_t> freq;    // expert -> 预测频次(LFU 分数;仅 SIDEREGION_LFU 用)
  bool inited = false;
  int base = 0;                    // 本代侧区起始物理行(= cap + gen*spec)
  int spec = 0;                    // 本代侧区行数
  std::set<int> dirty;             // 自上次 publish 起新写入字节、待发布进池的物理行
};
static std::mutex g_side_mutex;
static std::map<std::pair<int, int>, SideLayer> g_side;   // 键 (layer, gen)：双缓冲两代独立

// C++ 拥有的稳定字节缓冲：键 (layer, gen)，大小 spec*stride，按 (row-base)*stride 索引。
// 侧区异步预取只写这里(永不被 MLX 迁移)；消费时由 sideregion_publish 取出、Python 以 MLX scatter 落池。
static std::map<std::pair<int, int>, std::vector<uint8_t>> g_side_bytes;
```

确认文件顶部已 `#include <set>`；若无则加入。

- [ ] **Step 4：`reserve` 记录 base/spec 到 SideLayer**

在 `reserve`（468-544）里，把 init 段（485-488）改为同时记录 base/spec：

```cpp
    SideLayer& c = g_side[{layer, gen}];
    if (!c.inited) {
      for (int r = 0; r < spec; ++r) c.free_rows.push_back(base + r);
      c.base = base;
      c.spec = spec;
      c.inited = true;
    }
```

- [ ] **Step 5：`read_publish` 把字节写进稳定缓冲 + 标脏（不再写池 ptrs）**

把 `read_publish`（573-627）改为：I/O 前在锁内确保缓冲大小并取 base；I/O 循环把 `tmp` 整条写进稳定缓冲的 `(row-base)*stride` 处；发布 e2r 时把 row 记入 dirty。完整替换实现：

```cpp
  static void read_publish(const std::vector<uint8_t*>& ptrs, const std::vector<int>& seg,
                           const std::vector<std::pair<int, int>>& to_read,
                           const std::string& path, size_t stride, int layer, int gen) {
    (void)ptrs;   // Route 1：不再旁路写池；字节改写 C++ 稳定缓冲，消费侧以 MLX scatter 发布。
    int base = 0;
    {   // 锁内：确保稳定缓冲已按 spec*stride 预分配（只分配一次，之后永不 resize）。
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      base = c.base;
      auto& buf = g_side_bytes[{layer, gen}];
      size_t want = static_cast<size_t>(c.spec) * stride;
      if (buf.size() != want) buf.assign(want, 0);
    }
    uint8_t* bufp;
    {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      bufp = g_side_bytes[{layer, gen}].data();   // 预分配后地址稳定(不 resize)
    }
    int fd = open_blob_nocache(path.c_str());
    if (fd < 0) {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : to_read) c.free_rows.push_back(pr.second);
      return;
    }
    std::vector<std::pair<int, int>> done;
    for (auto& pr : to_read) {
      int e = pr.first, row = pr.second;
      uint8_t* dst = bufp + static_cast<size_t>(row - base) * stride;   // 直接读进稳定缓冲该行
      if (::pread(fd, dst, stride, static_cast<off_t>(static_cast<size_t>(e) * stride)) !=
          static_cast<ssize_t>(stride)) {
        std::lock_guard<std::mutex> lk(g_side_mutex);
        g_side[{layer, gen}].free_rows.push_back(row);
        continue;
      }
      if (side_trace_hit(layer, row))
        fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d READBUF row=%d expert=%d\n",
                (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, e);
      done.emplace_back(e, row);
    }
    ::close(fd);
    {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : done) {                    // 字节就绪后才发布 e2r + 标脏
        c.e2r[pr.first] = pr.second;
        if (!c.freq.count(pr.first)) c.freq[pr.first] = 1;
        c.dirty.insert(pr.second);
        if (side_trace_hit(layer, pr.second))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d PUBLISH row=%d expert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, pr.second,
                  pr.first);
      }
      if (std::getenv("SIDE_AUDIT")) side_audit(c, layer, gen, "publish", {});
    }
    g_pf_fires.fetch_add(1);
  }
```

注：`open_blob_nocache` / `g_pf_fires` / `side_trace_hit` / `side_audit` 沿用现有定义，签名不变。

- [ ] **Step 6：`sideregion_reset` 一并清字节缓冲**

把 `sideregion_reset`（702-705）改为：

```cpp
void sideregion_reset() {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  g_side.clear();
  g_side_bytes.clear();
}
```

- [ ] **Step 7：实现 `sideregion_publish`（取脏行字节，按段拆成 per-seg 连续数组，清脏）**

在 `sideregion_reset` 之后新增：

```cpp
// 取出本代侧区自上次发布起新写入的行，按段拆成 per-seg 连续 uint8 数组返回，并清脏。
// 返回 (rows int32[m], per_seg[每个 uint8 (m, seg_nbytes[k])])；供 Python 以 MLX scatter 落池。
std::pair<mx::array, std::vector<mx::array>> sideregion_publish(
    int layer, int gen, const std::vector<int>& seg_nbytes) {
  std::vector<int> seg_off(seg_nbytes.size(), 0);
  size_t stride = 0;
  for (size_t k = 0; k < seg_nbytes.size(); ++k) { seg_off[k] = static_cast<int>(stride); stride += seg_nbytes[k]; }

  std::vector<int32_t> rows;
  std::vector<std::vector<uint8_t>> segbuf(seg_nbytes.size());
  {
    std::lock_guard<std::mutex> lk(g_side_mutex);
    auto it = g_side.find({layer, gen});
    auto bit = g_side_bytes.find({layer, gen});
    if (it != g_side.end() && bit != g_side_bytes.end() && !it->second.dirty.empty()) {
      SideLayer& c = it->second;
      const std::vector<uint8_t>& buf = bit->second;
      int m = static_cast<int>(c.dirty.size());
      rows.reserve(m);
      for (size_t k = 0; k < seg_nbytes.size(); ++k)
        segbuf[k].resize(static_cast<size_t>(m) * seg_nbytes[k]);
      int i = 0;
      for (int row : c.dirty) {
        rows.push_back(row);
        size_t rbase = static_cast<size_t>(row - c.base) * stride;
        for (size_t k = 0; k < seg_nbytes.size(); ++k)
          std::memcpy(segbuf[k].data() + static_cast<size_t>(i) * seg_nbytes[k],
                      buf.data() + rbase + seg_off[k], seg_nbytes[k]);
        ++i;
      }
      c.dirty.clear();
    }
  }
  int m = static_cast<int>(rows.size());
  std::vector<mx::array> out;
  out.reserve(seg_nbytes.size());
  for (size_t k = 0; k < seg_nbytes.size(); ++k)
    out.push_back(mx::array(segbuf[k].data(), mx::Shape{m, seg_nbytes[k]}, mx::uint8));
  return {mx::array(rows.data(), mx::Shape{m}, mx::int32), out};
}
```

- [ ] **Step 8：导出绑定**

若 `native/ext/native_prefetch.h`（或对应头）声明了 `sideregion_*`，加上：

```cpp
std::pair<mx::array, std::vector<mx::array>> sideregion_publish(
    int layer, int gen, const std::vector<int>& seg_nbytes);
```

在 `native/ext/native_bindings.cpp` 的 `sideregion_reset` 绑定之后加：

```cpp
  m.def("sideregion_publish", &sideregion_publish, "layer"_a, "gen"_a, "seg_nbytes"_a);
```

- [ ] **Step 9：编译并运行测试确认通过**

Run:
```bash
cd native/ext && make native_moe_ext && cd ../.. && \
  .venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_publish_native.py -v
```
Expected: 2 passed。

- [ ] **Step 10：提交**

```bash
git add native/ext/native_prefetch.cpp native/ext/native_bindings.cpp \
  native/ext/native_prefetch.h mlx_streaming/tests/test_sideregion_publish_native.py
git commit -m "feat(native): 侧区字节改写 C++ 稳定缓冲 + sideregion_publish 脏行取出"
```

---

## Task 2：Python 发布适配（NativeStagingManager + _StagingSide.publish）

**Files:**
- Modify: `mlx_streaming/core/prefetch/native_staging.py:229-244`（`_StagingSide`）+ `NativeStagingManager` 委托方法
- Test: `mlx_streaming/tests/test_sideregion_publish_wiring.py`

- [ ] **Step 1：写失败测试（拆段/dtype/形状正确）**

新建 `mlx_streaming/tests/test_sideregion_publish_wiring.py`：

```python
import mlx.core as mx
import pytest


class _FakeSrc:
    # 两段：weight(uint32, (2,2)=16B) + scales(uint16->bf16, (2,2)=8B)
    _segs = [
        ("gate_proj", "weight", "uint32", (2, 2), 16),
        ("gate_proj", "scales", "uint16", (2, 2), 8),
    ]


class _FakeStg:
    def __init__(self):
        self.src = _FakeSrc()
    def sideregion_publish(self, layer, gen, seg_nbytes):
        assert seg_nbytes == [16, 8]
        rows = mx.array([5, 6], dtype=mx.int32)
        # 段0：uint32 全 1；段1：uint16 全 0x3f80(=bf16 1.0)
        w = mx.full((2, 16), 0, dtype=mx.uint8)
        s = mx.zeros((2, 8), dtype=mx.uint8)
        return rows, [w, s]


def test_staging_side_publish_splits_segments():
    from mlx_streaming.core.prefetch.native_staging import _StagingSide
    side = _StagingSide(_FakeStg(), gen=0)
    rows, out = side.publish(0)
    assert int(rows.shape[0]) == 2
    assert set(out.keys()) == {"gate_proj.weight", "gate_proj.scales"}
    assert out["gate_proj.weight"].shape == (2, 2, 2)
    assert out["gate_proj.weight"].dtype == mx.uint32
    assert out["gate_proj.scales"].shape == (2, 2, 2)
    assert out["gate_proj.scales"].dtype == mx.bfloat16


def test_staging_side_publish_empty():
    class _EmptyStg(_FakeStg):
        def sideregion_publish(self, layer, gen, seg_nbytes):
            return mx.array([], dtype=mx.int32), [mx.zeros((0, 16), dtype=mx.uint8),
                                                  mx.zeros((0, 8), dtype=mx.uint8)]
    from mlx_streaming.core.prefetch.native_staging import _StagingSide
    side = _StagingSide(_EmptyStg(), gen=0)
    rows, out = side.publish(0)
    assert int(rows.shape[0]) == 0
    assert out == {}
```

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_publish_wiring.py -v
```
Expected: FAIL，`AttributeError: '_StagingSide' object has no attribute 'publish'`。

- [ ] **Step 3：`NativeStagingManager` 加委托方法**

在 `native_staging.py` 的 `NativeStagingManager` 类里（与已有 `sideregion_kv` 委托并列）加：

```python
    def sideregion_publish(self, layer, gen, seg_nbytes):
        # C++ 取出本代侧区新到行字节，按段拆成 per-seg uint8 数组，供消费侧 MLX scatter 落池。
        return self._ext.sideregion_publish(int(layer), int(gen), list(seg_nbytes))
```

> 注：`self._ext` 为该类持有的 `native_moe_ext` 句柄；若类中用别的属性名（如 `self._stg` 指向底层对象），按现有 `sideregion_kv` 的委托写法对齐（读一下同类 `sideregion_kv` 方法即可确认属性名）。

- [ ] **Step 4：`_StagingSide` 加 `publish`（复用 `native_staging.py:210-226` 的按段拆分逻辑）**

在 `_StagingSide`（229-244）里加：

```python
    def publish(self, layer):
        """取本代侧区新到行字节，按段拆成 per-key 具类型数组，供消费侧 MLX scatter 落池。

        返回 (rows int32[m], {key: (m,*shape)})；m=0 表示无新行(返回空 dict)。
        dtype 分派与 native_staging 行装配一致：uint32 权重原样；uint16→bfloat16；uint8(mxfp4 scales)原样。
        """
        seg_nbytes = [nb for *_, nb in self._stg.src._segs]
        rows, seg_arrays = self._stg.sideregion_publish(layer, self._gen, seg_nbytes)
        m = int(rows.shape[0]) if rows.ndim else 0
        if m == 0:
            return rows, {}
        out = {}
        for (proj, tensor, dt, shape, _nb), seg in zip(self._stg.src._segs, seg_arrays):
            if dt == "uint32":
                arr = seg.view(mx.uint32).reshape((m,) + tuple(shape))
            elif dt == "uint16":
                arr = seg.view(mx.uint16).reshape((m,) + tuple(shape)).view(mx.bfloat16)
            else:  # uint8（mxfp4 scales）
                arr = seg.reshape((m,) + tuple(shape))
            out[f"{proj}.{tensor}"] = arr
        return rows, out
```

- [ ] **Step 5：运行测试确认通过**

Run:
```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_publish_wiring.py -v
```
Expected: 2 passed。

- [ ] **Step 6：提交**

```bash
git add mlx_streaming/core/prefetch/native_staging.py mlx_streaming/tests/test_sideregion_publish_wiring.py
git commit -m "feat(staging): _StagingSide.publish 按段拆出侧区新行供 MLX 追踪落池"
```

---

## Task 3：消费侧接入（acquire_gpu_dual 在 has_side 时发布落池）

**Files:**
- Modify: `mlx_streaming/core/cache/resident_pool.py:613-685`（`acquire_gpu_dual`）
- Test: `mlx_streaming/tests/test_sideregion_publish_wiring.py`（追加消费侧单测）

- [ ] **Step 1：追加失败测试（发布后池对应行字节 == 提供的 typed）**

在 `mlx_streaming/tests/test_sideregion_publish_wiring.py` 末尾追加：

```python
def test_acquire_gpu_dual_publishes_side_rows(monkeypatch):
    import mlx.core as mx
    from mlx_streaming.core.cache.resident_pool import ResidentExpertPool

    # 造一个最小池：cap=2, spec=2, 单 key。
    rp = ResidentExpertPool(loader=lambda l, e: {"gate_proj.weight": mx.zeros((2, 2), mx.uint32)},
                            capacity=2, spec_slots=2, spec_gens=1)
    layer, num_experts = 0, 8
    rp._pools[layer] = {"gate_proj.weight": mx.zeros((4, 2, 2), dtype=mx.uint32)}
    mx.eval(list(rp._pools[layer].values()))
    rp._alloc[layer] = 4
    rp._slot_table[layer] = mx.array([-1] * num_experts, dtype=mx.int32)

    class _Side:
        def kv(self, l):
            # 专家 5 → 物理行 2（侧区），6 → 行 3
            return mx.array([5, 6], dtype=mx.uint32), mx.array([2, 3], dtype=mx.int32)
        def publish(self, l):
            rows = mx.array([2, 3], dtype=mx.int32)
            payload = mx.stack([mx.full((2, 2), 55, dtype=mx.uint32),
                                mx.full((2, 2), 66, dtype=mx.uint32)], axis=0)
            return rows, {"gate_proj.weight": payload}

    inds = mx.array([[5, 6]], dtype=mx.int32)
    pool, local = rp.acquire_gpu_dual(layer, inds, num_experts, _Side())
    mx.eval(pool["gate_proj.weight"])
    # 行 2 应为全 55，行 3 应为全 66（发布已落池）。
    assert bool(mx.all(pool["gate_proj.weight"][2] == 55))
    assert bool(mx.all(pool["gate_proj.weight"][3] == 66))
```

> 注：`ResidentExpertPool.__init__` 的确切签名以源码为准（`capacity`/`spec_slots`/`spec_gens` 命名读一下构造函数确认；若不同则相应调整测试的构造实参）。

- [ ] **Step 2：运行测试确认失败**

Run:
```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_publish_wiring.py::test_acquire_gpu_dual_publishes_side_rows -v
```
Expected: FAIL（行 2/3 仍为 0，未发布）。

- [ ] **Step 3：在 `acquire_gpu_dual` 加发布 + 私有方法 `_publish_side`**

在 `resident_pool.py` 的 `acquire_gpu_dual`（613-685），把 `has_side = ...` 之后改为：

```python
        keys, vals = side.kv(layer)
        has_side = int(keys.size) > 0                    # .size 只读 shape，无 GPU 同步
        if has_side and layer in self._pools:
            self._publish_side(layer, side)              # 新到侧区行以 MLX 追踪 scatter 落池(随迁移存活)
        eff = mx.array(base) if has_side else base       # 无侧区条目时直接用 base，省掉每层一次 256-int 拷贝
```

在 `ResidentExpertPool` 类里（`_verify_side_bytes` 附近）新增：

```python
    def _publish_side(self, layer, side):
        """把侧区自上次起新到达的行(C++ 稳定缓冲)以 MLX 追踪 scatter 写进池。

        每行只在字节首次到达时发布一次；MLX 追踪的写随 buffer 迁移存活，故之后无需重发。
        仅当 side 适配器实现 publish 时生效(向后兼容旧适配器)。
        """
        pub = getattr(side, "publish", None)
        if pub is None:
            return
        rows, typed = pub(layer)
        m = int(rows.shape[0]) if rows.ndim else 0
        if m == 0:
            return
        pool = self._pools[layer]
        for k, arr in typed.items():
            pool[k][rows] = arr
```

- [ ] **Step 4：运行测试确认通过**

Run:
```bash
.venv/bin/python -m pytest mlx_streaming/tests/test_sideregion_publish_wiring.py -v
```
Expected: 3 passed。

- [ ] **Step 5：提交**

```bash
git add mlx_streaming/core/cache/resident_pool.py mlx_streaming/tests/test_sideregion_publish_wiring.py
git commit -m "feat(pool): acquire_gpu_dual 发布侧区新行到池(MLX 追踪 scatter)"
```

---

## Task 4：集成验收（DUAL_VERIFY 0 BAD）

**Files:**
- 无代码改动（若 Task 0 的 spike 未回退，先回退）。

- [ ] **Step 1：短复现，确认 0 BAD**

Run:
```bash
DUAL_VERIFY=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 \
  ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=8 \
  WARMUP_TOK=0 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | grep -E "BAD|VERIFY_SUMMARY"
```
Expected: 无 `[DUAL_VERIFY] BAD`；`VERIFY_SUMMARY` 中 `DUAL_VERIFY.resident(_verify_side_bytes)` 的 `bad=0`（`ok>0`）。

- [ ] **Step 2：长跑确认稳定 0 BAD**

Run:
```bash
DUAL_VERIFY=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 \
  ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=128 \
  WARMUP_TOK=8 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | grep -E "BAD|VERIFY_SUMMARY"
```
Expected: `bad=0`。锚点 `L1 e453 r32`（历史必崩点）不再出现。

- [ ] **Step 3：跑既有相关测试确认无回归**

Run:
```bash
.venv/bin/python -m pytest \
  mlx_streaming/tests/test_resident_sideregion.py \
  mlx_streaming/tests/test_pool_sideregion_native.py \
  mlx_streaming/tests/test_sideregion_gen_native.py \
  mlx_streaming/tests/test_sideregion_lfu_native.py \
  mlx_streaming/tests/test_dual_source_verify_shape.py \
  mlx_streaming/tests/test_dual_fallback_gpu_miss.py -v
```
Expected: all passed（如个别测试断言了「侧区字节写进池数组」的旧机制，需按新机制更新断言——见 Step 4）。

- [ ] **Step 4：修既有测试对旧机制的断言（若有）**

若 `test_pool_sideregion_native.py` 断言「prefetch 后 `pool[k][row]` 含专家字节」，改为断言「`sideregion_publish` 返回的字节含专家真值」（旧路径已不再写池）。逐个按失败信息更新，保持语义等价。

- [ ] **Step 5：提交**

```bash
git add mlx_streaming/tests/
git commit -m "test: 侧区测试对齐稳定缓冲+发布新机制"
```

---

## Task 5：容量不变性回归 + 收尾

**Files:**
- 无核心代码改动；仅诊断清理与文档。

- [ ] **Step 1：容量不变性抽查（cap 变、输出 token 不变）**

Run（两种 cap 各跑贪心、对比输出 token 序列一致）：
```bash
for CAP in 32 48; do
  DUMP_IDS=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=$CAP \
    ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=32 \
    WARMUP_TOK=0 REPEAT=1 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | grep DUMP_BASE_IDS
done
```
Expected: 两次 `DUMP_BASE_IDS` 序列一致（字节真值已修复后，容量不变性应成立；若仍有少量差异，记录并与 `benchmarks/reports/fp-noise-floor-2026-07-04.md` 的 N_floor 讨论对齐）。

- [ ] **Step 2：清理临时诊断（可选，按需保留门控探针）**

保留 `DUAL_VERIFY`/`SIDE_TRACE`/`SIDE_AUDIT` 等 env 门控诊断（默认关，零热路径开销）；确认无遗留 `SPIKE_PUBLISH` 临时代码。

- [ ] **Step 3：写验收报告**

新建 `benchmarks/reports/sideregion-route1-fix-2026-07-04.md`，记录：根因一句话、修法（稳定缓冲+MLX 追踪发布）、验收证据（短/长跑 0 BAD、容量不变性、既有测试通过）、以及遗留项（route 3 真零拷贝作为后续独立性能项目）。

- [ ] **Step 4：提交**

```bash
git add benchmarks/reports/sideregion-route1-fix-2026-07-04.md
git commit -m "docs: 侧区 route1 修复验收报告"
```

---

## Self-Review（作者自查结论）

**1. Spec 覆盖**：根因（旁路 memcpy 落到被迁移的孤儿 buffer）→ Task 1（改写稳定缓冲）；发布机制 → Task 2/3；验收 `DUAL_VERIFY 0 BAD` → Task 4；容量不变性 → Task 5。前提风险由 Task 0 spike 兜底。

**2. 占位符扫描**：无 TBD/TODO；每个改代码步骤均含真实代码；两处「以源码为准」的注记（`self._ext` 属性名、`ResidentExpertPool.__init__` 签名）已明确指出「读同类现有方法/构造函数确认」的具体动作，非占位。

**3. 类型一致性**：
- C++ `sideregion_publish(int layer,int gen,vector<int> seg_nbytes) -> pair<mx::array(int32[m]), vector<mx::array(uint8[m,nb])>>` 在 Task 1 定义、Task 2 绑定/委托、Task 3 消费，签名一致。
- Python `_StagingSide.publish(layer) -> (rows int32[m], {key:(m,*shape)})` 在 Task 2 定义、Task 3 `_publish_side` 消费，形状/dtype 一致（uint32/bf16/uint8 分派与 `native_staging.py:216-224` 对齐）。
- 物理行/base/spec 语义在 Task 1 各处一致（`row-base` 索引稳定缓冲）。

**潜在坑（实现时留意）**：
- MLX `array.view(dtype)` 要求源连续——Task 1 让 C++ 返回**每段独立连续** uint8 数组，故 Python `seg.view(...)` 安全（不做 strided 列切片）。
- 并发：稳定缓冲预分配后**永不 resize**；bg 线程写不同 row 与主线程 publish 读，靠 `g_side_mutex` 在 dirty 发布/读取处建立 happens-before。
- 向后兼容：`_publish_side` 对无 `publish` 的旧 side 适配器直接跳过。

---

## 执行交接

计划已存 `docs/superpowers/plans/2026-07-04-sideregion-route1-stable-publish.md`。两种执行方式：

1. **Subagent-Driven（推荐）** — 每个 Task 派新 subagent，任务间双阶段 review，快速迭代。
2. **Inline Execution** — 本会话内按 executing-plans 批量执行、检查点 review。
