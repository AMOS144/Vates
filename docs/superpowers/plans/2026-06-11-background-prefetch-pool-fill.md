# 后台预取 + 池预填(Background Prefetch Pool-Fill)实现计划

> **For agentic workers:** 用 superpowers:subagent-driven-development 或 executing-plans 逐任务执行；每步 checkbox。

**Goal:** 在后台线程(独立 MLX stream)上提前物化"预测的下一层专家",交接给主线程写入常驻池槽位;主线程算到该层时槽已填好 → `acquire_gpu` 全命中 → 消除每 miss 的 `.tolist` 编排。目标:小池(低内存)也能逼近大池吞吐。

**Architecture(已由 gating 测试 A/C 验证):**
- 后台预取线程持有 `s2 = mx.new_stream(gpu)`;`with mx.stream(s2):` 里把专家字节物化成**私有 mx.array** 并 `mx.eval`(与主线程计算重叠,Test A 证明可行)。
- **不**让后台直接写主线程共享池(Test B 报 stream-context 错);改为**交接**给主线程,主线程把已物化 array 拷进池槽(Test C 证明正确)。
- 复用现有 `BlobExpertSource`(并行 pread + 物化)、`FileExpertStore/ResidentExpertPool`(`acquire_gpu` GPU slot 表快路径)、`enable_cross_layer_prefetch`(跨层预测)。

**Tech Stack:** Python、MLX(`mx.new_stream`/`mx.stream`/`mx.eval`)、threading、现有 blob_loader / expert_store / streaming_moe。

**前置已验证:** gating A(后台 s2 物化+eval 重叠不崩)、C(私有物化→交接→主线程写池,正确)。见 `mlx_streaming/cli/probe_multistream_gate.py` / `probe_multistream_handoff.py`。

---

## 关键约束(从 gating 得到的硬规则)
1. 后台线程**只能物化私有 array**,且必须包在 `with mx.stream(s2):` 内并 `mx.eval`。
2. 后台**绝不**写主线程拥有的池张量(会 `no Stream(gpu,0)` 报错)。
3. 主线程**只消费已 eval(已物化)的交接 array** → 跨 stream 读安全。
4. 未及时就绪 → **优雅回退**到现有同步 demand 路径(不阻塞、不报错)。

## 关于 C++ / GIL / 拷池的取舍(决定哪里用 C++)
- **GIL(后台 load)**:后台用 Python `mx.array(np.frombuffer)` 物化时拷贝段持 GIL,与主线程 Python 抢。可选用 C++ `blob_load` primitive(eval_gpu 内 pread 进 `mlx::allocator::malloc` 自有 buffer,期间释放 GIL,不撞 Invalid Resource)做后台 load → 降 GIL 竞争。
  - 注意:实测 `blob_load` vs lazy 物化仅 **1.05×**(不加快 load 本身,价值只在释放 GIL);且 gating Test A 中纯 Python 后台物化**已能重叠**。→ **做成开关 `STREAM_BLOB_BG_NATIVE`,Task 4 实测了再决定是否默认开**。
- **拷池(promote)→ 保持在主线程**:Test C 的 handoff 已把贵活(物化)放后台、主线程只做**便宜的 scatter**(拷已物化数据,无 IO/大拷贝)。
  - **不要**用"后台 primitive 直接写池"消除它:那要么跨 stream 写共享池(Test B 脆),要么把 pread 拉回主线程(丢重叠)。核心矛盾:**要重叠→load 在后台;要安全写池→在主线程,同一操作不可兼得**。
  - 正确做法:Task 4 **先实测 promote 的主线程开销**;若确实显著,再单独 de-risk(不在本计划默认范围)。

---

## Task 1: BackgroundExpertPrefetcher(新模块)

**Files:**
- Create: `mlx_streaming/core/bg_prefetch.py`
- Test: `mlx_streaming/tests/test_bg_prefetch.py`

- [ ] **Step 1: 写失败测试**

```python
# test_bg_prefetch.py
import os, time, mlx.core as mx, pytest
from mlx_streaming.core.blob_loader import BlobExpertSource
from mlx_streaming.core.bg_prefetch import BackgroundExpertPrefetcher

BLOB = "/tmp/cb_2bit_blob"

@pytest.mark.skipif(not os.path.exists(os.path.join(BLOB, "layer15.blob")), reason="需要 blob")
def test_bg_prefetch_materializes_and_handoff():
    src = BlobExpertSource(BLOB, 2048, 512, 128, 2, num_experts=512)
    pf = BackgroundExpertPrefetcher(src)
    try:
        pf.submit(15, [3, 7, 100])
        # 等就绪(最多 2s)
        deadline = time.time() + 2
        while pf.ready_count(15) < 3 and time.time() < deadline:
            time.sleep(0.01)
        got = pf.take_ready(15, 7)
        assert got is not None
        ref = src.load_experts(15, [7])[7]
        for k in ref:
            assert bool(mx.all(got[k] == ref[k]).item())
    finally:
        pf.close(); src.close()
```

- [ ] **Step 2: 跑红** `uv run pytest mlx_streaming/tests/test_bg_prefetch.py -x`(ImportError)

- [ ] **Step 3: 实现 `BackgroundExpertPrefetcher`**

```python
# bg_prefetch.py
import os, threading, queue
from collections import OrderedDict
import mlx.core as mx

class BackgroundExpertPrefetcher:
    """后台线程在独立 stream 上物化预测专家(私有 array),交接给主线程。"""
    def __init__(self, blob_source, window=3):
        self._src = blob_source
        self._stream = mx.new_stream(mx.default_device())
        self._q = queue.Queue()
        self._ready = OrderedDict()          # (layer,e) -> {proj.tensor: mx.array}
        self._ready_layers = []
        self._window = window
        self._lock = threading.Lock()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def submit(self, layer, expert_ids):
        self._q.put((int(layer), [int(e) for e in expert_ids]))

    def _loop(self):
        while not self._stop:
            try:
                layer, ids = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                with mx.stream(self._stream):
                    experts = self._src.load_experts(layer, ids)   # 私有物化
                    mx.eval([v for d in experts.values() for v in d.values()])
                with self._lock:
                    for e, d in experts.items():
                        self._ready[(layer, e)] = d
                    if layer not in self._ready_layers:
                        self._ready_layers.append(layer)
                    while len(self._ready_layers) > self._window:
                        old = self._ready_layers.pop(0)
                        for k in [k for k in self._ready if k[0] == old]:
                            del self._ready[k]
            except Exception:
                pass

    def take_ready(self, layer, e):
        with self._lock:
            return self._ready.pop((int(layer), int(e)), None)

    def ready_count(self, layer):
        with self._lock:
            return sum(1 for k in self._ready if k[0] == int(layer))

    def close(self):
        self._stop = True
        self._t.join(timeout=1)
```

- [ ] **Step 4: 跑绿 + 提交** `feat: BackgroundExpertPrefetcher (s2 materialize + handoff)`

- [ ] **Step 5(可选,GIL 优化):C++ blob_load 后台 load 开关**

`STREAM_BLOB_BG_NATIVE=1` 时,worker 的物化改用已存在的 `native_moe_ext.blob_load(path, ids_u32, stride)`(C++ primitive,eval_gpu 内 pread 进 MLX 自有 buffer,释放 GIL),再在 Python 侧按段 `.view` 切成 9 个 typed 数组(见 `cli/probe_blob_load_primitive.py` 的 `view_expert`)。其余(交接/window)不变。默认关;Task 4 实测 GIL 是否真瓶颈后再决定默认值。

---

## Task 2: 池接受预物化专家(promote)

**Files:**
- Modify: `mlx_streaming/core/expert_store.py`(加 `promote_prefetched`)
- Test: `mlx_streaming/tests/test_bg_prefetch.py`(加用例)

- [ ] **Step 1: 写失败测试**：`store.promote_prefetched(layer)` 把 prefetcher 里该层就绪专家 `_place_expert` 进常驻池;之后 `_resident.resident_experts(layer)` 含这些专家,且 `acquire_gpu` 命中(miss=0)。

- [ ] **Step 2: 跑红**

- [ ] **Step 3: 实现**：`FileExpertStore` 持有可选 `self._bg = None`;`promote_prefetched(layer)`:

```python
def promote_prefetched(self, layer, expert_ids):
    if self._bg is None:
        return
    for e in expert_ids:
        d = self._bg.take_ready(layer, e)
        if d is not None:
            # 已物化的 dict 直接写槽(主线程、默认 stream，便宜 scatter)
            self._resident._ensure_layer(layer)
            self._resident._place_expert(layer, int(e), d)
```

- [ ] **Step 4: 跑绿 + 提交** `feat: FileExpertStore.promote_prefetched places bg-materialized experts into pool`

---

## Task 3: 接入 streaming + 跨层预测

**Files:**
- Modify: `mlx_streaming/core/streaming_moe.py`(cross-layer hook 提交预取 + 本层 acquire 前 promote)
- Modify: `mlx_streaming/model_builder.py`(`STREAM_BLOB_BG=1` 时构造 prefetcher 挂到 store)

- [ ] **Step 1: 写等价测试**(小模型):`STREAM_BLOB_BG=1` 输出与常驻路径逐元素一致(<1e-4)。

- [ ] **Step 2: 跑红**

- [ ] **Step 3: 实现**：
  - `model_builder`：`STREAM_BLOB_BG=1` 时 `store._bg = BackgroundExpertPrefetcher(BlobExpertSource(...))`，并 `enable_cross_layer_prefetch()`。
  - cross-layer hook：`STREAM_BLOB_BG` 分支调用 `target_mlp.store._bg.submit(target_layer, picked)`（预测整层激活集，budget≈top_k×mult）。
  - `FileStreamingMoeBlock.__call__`：进入专家计算前 `if store._bg: store.promote_prefetched(self.layer_idx, <预测/上次该层专家>)`。注意：promote 需要"本层要用的专家 id"；用上一次该层路由或当前预测结果。最稳妥：promote 该层**所有就绪**专家（`take_ready` 已就绪的），不依赖精确 id。
  - 未就绪 → 走现有 demand(acquire_gpu fallback)。

- [ ] **Step 4: 跑绿 + 提交** `feat: opt-in STREAM_BLOB_BG background pool-fill path`

---

## Task 4: 端到端 A/B + 报告

**Files:**
- Create: `mlx_streaming/cli/probe_stream_blob_bg.py`
- Append: `benchmarks/reports/low-memory-streaming-moe-2026-06-11.md`

- [ ] **Step 1: probe**：同 prompt/K/MAXTOK，A=常驻大池(256)，B=小池(32)+blob loader，C=小池(32)+`STREAM_BLOB_BG`。记录 `spec_tok_per_s`/`mlx_peak_gb`/`spec_hit_rate`/`exact_match`。

- [ ] **Step 2: 跑 A/B/C**（每次前 `sudo purge`）

Run: `uv run python -m mlx_streaming.tools.probe_stream_blob_bg`
Expected: C 的 `spec_hit_rate` 显著高于 B（后台预填生效）；C 的 tok/s 高于 B；`exact_match` 与常驻一致；峰值内存 ≈ B（小池）。

- [ ] **Step 3: 拆解瓶颈(决定要不要上 C++)**：在 C 上分别测量
  - **promote 主线程开销**(每层 scatter + Python 循环耗时)——决定"拷池"是否值得进一步优化(默认判断:便宜,不优化)。
  - **GIL 竞争**:对比 `STREAM_BLOB_BG_NATIVE` 关/开的 tok/s——决定 C++ 后台 load 是否默认开。
  - 后台填充命中率(promote 命中 vs demand)、预测命中率。

- [ ] **Step 4: 写报告**：记录 C 是否把小池 tok/s 拉近大池、命中率提升多少;以及上面拆解结论(C++ 在 GIL/拷池上各值不值)。若仍不及大池，记录残余瓶颈。

---

## 自检
- 类型一致：`take_ready` 返回的 dict 键(`gate_proj.weight` 等)必须与 `_place_expert` 期望一致(= `BlobExpertSource._materialize` 产出，已在 Task1 测试核对)。
- 正确性贯穿：Task 3 等价测试 + Task 4 `exact_match`。
- 优雅回退:prefetcher 未就绪/异常 → 现有 demand 路径，绝不崩(gating 已验证后台异常被吞、主线程不受影响)。
- 风险点显式测量(Task 4)：后台填充命中率、promote 主线程开销、与大池的 tok/s 差距——都靠实测，不靠假设。
- 关键不变量:后台线程只物化私有 array(规则1)、绝不写共享池(规则2)、主线程只消费已 eval 的交接(规则3)。
