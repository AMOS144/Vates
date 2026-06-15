# 低内存全流式 MoE 模式 实现计划（Plan）

> **For agentic workers:** 用 superpowers:subagent-driven-development 或 executing-plans 逐任务执行。每步带 checkbox。

**Goal:** 新增可选的"全流式 blob"模式：每层只临时持有当前+预取窗口的专家，其余按需从 SSD 流式读取，峰值内存 11GB→<3GB，吞吐 ≥14 tok/s，输出与常驻路径逐 token 一致。

**Architecture:** blob 格式（每专家连续、页对齐）→ 并行 `pread`(+F_NOCACHE) → `np.frombuffer`+`mx.array`(scales/biases `.view(bfloat16)`) → 复用 MLX `quantized_matmul`；跨层预测驱动后台预取与计算重叠；滚动窗口释放旧层。

**Tech Stack:** Python、MLX（`mx.quantized_matmul`/`mx.view`）、numpy、`os.pread`、`concurrent.futures`、macOS `fcntl F_NOCACHE`。

**前置：** M0 de-risk 已通过（blob→mx.array→quantized_matmul 0 误差、不崩、可重叠）。spec 见 `docs/superpowers/specs/2026-06-11-low-memory-streaming-moe-design.md`。

---

## 约定常量
`HIDDEN=2048, INTER=512, GROUP=128, BITS=2, NUM_EXPERTS=512`；blob 段顺序 `gate[w,s,b],up[w,s,b],down[w,s,b]`，`stride=884736`（16KB 对齐）。`weight` 为 uint32，`scales/biases` 存 uint16 原始位、用时 `.view(mx.bfloat16)`。

---

## Task 1: blob prep 扩成全层 + CLI（已有雏形）

**Files:**
- Modify: `mlx_streaming/prep/repack_expert_blobs.py`
- Test: `mlx_streaming/tests/test_repack_blobs.py`

- [ ] **Step 1: 写失败测试**

```python
# test_repack_blobs.py
import json, os, numpy as np, mlx.core as mx
from mlx_streaming.prep.repack_expert_blobs import repack_layer, _layout

def test_blob_roundtrip_matches_compute_buffer(tmp_path, monkeypatch):
    # 用已有 /tmp/cb_2bit_g128 的一层，重打包后逐字节核对 expert 5 的 gate.weight
    monkeypatch.setenv("BLOB_DIR", str(tmp_path))
    idx = repack_layer(15)
    segs, stride = _layout()
    assert idx["stride"] == stride and idx["page_aligned"]
    blob = open(os.path.join(tmp_path, "layer15.blob"), "rb")
    raw = blob.read(stride); blob.close()
    # 第一段就是 gate.weight，比对源 compute buffer
    src = np.memmap("/tmp/cb_2bit_g128/layer15.gate_proj.weight.bin", dtype=np.uint8, mode="r")
    assert raw[:segs[0][2]] == src[:segs[0][2]].tobytes()
```

- [ ] **Step 2: 跑测试确认红**（`/tmp/cb_2bit_g128` 不在则先 `pack_compute_buffers` 全层）

Run: `uv run pytest mlx_streaming/tests/test_repack_blobs.py -x`
Expected: 通过（repack_layer 已实现）或因缺源数据红 → 先补源。

- [ ] **Step 3: 加 `--all` 全层 CLI + 落 index 汇总**

在 `main()` 支持 `LAYERS=all` 时遍历 `_split_meta.json["moe_layers"]`，并写 `blob_index.json` 汇总（层→文件、stride、num_experts）。

- [ ] **Step 4: 跑通 + 提交**

Run: `LAYERS=all uv run python -m mlx_streaming.prep.repack_expert_blobs`
Expected: 48 个 `layerXX.blob` + index。
Commit: `feat: repack experts into per-expert contiguous blobs (all layers)`

---

## Task 2: BlobExpertSource —— 并行读 + 物化（核心新模块）

**Files:**
- Create: `mlx_streaming/core/blob_loader.py`
- Test: `mlx_streaming/tests/test_blob_loader.py`

- [ ] **Step 1: 写失败测试（正确性 vs safetensors）**

```python
# test_blob_loader.py
import os, mlx.core as mx
from mlx_streaming.core.blob_loader import BlobExpertSource

def _moe(x, w, G=128, B=2):
    g = mx.quantized_matmul(x, w["gate_proj.weight"], w["gate_proj.scales"], w["gate_proj.biases"], transpose=True, group_size=G, bits=B)
    u = mx.quantized_matmul(x, w["up_proj.weight"], w["up_proj.scales"], w["up_proj.biases"], transpose=True, group_size=G, bits=B)
    a = g * mx.sigmoid(g) * u
    return mx.quantized_matmul(a, w["down_proj.weight"], w["down_proj.scales"], w["down_proj.biases"], transpose=True, group_size=G, bits=B)

def test_blob_source_matches_safetensors():
    src = BlobExpertSource("/tmp/cb_2bit_blob", hidden=2048, inter=512, group=128, bits=2, num_experts=512)
    x = (mx.random.normal((1, 2048)) * 0.1).astype(mx.float32); mx.eval(x)
    experts = src.load_experts(15, [3, 7, 100])           # 返回 {e: {proj.tensor: mx.array}}
    for e, wb in experts.items():
        wr = mx.load(f"models/qwen3_next_experts_2bit_g128/layer15_expert{e:03d}.safetensors")
        yb, yr = _moe(x, wb), _moe(x, wr); mx.eval(yb, yr)
        assert float(mx.max(mx.abs(yb - yr))) < 1e-5
```

- [ ] **Step 2: 跑测试确认红**

Run: `uv run pytest mlx_streaming/tests/test_blob_loader.py -x`
Expected: ImportError（模块不存在）。

- [ ] **Step 3: 实现 `BlobExpertSource`**

```python
# blob_loader.py
import os, fcntl
from concurrent.futures import ThreadPoolExecutor
import numpy as np, mlx.core as mx

F_NOCACHE = 48  # macOS

class BlobExpertSource:
    def __init__(self, blob_dir, hidden, inter, group, bits, num_experts, workers=8, nocache=True):
        self.dir, self.workers, self.nocache = blob_dir, workers, nocache
        self.h, self.i, self.g, self.b, self.ne = hidden, inter, group, bits, num_experts
        self._segs, self.stride = self._layout()
        self._fds = {}

    def _layout(self):
        segs, projs = [], (("gate_proj", self.i, self.h), ("up_proj", self.i, self.h), ("down_proj", self.h, self.i))
        for proj, out_dim, in_dim in projs:
            words, groups = in_dim * self.b // 32, in_dim // self.g
            segs.append((proj, "weight", np.uint32, (out_dim, words), out_dim * words * 4))
            segs.append((proj, "scales", np.uint16, (out_dim, groups), out_dim * groups * 2))
            segs.append((proj, "biases", np.uint16, (out_dim, groups), out_dim * groups * 2))
        return segs, sum(s[4] for s in segs)

    def _fd(self, layer):
        fd = self._fds.get(layer)
        if fd is None:
            fd = os.open(os.path.join(self.dir, f"layer{layer:02d}.blob"), os.O_RDONLY)
            if self.nocache:
                try: fcntl.fcntl(fd, F_NOCACHE, 1)
                except OSError: pass
            self._fds[layer] = fd
        return fd

    def _materialize(self, raw):
        out, off = {}, 0
        for proj, tensor, dt, shape, nb in self._segs:
            v = np.frombuffer(raw, dtype=dt, count=nb // np.dtype(dt).itemsize, offset=off).reshape(shape)
            a = mx.array(v)
            if tensor in ("scales", "biases"): a = a.view(mx.bfloat16)
            out[f"{proj}.{tensor}"] = a; off += nb
        return out

    def load_experts(self, layer, expert_ids):
        fd = self._fd(layer)
        def rd(e): return e, self._materialize(os.pread(fd, self.stride, e * self.stride))
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return dict(ex.map(rd, expert_ids))

    def close(self):
        for fd in self._fds.values(): os.close(fd)
        self._fds.clear()
```

- [ ] **Step 4: 跑测试确认绿**

Run: `uv run pytest mlx_streaming/tests/test_blob_loader.py -x`
Expected: PASS（<1e-5）。

- [ ] **Step 5: 提交**

Commit: `feat: BlobExpertSource parallel pread + F_NOCACHE + mx.array materialize`

---

## Task 3: stacked 适配，复用 _sub.forward

**Files:**
- Modify: `mlx_streaming/core/blob_loader.py`（加 `acquire`）
- Test: `mlx_streaming/tests/test_blob_loader.py`（加用例）

- [ ] **Step 1: 写失败测试**：`acquire(layer, expert_ids)` 返回 `(pool_arrays, slots)`，`pool_arrays` 结构与 `ResidentExpertPool.acquire` 一致（按 unique 专家 stack 成 `[n,...]` 的 gate/up/down weight/scales/biases），`slots` 是每个 `expert_id` 在 unique 列表的下标。断言 stack 后用 `_sub.forward` 与逐专家 `_moe` 求和一致。

- [ ] **Step 2: 跑红**：`uv run pytest mlx_streaming/tests/test_blob_loader.py::test_acquire_matches -x`

- [ ] **Step 3: 实现 `acquire`**：去重→`load_experts`→对 9 个 key 各 `mx.stack([...], axis=0)`→`slots=[uniq.index(e) for e in expert_ids]`。

- [ ] **Step 4: 跑绿 + 提交**：`feat: BlobExpertSource.acquire stacked arrays compatible with SwitchGLU forward`

---

## Task 4: 接入 streaming 路径（opt-in `STREAM_BLOB=1`）

**Files:**
- Modify: `mlx_streaming/core/streaming_moe.py`（`FileStreamingMoeBlock.__call__`）
- Modify: `mlx_streaming/model_builder.py`（按 flag 注入 `BlobExpertSource`）
- Test: `mlx_streaming/tests/test_stream_blob_equiv.py`

- [ ] **Step 1: 写等价测试（小模型）**：构造一个 2 层小 MoE + 临时 blob，`STREAM_BLOB=1` 与常驻路径对同一输入输出逐元素一致（<1e-5）。

- [ ] **Step 2: 跑红**。

- [ ] **Step 3: 实现 flag 分支**：`__call__` 里当 `os.environ.get("STREAM_BLOB")=="1"` 且 `self._blob is not None` 时，用 `self._blob.acquire(self.layer_idx, flat)` 取代 resident pool 的 `acquire`，其余（`_sub.forward`、score 合并、shared expert）不变。

- [ ] **Step 4: 跑绿 + 提交**：`feat: opt-in STREAM_BLOB path in FileStreamingMoeBlock`

---

## Task 5: 后台预取重叠 + 滚动窗口

**Files:**
- Modify: `mlx_streaming/core/blob_loader.py`（加 `prefetch_async` + 已读缓存 + 窗口驱逐）
- Modify: `mlx_streaming/core/streaming_moe.py`（跨层预测处调用 `prefetch_async`）
- Test: `mlx_streaming/tests/test_blob_prefetch.py`

- [ ] **Step 1: 写测试**：`prefetch_async(layer, ids)` 后 `acquire(layer, ids)` 不再触发新 pread（命中预取），且窗口外的层被释放（`fd`/缓存数量有界）。

- [ ] **Step 2: 跑红**。

- [ ] **Step 3: 实现**：单后台线程（或 1-worker executor）按预测读下一层 → 存 `{(layer,e): materialized}`；`acquire` 优先消费；保留最近 `WINDOW`(默认 2) 层，其余 `close`/丢弃。复用现有 `enable_cross_layer_prefetch` 的预测，新增 `STREAM_BLOB` 分支调用 `prefetch_async`。

- [ ] **Step 4: 跑绿 + 提交**：`feat: async blob prefetch overlap + rolling window eviction`

---

## Task 6: 端到端实测 + 报告

**Files:**
- Create: `mlx_streaming/cli/probe_stream_blob.py`
- Create: `benchmarks/reports/low-memory-streaming-moe-2026-06-11.md`

- [ ] **Step 1: probe**：同 prompt/K/MAXTOK 跑 `run_mtp_spec`，A/B = 常驻 vs `STREAM_BLOB=1`，记录 `spec_tok_per_s`、`mlx_peak_gb`、`rss_gb`、`exact_match`、预取命中/未命中。

- [ ] **Step 2: 全层 blob 准备**：`LAYERS=all` 打包到 `/tmp/cb_2bit_blob`。

- [ ] **Step 3: 跑 A/B**

Run: `uv run python -m mlx_streaming.tools.probe_stream_blob`
Expected: `exact_match=true`；峰值内存显著下降；tok/s ≥ 常驻 80%。

- [ ] **Step 4: 写报告 + 提交**：记录是否达成"内存 <3GB、tok/s ≥14、exact_match"。若 F_NOCACHE 反伤带宽或预测未命中拖速，记录并给参数建议。

---

## 自检
- 类型/签名一致：`BlobExpertSource.acquire` 返回结构必须与 `ResidentExpertPool.acquire` 完全对齐（`_sub.forward` 才能复用）。Task 3 必须先与 `expert_store.py` 的 `acquire` 实际返回结构核对再实现。
- 正确性贯穿每个 milestone：Task 2/4 都有 `exact_match`/数值断言。
- 风险点（F_NOCACHE 行为、预测未命中冷 stall、GIL 重叠折扣）在 Task 5/6 显式测量，不靠假设。
