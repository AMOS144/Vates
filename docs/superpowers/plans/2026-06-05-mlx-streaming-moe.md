# MLX Streaming-MoE 实现方案

> **给执行者：** 必备子技能 —— 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务逐步实现。步骤用 `- [ ]` 复选框跟踪。
>
> **本文档用中文。Spec（第 1~4 节）力求易懂；Plan（第 5 节起）力求严谨、可直接照做。**

**Goal（一句话）：** 在 Apple Silicon 上用 MLX 跑 Qwen3 MoE，让模型「即使能装进统一内存也不全量驻留」——把 MoE 专家权重留在磁盘、按需流式、在 GPU 上计算，并能用一个内存预算把常驻 RAM 压到目标值，从而把内存让给本机其它软件。

**Architecture（2~3 句）：** 先用一个 Phase 0 探针实测「mmap 惰性专家张量 + `gather_qmm` 按需换页」能否天然压低 RSS；若能，则走「库存 mlx-lm + `use_mmap` + `set_wired_limit`」的便宜路线（路线 A）；若不能，则替换 Qwen3 MoE 块为自定义流式 MoE（显式按需加载选中专家 + LRU 驱逐 + `clear_cache`），路线 B 必要时配一个离线「按专家拆分权重」的一次性重打包。

**Tech Stack：** MLX (`mlx-core`)、mlx-lm、Python 3.11+、macOS 15+（`set_wired_limit` 需要）、mlx-community 预量化 4bit Qwen3-MoE safetensors。

---

## 1. 背景与动机（易懂版）

我们在 Rust 工程 `hypura`（基于 ggml/llama.cpp）里验证过「专家流式 → NVMe」，但撞到一个**架构死结**：自定义 NVMe buffer 在 ggml 里报告 `is_host=true`，导致 MoE 专家矩阵乘**回退到 CPU**，比全 GPU 慢约 27 倍（1.3 tok/s vs 35 tok/s）。I/O 不是瓶颈，**计算回退**才是。

根因有两条，都和 ggml/Metal 的执行模型耦合：
1. 自定义 host buffer 强制 CPU 计算（无法零拷贝交给 Metal 在 GPU 上算）。
2. 专家路由是数据相关的、在 GPU 图中途算出来的；ggml 的批量 command buffer 难以「中途按路由结果去 staging 专家」。

**MLX 为什么有机会解开这个结**（均来自官方文档，见第 3 节出处）：
- **统一内存**：MLX 数组天生活在 CPU/GPU 共享内存里，算子在哪个 device 跑由调用方指定，**不存在 host buffer 强制 CPU 回退**。
- **惰性加载**：`mx.load(path)` 立即返回、只读 metadata，数组在 `mx.eval` 前不从磁盘读。
- **动态图 / define-by-run**：可以 `eval` 出路由结果、在 Python 里读出选中的专家 id、再按需加载这几个专家——控制流和 GPU 计算可自由交错。
- **显式内存上限**：`set_wired_limit`（常驻上限，macOS 15+）、`set_memory_limit`、`set_cache_limit`、`clear_cache`，可以「告诉 MLX 最多常驻 N GB，其余还给系统」。

## 2. 目标（成功判据）

用「能装进统一内存」的 Qwen3-MoE 4bit 来验证「能装也不驻留」这个核心诉求。成功判据：

- **G1（必须）：** 在 MLX 上能正常生成、且专家计算跑在 **GPU** 上（不是 CPU 回退）。判据：`mx.metal` 可用、生成期间 GPU 有占用、decode 速度量级与全驻留 MLX 同档（不出现 10x+ 的崩塌）。
- **G2（必须）：** 推理进程的常驻内存（`ru_maxrss` 与 `mx.get_peak_memory()`）**明显低于**「全量驻留」基线。量化目标：专家相关常驻随预算下降而下降，存在可观测的「内存 ↓ / tok/s」帕累托前沿。
- **G3（必须）：** 提供一个**可扫描的内存预算旋钮**（路线 A 是 `set_wired_limit`/`set_cache_limit`；路线 B 是 LRU 专家槽数），并产出一份「预算 × RSS × decode tok/s × 专家命中率」的实测报告。
- **G4（应当）：** 与 `hypura` 的概念对齐（专家池 ↔ LRU、命中率统计口径一致），便于横向对比两条技术路线。

## 3. 已核实的关键技术事实（带出处）

> 这些是方案的硬依据。执行时若版本不同，以 Phase 0 探针的实测为准。

1. **`mx.load` 惰性**：`mx.load("model.safetensors")` 立即返回，只读 metadata；数组直到 `mx.eval(weights)` 才从磁盘读。
   出处：官方 [Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)；维护者 [Discussion #1292](https://github.com/ml-explore/mlx/discussions/1292)。
   ⚠️ 注意：传入「已打开的文件对象」会**立即求值**（生命周期问题），必须传**字符串路径**才惰性。

2. **`use_mmap` 零拷贝（不确定项）**：stock MLX 0.31.2 的 [`mlx.core.load` 文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.load.html) 签名只有 `file / format / return_metadata`，**未列出 `use_mmap`**。`use_mmap`（mmap → Metal 共享 buffer 零拷贝）出现在社区 fork [OptMLX](https://github.com/AtomGradient/OptMLX) 与较新的 mlx-lm（`--use-mmap`）。**必须在 Phase 0 用 `inspect.signature` 核对本机版本是否真有这个参数。**

3. **内存管理 API**：官方 [Memory Management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html) 列出：取活跃内存、取峰值内存、重置峰值、取缓存大小、设内存上限、设缓存上限、设常驻上限、清缓存。
   - [`set_wired_limit(limit)`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)：常驻（pinned）内存上限字节数，默认 `0`；**仅 macOS 15.0+ 有效**；必须严格小于系统 wired 上限；可用 `device_info()` 的 `max_recommended_working_set_size` / `memory_size` 查询；系统上限可 `sudo sysctl iogpu.wired_limit_mb=<MB>` 调高。
   - [`set_memory_limit(limit)`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_memory_limit.html)：图求值的内存上限指引；超限且无 RAM（含 swap）则分配抛异常；Metal 下默认 = 1.5× max recommended working set size。
   - ⚠️ 上述函数的**确切命名空间/名字**（`mx.*` vs `mx.metal.*`、`get_cache_memory` vs `get_cache_size`）随版本变动，Phase 0 必须 `print([n for n in dir(mx) if 'mem' in n or 'cache' in n or 'wired' in n])` 核对。

4. **mlx-lm 的 Qwen3 MoE 结构（已读源码 `mlx_lm/models/qwen3_moe.py` + `switch_layers.py`）**：
   - 专家**堆叠存储**：每层每投影是一个张量 `model.layers.{l}.mlp.switch_mlp.{gate_proj,up_proj,down_proj}.weight`，形状 `(num_experts, output_dims, input_dims)`；量化版还有同前缀的 `.scales` 与 `.biases`。
   - 路由块 `Qwen3MoeSparseMoeBlock`：`gate`（小 Linear）→ `softmax` → `argpartition` 取 top-k → `switch_mlp(x, inds)`。
   - `SwitchGLU` / `QuantizedSwitchLinear.__call__` 调 `mx.gather_qmm(x, weight, scales, biases, rhs_indices=indices, transpose=True, ...)`，**只 gather 选中专家的行**。
   - `Model.sanitize` 会把 HF 的 per-expert 张量 `mx.stack` 成 `switch_mlp` 堆叠张量（说明 mlx-community 预量化模型里专家已是堆叠布局）。
   - 注意：专家堆叠张量内存布局是 `(num_experts, ...)` 连续，**专家 e 占一段连续 slab**——这是「按专家换页/切片」可行的前提。

## 4. 架构决策：Phase 0 探针决定路线

```
                ┌─────────────────────────────────────────────┐
                │ Phase 0 探针（半天）                          │
                │ lazy=True + use_mmap 加载 → 单步 decode →     │
                │ 测 RSS / mx.get_peak_memory / active_memory  │
                └───────────────┬─────────────────────────────┘
                                │
          RSS ≈ 活跃工作集？     │     RSS ≈ 整模型？
        （专家按需换页生效）     │   （Metal 提交整 buffer / load 强制 eval）
              ┌─────────────────┴─────────────────┐
              ▼                                     ▼
   路线 A（便宜）                          路线 B（自定义流式）
   库存 mlx-lm + use_mmap +               替换 Qwen3MoeSparseMoeBlock：
   set_wired_limit/cache_limit            显式按需加载选中专家 +
   + 预算扫描报告                          LRU 驱逐 + clear_cache
                                          （必要时 + 离线按专家拆分重打包）
```

**非目标（YAGNI）：** 不做训练/微调；不做多模型并发；不追求超越 NVMe 物理带宽；不在本方案里重写注意力或 KV cache（KV 留统一内存，沿用 mlx-lm 默认）。

---

## 5. 文件结构

所有新代码放在仓库的 `mlx_streaming/` 子目录（与 Rust 工程隔离，互不影响构建）：

- 新建 `mlx_streaming/pyproject.toml` — 声明 `mlx`、`mlx-lm`、`numpy` 依赖（用 pip/uv 装最新版）。
- 新建 `mlx_streaming/env_probe.py` — Phase 0 环境探针：核对 API 名、`use_mmap`/`lazy` 参数、`set_wired_limit` 可用性、`device_info()`。
- 新建 `mlx_streaming/mem.py` — 内存度量工具：封装 `ru_maxrss`、`mx.get_peak_memory()`、`mx.get_active_memory()`、`mx.clear_cache()`，统一口径。
- 新建 `mlx_streaming/run_baseline.py` — 全量驻留基线：正常 `load` + 生成，记录 RSS/峰值/速度。
- 新建 `mlx_streaming/run_mmap.py` — 路线 A：`lazy=True` + `use_mmap` + 预算旋钮（`set_wired_limit`/`set_cache_limit`），单进程跑一组预算。
- 新建 `mlx_streaming/streaming_moe.py` — 路线 B：自定义 `StreamingSwitchGLU` / `StreamingMoeBlock`（LRU 专家缓存 + 按需加载 + 驱逐），以及把模型里 MoE 块替换为流式块的 `patch_model()`。
- 新建 `mlx_streaming/expert_store.py` — 路线 B 的专家权重后端：从 mmap 堆叠张量切片，或从「离线拆分」的 per-expert 文件加载；带命中率统计。
- 新建 `mlx_streaming/split_experts.py` — 路线 B 可选离线工具：把堆叠 `switch_mlp.*.weight/scales/biases` 重打包成 per-expert 小 safetensors（一次性，便于干净的按需加载）。
- 新建 `mlx_streaming/sweep.py` — 预算扫描驱动：对一组预算逐个起子进程跑 `run_mmap.py`/路线 B，汇总成表。
- 新建 `mlx_streaming/tests/test_mem.py`、`tests/test_expert_store.py`、`tests/test_streaming_equiv.py` — 单元/等价性测试。
- 新建 `benchmarks/reports/mlx-streaming-moe-<date>.md` — 最终实测报告（套用 `benchmarks/reports/` 既有风格）。

每个文件单一职责；度量逻辑集中在 `mem.py`，避免到处复制 RSS 读取。

---

## 6. 任务分解

> 约定：模型路径用环境变量 `MODEL`（默认 `mlx-community/Qwen3-30B-A3B-4bit`，即你一直测的 Qwen3-MoE 的 mlx-community 4bit 版；若你的确切型号不同，替换为对应的 mlx-community 仓库名）。每个任务结束都 commit。

### Task 0：项目骨架与依赖

**Files:**
- Create: `mlx_streaming/pyproject.toml`
- Create: `mlx_streaming/__init__.py`（空）

- [ ] **Step 1：写 `pyproject.toml`**

```toml
[project]
name = "mlx-streaming"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["mlx", "mlx-lm", "numpy"]

[tool.pytest.ini_options]
addopts = "-q"
```

- [ ] **Step 2：建虚拟环境并安装最新依赖**

Run:
```bash
cd mlx_streaming && python3 -m venv .venv && . .venv/bin/activate && pip install -U mlx mlx-lm numpy pytest
```
Expected：安装成功，`pip show mlx mlx-lm` 打印出版本号（记录下来，后续探针对照）。

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/pyproject.toml mlx_streaming/__init__.py
git commit -m "chore(mlx): 初始化 mlx_streaming 子项目骨架"
```

---

### Task 1：环境探针（核实文档里的不确定项）

**Files:**
- Create: `mlx_streaming/env_probe.py`

- [ ] **Step 1：写探针脚本**

```python
"""Phase 0 环境探针：核对 MLX 版本、内存 API、use_mmap/lazy 是否可用。"""
import inspect
import mlx.core as mx

def main():
    print("== 内存/缓存/常驻相关符号 ==")
    print([n for n in dir(mx) if any(k in n for k in ("mem", "cache", "wired", "peak"))])

    print("== metal 可用性与设备信息 ==")
    print("metal.is_available:", mx.metal.is_available())
    try:
        print("device_info:", mx.metal.device_info())
    except Exception as e:  # 不同版本可能在 mx.device_info
        print("mx.metal.device_info 失败:", e)

    print("== mlx_lm.load 签名（确认 lazy / use_mmap）==")
    from mlx_lm import load
    print(inspect.signature(load))
    try:
        from mlx_lm.utils import load_model
        print("load_model 签名:", inspect.signature(load_model))
    except Exception as e:
        print("load_model 取签名失败:", e)

    print("== mx.load 是否支持 use_mmap ==")
    print(inspect.signature(mx.load))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2：运行并记录结果**

Run: `python -m mlx_streaming.env_probe`（在 `mlx_streaming` 上层目录、激活 venv 后）
Expected：打印出实际的内存 API 名字、`metal.is_available() == True`、`load`/`mx.load` 的真实签名。**把输出贴进最终报告的「环境」一节。** 据此确定后续代码用 `mx.set_wired_limit` 还是 `mx.metal.set_wired_limit`、`use_mmap` 是否存在。

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/env_probe.py
git commit -m "feat(mlx): 加环境探针核对内存 API 与 lazy/use_mmap 可用性"
```

---

### Task 2：内存度量工具（TDD）

**Files:**
- Create: `mlx_streaming/mem.py`
- Test: `mlx_streaming/tests/test_mem.py`

- [ ] **Step 1：写失败测试**

```python
from mlx_streaming.mem import MemSnapshot, snapshot, rss_bytes

def test_rss_is_positive():
    assert rss_bytes() > 0

def test_snapshot_has_fields():
    s = snapshot()
    assert isinstance(s, MemSnapshot)
    assert s.rss_bytes > 0
    assert s.mlx_active_bytes >= 0
    assert s.mlx_peak_bytes >= 0
```

- [ ] **Step 2：跑测试确认失败**

Run: `pytest mlx_streaming/tests/test_mem.py -v`
Expected：FAIL，`ModuleNotFoundError: mlx_streaming.mem`。

- [ ] **Step 3：实现 `mem.py`**

> 注：`mx` 内存函数名以 Task 1 探针结果为准；下面用 `getattr` 容错，缺失的就返回 0。

```python
"""内存度量统一口径。macOS 上 ru_maxrss 单位是字节。"""
import resource
from dataclasses import dataclass
import mlx.core as mx

def rss_bytes() -> int:
    # macOS: ru_maxrss 已是字节；Linux 是 KB（本项目目标是 macOS）
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def _call(name: str) -> int:
    fn = getattr(mx, name, None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", object()), name, None)
    try:
        return int(fn()) if fn else 0
    except Exception:
        return 0

@dataclass
class MemSnapshot:
    rss_bytes: int
    mlx_active_bytes: int
    mlx_peak_bytes: int

def snapshot() -> MemSnapshot:
    return MemSnapshot(
        rss_bytes=rss_bytes(),
        mlx_active_bytes=_call("get_active_memory"),
        mlx_peak_bytes=_call("get_peak_memory"),
    )

def clear_cache() -> None:
    fn = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", object()), "clear_cache", None)
    if fn:
        fn()

def reset_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", object()), "reset_peak_memory", None)
    if fn:
        fn()
```

- [ ] **Step 4：跑测试确认通过**

Run: `pytest mlx_streaming/tests/test_mem.py -v`
Expected：PASS（2 passed）。

- [ ] **Step 5：Commit**

```bash
git add mlx_streaming/mem.py mlx_streaming/tests/test_mem.py
git commit -m "feat(mlx): 加内存度量工具(mem.py)，统一 RSS/active/peak 口径"
```

---

### Task 3：全量驻留基线

**Files:**
- Create: `mlx_streaming/run_baseline.py`

- [ ] **Step 1：写基线脚本**

```python
"""全量驻留基线：正常加载 + 生成，记录 RSS/峰值/decode 速度。"""
import os, time, json
import mlx.core as mx
from mlx_lm import load, generate
from mlx_streaming.mem import snapshot, reset_peak

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))

def main():
    reset_peak()
    t0 = time.perf_counter()
    model, tok = load(MODEL)            # 默认全量加载
    mx.eval(model.parameters())         # 强制全部驻留，作为对照上界
    load_done = snapshot()
    t1 = time.perf_counter()

    # 计时 decode：先 warmup 一个 token，再正式跑
    text = generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK, verbose=False)
    t2 = time.perf_counter()
    after = snapshot()

    out = {
        "mode": "baseline_resident",
        "model": MODEL,
        "load_s": round(t1 - t0, 2),
        "gen_s": round(t2 - t1, 2),
        "tok_per_s": round(MAXTOK / (t2 - t1), 2),
        "rss_gb_after_load": round(load_done.rss_bytes / 1e9, 2),
        "rss_gb_after_gen": round(after.rss_bytes / 1e9, 2),
        "mlx_peak_gb": round(after.mlx_peak_bytes / 1e9, 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2：运行并记录（用外部 RSS 双保险）**

Run: `/usr/bin/time -l python -m mlx_streaming.run_baseline 2>&1 | tail -n 30`
Expected：打印 JSON，含 `rss_gb_after_gen`、`tok_per_s`；`/usr/bin/time -l` 的 `maximum resident set size` 与脚本内 `rss` 量级一致。**这是 G2 的对照上界（全量驻留 RSS）。**

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/run_baseline.py
git commit -m "feat(mlx): 加全量驻留基线脚本，建立 RSS/速度对照上界"
```

---

### Task 4：Phase 0 探针——判定走路线 A 还是 B（关键决策点）

**Files:**
- Create: `mlx_streaming/run_mmap.py`

- [ ] **Step 1：写 mmap/lazy 加载脚本（带预算旋钮）**

> ⚠️ `load(..., lazy=True)` 与 `use_mmap=...` 的参数名以 Task 1 探针为准。下面用 `**kwargs` 容错传参；不支持的参数在调用前剔除。

```python
"""路线 A：lazy + mmap 加载，可选设常驻/缓存上限，单步与多步 decode 量内存。"""
import os, time, json, inspect
import mlx.core as mx
from mlx_lm import load, generate
from mlx_streaming.mem import snapshot, reset_peak, clear_cache

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
WIRED_GB = os.environ.get("WIRED_GB")      # 例如 "6"；空=不设
CACHE_GB = os.environ.get("CACHE_GB")      # 例如 "1"；空=不设

def _maybe(fn_name, val_bytes):
    fn = getattr(mx, fn_name, None) or getattr(getattr(mx, "metal", object()), fn_name, None)
    if fn and val_bytes is not None:
        prev = fn(int(val_bytes))
        print(f"{fn_name}({val_bytes}) 旧值={prev}")

def _filtered_load_kwargs():
    sig = inspect.signature(load)
    want = {"lazy": True, "use_mmap": True}
    return {k: v for k, v in want.items() if k in sig.parameters}

def main():
    if WIRED_GB: _maybe("set_wired_limit", float(WIRED_GB) * 1e9)
    if CACHE_GB: _maybe("set_cache_limit", float(CACHE_GB) * 1e9)

    reset_peak()
    kw = _filtered_load_kwargs()
    print("load kwargs:", kw)
    t0 = time.perf_counter()
    model, tok = load(MODEL, **kw)       # 不强制 eval 全部参数！
    t1 = time.perf_counter()
    after_load = snapshot()              # 关键：此时 RSS 应远低于基线（专家还没换入）

    text = generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK, verbose=False)
    t2 = time.perf_counter()
    after_gen = snapshot()

    out = {
        "mode": "mmap_lazy", "model": MODEL,
        "wired_gb": WIRED_GB, "cache_gb": CACHE_GB,
        "load_s": round(t1 - t0, 2), "gen_s": round(t2 - t1, 2),
        "tok_per_s": round(MAXTOK / (t2 - t1), 2),
        "rss_gb_after_load": round(after_load.rss_bytes / 1e9, 2),
        "rss_gb_after_gen": round(after_gen.rss_bytes / 1e9, 2),
        "mlx_peak_gb": round(after_gen.mlx_peak_bytes / 1e9, 2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2：运行并与基线对比（决策判据）**

Run: `/usr/bin/time -l python -m mlx_streaming.run_mmap 2>&1 | tail -n 30`
Expected 与**决策规则**：
- 若 `rss_gb_after_gen`（及 `/usr/bin/time` 的 max RSS）**明显低于** Task 3 基线（例如基线 ~14 GB，这里降到接近「非专家权重 + 少量活跃专家」量级）且 `tok_per_s` 没有 10x 崩塌 → **走路线 A**（Task 5、6），跳过路线 B。
- 若 RSS 仍 ≈ 整模型（说明 `load` 强制 eval 了全部，或 Metal 在使用时把整堆叠 buffer 提交常驻）→ **走路线 B**（Task 7~10）。
- 记录两种情况下的数字到报告。

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/run_mmap.py
git commit -m "feat(mlx): 加 mmap/lazy 加载探针，作为路线 A/B 的判定脚本"
```

---

### Task 5（仅路线 A）：预算扫描驱动

**Files:**
- Create: `mlx_streaming/sweep.py`

- [ ] **Step 1：写扫描脚本（逐预算起子进程，避免进程内状态污染）**

```python
"""对一组 WIRED_GB 预算逐个起子进程跑 run_mmap，汇总 RSS/速度。"""
import os, sys, json, subprocess

BUDGETS = os.environ.get("BUDGETS", "4,6,8,12,").split(",")  # 末尾空串=不设上限

def run_one(wired_gb: str):
    env = dict(os.environ)
    if wired_gb:
        env["WIRED_GB"] = wired_gb
    else:
        env.pop("WIRED_GB", None)
    p = subprocess.run(
        [sys.executable, "-m", "mlx_streaming.run_mmap"],
        env=env, capture_output=True, text=True,
    )
    # run_mmap 末尾打印 JSON；取最后一个 JSON 块
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            pass
    # 简化：解析整段 stdout 里最后的 JSON 对象
    start = p.stdout.rfind("{")
    end = p.stdout.rfind("}")
    try:
        return json.loads(p.stdout[start:end+1])
    except Exception:
        return {"wired_gb": wired_gb, "error": p.stderr[-500:]}

def main():
    rows = [run_one(b) for b in BUDGETS]
    print(json.dumps(rows, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2：运行扫描**

Run: `python -m mlx_streaming.sweep | tee /tmp/mlx_sweep.json`
Expected：得到一组 `{wired_gb, rss_gb_after_gen, tok_per_s, mlx_peak_gb}`，能画出「预算 ↓ → RSS ↓、tok/s 变化」的前沿（满足 G3）。

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/sweep.py
git commit -m "feat(mlx): 加预算扫描脚本，产出内存/速度帕累托前沿"
```

---

### Task 6（仅路线 A）：实测报告

**Files:**
- Create: `benchmarks/reports/mlx-streaming-moe-<date>.md`

- [ ] **Step 1：汇总报告**

把以下内容写进报告：环境（Task 1 探针输出、MLX/mlx-lm 版本、机器内存）、基线（Task 3）、mmap/lazy（Task 4）、预算扫描表（Task 5），并给出结论：路线 A 是否满足 G1/G2/G3，最优预算建议。套用 `benchmarks/reports/` 既有 md 风格。

- [ ] **Step 2：Commit**

```bash
git add benchmarks/reports/mlx-streaming-moe-*.md
git commit -m "docs(mlx): 路线 A 实测报告（内存/速度/预算前沿）"
```

> **若 Task 4 判定为路线 A，则到此结束。下面 Task 7~11 仅在判定为路线 B 时执行。**

---

### Task 7（仅路线 B）：专家权重后端 + 命中率（TDD）

**Files:**
- Create: `mlx_streaming/expert_store.py`
- Test: `mlx_streaming/tests/test_expert_store.py`

设计要点：`ExpertStore` 持有每层堆叠权重的**惰性句柄**（不 eval），`fetch(layer, expert_ids) -> dict[str, mx.array]` 返回这些专家的 `weight/scales/biases` 子张量；内部 LRU 按 `(layer, expert_id)` 缓存、容量 = 预算；统计 `hits/misses`。两种取数实现：(a) 从 mmap 堆叠张量做 `arr[expert_ids]` 切片；(b) 从离线 per-expert 文件 `mx.load`。

- [ ] **Step 1：写失败测试（用一个小的合成堆叠张量，不依赖真实模型）**

```python
import mlx.core as mx
from mlx_streaming.expert_store import LruExpertStore

def _fake_stacked(num_experts=8, out=16, inp=12):
    return {"weight": mx.arange(num_experts*out*inp).reshape(num_experts, out, inp).astype(mx.float32)}

def test_fetch_returns_only_requested_experts():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=4)
    got = store.fetch(layer=0, expert_ids=[2, 5])
    assert got["weight"].shape[0] == 2
    assert mx.array_equal(got["weight"][0], _fake_stacked()["weight"][2])

def test_lru_hits_and_misses():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=4)
    store.fetch(layer=0, expert_ids=[1, 2])     # 2 misses
    store.fetch(layer=0, expert_ids=[1, 2])     # 2 hits
    assert store.misses == 2
    assert store.hits == 2

def test_capacity_evicts():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=2)
    store.fetch(layer=0, expert_ids=[0, 1])
    store.fetch(layer=0, expert_ids=[2, 3])     # 触发驱逐
    assert store.resident_count() <= 2
```

- [ ] **Step 2：跑测试确认失败**

Run: `pytest mlx_streaming/tests/test_expert_store.py -v`
Expected：FAIL，`ModuleNotFoundError: mlx_streaming.expert_store`。

- [ ] **Step 3：实现 `expert_store.py`**

```python
"""路线 B 专家后端：LRU 缓存 + 按需切片，统计命中率。"""
from collections import OrderedDict
from typing import Dict, List
import mlx.core as mx
from mlx_streaming.mem import clear_cache

class LruExpertStore:
    def __init__(self, stacked: Dict[int, Dict[str, mx.array]], capacity: int):
        # stacked[layer] = {"weight": (E,O,I) [, "scales", "biases"]}（惰性，未 eval）
        self._stacked = stacked
        self.capacity = capacity
        self._cache: "OrderedDict[tuple, Dict[str, mx.array]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def resident_count(self) -> int:
        return len(self._cache)

    def _load_one(self, layer: int, e: int) -> Dict[str, mx.array]:
        src = self._stacked[layer]
        out = {}
        for k, arr in src.items():
            sub = arr[e]            # 切第 e 个专家；mmap 下只触碰该 slab 的页
            mx.eval(sub)            # 显式物化这一个专家
            out[k] = sub
        return out

    def fetch(self, layer: int, expert_ids: List[int]) -> Dict[str, mx.array]:
        picked = []
        for e in expert_ids:
            key = (layer, int(e))
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
            else:
                self.misses += 1
                self._cache[key] = self._load_one(layer, int(e))
                self._cache.move_to_end(key)
                while len(self._cache) > self.capacity:
                    self._cache.popitem(last=False)   # 驱逐最久未用
                    clear_cache()                      # 把缓存还给系统
            picked.append(self._cache[(layer, int(e))])
        # 把选中的若干专家在第 0 维堆叠回 (k_selected, O, I)
        keys = picked[0].keys()
        return {k: mx.stack([p[k] for p in picked]) for k in keys}

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
```

- [ ] **Step 4：跑测试确认通过**

Run: `pytest mlx_streaming/tests/test_expert_store.py -v`
Expected：PASS（3 passed）。

- [ ] **Step 5：Commit**

```bash
git add mlx_streaming/expert_store.py mlx_streaming/tests/test_expert_store.py
git commit -m "feat(mlx): 加 LRU 专家后端(expert_store)，按需切片+命中率统计"
```

---

### Task 8（仅路线 B）：流式 MoE 块 + 数值等价测试（TDD）

**Files:**
- Create: `mlx_streaming/streaming_moe.py`
- Test: `mlx_streaming/tests/test_streaming_equiv.py`

设计要点：`StreamingMoeBlock` 复刻 `Qwen3MoeSparseMoeBlock` 的前向（gate→softmax→top-k），但专家计算改为：取出本次 token 选中的**唯一**专家集合 → `ExpertStore.fetch` → 用 `mx.gather_qmm`（量化）或手算（非量化）只在这些专家上算 → 用「全局专家 id → 本地下标」的重映射喂给 gather。要求与原 `SwitchGLU` 输出数值等价。

- [ ] **Step 1：写等价性失败测试（小型合成 MoE，对照 mlx-lm 的 SwitchGLU）**

```python
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU
from mlx_streaming.streaming_moe import streaming_switch_glu_forward

def test_streaming_matches_switchglu():
    mx.random.seed(0)
    dim, hidden, E, k = 16, 32, 8, 2
    glu = SwitchGLU(dim, hidden, E)
    mx.eval(glu.parameters())
    x = mx.random.normal((1, 5, dim))
    inds = mx.argpartition(mx.random.normal((1, 5, E)), kth=-k, axis=-1)[..., -k:]

    ref = glu(x, inds)
    got = streaming_switch_glu_forward(glu, x, inds)   # 只算选中专家，结果须等价
    assert mx.allclose(ref, got, atol=1e-4).item()
```

- [ ] **Step 2：跑测试确认失败**

Run: `pytest mlx_streaming/tests/test_streaming_equiv.py -v`
Expected：FAIL，`ImportError`/`AttributeError`（函数未定义）。

- [ ] **Step 3：实现 `streaming_moe.py`**

> 第一版先做「数值等价 + 只算选中专家」的纯函数 `streaming_switch_glu_forward`（不接 ExpertStore，先证明 gather 重映射正确）；再写 `StreamingMoeBlock` 把它接到 `ExpertStore`。

```python
"""路线 B：流式 MoE。先证明「只算选中专家」与 SwitchGLU 等价。"""
import mlx.core as mx

def _unique_sorted(inds: mx.array):
    flat = inds.reshape(-1)
    uniq = mx.array(sorted(set(flat.tolist())))      # 选中的全局专家 id
    remap = {int(g): i for i, g in enumerate(uniq.tolist())}
    local = mx.array([remap[int(g)] for g in flat.tolist()]).reshape(inds.shape)
    return uniq, local

def streaming_switch_glu_forward(glu, x, inds):
    """只在 inds 涉及到的唯一专家上做 SwitchGLU 等价计算。"""
    uniq, local = _unique_sorted(inds)               # uniq: (S,), local: 同 inds 形状
    # 取子集专家权重（gate/up/down 三个 SwitchLinear 的 weight，按需含 scales/biases）
    def sub_linear(lin):
        sub = type(lin).__new__(type(lin))
        for name, p in lin.parameters().items():
            # 第 0 维是专家维，按 uniq 取子集
            setattr(sub, name, p[uniq] if p.ndim >= 1 and p.shape[0] == lin.num_experts else p)
        # 复制超参
        for attr in ("group_size", "bits", "mode"):
            if hasattr(lin, attr):
                setattr(sub, attr, getattr(lin, attr))
        sub.__class__ = type(lin)
        return sub
    sub_glu = type(glu).__new__(type(glu))
    sub_glu.gate_proj = sub_linear(glu.gate_proj)
    sub_glu.up_proj = sub_linear(glu.up_proj)
    sub_glu.down_proj = sub_linear(glu.down_proj)
    sub_glu.activation = glu.activation
    sub_glu.training = False
    return sub_glu(x, local)                          # 用本地下标 gather，仅 S 个专家参与
```

> 注：`SwitchLinear`/`QuantizedSwitchLinear` 的参数名（`weight`/`scales`/`biases`/`bias`）按第 3 节源码；上面用 `lin.parameters()` 遍历并对「第 0 维==num_experts」的张量取子集，量化与非量化都覆盖。若 `__new__` 重建对象在你的 mlx-lm 版本上别扭，可改为：直接 `mx.gather_qmm`/`mx.gather_mm` 手写，传 `rhs_indices=local`、权重传 `weight[uniq]`。

- [ ] **Step 4：跑测试确认通过（必要时按上面注释切到手写 gather 版）**

Run: `pytest mlx_streaming/tests/test_streaming_equiv.py -v`
Expected：PASS（数值等价，atol 1e-4）。

- [ ] **Step 5：实现 `StreamingMoeBlock`（接 ExpertStore）并 Commit**

在 `streaming_moe.py` 追加：

```python
class StreamingMoeBlock:
    """包住原 Qwen3MoeSparseMoeBlock，专家计算改为按需子集 + ExpertStore 命中率统计。"""
    def __init__(self, orig_block, store, layer_idx):
        self.gate = orig_block.gate            # 路由器常驻（很小）
        self.top_k = orig_block.top_k
        self.norm_topk_prob = orig_block.norm_topk_prob
        self.switch_mlp = orig_block.switch_mlp
        self.store = store
        self.layer_idx = layer_idx

    def __call__(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        mx.eval(inds)                          # 先物化路由结果，才能按结果取专家
        y = streaming_switch_glu_forward(self.switch_mlp, x, inds)
        return (y * scores[..., None]).sum(axis=-2)
```

```bash
git add mlx_streaming/streaming_moe.py mlx_streaming/tests/test_streaming_equiv.py
git commit -m "feat(mlx): 加流式 MoE 块，只算选中专家且与 SwitchGLU 数值等价"
```

---

### Task 9（仅路线 B）：把模型里的 MoE 块替换为流式块

**Files:**
- Modify: `mlx_streaming/streaming_moe.py`（追加 `patch_model`）

- [ ] **Step 1：实现 `patch_model`**

```python
def patch_model(model, store_factory):
    """把每个 Qwen3MoeSparseMoeBlock 换成 StreamingMoeBlock。store_factory(layer_idx)->store。"""
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            layer.mlp = StreamingMoeBlock(mlp, store_factory(i), i)
    return model
```

- [ ] **Step 2：冒烟测试（真实模型，先确认能跑通、输出像样）**

新建临时脚本或在 REPL：`load(MODEL, lazy=True)` → `patch_model(...)` → `generate(...)` 跑 32 token。
Expected：生成不报错、文本通顺；`store.hit_rate()` > 0。

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/streaming_moe.py
git commit -m "feat(mlx): patch_model 把 Qwen3 MoE 块替换为流式块"
```

---

### Task 10（仅路线 B）：端到端流式运行 + 预算扫描

**Files:**
- Create: `mlx_streaming/run_streaming.py`
- Modify: `mlx_streaming/sweep.py`（支持调用流式入口）

- [ ] **Step 1：写 `run_streaming.py`**

复用 `run_mmap.py` 的结构，但加载后 `patch_model`，预算旋钮改为 `EXPERT_SLOTS`（LRU 容量），输出额外字段 `hit_rate`、`store_resident`。生成后调用一次 `clear_cache()`，并在结束时打印 `snapshot()`。

- [ ] **Step 2：跑单次 + 扫描**

Run（单次）：`/usr/bin/time -l EXPERT_SLOTS=8 python -m mlx_streaming.run_streaming 2>&1 | tail -n 30`
Run（扫描）：`BUDGETS=4,8,16,32 python -m mlx_streaming.sweep`（sweep 改为驱动 `run_streaming`，预算环境变量换成 `EXPERT_SLOTS`）
Expected：得到「专家槽数 ↑ → RSS ↑、命中率 ↑、tok/s 变化」的曲线（满足 G3、G4）。

- [ ] **Step 3：Commit**

```bash
git add mlx_streaming/run_streaming.py mlx_streaming/sweep.py
git commit -m "feat(mlx): 路线 B 端到端流式运行 + 专家槽预算扫描"
```

---

### Task 11（仅路线 B，可选）：离线按专家拆分重打包

**Files:**
- Create: `mlx_streaming/split_experts.py`

仅当 Task 10 显示「从 mmap 堆叠张量切片」仍换页不干净（RSS 压不下去）时才做：把 `switch_mlp.*.weight/scales/biases` 沿专家维拆成 per-expert 小 safetensors（如 `layer{l}_expert{e}.safetensors`），`ExpertStore` 改为从这些小文件 `mx.load`，实现真正「只读用到的文件」。这与 `hypura` 的 per-expert NVMe 切片一一对应（见第 7 节）。

- [ ] **Step 1：写拆分脚本** —— 用 `mx.load(shard, return_metadata=True)` 读堆叠张量，按专家维切片，逐个 `mx.save_safetensors`。
- [ ] **Step 2：让 `ExpertStore` 支持「文件后端」**，`fetch` 走 `mx.load(per_expert_path)`。
- [ ] **Step 3：复跑 Task 10 扫描**，对比切片 vs 文件两种后端的 RSS/速度。
- [ ] **Step 4：Commit。**

---

## 7. 与 hypura 的概念映射（G4）

| hypura（ggml/Rust） | 本方案（MLX/Python） | 说明 |
|---|---|---|
| `ExpertPool`（NVMe staging 缓冲） | `LruExpertStore` 的 LRU 容量 | 都是「最多常驻多少专家」的预算旋钮 |
| `NeuronCache`（LRU 专家切片） | `LruExpertStore`（OrderedDict LRU） | 命中率口径对齐：`hits/(hits+misses)` |
| `Hypura_NVMe` 自定义 buffer（`is_host=true`→CPU） | mmap 张量 / per-expert 文件（统一内存→**GPU** `gather_qmm`） | **这是 MLX 的核心优势：专家计算不回退 CPU** |
| `gpu_layers_from_placement` / `SparseMoeMmap` | `lazy=True` + `use_mmap`（路线 A） | 都靠按需换页降低常驻 |
| `--expert-pool-slots` | `EXPERT_SLOTS`（路线 B）/ `set_wired_limit`（路线 A） | 预算扫描参数 |
| 「Layer prefetch hit rate」报告项 | `store.hit_rate()` + RSS/速度表 | 报告横向可比 |

## 8. 风险与回退

- **R1：`gather_qmm` 在堆叠张量上 gather 时会先把整张量拷进连续 Metal buffer** → 路线 A 的 mmap 省内存失效。回退：路线 B + Task 11 per-expert 文件后端（每个专家独立小张量，gather 前只 `eval` 选中的）。
- **R2：`use_mmap`/`lazy` 在本机 mlx-lm 版本不存在** → Task 1 探针会暴露；路线 A 退化为「靠 `set_wired_limit` 限制常驻 + OS 换页」，或直接走路线 B。
- **R3：`set_wired_limit` 需 macOS 15+** → 低版本只能靠路线 B 的显式驱逐 + `clear_cache` 控常驻。
- **R4：NVMe 带宽上限** → 命中率低时 tok/s 受 I/O 限制，这是物理约束；用预算扫描找「可接受速度下的最低常驻」即可，不强求两全。
- **R5：量化 gather kernel 覆盖** → `mx.gather_qmm` 须支持模型的 `group_size/bits/mode`；Task 8 等价测试会暴露不支持的情况，回退到 per-expert 反量化后普通 matmul。

## 9. 自检（写完计划后用 spec 复查）

- **覆盖：** G1（GPU 计算不回退）→ Task 4/8/9；G2（低常驻）→ Task 3 基线 + Task 4/10；G3（预算旋钮 + 报告）→ Task 5/6/10；G4（与 hypura 对齐）→ 第 7 节 + Task 7 命中率。无遗漏。
- **占位符扫描：** 各步均含真实代码/命令/预期；唯一「按版本择一」处（内存 API 名、`use_mmap`/`lazy`、`__new__` vs 手写 gather）都给了 Task 1 探针依据与明确回退，非 TODO。
- **类型/命名一致：** `LruExpertStore.fetch/hit_rate/resident_count`、`streaming_switch_glu_forward`、`StreamingMoeBlock`、`patch_model`、环境变量 `WIRED_GB/CACHE_GB/EXPERT_SLOTS/BUDGETS/MODEL` 在各任务间一致。

---

## 10. 执行交接

计划已保存。两种执行方式：

1. **Subagent-Driven（推荐）** —— 每个任务派新子代理、任务间评审、迭代快。
2. **Inline Execution** —— 本会话内按 executing-plans 批量执行 + 检查点评审。

选哪个？另外强烈建议先只跑 **Task 0~4**（半天内出 Phase 0 判定），用实测决定走路线 A 还是 B，再继续。
