# 专家池管理统一迁移到 C++（单一权威）实现 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把真实区+侧区+demand+prefetch 的槽管理统一收进一套 C++ 池管理器（单一权威），主线程每层只做 1 次不可避免的 `inds.eval()`、demand 工作全移出热路径，迁移后用「容量不变性 + 字节真值」严格保证正确。

**Architecture:** 分 5 个 Phase：Phase 0 实测前提（并集/FP 噪声地板/可跑 cap）→ Phase 1 正确性护栏（容量不变性 + STG_VERIFY 测试）→ Phase 2 root-cause 并修复现有方案 B 的确定性错槽到 oracle 干净（兑现 exact 收益）→ Phase 3 统一权威 + 降耦合（P3-a~f）→ Phase 4 收尾默认开。每 Phase 独立可验证、config 开关可秒回退。

**Tech Stack:** MLX 0.31.2（Python + Metal）、C++ nanobind 扩展（`native_moe_ext`）、pytest；模型 Qwen3-Next-80B-A3B（512 专家、top_k=10、48 层、12 全注意力）。

**关联 spec:** `docs/superpowers/specs/2026-07-04-cpp-unified-pool-authority-design.md`

> **Phase 0 实测结论回填（2026-07-04）**：MTP verify 前向实际喂 **K=3** 个 token（`verify_in=[x, d_1..d_{K-1}]`，`mtp/generate.py:296`），非早前假设的 K+1=4。故单前向并集理论上界 = `K×top_k = 3×10 = 30`；**实测 U_max=30（恰触顶、零裕度）、p99=29、均值≈19.5**（详见 `benchmarks/reports/union-cap-floor-2026-07-04.md`）。**正确性下限：cap（真实区+侧区可寻址）≥ 30**。→ 后续所有实测/测试的 cap 一律取 ≥ 30；`EXPERT_SLOTS=32` 满足但仅 2 裕度，`cap<30` 必然溢出错槽。

---

## 前置约定（所有 Phase 通用）

- **虚拟环境**：命令一律用 `.venv/bin/python`。
- **native 重编译**：改了 `native/ext/*.cpp|*.h` 后必须 `cd native/ext && make native_moe_ext`（首次若 build 缓存指向旧路径，先 `rm -rf native/ext/build`）。
- **基线跑法**（80B 真实模型，稳态 tok/s）：
  ```
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
    SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
  ```
- **MLX 限制**：不支持布尔索引 `arr[mask]`，一律 `mx.where` / `mx.take`。
- **报告落盘**：每个实测 Phase 的结论写进 `benchmarks/reports/<topic>-2026-07-04.md`。
- **提交粒度**：每个 Task 末尾 commit；不改 git config。

---

## Phase 0 —— 实测前提（写实现前必做，产出验收基准）

### Task 0.1：实测各层单前向最大并集（坐实 cap 下限）

**Files:**
- 只读运行 + 分析：`mlx_streaming/config.py`（`union_prof()` 开关，L208）、`mlx_streaming/core/moe/block.py`（`note_union`，L172-174/212-213）
- 产出：`benchmarks/reports/union-cap-floor-2026-07-04.md`

- [ ] **Step 1: 跑 union_prof 采集并集分布**

Run:
```
UNION_PROF=1 STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=64 \
  ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=64 K=3 MAXTOK=64 WARMUP_TOK=0 REPEAT=1 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/union_prof.log
```
Expected: 日志含 UNION_PROF 分桶输出（按 seq 分桶记每层路由专家并集大小）。用 cap=64 避免采集期本身溢出污染。

- [ ] **Step 2: 提取每层 verify(seq=K) 桶的最大并集**

从 `/tmp/union_prof.log` 读出 seq=3（K=3）桶各层并集的 **max** 与 **p99**。记录：全 48 层里的最大值 `U_max`、以及分层分布。

- [ ] **Step 3: 写结论报告**

在 `benchmarks/reports/union-cap-floor-2026-07-04.md` 记录：`U_max`、p99、分层分布，结论「正确性要求 cap（真实区+侧区可寻址）≥ `U_max`」。**实测 `U_max=30`（= 理论上界 `K×top_k=3×10`，零裕度）**。

> ✅ 已完成（commit 49fe63b + 6b2c1b5）：U_max=30、p99=29、均值≈19.5，浅层并集最大（L0/L2 触顶 30）。附带修正了 UNION_PROF 探针（原只给均值→现出 max/p99/per_layer）与 verify=K 口径注释。

- [ ] **Step 4: Commit**

```bash
git add benchmarks/reports/union-cap-floor-2026-07-04.md
git commit -m "bench(phase0): 实测各层单前向最大并集，坐实 cap 下限"
```

### Task 0.2：实测 FP 噪声地板（确定性 delta 的验收基准）

**Files:**
- 只读运行：`mlx_streaming/runtime/run_mtp_spec.py`（`_baseline_greedy` L42-57）
- 产出：`benchmarks/reports/fp-noise-floor-2026-07-04.md`

- [ ] **Step 1: 基线 greedy 背靠背自比（同配置多次）**

Run（连跑 3 次，记录每次相对首次的 token 差异位数）：
```
for i in 1 2 3; do \
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
    SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=64 WARMUP_TOK=0 REPEAT=1 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/noise_$i.log ; done
```
Expected: 每次日志含 `n_mismatch`（spec 相对 greedy 参照）与输出 token。

- [ ] **Step 2: 量化噪声地板**

比较 3 次运行的 `n_mismatch` 与实际 token 序列：记录「run-to-run 是否变化」。若变化 → 存在 GPU 浮点非确定性；记录其量级 `N_floor`（波动范围）。若稳定 → `N_floor=0`（确定性）。

- [ ] **Step 3: 写结论报告**

在 `benchmarks/reports/fp-noise-floor-2026-07-04.md` 记录 `N_floor` 与判定方法，作为后续「确定性 delta 验收线 = 高于 `N_floor` 即为 bug」。

- [ ] **Step 4: Commit**

```bash
git add benchmarks/reports/fp-noise-floor-2026-07-04.md
git commit -m "bench(phase0): 实测 FP 噪声地板，确立确定性 delta 验收基准"
```

### Task 0.3：实测 32GB 上可安全跑的 cap

**Files:**
- 只读运行：`mlx_streaming/core/mem.py`（`snapshot`/`reset_peak`）
- 产出：`benchmarks/reports/cap-memory-32gb-2026-07-04.md`

- [ ] **Step 1: cap 扫描测峰值内存与 tok/s**

Run（cap ∈ {32, 48, 64}，各记录 `mlx_peak_gb` 与 `spec_tok_per_s`）：
```
for CAP in 32 48 64; do \
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=$CAP ZEROCOPY_DUAL_SOURCE=1 \
    SIDEREGION_LFU=1 POOL_SPEC_SLOTS=$CAP K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
    .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/cap_$CAP.log ; done
```
Expected: 各 cap 的峰值内存与 tok/s。

- [ ] **Step 2: 判定安全 cap**

记录哪个 cap 峰值 < 系统建议工作集（26.8GB）且留有余量。结论：后续 Phase 2 收益实测用「安全且 ≥ `U_max`」的 cap（预期 cap=32 或 48 满足）。

- [ ] **Step 3: 写报告 + Commit**

```bash
git add benchmarks/reports/cap-memory-32gb-2026-07-04.md
git commit -m "bench(phase0): 32GB 上 cap 扫描峰值内存与 tok/s，定安全 cap"
```

### Phase 0 出口判据
`U_max` 已知（cap 下限）、`N_floor` 已知（验收基准）、安全 cap 已定。三者写入报告后进入 Phase 1。

> **⚠️ Phase 0 中途重大发现 + 方向调整（2026-07-04，用户拍板 pivot）**
>
> Task 0.2 实测 N_floor 时发现：**当前默认路径 `ZEROCOPY_DUAL_SOURCE=1` 的侧区存在可复现的字节错槽真 bug**——`DUAL_VERIFY=1` 连跑 3 次每次都 BAD（31/62/54 条），`L1 e453 r32` 跨三次运行复现（确定性）：被路由命中的专家读到了别的专家的权重字节（详见 `benchmarks/reports/fp-noise-floor-2026-07-04.md`）。此外：STG_VERIFY 不覆盖该默认路径（其正确开关是 `DUAL_VERIFY`）；约 70% 走慢回退路径的 acquire 当前无 live 字节校验。
>
> **后果**：① 待对齐的「正确性基线」本身是错的；② token 级容量不变性（原 Task 1.1）在修复前不可用；③ 原 Phase 2 的错槽 root-cause **从「方案 B（默认关）」升级为「默认热路径真 bug」，提到 Phase 0.3/1 之前先做**。
>
> **调整后顺序**：Phase 2（改名 **Phase 2′：默认双源侧区错槽 root-cause + 修复到 DUAL_VERIFY 0 BAD**）→ 回头补 Phase 0.3 + 干净 N_floor + Phase 1 oracle → 原 Phase 3/4 不变。

---

## Phase 1 —— 正确性护栏（不改行为，只建 oracle 与量具）

### Task 1.1：容量不变性回归测试（真实 80B，端到端）

**Files:**
- Create: `mlx_streaming/tests/test_capacity_invariance.py`
- 参考：`mlx_streaming/runtime/run_mtp_spec.py`（`_baseline_greedy` L42-57, `build_streaming_model`/`load_mtp` 导入 L15-18）

- [ ] **Step 1: 写失败测试——同 prompt greedy 在两个 cap 下 token 必须一致**

```python
"""容量不变性 oracle：正确的流式缓存对 cap 不变——同一 prompt greedy 在不同 cap 下
token 序列必须逐位一致。cap 只影响命中率/速度，绝不影响输出。
需真实 80B 模型（QN_CONFIG/MTP_OUT 就位）；无模型环境自动 skip。"""
import os
import pytest
import mlx.core as mx

from mlx_streaming import config as _cfg

_HAS_MODEL = os.path.exists(_cfg.qn_config()) and os.path.exists(_cfg.mtp_out())
pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="需真实 80B 模型权重")

PROMPT = "用三句话解释什么是混合专家模型。"
N = 48


def _greedy_tokens_at_cap(cap: int) -> list:
    """在指定 EXPERT_SLOTS 下跑非投机 greedy，返回前 N 个 token id。"""
    os.environ["EXPERT_SLOTS"] = str(cap)
    os.environ["POOL_SPEC_SLOTS"] = str(cap)
    os.environ.setdefault("STREAM_BLOB_LOADER", "1")
    os.environ.setdefault("NATIVE_FUSED_PREFETCH", "1")
    os.environ.setdefault("ZEROCOPY_DUAL_SOURCE", "1")
    os.environ.setdefault("SIDEREGION_LFU", "1")
    from mlx_streaming.model_builder import build_streaming_model
    from mlx_streaming.mtp.generate import prefill_chunked, forward_with_hidden
    model, tok = build_streaming_model()
    cache = model.make_cache()
    ids = mx.array([tok.encode(PROMPT)])
    logits, _ = prefill_chunked(model, ids, cache)
    out = []
    for _ in range(N):
        nxt = int(mx.argmax(logits[:, -1, :]))
        out.append(nxt)
        cur = mx.array([[nxt]]); mx.eval(cur)
        logits, _ = forward_with_hidden(model, cur, cache)
    return out


def test_greedy_capacity_invariant():
    # cap 必须都 ≥ Phase 0 实测 U_max（默认 32/48 满足）
    toks_a = _greedy_tokens_at_cap(32)
    toks_b = _greedy_tokens_at_cap(48)
    assert toks_a == toks_b, f"容量不变性被破坏：cap32 与 cap48 token 序列不同\n{toks_a}\n{toks_b}"
```

- [ ] **Step 2: 运行确认当前是否已满足（基线体检）**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_capacity_invariance.py -v -s`
Expected: 若当前基线本就容量不变 → PASS（护栏成立）；若 FAIL → 记录，说明基线已有 cap 相关错槽（Phase 2 一并修）。**无论结果都保留此测试作为 oracle。**

> 注：本 Task 的「失败测试」语义是「护栏」——它断言的是不变量，不是新功能。当前基线大概率已满足（除非 cap 低于 `U_max`）。

- [ ] **Step 3: Commit**

```bash
git add mlx_streaming/tests/test_capacity_invariance.py
git commit -m "test(phase1): 容量不变性 oracle——同 prompt greedy 跨 cap token 一致"
```

### Task 1.2：STG_VERIFY 字节真值提为一等公民测试

**Files:**
- Create: `mlx_streaming/tests/test_pool_byte_truth.py`
- 参考：`mlx_streaming/core/cache/resident_pool.py`（`_STG_VERIFY` L23, `verify_acquire_bytes` 用法 L202-205）、`native_prefetch.cpp`（`sideregion_kv`/`real_region_contents`/`blob_load`）

- [ ] **Step 1: 写失败测试——开 STG_VERIFY 跑一段 decode，断言 0 BAD**

```python
"""字节真值 oracle：开 STG_VERIFY 跑一段 decode，断言每个占用槽的池字节 ==
其属主专家磁盘真值（0 BAD）。需真实 80B 模型；无则 skip。"""
import os
import pytest

from mlx_streaming import config as _cfg

_HAS_MODEL = os.path.exists(_cfg.qn_config()) and os.path.exists(_cfg.mtp_out())
pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="需真实 80B 模型权重")


def test_stg_verify_zero_bad(capfd):
    os.environ["STG_VERIFY"] = "1"
    os.environ["EXPERT_SLOTS"] = "48"
    os.environ["POOL_SPEC_SLOTS"] = "48"
    os.environ.setdefault("STREAM_BLOB_LOADER", "1")
    os.environ.setdefault("NATIVE_FUSED_PREFETCH", "1")
    os.environ.setdefault("ZEROCOPY_DUAL_SOURCE", "1")
    os.environ.setdefault("SIDEREGION_LFU", "1")
    import mlx.core as mx
    from mlx_streaming.model_builder import build_streaming_model
    from mlx_streaming.mtp.generate import prefill_chunked, forward_with_hidden
    model, tok = build_streaming_model()
    cache = model.make_cache()
    ids = mx.array([tok.encode("用三句话解释什么是混合专家模型。")])
    logits, _ = prefill_chunked(model, ids, cache)
    for _ in range(32):
        nxt = int(mx.argmax(logits[:, -1, :]))
        cur = mx.array([[nxt]]); mx.eval(cur)
        logits, _ = forward_with_hidden(model, cur, cache)
    out = capfd.readouterr().out
    assert "[STG_VERIFY] BAD" not in out, f"检测到池槽字节污染:\n{out}"
```

- [ ] **Step 2: 运行确认**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_pool_byte_truth.py -v -s`
Expected: PASS（基线字节应为真值）。若 FAIL → 记录 BAD 详情，Phase 2 一并查。

- [ ] **Step 3: Commit**

```bash
git add mlx_streaming/tests/test_pool_byte_truth.py
git commit -m "test(phase1): STG_VERIFY 字节真值 oracle 提为一等公民测试"
```

### Phase 1 出口判据
两个 oracle 测试就位并在基线上跑过（PASS 或已记录 FAIL 原因）。这是 Phase 2 判定「修好了没」的唯一依据。

---

## Phase 2 —— root-cause 并修复方案 B 确定性错槽（走 systematic-debugging）

> **性质**：这是调试任务。以下 Task 给出**结构化调查步骤 + 验收**，具体 fix 代码由 root-cause 结论决定，不预写占位代码。
> **REQUIRED SUB-SKILL**：执行本 Phase 必须用 superpowers:systematic-debugging。

### Task 2.1：用 oracle 复现方案 B 的 n_mismatch（RED）

**Files:** 只读 `native/ext/native_prefetch.cpp`（`demand_core_locked` L757-798, `demand_dual` L800-857）、`mlx_streaming/core/cache/resident_pool.py`（`acquire_gpu_dual` L567-628）

- [ ] **Step 1: 在 cap ≥ U_max（无溢出）下开 NATIVE_DEMAND_DUAL 跑 oracle**

Run（cap 用 Phase 0 安全 cap，须 ≥ `U_max`）：
```
NATIVE_DEMAND_DUAL=1 EXPERT_SLOTS=48 POOL_SPEC_SLOTS=48 STG_VERIFY=1 \
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 \
  K=3 MAXTOK=64 WARMUP_TOK=64 REPEAT=2 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec 2>&1 | tee /tmp/schemeB_repro.log
```
Expected: 复现 `n_mismatch` 高于 `N_floor`（确定性 delta），且 STG_VERIFY 0 BAD（字节对、映射错）。**这确认 bug 是「错槽而非坏字节」。**

- [ ] **Step 2: 记录复现事实**

写 `benchmarks/reports/schemeB-mismatch-rootcause-2026-07-04.md` 开头：复现的 cap、n_mismatch、delta、STG_VERIFY 结果、是否无溢出（cap ≥ U_max）。

### Task 2.2：定位分叉点（systematic-debugging 二分）

- [ ] **Step 1: 逐 token 对比 B vs 基线的 `local`**

方法：在 `acquire_gpu_dual`（Python 基线）与 `demand_dual`（C++）路径各加临时诊断，dump 首个分叉层/token 的 `inds`、side 快照、real e2r、最终 `local`。找到第一个 `local` 不同的位置。

- [ ] **Step 2: 判定分叉根因（候选逐一排除）**

按 spec §4 嫌疑排查：
1. **侧区 gen 新鲜度**：该槽在当前 read-gen 是否已填成该专家字节？（对比 `sideregion_kv` 的 gen 与实际字节）
2. **side/real 优先级**：expert 同时在 side 和 real 时，B 与基线 `eff[keys]=vals` 是否选同一槽？
3. **LFU 记账分叉**：canonical（C++ 全 bump）vs 基线回退层省略 bump 是否导致不同驱逐 → 不同池内容？（注意：无溢出时不同池内容仍应输出中性，除非叠加了 1/2）
记录哪条是首因。

- [ ] **Step 2 出口**：`benchmarks/reports/schemeB-mismatch-rootcause-2026-07-04.md` 写明确定的首因 + 证据。

### Task 2.3：按 root-cause 修复（GREEN）+ oracle 验收

- [ ] **Step 1: 写针对性单测复现该错槽**（在 `mlx_streaming/tests/test_demand_dual_native.py` 扩展）

按首因构造最小合成场景（沿用该文件 `demand_dual` 合成池风格），断言 B 路径的 `local` 与预期真值一致。先 RED。

- [ ] **Step 2: 修 C++ `demand_core_locked` / 侧区合并逻辑**（据首因）

改 `native/ext/native_prefetch.cpp` 相应处 → `cd native/ext && make native_moe_ext`。

- [ ] **Step 3: 单测转 GREEN**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_demand_dual_native.py -v`
Expected: 全 PASS。

- [ ] **Step 4: oracle 端到端验收**

Run Task 2.1 的命令 + Task 1.1/1.2 的两个 oracle 测试。
Expected: `NATIVE_DEMAND_DUAL=1` 下 `n_mismatch` 回落到 `≤ N_floor`（确定性 delta 消失）、STG_VERIFY 0 BAD、容量不变性测试 PASS。

- [ ] **Step 5: 收益实测 + Commit**

Run 安全 cap 下 A/B（ON vs OFF）tok/s。记录净收益。
```bash
git add native/ext/native_prefetch.cpp mlx_streaming/tests/test_demand_dual_native.py \
  benchmarks/reports/schemeB-mismatch-rootcause-2026-07-04.md
git commit -m "fix(phase2): 根治方案B确定性错槽——NATIVE_DEMAND_DUAL 达 oracle 干净"
```

### Phase 2 出口判据
`NATIVE_DEMAND_DUAL=1` 在安全 cap 下：容量不变性 PASS + STG_VERIFY 0 BAD + n_mismatch ≤ `N_floor` + tok/s 净提升。**此时现有方案 B 已是 exact 收益，可作为 Phase 3 的稳定基座。**

---

## Phase 3 —— 统一权威 + 降耦合（P3-a~f，每子项独立可验证）

> 每子项：先在合成/端到端加断言（RED）→ 迁移实现（GREEN）→ 全程保持 Phase 1 oracle 绿 + Phase 2 收益不回退。每子项单独 config 开关灰度、独立 commit。
> 详细 bite-sized 步骤在 Phase 2 root-cause 完成后据实际接口细化（避免此刻臆测 C++ 接口签名写占位）。

- [ ] **Task 3.1（P3-a）promote 落真实区迁 C++**：新增 C++ `promote_to_real(layer, route_inds)` 直写 `g_real`，替换 `native_staging.py:143-177` 的 Python `_place_expert`。验收：oracle 绿 + promote 命中率不降。
- [ ] **Task 3.2（P3-b）`route_used_subset` 并入 C++ promote**：消 `native_staging.py:46` 的 per-layer `.tolist()`。验收：oracle 绿 + 该 drain 消失（PREFETCH_TPROF 佐证）。
- [ ] **Task 3.3（P3-c）侧区∪真实区 overlay 迁 C++**：`acquire_gpu_dual` 的 `eff[keys]=vals`（`resident_pool.py:576-578`）→ C++ 单次 gather。验收：oracle 绿。
- [ ] **Task 3.4（P3-e）删 resident 快照跨界**：`submit`/`promote` 不再从 Python 传 `resident_experts` list（`block.py:317-318`），C++ 内读 `g_real`。验收：oracle 绿 + block.py 该处调用简化。
- [ ] **Task 3.5（P3-f）Python 读 C++ 状态降级为只读诊断**：`real_region_contents/_count/real_freq/...` 统一为只读 API；删 Python `_slot_of/_free/_freq/_slot_table` 死影子（dual 路径）。验收：oracle 绿 + 单一权威（无双份状态）。
- [ ] **Task 3.6（P3-d，风险最高、排末尾、单独开关灰度）统一并行读**：`blob_loader` 的 `ThreadPoolExecutor(8)` → 统一到 C++ `BgReader`；Python demand/host 只提交 ticket + wait；物化保留 lazy 切片。验收：oracle 绿 + tok/s 不回退 + 无死锁/竞态（STG_VERIFY 0 BAD、多次跑稳定）。
- [ ] **Task 3.7 收敛 block.py**：确认计算段塌缩为 `pm.acquire` / `pm.prefetch`，无 §5.3 所列内部窥探。验收：grep 确认 `block.py` 不再引用 `_slot_of/_stg_mgr.last_ready/stg.src._segs` 等。

### Phase 3 出口判据
`block.py` 塌缩为「gate → 一次 acquire / 一次 prefetch」；Python 无死影子槽状态；oracle 全绿；tok/s ≥ Phase 2。

---

## Phase 4 —— 收尾

- [ ] **Task 4.1**：统一 C++ 路径设为默认（评估 `NATIVE_DEMAND_DUAL` 默认值翻转为 1），保留开关兜底一版。验收：默认配置下 oracle 全绿、tok/s 达 Phase 2 收益。
- [ ] **Task 4.2**：退役 Python 槽路径的死代码（保留 prefill host 路径 + 诊断），更新 `config.py` 注释。
- [ ] **Task 4.3**：归档最终收益报告 `benchmarks/reports/cpp-unified-pool-final-2026-07-04.md`（baseline vs 统一后 tok/s、内存、n_mismatch）。

---

## Self-Review（对照 spec）

- **Spec §3 正确性口径（容量不变性 + 字节真值）** → Task 1.1 / 1.2 覆盖。
- **Spec §4 待根治错槽** → Phase 2 覆盖（含无溢出前提、侧区 gen 首要嫌疑）。
- **Spec §5 架构 / PoolManager 接口** → Phase 3 各子项 + Task 3.7 收敛覆盖。
- **Spec §6 Phase 0-4** → 本 plan Phase 0-4 一一对应。
- **Spec §6 Phase 3 的 P3-a~f** → Task 3.1-3.6 一一对应（P3-a=3.1, b=3.2, c=3.3, e=3.4, f=3.5, d=3.6）。
- **Spec §7 内存约束** → Task 0.3 覆盖。
- **Spec §9 成功标准** → Phase 2/3 出口判据 + Task 4.3 覆盖。
- **占位符扫描**：Phase 0/1/2 为具体命令与代码；Phase 3 明确标注「接口签名待 Phase 2 后细化」，属有意的增量而非占位（依赖 root-cause 结论，此刻臆测反而有害）。
- **类型一致性**：`U_max` / `N_floor` / 安全 cap 三个 Phase 0 产出贯穿 Phase 1/2 验收，命名一致。
