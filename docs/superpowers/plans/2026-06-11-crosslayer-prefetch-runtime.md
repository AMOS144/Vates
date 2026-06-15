# 同 token 跨层预取运行时 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用「同 token、提前 AHEAD 层的真实 hidden + 目标层 gate」预测目标层专家，异步预取 `pred∩非常驻`，在 AHEAD 层窗口内物化好、目标层 MoE 前 promote，把中池 miss 转 hit。

**Architecture:** 复用已建好测过的预取设施（`BackgroundExpertPrefetcher` / `promote_prefetched`(永不驱逐 current) / `_submit_missing_prefetch` / `bg_stats.ready_on_time` / `WINDOW_PROF`）。唯一实质改动：把现有跨层 hook 的预测 norm 从「预测层自己的 `post_attention_layernorm`」改成「**目标层的** `post_attention_layernorm`」——这是 probe 验证 recall_miss≈0.95 的配置（历史 AHEAD=1 只有 0.83 疑似就是用错了 norm）。

**Tech Stack:** Python、MLX、现有 `streaming_moe.py` 跨层 hook / `expert_store` / `bg_prefetch` / `run_mtp_spec`。

---

## 背景（执行者必读）

probe `probe_crosslayer_miss_recall.py` 已验证（MAXTOK=64/K=2，中池 cap∈{64,96}）：
- 用第 `L-AHEAD` 层真实 hidden 经 `gate_L(post_attention_layernorm_L(h_{L-AHEAD}))` 预测第 L 层专家。
- `recall_miss`@mult=2：AHEAD=1→0.953、AHEAD=2→0.931、AHEAD=3→0.934（pred≈32/层）。
- 窗口：AHEAD=1≈1.1ms、AHEAD=2≈2.2ms，均 ≫ 物化 340µs。
- 对比死路：同层 AHEAD=0 窗口仅 70µs；跨-token 历史信号 recall_miss≈0。

现有 hook（`mlx_streaming/core/streaming_moe.py` `enable_cross_layer_prefetch` 内 `patched_call`，约 837–870 行）已支持 AHEAD≥1 + `STREAM_BLOB_BG`，但第 853 行用 `self.post_attention_layernorm(x)`（预测层 norm）。需改为目标层 norm 以匹配验证配置。

---

## Task 1：预测改用目标层 norm（匹配验证的 0.95 配置）

**Files:**
- Modify: `mlx_streaming/core/streaming_moe.py`（抽出 `_predict_layer_experts` 辅助 + hook 用目标层 norm）
- Test: `mlx_streaming/tests/test_predict_layer_experts.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_predict_layer_experts.py
import mlx.core as mx
from mlx_streaming.core.streaming_moe import _predict_layer_experts


def test_predict_layer_experts_returns_topk_with_scores():
    # norm/gate 都用恒等：gates=softmax(x)，top_k=1、mult=2 → k=2 → 取最大两个下标 {1,3}
    ident = lambda t: t
    x = mx.array([[[1.0, 5.0, 2.0, 9.0, 3.0]]])
    best = _predict_layer_experts(ident, ident, top_k=1, x=x, mult=2)
    assert set(best.keys()) == {1, 3}
    # 分数为 softmax 概率，下标 3（logit 9）应高于下标 1（logit 5）
    assert best[3] > best[1]
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest mlx_streaming/tests/test_predict_layer_experts.py -p no:cacheprovider`
Expected: FAIL（`ImportError: cannot import name '_predict_layer_experts'`）

- [ ] **Step 3: 实现辅助函数**（加在 `streaming_moe.py` 的 `_submit_missing_prefetch` 之后）

```python
def _predict_layer_experts(norm, gate, top_k: int, x: mx.array, mult: int) -> "dict[int, float]":
    """用 gate(norm(x)) 预测专家集，返回 {expert_id: 最大 softmax 分数}。

    norm/gate 必须取自**目标层**（被预测的那层），以匹配 probe 验证的
    gate_L(post_attention_layernorm_L(h)) 配置（recall_miss≈0.95）。
    """
    gates = mx.softmax(gate(norm(x)), axis=-1, precise=True)
    k = min(gates.shape[-1], top_k * mult)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    vals = mx.take_along_axis(gates, inds, axis=-1)
    mx.eval(inds, vals)
    best: dict[int, float] = {}
    for e, s in zip(inds.reshape(-1).tolist(), vals.reshape(-1).tolist()):
        e, s = int(e), float(s)
        if s > best.get(e, -1.0):
            best[e] = s
    return best
```

- [ ] **Step 4: 跑绿（辅助函数）**

Run: `uv run pytest mlx_streaming/tests/test_predict_layer_experts.py -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: hook 改用目标层 norm + 辅助函数**

把 `patched_call` 中下面这段（约 845–863 行）：

```python
            target_mlp = None
            if ahead == 0 and isinstance(mlp, FileStreamingMoeBlock):
                target_mlp = mlp
            elif target_layer is not None:
                layers = getattr(getattr(self, "_prefetch_model_ref", None), "layers", [])
                if 0 <= target_layer < len(layers):
                    target_mlp = getattr(layers[target_layer], "mlp", None)
            if isinstance(target_mlp, FileStreamingMoeBlock):
                h_pred = self.post_attention_layernorm(x)
                gates = mx.softmax(target_mlp.gate(h_pred), axis=-1, precise=True)
                k = min(gates.shape[-1], target_mlp.top_k * _cross_layer_prefetch_mult())
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                vals = mx.take_along_axis(gates, inds, axis=-1)
                mx.eval(inds, vals)
                best = {}
                for e, s in zip(inds.reshape(-1).tolist(), vals.reshape(-1).tolist()):
                    e, s = int(e), float(s)
                    if s > best.get(e, -1.0):
                        best[e] = s
```

替换为（目标层 norm；AHEAD=0 时目标层即 self）：

```python
            target_mlp = None
            target_decoder = None
            if ahead == 0 and isinstance(mlp, FileStreamingMoeBlock):
                target_mlp = mlp
                target_decoder = self
            elif target_layer is not None:
                layers = getattr(getattr(self, "_prefetch_model_ref", None), "layers", [])
                if 0 <= target_layer < len(layers):
                    target_decoder = layers[target_layer]
                    target_mlp = getattr(target_decoder, "mlp", None)
            if isinstance(target_mlp, FileStreamingMoeBlock):
                # 关键：用**目标层**的 post_attention_layernorm（匹配 probe 验证的
                # gate_L(post_norm_L(h_{L-ahead})) 配置，recall_miss≈0.95）。
                best = _predict_layer_experts(
                    target_decoder.post_attention_layernorm, target_mlp.gate,
                    target_mlp.top_k, x, _cross_layer_prefetch_mult())
```

- [ ] **Step 6: 跑回归**（确认 hook 重构没破坏现有跨层/预取测试）

Run: `uv run pytest mlx_streaming/tests/test_bg_prefetch.py mlx_streaming/tests/test_predict_layer_experts.py -p no:cacheprovider`
Expected: PASS（全绿）

- [ ] **Step 7: 提交**

```bash
git add mlx_streaming/core/streaming_moe.py mlx_streaming/tests/test_predict_layer_experts.py
git commit -m "feat: cross-layer prefetch predicts with target-layer norm (validated 0.95 recall_miss)"
```

---

## Task 2：端到端 A/C × cap × AHEAD 实测

**Files:**
- Create: `mlx_streaming/cli/probe_crosslayer_prefetch.py`
- Append: `benchmarks/reports/low-memory-streaming-moe-2026-06-11.md`

- [ ] **Step 1: 写 probe**

```python
# mlx_streaming/cli/probe_crosslayer_prefetch.py
"""端到端 A(plain 中池) vs C(同 token 跨层预取) × cap∈{64,96} × AHEAD∈{1,2}。

判定：C 的 tok/s 是否 > A（同 cap）、hit 是否提升、ready_on_time 率是否高（窗口够）、
内存仍是中池水平、exact_match 与 A 一致。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXPERT_DIR = os.path.join(ROOT, "models", "qwen3_next_experts_2bit_g128")
BLOB_DIR = "/tmp/cb_2bit_blob"

COMMON = {
    "MODEL": "/tmp/qwen3_next_80b_4bit",
    "EXPERT_DIR": EXPERT_DIR,
    "RESIDENT_POOL": "1",
    "MTP_VERIFY_MODE": "batch",
    "MTP_ARRAY_COMMIT": "1",
    "K": "2",
    "MAXTOK": "64",
    "NATIVE_MOE": "0",
    "STREAM_BLOB": "0",
}
CAPS = [64, 96]
AHEADS = [1, 2]
FIELDS = ["spec_tok_per_s", "spec_hit_rate", "mlx_peak_gb", "rss_gb",
          "exact_match", "n_mismatch", "bg_stats", "window_prof"]


def _run(name, overrides):
    env = os.environ.copy()
    env.update(COMMON)
    env.update(overrides)
    out = subprocess.check_output(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"], env=env, text=True)
    rec = json.loads(out)
    row = {"variant": name}
    row.update({k: rec.get(k) for k in FIELDS})
    bg = rec.get("bg_stats") or {}
    rot, nr = bg.get("ready_on_time", 0), bg.get("not_ready", 0)
    row["ready_rate"] = round(rot / max(rot + nr, 1), 3) if (rot + nr) else None
    return row


def main():
    rows = []
    for cap in CAPS:
        a = _run(f"A_plain_{cap}", {
            "EXPERT_SLOTS": str(cap), "STREAM_BLOB_LOADER": "1",
            "STREAM_BLOB_BG": "0", "BLOB_DIR": BLOB_DIR, "CROSS_LAYER_PREFETCH": "0"})
        rows.append(a)
        base = a.get("spec_tok_per_s") or 1e-9
        for ahead in AHEADS:
            c = _run(f"C_ahead{ahead}_{cap}", {
                "EXPERT_SLOTS": str(cap), "STREAM_BLOB_BG": "1", "BLOB_DIR": BLOB_DIR,
                "STREAM_BLOB_WINDOW": str(ahead + 1), "CROSS_LAYER_PREFETCH": "1",
                "CROSS_LAYER_PREFETCH_AHEAD": str(ahead), "CROSS_LAYER_PREFETCH_MULT": "2",
                "STREAM_BLOB_BG_BUDGET": "32"})
            c["tok_s_vs_A"] = round((c.get("spec_tok_per_s") or 0) / base, 3)
            rows.append(c)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑**

Run: `uv run python -m mlx_streaming.tools.probe_crosslayer_prefetch`
Expected: 打出 A/C 各变体的 `spec_tok_per_s / spec_hit_rate / ready_rate / mlx_peak_gb / exact_match`。

- [ ] **Step 3: 判定**
  - **正确性**：每个 C 的 `n_mismatch` 与同 cap 的 A 一致（数值等价）。
  - **窗口够不够**：C 的 `ready_rate` 高（≫ 同层方案的 ~0）。
  - **净收益**：C 的 `spec_tok_per_s` ≥ A（同 cap）、`spec_hit_rate` 提升、`mlx_peak_gb` 仍是中池水平。
  - AHEAD=1 vs 2 比较：窗口更大（2）是否换来更高 ready_rate / 净收益。

- [ ] **Step 4: 写报告**（把 A/C × cap × AHEAD 表 + ready_rate + 结论追加到报告；明确推荐配置或记录瓶颈）

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/cli/probe_crosslayer_prefetch.py benchmarks/reports/low-memory-streaming-moe-2026-06-11.md
git commit -m "feat: end-to-end cross-layer prefetch A/C probe + results"
```

---

## 自检

- **Spec/验证覆盖**：Task 1 把运行时预测对齐到 probe 验证的目标层-norm 配置（recall_miss 0.95）；Task 2 端到端实测净收益 + ready_rate（窗口够不够）+ 正确性。
- **占位符**：无；所有代码步给出完整代码与可跑命令。
- **类型一致**：`_predict_layer_experts(norm, gate, top_k, x, mult) -> dict[int,float]` 在 Task 1 定义并被 hook 调用；hook 用 `best`（同名同结构）排序后传 `_submit_missing_prefetch`（已存在，收 `list[int]`）。
- **复用既有不变量**：后台只物化私有 array、promote 永不驱逐 current、`ready_on_time`/`WINDOW_PROF` 埋点已就绪。
- **风险显式实测**：每层 `pred∩非常驻` 体量 vs AHEAD 窗口能否物化完，由 Task 2 的 `ready_rate` 给出判决；不足则 AHEAD=2 拉长窗口或调小 `STREAM_BLOB_BG_BUDGET`。
