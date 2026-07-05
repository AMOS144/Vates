# 最小树 top-2 救回评测（方案 B）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已有但从未干净评测的最小树 top-2 串行救回（`TREE_TOP2`），先补齐单测锁死 lossless、修掉救回后 `accepted_in` 用错链的 bug，再用稳态多 prompt A/B 得出净 tok/s 的 go/no-go 结论。

**Architecture:** 先在玩具模型上用 TDD 锁住"最小树输出逐 token 等于贪婪"与"救回确实触发"，并修 `generate.py` 救回分支的 `accepted_in` 陈旧 bug；抽出可测的纯裁决逻辑（中位数 + go/no-go）；再把 `benchmarks/bench_tree.py` 升级成稳态（warmup + 多 prompt + repeat 取中位）A/B harness；最后跑真实 80B 出报告。

**Tech Stack:** Python 3.12、MLX、mlx-lm（Qwen3-Next）、pytest。运行前置：`uv pip install -e .`，模型在 `models/`。

---

## 背景与约束（来自 spec）

spec：`docs/superpowers/specs/2026-07-05-minimal-tree-top2-rescue-design.md`。

- **lossless 红线**：最小树只接受"等于主模型贪婪 argmax"的 token，输出必须逐 token 等于非投机贪婪。
- **不放大并集**：救回是串行第二次 `1×K` 前向，绝不 batch 多路径（区别于已证伪的 `tree_verify`）。
- **裁决**：所有 prompt `exact_match==true`（硬门）；跨 prompt 中位 tok/s 提升 `>+5%` → go，`-5%~+5%` → 持平，`<-5%` → no-go。
- **生产配置**：`STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3`。

## 现状关键事实（已核对）

- `mlx_streaming/mtp/generate.py`：主循环里 `tree_mode = config.tree_top2()`。救回条件 `matched==0 and tree_b[0]==preds[0]`，命中则 `_restore(main_cache, snap_m)` 后改验 B 链。
- **待修 bug**：救回分支（`generate.py` 约 319-327 行）重算了 `vlogits/vH/preds/drafts/matched`，但**没有重建 `verify_in`**；随后（约 364 行）`accepted_in = verify_in[:, :accepted_len]` 仍取 chainA 的旧数组（含被拒错 token）。该 `accepted_in` 只喂 `drafter.sync`（约 408 行），故主输出仍 lossless，但会用错 token 推进 MTP cache → 后续草稿变差 → 低估救回收益。
- `mlx_streaming/mtp/drafter.py::draft_tree`：位置 1 取 top-2，返回 (chainA, chainB) 各长 K。
- 测试收口：`pyproject.toml` 的 `testpaths = ["mlx_streaming/tests"]`；`mlx_streaming/tests/` 与 `benchmarks/` 均**非包**（无 `__init__.py`）。
- 现有 `benchmarks/bench_tree.py`：单 prompt、无 warmup、无 repeat——tok/s 不可信，需升级。
- 现有 `router_pred_prompts.txt`：50 条多样中文 prompt，可选子集复用。

## 文件结构

- **修改** `mlx_streaming/mtp/generate.py`：救回分支重建 `verify_in`，修 `accepted_in` 陈旧 bug（Task 2）。
- **修改** `mlx_streaming/tests/test_mtp_generate.py`：新增最小树 lossless + 救回触发 + `accepted_in` 正确性三个玩具单测（Task 1、Task 2）。
- **新建** `mlx_streaming/mtp/bench_verdict.py`：纯函数 `median` / `verdict_from_delta`（Task 3）。
- **新建** `mlx_streaming/tests/test_bench_verdict.py`：`bench_verdict` 单测（Task 3）。
- **重写** `benchmarks/bench_tree.py`：稳态多 prompt A/B harness（Task 4）。
- **新建** `benchmarks/reports/minimal-tree-top2-2026-07-05.md`：评测报告（Task 5）。

---

## Task 1: 玩具模型锁死最小树 lossless + 救回触发

**Files:**
- Modify: `mlx_streaming/tests/test_mtp_generate.py`（在文件末尾追加）

说明：现有测试已有 `_ToyModel`、`_naive_greedy`（KV-only 玩具自回归模型 + 逐 token 贪婪参考）。本任务加一个 **oracle drafter**：它按贪婪参考序列 `ref` 精确给出主模型真实 preds，从而**保证**每步 chainA 首草稿错（触发 `matched==0`）、chainB 首草稿 = 真实 preds[0]（触发救回并全命中）。这样能在无 80B 的前提下确定性地跑通救回分支并断言 lossless。

- [ ] **Step 1: 写失败测试**

在 `mlx_streaming/tests/test_mtp_generate.py` 末尾追加：

```python
# ---------------------------------------------------------------- 最小树 top-2 救回
class _OracleTreeDraft:
    """玩具 drafter：用贪婪参考 ref 当"主模型真实 preds"的 oracle。

    每步 chainA 首草稿故意置错(必触发 matched==0)、chainB = 从当前位置起的真实贪婪续,
    故 chainB[0]==preds[0] 必触发救回且全命中。用于确定性地覆盖 tree_top2 救回分支,
    验证主输出仍逐 token 等于贪婪(lossless)。pos 靠 sync 每步按接受长度前进。
    """

    def __init__(self, ref, vocab):
        self.ref = list(ref)      # 完整贪婪续(主模型真值)
        self.pos = 0              # 当前 x 在 ref 中的下标
        self.vocab = vocab

    def make_cache(self):
        # 返回非 None,使 generate 触发 snap_d 与 sync(sync 用于推进 pos)。
        from mlx_lm.models import cache as kvcache
        return [kvcache.KVCache()]

    def draft_tree(self, H_last, x_ids, mtp_cache, K):
        nxt = self.ref[self.pos + 1: self.pos + 1 + K]   # 真实 preds[0..K-1]
        while len(nxt) < K:                              # 末尾补齐(会被拒,不破坏 lossless)
            nxt.append(0)
        wrong0 = (nxt[0] + 1) % self.vocab               # 保证 != preds[0]
        chain_a = [wrong0] + nxt[1:]                     # 首错 → matched==0
        chain_b = list(nxt)                              # 首=真实 → 救回且全命中
        return chain_a, chain_b

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        self.pos += int(replay_in.shape[1])              # 按本步接受长度前进


def test_mtp_generate_tree_top2_lossless_and_rescues(monkeypatch):
    """最小树 top-2:强制每步走救回分支,主输出必须逐 token 等于贪婪,且救回确有触发。"""
    from mlx_streaming.mtp.generate import mtp_generate

    monkeypatch.setenv("TREE_TOP2", "1")
    monkeypatch.delenv("MTP_VERIFY_MODE", raising=False)
    monkeypatch.delenv("TREE_VERIFY", raising=False)
    mx.random.seed(0)
    model = _ToyModel(nl=2, vocab=40)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    prompt = mx.array([[1, 5, 9]])
    ref = _naive_greedy(model, prompt, 16)

    drafter = _OracleTreeDraft(ref, vocab=40)
    got, stats = mtp_generate(model, drafter, None, prompt, 16, K=3, ids_mode=True)

    assert got == ref                       # lossless
    assert stats["tree_rescues"] > 0        # 救回分支确实被覆盖
```

- [ ] **Step 2: 跑测试确认结果**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py::test_mtp_generate_tree_top2_lossless_and_rescues -v`
Expected: PASS（若 FAIL，说明最小树主输出 lossless 被破坏，属真实 bug，需先在 `generate.py` 定位修复再继续）。

- [ ] **Step 3: 提交**

```bash
git add mlx_streaming/tests/test_mtp_generate.py
git commit -m "test(mtp): 最小树 top-2 救回 lossless + 触发覆盖(玩具 oracle)"
```

---

## Task 2: 修救回后 `accepted_in` 取错链的 bug

**Files:**
- Modify: `mlx_streaming/tests/test_mtp_generate.py`（追加 spy 测试）
- Modify: `mlx_streaming/mtp/generate.py`（救回分支重建 `verify_in`）

原理：救回后 `drafts` 已是 chainB，但 `accepted_in` 仍由 chainA 的 `verify_in` 切片得到。修法是在救回分支内把 B 链数组赋回 `verify_in`，令后续 `accepted_in = verify_in[:, :accepted_len]` 自动取到 chainB。用一个记录 `sync` 入参的 spy drafter 断言 `accepted_in` 的 token 来自 chainB。

- [ ] **Step 1: 写失败测试**

在 `mlx_streaming/tests/test_mtp_generate.py` 末尾追加：

```python
class _SpyTreeDraft(_OracleTreeDraft):
    """在 oracle 基础上记录每步 sync 收到的 replay_in(即 accepted_in)的 token 序列。"""

    def __init__(self, ref, vocab):
        super().__init__(ref, vocab)
        self.synced = []                    # 每步: (replay_in 的 python list)

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        self.synced.append([int(t) for t in replay_in[0].tolist()])
        super().sync(prev_H, rH, replay_in, mtp_cache)


def test_tree_top2_rescue_accepted_in_uses_chain_b(monkeypatch):
    """救回后喂给 sync 的 accepted_in 必须是 B 链 token,不能残留 chainA 的错误首草稿。

    构造:每步必救回且全命中 → 接受长度=K=3 → accepted_in==[x, chain_b[0], chain_b[1]]。
    chainA 首为 wrong0=(preds0+1)%vocab,若 bug 存在(用旧 verify_in),accepted_in[1] 会等于
    wrong0 != preds0,断言失败。
    """
    from mlx_streaming.mtp.generate import mtp_generate

    monkeypatch.setenv("TREE_TOP2", "1")
    monkeypatch.delenv("MTP_VERIFY_MODE", raising=False)
    monkeypatch.delenv("TREE_VERIFY", raising=False)
    mx.random.seed(0)
    model = _ToyModel(nl=2, vocab=40)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    prompt = mx.array([[1, 5, 9]])
    ref = _naive_greedy(model, prompt, 16)

    drafter = _SpyTreeDraft(ref, vocab=40)
    got, stats = mtp_generate(model, drafter, None, prompt, 16, K=3, ids_mode=True)

    assert got == ref
    assert stats["tree_rescues"] > 0
    # 至少一整步接受长度为 K:该步 accepted_in[1:] 必须等于 B 链真值(即 ref 的对应两个 token),
    # 且不得等于 chainA 的 wrong0。逐步用其起始 x 复核。
    checked_full_step = False
    pos = 0
    for step in drafter.synced:
        assert step[0] == ref[pos]                 # accepted_in 首元恒为该步起始 x
        acc_len = len(step)
        if acc_len == 3:
            wrong0 = (ref[pos + 1] + 1) % 40
            assert step[1] == ref[pos + 1]         # 必为 B 链真值(bug 下会是 wrong0)
            assert step[1] != wrong0
            checked_full_step = True
        pos += acc_len
    assert checked_full_step                        # 至少覆盖到一整步 K 长接受
```

- [ ] **Step 2: 跑测试确认它失败（暴露 bug）**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py::test_tree_top2_rescue_accepted_in_uses_chain_b -v`
Expected: FAIL —— `step[1]` 等于 `wrong0` 而非 `ref[pos+1]`（证明 `accepted_in` 取了 chainA 旧 `verify_in`）。

- [ ] **Step 3: 修 `generate.py` 救回分支**

在 `mlx_streaming/mtp/generate.py` 的最小树救回分支里，把内联的 B 链前向输入改为赋回 `verify_in`。找到：

```python
        if tree_b is not None and matched == 0 and tree_b[0] == preds[0]:
            _restore(main_cache, snap_m)
            begin_speculative_checkpoints(main_cache)
            vlogits, vH = forward_with_hidden(model, mx.array([[x] + tree_b[:K - 1]]), main_cache)
            mx.eval(vlogits, vH)
            preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]
            drafts = tree_b
            matched = accept_prefix(drafts, preds)
            tree_rescues += 1
```

改为（把 B 链输入落到 `verify_in`，使后续 `accepted_in`/`accepted_in = verify_in[:, :accepted_len]` 自动取 chainB）：

```python
        if tree_b is not None and matched == 0 and tree_b[0] == preds[0]:
            _restore(main_cache, snap_m)
            begin_speculative_checkpoints(main_cache)
            # 重建 verify_in 为 B 链:后续 accepted_in=verify_in[:, :accepted_len] 才会取到 chainB,
            # 否则残留 chainA 的错误首草稿会经 sync 污染 MTP cache、拉低后续草稿质量(低估救回收益)。
            verify_in = mx.array([[x] + tree_b[:K - 1]])
            vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
            mx.eval(vlogits, vH)
            preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]
            drafts = tree_b
            matched = accept_prefix(drafts, preds)
            tree_rescues += 1
```

- [ ] **Step 4: 跑测试确认通过（含 Task 1 不回归）**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py -k "tree_top2 or rescue" -v`
Expected: 两个测试全 PASS。

- [ ] **Step 5: 跑整份 MTP 测试防回归**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py -v`
Expected: 全 PASS。

- [ ] **Step 6: 提交**

```bash
git add mlx_streaming/mtp/generate.py mlx_streaming/tests/test_mtp_generate.py
git commit -m "fix(mtp): 最小树救回后 accepted_in 取 chainB(修 sync 喂错 token 污染 MTP cache)"
```

---

## Task 3: 抽出可测的纯裁决逻辑（中位数 + go/no-go）

**Files:**
- Create: `mlx_streaming/mtp/bench_verdict.py`
- Create: `mlx_streaming/tests/test_bench_verdict.py`

- [ ] **Step 1: 写失败测试**

新建 `mlx_streaming/tests/test_bench_verdict.py`：

```python
"""bench_verdict 纯裁决逻辑单测:中位数 + go/no-go 判定。"""
import pytest

from mlx_streaming.mtp.bench_verdict import median, verdict_from_delta


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_median_single():
    assert median([7.5]) == 7.5


def test_verdict_bug_when_not_exact():
    # exact_all=False 一票否决,无论提速多少都判 bug
    assert verdict_from_delta(0.5, exact_all=False) == "bug"


def test_verdict_go_above_margin():
    assert verdict_from_delta(0.06, exact_all=True, margin=0.05) == "go"


def test_verdict_even_within_margin():
    assert verdict_from_delta(0.02, exact_all=True, margin=0.05) == "even"
    assert verdict_from_delta(-0.02, exact_all=True, margin=0.05) == "even"


def test_verdict_nogo_below_margin():
    assert verdict_from_delta(-0.06, exact_all=True, margin=0.05) == "no-go"


def test_verdict_boundary_is_even():
    # 恰好等于 margin 不算 go(严格大于才 go)
    assert verdict_from_delta(0.05, exact_all=True, margin=0.05) == "even"
    assert verdict_from_delta(-0.05, exact_all=True, margin=0.05) == "even"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_bench_verdict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_streaming.mtp.bench_verdict'`

- [ ] **Step 3: 写实现**

新建 `mlx_streaming/mtp/bench_verdict.py`：

```python
"""最小树 A/B 评测的纯裁决逻辑:中位数 + go/no-go 判定(无副作用,可单测)。"""


def median(xs):
    """样本中位数(偶数个取中间两数均值)。xs 非空。"""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def verdict_from_delta(delta, exact_all, margin=0.05):
    """按跨 prompt 中位相对提速 delta 与 lossless 门给出裁决。

    - exact_all=False → "bug"(lossless 硬门一票否决)
    - delta >  margin → "go"
    - delta < -margin → "no-go"
    - 其余(含边界) → "even"
    """
    if not exact_all:
        return "bug"
    if delta > margin:
        return "go"
    if delta < -margin:
        return "no-go"
    return "even"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_bench_verdict.py -v`
Expected: 全 PASS。

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/mtp/bench_verdict.py mlx_streaming/tests/test_bench_verdict.py
git commit -m "feat(mtp): 最小树 A/B 纯裁决逻辑(中位数+go/no-go)+ 单测"
```

---

## Task 4: 升级 `bench_tree.py` 为稳态多 prompt A/B harness

**Files:**
- Modify（整体重写）: `benchmarks/bench_tree.py`

要点：模型只加载一次；先跑一次 warmup（两条路径都热）；固定 6 条多样中文 prompt；每 prompt × {tree-off, tree-on} 各重复 `REPEAT=3` 取 tok/s 中位数；tree-off 输出作 `ref`，tree-on 必须逐 token 相等（lossless 门）；用 `bench_verdict` 出跨 prompt 中位提速与裁决。

- [ ] **Step 1: 重写 harness**

把 `benchmarks/bench_tree.py` 整体替换为：

```python
"""最小树 top-2(TREE_TOP2)稳态多 prompt A/B:tree-off vs tree-on 的净 tok/s 与 lossless 裁决。

模型只加载一次;每 prompt 先以 tree-off 输出为 lossless 参考 ref,tree-on 必须逐 token 相等。
tok/s 每配置每 prompt 重复 REPEAT 次取中位数,抵抗 run-to-run 抖动;跑前 warmup 一遍热 kernel/池。
生产配置(env,与 run_mtp_spec 一致):
  STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 .venv/bin/python benchmarks/bench_tree.py
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core.mem import reset_peak
from mlx_streaming.mtp.bench_verdict import median, verdict_from_delta
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

K = int(os.environ.get("K", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "96"))
REPEAT = int(os.environ.get("REPEAT", "3"))
MARGIN = float(os.environ.get("MARGIN", "0.05"))

# 固定 6 条多样中文 prompt(取自 router_pred_prompts.txt:概念/代码/英文/叙事/JSON 等风格)。
PROMPTS = [
    "用三句话解释什么是混合专家模型。",
    "写一段 Python 代码，演示如何用 LRU 缓存函数结果。",
    "为什么模型量化会影响困惑度和生成质量？",
    "用英文写一段关于 speculative decoding 的技术摘要。",
    "请写一个短故事，主题是工程师在午夜调试模型推理性能。",
    "请给出一个使用 Python 解析 JSONL 文件并统计字段频率的例子。",
]


def _run_once(model, drafter, tok, enc, store):
    store.reset_stats()
    ids, stats = mtp_generate(model, drafter, tok, enc, MAXTOK, K=K,
                              ids_mode=True, profile=True)
    tps = round(stats["tokens"] / stats["wall_s"], 2)
    return ids, stats, tps


def _bench_prompt(model, drafter, tok, prompt, store):
    enc = mx.array([tok.encode(prompt)])
    # tree-off 参考(取第一次输出作 ref);tree-on 必须逐 token 等于它。
    os.environ["TREE_TOP2"] = "0"
    off_tps, ref = [], None
    for r in range(REPEAT):
        ids, _stats, tps = _run_once(model, drafter, tok, enc, store)
        if ref is None:
            ref = ids
        off_tps.append(tps)
    os.environ["TREE_TOP2"] = "1"
    on_tps, on_ids, on_rescues = [], None, 0
    for r in range(REPEAT):
        ids, stats, tps = _run_once(model, drafter, tok, enc, store)
        on_ids = ids
        on_rescues = stats["tree_rescues"]
        on_tps.append(tps)
    off_med, on_med = median(off_tps), median(on_tps)
    exact = on_ids == ref
    n_mismatch = sum(1 for a, b in zip(on_ids, ref) if a != b)
    delta = (on_med - off_med) / max(off_med, 1e-6)
    return {
        "prompt": prompt[:16],
        "off_tps_med": off_med,
        "on_tps_med": on_med,
        "off_tps_runs": off_tps,
        "on_tps_runs": on_tps,
        "delta_pct": round(delta * 100, 2),
        "tree_rescues": on_rescues,
        "exact_match": exact,
        "n_mismatch": n_mismatch,
    }


def main():
    reset_peak()
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    # warmup:两条路径各跑一遍第一个 prompt,热 Metal kernel + 专家常驻池 + 预取。
    print(f"[warmup] slots={store.capacity} K={K} maxtok={MAXTOK} repeat={REPEAT}", flush=True)
    _enc = mx.array([tok.encode(PROMPTS[0])])
    for _t in ("0", "1"):
        os.environ["TREE_TOP2"] = _t
        _run_once(model, drafter, tok, _enc, store)

    rows = []
    for p in PROMPTS:
        row = _bench_prompt(model, drafter, tok, p, store)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    deltas = [r["delta_pct"] / 100 for r in rows]
    agg_delta = median(deltas)
    exact_all = all(r["exact_match"] for r in rows)
    verdict = verdict_from_delta(agg_delta, exact_all, margin=MARGIN)
    summary = {
        "n_prompts": len(rows),
        "repeat": REPEAT,
        "maxtok": MAXTOK,
        "margin_pct": round(MARGIN * 100, 1),
        "median_delta_pct": round(agg_delta * 100, 2),
        "exact_all": exact_all,
        "total_mismatch": sum(r["n_mismatch"] for r in rows),
        "verdict": verdict,
    }
    print("=== SUMMARY (minimal-tree top-2 off vs on) ===")
    print(json.dumps({"rows": rows, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法/导入自检（不加载 80B，仅编译）**

Run: `.venv/bin/python -c "import ast; ast.parse(open('benchmarks/bench_tree.py').read()); print('OK')"`
Expected: 打印 `OK`（确保无语法错误；真实运行在 Task 5）。

- [ ] **Step 3: 提交**

```bash
git add benchmarks/bench_tree.py
git commit -m "bench(mtp): bench_tree 升级为稳态多 prompt A/B(warmup+repeat 中位+裁决)"
```

---

## Task 5: 跑真实 80B 评测并写报告

**Files:**
- Create: `benchmarks/reports/minimal-tree-top2-2026-07-05.md`

- [ ] **Step 1: 跑稳态 A/B（真实 80B，耗时约 10-15 分钟）**

Run:
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=96 REPEAT=3 \
  .venv/bin/python benchmarks/bench_tree.py 2>&1 | tee /tmp/bench_tree_out.txt
```
Expected: 逐 prompt JSON + 末尾 `=== SUMMARY ===`，含 `verdict` 字段。

- [ ] **Step 2: 核对 lossless 硬门**

从输出确认 `summary.exact_all == true`（`total_mismatch == 0`）。
- 若为 false：说明救回实现仍破坏 lossless（Task 1/2 应已挡住；若此处才暴露，回到 `generate.py` 用 systematic-debugging 定位，不得跳过）。此为 **bug 结论**，报告如实记录，暂不下 go/no-go。

- [ ] **Step 3: 写报告**

用真实输出的数字填入，新建 `benchmarks/reports/minimal-tree-top2-2026-07-05.md`：

```markdown
# 最小树 top-2 救回稳态 A/B 实测

模型 qwen3-next-80b-a3b-4bit,g64 专家。配置:STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1
EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=96 REPEAT=3。
6 条多样中文 prompt。ref=tree-off 输出,tree-on 必须逐 token 相等(lossless 门)。tok/s 取每配置每
prompt REPEAT 次中位数;裁决用跨 prompt 中位相对提速,阈值 ±5%。

## 逐 prompt 结果

| prompt | off tok/s(中位) | on tok/s(中位) | Δ% | tree_rescues | exact_match | n_mismatch |
|---|---|---|---|---|---|---|
| <填> | <填> | <填> | <填> | <填> | <填> | <填> |
（每条 prompt 一行,从 /tmp/bench_tree_out.txt 的逐行 JSON 抄入）

## 汇总

- 跨 prompt 中位 Δ% = <填 summary.median_delta_pct>
- exact_all = <填>，total_mismatch = <填>
- **裁决 = <填 summary.verdict>**

## 结论

<按 verdict 三选一写明:>
- go(Δ>+5% 且全 lossless):最小树 top-2 串行救回净赚,建议并入生产路径并进入方案 A 叠加;
  可选继续 Task 6(top-3 扩展)。
- even(-5%~+5%):串行救回收益被额外前向 IO 抵消,持平。归档负结论,转方案 A(置信度门控)。
- no-go(Δ<-5%):救回代价大于收益,归档。转方案 A。
- bug(任一 prompt 失配):实现破坏 lossless,记录复现 env 与失配位置,回到 generate.py 修复后重测。

## 备注

- 与并集放大的 tree_verify(见 tree-verify-2026-07-01.md)对照:本方案每次前向恒 1×K,不溢出 cap;
  若 tok/s 仍未净赚,原因应为"救回步的额外一次前向 IO"而非"cap 溢出",disk_loads 增量可佐证。
```

- [ ] **Step 4: 提交**

```bash
git add benchmarks/reports/minimal-tree-top2-2026-07-05.md
git commit -m "docs(bench): 最小树 top-2 救回稳态 A/B 实测 + go/no-go 结论"
```

---

## Task 6（条件执行）: top-3 位置 1 救回扩展

**仅当 Task 5 裁决为 `go` 或 `even`（且 Δ 非明显为负）时执行；若 `no-go` 或 `bug` 则跳过本任务，plan 到此结束。**

原理：位置 1 从 top-2 扩到 top-3（探针 pos1 top-3 覆盖 87.7% vs top-2 81.5%，多约 6pp 救回空间）。仍是串行、`1×K` 前向、并集不放大：首草稿被拒时最多再串行验一次 top-2 候选链、再一次 top-3 候选链，取首个命中 `preds[0]` 的候选救回。

**Files:**
- Modify: `mlx_streaming/mtp/drafter.py`（新增 `draft_tree3` 返回 3 条链）
- Modify: `mlx_streaming/mtp/generate.py`（救回循环支持多候选链）
- Modify: `mlx_streaming/config.py`（新增 `tree_top3()` 开关）
- Modify: `mlx_streaming/tests/test_mtp_generate.py`（top-3 lossless + 救回覆盖玩具单测）
- Modify: `benchmarks/bench_tree.py`（加一档 `tree-top3` 配置对比）

- [ ] **Step 1: 写失败测试（top-3 lossless + 救回）**

在 `mlx_streaming/tests/test_mtp_generate.py` 末尾追加（复用 `_OracleTreeDraft` 思路，扩到 3 链）：

```python
class _OracleTree3Draft(_OracleTreeDraft):
    """扩展 oracle:draft_tree3 返回 3 条链,chainC 首=真实 preds[0](chainA/B 首均错),
    覆盖"前两候选均未命中 preds[0]、第三候选救回"的最深路径。"""

    def draft_tree3(self, H_last, x_ids, mtp_cache, K):
        nxt = self.ref[self.pos + 1: self.pos + 1 + K]
        while len(nxt) < K:
            nxt.append(0)
        wrong_a = (nxt[0] + 1) % self.vocab
        wrong_b = (nxt[0] + 2) % self.vocab
        chain_a = [wrong_a] + nxt[1:]     # 首错
        chain_b = [wrong_b] + nxt[1:]     # 首错(与 A 不同)
        chain_c = list(nxt)               # 首=真实 → 第三候选救回
        return chain_a, chain_b, chain_c


def test_mtp_generate_tree_top3_lossless_and_rescues(monkeypatch):
    from mlx_streaming.mtp.generate import mtp_generate

    monkeypatch.setenv("TREE_TOP3", "1")
    monkeypatch.delenv("TREE_TOP2", raising=False)
    monkeypatch.delenv("MTP_VERIFY_MODE", raising=False)
    monkeypatch.delenv("TREE_VERIFY", raising=False)
    mx.random.seed(0)
    model = _ToyModel(nl=2, vocab=40)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    prompt = mx.array([[1, 5, 9]])
    ref = _naive_greedy(model, prompt, 16)

    drafter = _OracleTree3Draft(ref, vocab=40)
    got, stats = mtp_generate(model, drafter, None, prompt, 16, K=3, ids_mode=True)
    assert got == ref
    assert stats["tree_rescues"] > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py::test_mtp_generate_tree_top3_lossless_and_rescues -v`
Expected: FAIL（`TREE_TOP3` 开关与 `draft_tree3` 尚未实现）。

- [ ] **Step 3: 加 config 开关**

在 `mlx_streaming/config.py` 的 MTP 区（`tree_top2` 附近）新增：

```python
def tree_top3() -> bool: return _b("TREE_TOP3", "0")  # 位置1 top-3 串行救回(仍 1×K 前向,并集不放大)
```

- [ ] **Step 4: 加 `draft_tree3`**

在 `mlx_streaming/mtp/drafter.py` 的 `draft_tree` 之后新增（结构与 `draft_paths` 一致,但只在位置 1 展开 top-3,各链从共享 mh1 分叉、快照隔离）：

```python
    def draft_tree3(self, H_last, x_ids, mtp_cache, K):
        """位置1 展开 top-3,返回三条链(各长 K),供串行救回。与 draft_tree 同构,P=3。"""
        logits1, mh1 = mtp_step(self.mtp, H_last, x_ids, self.lm_head, mtp_cache[0])
        lg = logits1[0].reshape(-1)
        top3 = [int(i) for i in mx.argsort(lg)[-3:].tolist()][::-1]   # [d1a, d1b, d1c] 降序
        snap_pos1 = _snapshot(mtp_cache)

        def _continue(first):
            chain = [first]
            h, cur = mh1, mx.array([[first]])
            for _ in range(K - 1):
                lo, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
                d = int(mx.argmax(lo[0]))
                chain.append(d)
                h, cur = mh, mx.array([[d]])
            return chain

        chains = []
        for j, f in enumerate(top3):
            if j > 0:
                _restore(mtp_cache, snap_pos1)
            chains.append(_continue(f))
        return chains[0], chains[1], chains[2]
```

- [ ] **Step 5: 在 `generate.py` 接入 top-3 串行救回**

在 `mlx_streaming/mtp/generate.py` 循环外读开关（`tree_mode` 附近）：

```python
    tree3_mode = config.tree_top3()
```

在最小树抽草稿分支（`if tree_mode and verify_mode != "step":` 之前）加 top-3 抽取分支：

```python
        tree_cands = None
        if tree3_mode and verify_mode != "step":
            ca, cb, cc = drafter.draft_tree3(H_last, x_ids, mtp_cache, K)
            drafts, tree_cands = ca, [cb, cc]   # 主链 A + 两条备选
            draft_cands = None
        elif tree_mode and verify_mode != "step":
            drafts, tree_b = drafter.draft_tree(H_last, x_ids, mtp_cache, K)
            draft_cands = None
        elif topk_probe > 0:
            drafts, draft_cands = drafter.draft(H_last, x_ids, mtp_cache, K, topk=topk_probe)
        else:
            drafts, draft_cands = drafter.draft(H_last, x_ids, mtp_cache, K), None
```

把原最小树救回块（`if tree_b is not None and matched == 0 and tree_b[0] == preds[0]:`）替换为**统一的多候选串行救回循环**（top-2 是 `tree_cands=[tree_b]` 的特例）：

```python
        # 统一串行救回:首草稿被拒(matched==0)时,依次试每条备选链,首个"首候选==preds[0]"者救回。
        # 每次仍是 1×K 前向、并集不放大。tree_top2 → rescue_cands=[tree_b];tree_top3 → [cb, cc]。
        rescue_cands = None
        if tree_cands is not None:
            rescue_cands = tree_cands
        elif tree_b is not None:
            rescue_cands = [tree_b]
        if rescue_cands is not None and matched == 0:
            for cand in rescue_cands:
                if cand[0] != preds[0]:
                    continue
                _restore(main_cache, snap_m)
                begin_speculative_checkpoints(main_cache)
                verify_in = mx.array([[x] + cand[:K - 1]])
                vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
                mx.eval(vlogits, vH)
                preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]
                drafts = cand
                matched = accept_prefix(drafts, preds)
                tree_rescues += 1
                break
```

同时把循环外 `tree_b` 初始化补上（避免未定义）：在 `tree_cands = None` 之后确保 `tree_b = None` 每步复位——即在抽草稿分支前加 `tree_b = None`（若原代码已有则保留）。

- [ ] **Step 6: 跑玩具测试确认通过 + 防回归**

Run: `.venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py -v`
Expected: 全 PASS（含 top-2、top-3、accepted_in、原有等价性测试）。

- [ ] **Step 7: 提交实现**

```bash
git add mlx_streaming/config.py mlx_streaming/mtp/drafter.py mlx_streaming/mtp/generate.py mlx_streaming/tests/test_mtp_generate.py
git commit -m "feat(mtp): 位置1 top-3 串行救回(统一多候选救回循环,并集不放大)"
```

- [ ] **Step 8: harness 加 top-3 档并重测**

在 `benchmarks/bench_tree.py` 的 `_bench_prompt` 里，tree-on 之后再加一档 top-3（`os.environ["TREE_TOP2"]="0"; os.environ["TREE_TOP3"]="1"` 跑 REPEAT 次取中位），并在返回 dict 增加 `on3_tps_med`/`delta3_pct`/`exact3`/`n_mismatch3`；`main` 的 summary 增加 top-3 的 `median_delta3_pct` 与 `verdict3`（复用 `verdict_from_delta`）。每档结束务必把两个开关都置 0 复位，避免串档。

Run:
```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 MAXTOK=96 REPEAT=3 \
  .venv/bin/python benchmarks/bench_tree.py 2>&1 | tee /tmp/bench_tree3_out.txt
```
Expected: 输出含 top-2 与 top-3 两档的中位 Δ% 与各自 verdict。

- [ ] **Step 9: 报告追加 top-3 小节并提交**

在 `benchmarks/reports/minimal-tree-top2-2026-07-05.md` 追加"## top-3 扩展"小节（同样的表 + 裁决），据 top-2/top-3 Δ% 对比给出最终推荐档位。

```bash
git add benchmarks/bench_tree.py benchmarks/reports/minimal-tree-top2-2026-07-05.md
git commit -m "bench(mtp): 最小树 top-3 档 A/B + 报告追加 top-2/top-3 对比"
```

---

## 自检（写完计划后对照 spec）

- **spec 覆盖**：
  - lossless 红线 → Task 1（玩具锁死）+ Task 5 Step 2（真实硬门）✓
  - 不放大并集（串行 1×K）→ Task 2 修复保持 1×K；Task 6 top-3 同为串行 ✓
  - 稳态多 prompt/warmup/repeat 中位 → Task 4 ✓
  - go/no-go 判据（±5%）→ Task 3（纯逻辑）+ Task 5（应用）✓
  - 报告归档（正/负都留）→ Task 5 Step 3 ✓
  - 可选 top-3（仅 go/even 才做）→ Task 6 条件门 ✓
  - 范围外（不碰 tree_verify/不改 cap）→ 计划未触及 ✓
- **额外必要项（spec 未显式但实现必需）**：`generate.py` 救回 `accepted_in` bug → Task 2（否则评测的是被污染 MTP cache 的退化版本，结论失真）✓
- **占位符扫描**：报告模板里的 `<填>` 是 Task 5 运行后按实测填入的数据位，非计划占位；其余步骤均含完整代码/命令。
- **类型/命名一致**：`median`/`verdict_from_delta` 在 Task 3 定义、Task 4/6 使用签名一致；`_OracleTreeDraft`(Task1)→`_SpyTreeDraft`(Task2)→`_OracleTree3Draft`(Task6)继承链一致；`tree_top2`(既有)/`tree_top3`(Task6 新增)命名对齐既有 config 风格。
