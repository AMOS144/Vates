# 跨-token 大窗口预取（测量门控）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先用离线 probe 判定「上一/前几次同层路由」这一大窗口信号对当前 token 各层 **miss 子集** 的 recall（命门指标），过关才落运行时预取。

**Architecture:** Milestone 1 纯离线分析现有 `route_trace` 事件流（逐层已含 `experts`/`miss`），零热路径改动；新增一个纯函数 `crosstoken_recall` 做聚合，一个 probe CLI 跑真实 decode 并输出 `recall_full`/`recall_miss`。Milestone 2（运行时）显式延后到 M1 数据出来后再写计划，因为其参数（用哪个 history_n、AHEAD、cap）取决于 M1 结论。

**Tech Stack:** Python、MLX、现有 `route_trace` / `build_streaming_model` / `mtp_generate` / `FileExpertStore`。

---

## 背景（执行者必读）

- 物化一个专家 ≈ 85µs；同层（AHEAD=0）窗口仅 70µs，物理上盖不住 → 已验证 `ready_on_time≈0`。
- 中池（cap≈64–96）每 token 仅 ~19 个 miss，物化总量 ~1.6ms ≪ 算力（~55ms/token），**只要窗口够大就能藏**。
- **命门**：中池 + LRU 稳态下，miss = 「新需要、最近没用过」的专家，与历史信号弱相关。需要预取的 miss 很可能 ⊥ 大窗口能预测的专家。本计划 Milestone 1 就是低成本证伪/证实这一点。
- `route_trace`（`mlx_streaming/core/route_trace.py`）在 MoE forward（`mlx_streaming/core/streaming_moe.py` 中 `ROUTE_TRACE=1` 分支）已逐层记录 `{"layer","experts"(routed),"miss"(routed−resident),"resident"}`，事件按执行顺序追加。Milestone 1 只需离线分析这串事件。

---

## Milestone 1：判生死（离线测量）

### Task 1：跨-occurrence recall 纯函数

**Files:**
- Create: `mlx_streaming/core/crosstoken_recall.py`
- Test: `mlx_streaming/tests/test_crosstoken_recall.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_crosstoken_recall.py
from mlx_streaming.core.crosstoken_recall import crosstoken_recall


def test_crosstoken_recall_n1_scores_from_previous_occurrence():
    # 同一层 L=0 的三次出现（执行顺序）。首次无历史→不计分。
    events = [
        {"layer": 0, "experts": [1, 2, 3], "miss": [1, 2, 3]},
        {"layer": 0, "experts": [2, 3, 4], "miss": [4]},   # pred(前1次)={1,2,3}
        {"layer": 0, "experts": [3, 4, 5], "miss": [5]},   # pred(前1次)={2,3,4}
    ]
    r = crosstoken_recall(events, history_n=1)
    assert r["n_scored"] == 2
    # full = (|{1,2,3}∩{2,3,4}|=2 + |{2,3,4}∩{3,4,5}|=2)/(3+3) = 4/6
    assert r["recall_full"] == 0.6667
    # miss = ({4}覆盖0 + {5}覆盖0)/2 = 0 —— 历史覆盖不到新颖 miss
    assert r["recall_miss"] == 0.0
    assert r["tot_miss"] == 2


def test_crosstoken_recall_n2_unions_two_previous_occurrences():
    events = [
        {"layer": 0, "experts": [1], "miss": []},
        {"layer": 0, "experts": [2], "miss": []},
        {"layer": 0, "experts": [1, 2], "miss": [1, 2]},  # pred=union({1},{2})={1,2}
    ]
    r = crosstoken_recall(events, history_n=2)
    assert r["n_scored"] == 2
    # 复发 miss 被前两次并集覆盖：{1,2}∩{1,2} = 2/2
    assert r["recall_miss"] == 1.0
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest mlx_streaming/tests/test_crosstoken_recall.py -p no:cacheprovider`
Expected: FAIL（`ModuleNotFoundError: mlx_streaming.core.crosstoken_recall`）

- [ ] **Step 3: 实现**

```python
# mlx_streaming/core/crosstoken_recall.py
"""离线分析：从 route_trace 事件流计算「跨-occurrence 同层」大窗口信号对 routed/miss 的 recall。

事件流按执行顺序排列。对层 L 的第 i 次出现，用其前 history_n 次出现的 routed 并集作为预测集
pred（= 上一个/前几个 token 同层路由，窗口可达整 token，最大）。命门指标是 recall_miss：
pred 能否覆盖「本次 routed 但当前不常驻」的 miss——即真正需要预取的那部分专家。
"""
from collections import defaultdict, deque


def crosstoken_recall(events, history_n: int = 1) -> dict:
    hist = defaultdict(lambda: deque(maxlen=history_n))
    hit_full = tot_routed = 0
    hit_miss = tot_miss = 0
    pred_size_sum = miss_sum = 0
    n_scored = 0
    for ev in events:
        layer = int(ev["layer"])
        routed = {int(e) for e in ev.get("experts", [])}
        miss = {int(e) for e in ev.get("miss", [])}
        h = hist[layer]
        if h:  # 有历史才计分（首次出现无预测来源）
            pred = set().union(*h)
            hit_full += len(pred & routed)
            tot_routed += len(routed)
            hit_miss += len(pred & miss)
            tot_miss += len(miss)
            pred_size_sum += len(pred)
            miss_sum += len(miss)
            n_scored += 1
        h.append(routed)
    return {
        "history_n": history_n,
        "n_scored": n_scored,
        "recall_full": round(hit_full / max(1, tot_routed), 4),
        "recall_miss": round(hit_miss / max(1, tot_miss), 4),
        "tot_miss": tot_miss,
        "avg_pred_size": round(pred_size_sum / max(1, n_scored), 2),
        "avg_miss": round(miss_sum / max(1, n_scored), 2),
    }
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest mlx_streaming/tests/test_crosstoken_recall.py -p no:cacheprovider`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/core/crosstoken_recall.py mlx_streaming/tests/test_crosstoken_recall.py
git commit -m "feat: crosstoken_recall offline analysis for miss-prefetch gating"
```

---

### Task 2：probe CLI（真实 decode + 离线分析）

**Files:**
- Create: `mlx_streaming/cli/probe_crosstoken_miss_recall.py`

- [ ] **Step 1: 写 probe**（无新单测——纯离线测量脚本；分析逻辑已被 Task 1 覆盖）

```python
# mlx_streaming/cli/probe_crosstoken_miss_recall.py
"""Milestone 1：测「上一/前几次同层路由」对当前 token 各层 miss 的 recall（命门 recall_miss）。

中池由 EXPERT_SLOTS 控制（跑 64 / 96 两档）。high→大窗口预取能藏住中池 miss；
low→miss ⊥ 历史信号（墙坐实），转 Plan B。
history_n>1 的并集天然涵盖 MTP 草稿/验证多次 occurrence。
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core import route_trace
from mlx_streaming.core.crosstoken_recall import crosstoken_recall
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.mtp_generate import MTPDrafter, mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "64"))
K = int(os.environ.get("K", "2"))
HISTORY_NS = [int(x) for x in os.environ.get("HISTORY_NS", "1,2,3").split(",")]


def main():
    os.environ["ROUTE_TRACE"] = "1"
    os.environ.setdefault("RESIDENT_POOL", "1")
    os.environ.setdefault("MTP_VERIFY_MODE", "batch")
    os.environ.setdefault("MTP_ARRAY_COMMIT", "1")
    model, tok, store = build_streaming_model()
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    route_trace.enable()
    ids, mtp_stats = mtp_generate(
        model, drafter, tok, mx.array([tok.encode(PROMPT)]),
        MAXTOK, K=K, ids_mode=True, profile=False)
    events = route_trace.events()
    route_trace.disable()

    rows = [crosstoken_recall(events, history_n=n) for n in HISTORY_NS]
    print(json.dumps({
        "expert_slots": os.environ.get("EXPERT_SLOTS"),
        "K": K,
        "tokens": len(ids),
        "n_events": len(events),
        "store_hits": store.hits,
        "store_misses": store.misses,
        "hit_rate": round(store.hits / max(1, store.hits + store.misses), 3),
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟（小 MAXTOK，确认能跑通且 events 有 miss 字段）**

Run:
```bash
MODEL=/tmp/qwen3_next_80b_4bit EXPERT_DIR=$PWD/models/qwen3_next_experts_2bit_g128 \
EXPERT_SLOTS=64 MAXTOK=8 K=2 \
uv run python -m mlx_streaming.tools.probe_crosstoken_miss_recall
```
Expected: 打出 JSON，`n_events>0`，每个 row 含 `recall_full`/`recall_miss`/`tot_miss>0`。
（若 `tot_miss==0`：说明 cap 太大全命中，调小 `EXPERT_SLOTS` 重试。）

- [ ] **Step 3: 提交**

```bash
git add mlx_streaming/cli/probe_crosstoken_miss_recall.py
git commit -m "feat: probe_crosstoken_miss_recall (Milestone 1 gating probe)"
```

---

### Task 3：跑实测 + 写判定

**Files:**
- Append: `benchmarks/reports/low-memory-streaming-moe-2026-06-11.md`

- [ ] **Step 1: 正式跑两档中池**（MAXTOK=64）

Run（cap=64）：
```bash
MODEL=/tmp/qwen3_next_80b_4bit EXPERT_DIR=$PWD/models/qwen3_next_experts_2bit_g128 \
EXPERT_SLOTS=64 MAXTOK=64 K=2 \
uv run python -m mlx_streaming.tools.probe_crosstoken_miss_recall
```
Run（cap=96）：把上面 `EXPERT_SLOTS=64` 改成 `EXPERT_SLOTS=96` 再跑一次。

- [ ] **Step 2: 判定（用数据回答命门）**

判据（对每档 cap、每个 history_n 看 `recall_miss`）：
- `recall_miss ≥ 0.8` 且 `avg_pred_size` 合理（不爆）→ **过关**：大窗口信号能覆盖中池 miss，进入 Milestone 2（届时按"哪个 history_n / cap 最优"写运行时计划）。
- `recall_miss` 明显低（预期结果）→ **命门坐实**：miss ⊥ 历史信号。不再造预测器，转 Plan B（见 spec），把量级事实写进报告。

- [ ] **Step 3: 写报告**（把两档 cap × history_n 的 `recall_full` vs `recall_miss` 表、`avg_pred_size`、`tot_miss`、结论追加到报告；明确"过关/不过关"与下一步）

- [ ] **Step 4: 提交**

```bash
git add benchmarks/reports/low-memory-streaming-moe-2026-06-11.md
git commit -m "docs: crosstoken miss-recall gating results + verdict"
```

---

## 决策门（Milestone 1 之后）

**只有 Task 3 判定"过关"（`recall_miss ≥ 0.8`）才进入 Milestone 2。**

Milestone 2（运行时跨-token 预取）**有意延后到 M1 数据出来后再用 writing-plans 单独成计划**，因为其关键参数取决于 M1 结论：
- 用哪个 `history_n`（前 1 次 vs 前 N 次并集）。
- `AHEAD`（提前几层提交）——跨-token 信号来自上一 token，**精度与 AHEAD 无关**，故可取大 AHEAD（如 8–16 层）拿大窗口而不掉 recall。
- 在哪档 cap 上做。

Milestone 2 届时复用既有、已测组件：`BackgroundExpertPrefetcher` / `promote_prefetched`（永不驱逐 current）/ `_submit_missing_prefetch` / `bg_stats.ready_on_time` / `WINDOW_PROF`；新增的仅是「记录上一 occurrence 同层 routed」+「在 decoder 层 hook 用 `last_routing[L+AHEAD]` 提交预取」。等价性 `exact_match` 与中池 plain 一致 + 端到端 A/C 净收益判定。

若"不过关"：执行 spec 的 Plan B 方向（减少 miss 体量 / 主模型基座量化），不做 Milestone 2。

---

## 自检

- **Spec 覆盖**：M1（命门 `recall_miss` 测量）= Task 1–3，完整；M2 按 spec「过关才做」显式延后并说明理由（参数依赖 M1 数据，非占位符）；Plan B 在决策门引用 spec。
- **占位符**：无 TBD/TODO；所有代码步给出完整代码与可跑命令。
- **类型一致**：`crosstoken_recall(events, history_n)` 在 Task 1 定义、Task 2 调用，签名/返回键（`recall_miss`/`recall_full`/`tot_miss`/`avg_pred_size`/`n_scored`/`history_n`）一致；事件键 `layer`/`experts`/`miss` 与 `route_trace.record` 写入一致。
