# 同层预测 + attention 窗口预取（Same-Layer Predict-Prefetch）实现计划

> **For agentic workers:** 用 superpowers:subagent-driven-development 或 executing-plans 逐任务执行；每步 checkbox。**本计划纠正了之前用错探针(AHEAD=1 / 弱 proxy)的方向。**

**Goal:** 在第 L 层 attention/GDN 之前，用 L 的输入 hidden 经 `post_attention_layernorm + gate` 预测 L 自己的专家（已实测 recall 0.984@x2），把"预测到但不在常驻池"的少数专家在后台 stream 上物化、藏进 attention/GDN 计算窗口；到 L 的 MoE 时 promote 进池 → 命中。目标：中等池(低于全常驻)也能接近全常驻吞吐。

**已验证前提（probe_cross_layer_predict，2bit-g128，K=3/MAXTOK=48）：**

| 预测量 | recall | 预测数 | 实际 | 漏≥1 专家的层 |
|---|---|---|---|---|
| post_norm_x1 | 0.897 | 21.6 | 21.05 | 84% |
| post_norm_x2 | 0.984 | 40.6 | 21.05 | 13% |
| post_norm_x4 | 0.994 | 75.8 | 21.05 | 4.5% |

后台并发物化(独立 stream)已由 gating 测试 A/C 验证可行、可重叠、不崩。

**Architecture:**
- 复用 `enable_cross_layer_prefetch` 的 **AHEAD=0**（同层）分支：hook 在 decoder 层 `__call__` 开头(attention 前)用 `post_attention_layernorm(x)` 预测**当前层**专家。
- 复用 `BackgroundExpertPrefetcher`(s2 物化私有 array) + `FileExpertStore.promote_prefetched`(主线程写池槽)。
- **关键纠正(相对上一版 bg-fill)**：① AHEAD=0 而非 1；② 只预取"预测 ∩ 非常驻"(去重，避免重载常驻、避免超出 attention 窗口)。

**Tech Stack:** Python、MLX(`mx.new_stream`/`mx.stream`)、threading、现有 blob_loader / bg_prefetch / expert_store / streaming_moe。

---

## 关键约束与纠正
1. **AHEAD=0**：预测并预取**当前层**(窗口=该层 attention/GDN，~1ms 量级，比一整层短)。
2. **只预取缺的**：submit 前用 `store.resident_experts(layer)` 过滤，只取 `predicted − resident`。否则会 load 几十个已常驻专家、撑爆 attention 窗口。
3. **promote 语义**：ahead=0 预取的是**当前层要用的**专家(非投机)，所以 promote 必须能把它们放进池(会被 acquire 立即用)。需保证 acquire/promote **绝不驱逐当前请求专家**(之前 KeyError 的根因)——通过"中等池容量 + `_choose_victim` 永不驱逐 current"双保险。
4. **优雅回退**：未及时就绪/异常 → 现有 demand 路径，不崩不阻塞。

---

## Task 1: hook AHEAD=0 + 只预取缺失专家

**Files:**
- Modify: `mlx_streaming/core/streaming_moe.py`(cross-layer hook 的 STREAM_BLOB_BG 分支)
- Test: `mlx_streaming/tests/test_bg_prefetch.py`(加 submit-dedup 用例)

- [ ] **Step 1: 写失败测试**：构造 store(含 `_bg` + resident 池预置专家 {1,2,3}),调用一个辅助 `submit_missing(store, layer, predicted=[1,2,3,7,8])`，断言 bg 只收到 `{7,8}`（已常驻的 1,2,3 被过滤）。

- [ ] **Step 2: 跑红**

- [ ] **Step 3: 实现**：在 hook 的 `STREAM_BLOB_BG` 分支(已存在),改为：
```python
bg = getattr(target_mlp.store, "_bg", None)
if os.environ.get("STREAM_BLOB_BG", "0") == "1" and bg is not None:
    budget = int(os.environ.get("STREAM_BLOB_BG_BUDGET", str(target_mlp.top_k * 2)))
    picked = [e for e, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:budget]
    resident = target_mlp.store.resident_experts(target_mlp.layer_idx)
    missing = [e for e in picked if e not in resident]   # 只预取缺的
    if missing:
        bg.submit(target_mlp.layer_idx, missing)
    return orig_call(self, x, mask=mask, cache=cache)
```
并确保 `model_builder` 默认 `CROSS_LAYER_PREFETCH_AHEAD=0`（同层）当 STREAM_BLOB_BG=1。

- [ ] **Step 4: 跑绿 + 提交** `feat: same-layer ahead=0 prefetch only missing experts`

---

## Task 2: promote 放当前层专家 + 永不驱逐 current

**Files:**
- Modify: `mlx_streaming/core/expert_store.py`(`_choose_victim` 永不驱逐 current；`promote_prefetched` 允许放当前层专家)
- Test: `mlx_streaming/tests/test_bg_prefetch.py`

- [ ] **Step 1: 写失败测试**：cap=4 的池，先 promote {10,11,12,13} 填满，再 `acquire(flat=[10,11,90,91])`（uniq=4≤cap）。断言不抛 KeyError、且 slots 全部存在（验证 acquire 不会驱逐 current）。

- [ ] **Step 2: 跑红**（当前 `_choose_victim` 二次兜底会驱逐 current → KeyError）

- [ ] **Step 3: 实现**：
  - `_choose_victim`：去掉"二次兜底驱逐 current"——若无非 current 可驱逐，**抛清晰错误**(调用方 acquire 已有 `len(uniq)>cap` 守卫，正常不会触发)。
  - `promote_prefetched`：去掉"只填空闲"限制（那是给投机 ahead=1 的）；ahead=0 预取的是当前层专家，应允许 `_place_expert`（current 保护由 Task 2 的 `_choose_victim` 保证不误伤）。但仍跳过已在池中的。

- [ ] **Step 4: 跑绿 + 回归** `uv run pytest mlx_streaming/tests/test_bg_prefetch.py mlx_streaming/tests/test_resident_pool.py`

- [ ] **Step 5: 提交** `fix: never evict current-request experts; promote current-layer prefetch`

---

## Task 3: 就绪率/命中转化 instrumentation

**Files:**
- Modify: `mlx_streaming/core/bg_prefetch.py`(已有 stats)、`mlx_streaming/cli/run_mtp_spec.py`(输出 bg stats)

- [ ] **Step 1**：`BackgroundExpertPrefetcher.stats()` 增加 `ready_on_time`(promote 时已就绪数) vs `not_ready`(promote 时尚未物化完→本层仍 demand)。`promote_prefetched` 累计这两个计数。

- [ ] **Step 2**：`run_mtp_spec` 输出 `bg_stats`(submitted/materialized/taken/ready_on_time/not_ready)。

- [ ] **Step 3: 提交** `feat: instrument bg prefetch ready-on-time rate`

---

## Task 4: 端到端 A/B/C + 窗口够不够的判定

**Files:**
- Create: `mlx_streaming/cli/probe_samelayer_prefetch.py`
- Append: `benchmarks/reports/low-memory-streaming-moe-2026-06-11.md`

- [ ] **Step 1: probe**：固定 EXPERT_DIR=2bit-g128、K=2、MAXTOK=64，对每个 cap ∈ {32,64,96}：
  - A：plain（无 bg、无 cross-layer）。
  - C：`STREAM_BLOB_BG=1 CROSS_LAYER_PREFETCH=1 CROSS_LAYER_PREFETCH_AHEAD=0 STREAM_BLOB_BG_BUDGET=2×top_k`。
  记录 `spec_tok_per_s / spec_hit_rate / mlx_peak_gb / exact_match / bg_stats`。

- [ ] **Step 2: 跑**（每次前 `sudo purge`）。

- [ ] **Step 3: 判定**（用数据回答两个核心问题）：
  - **窗口够不够**：`ready_on_time / (ready_on_time + not_ready)` 高 → attention 窗口足够藏住缺失专家的物化；低 → 窗口太短(预取没赶上 MoE)。
  - **净收益**：C 的 tok/s 是否 > A（同 cap）；hit 是否提升；内存是否仍是该 cap 水平。

- [ ] **Step 4: 写报告**：结论 = 同层预取在哪个 cap 下净正收益、就绪率多少、与全常驻的差距。若就绪率低(窗口不够) → 记录"attention 窗口 < 缺失专家物化时间"这个量级事实。

---

## 自检
- **方向纠正**：本计划用 AHEAD=0(同层，recall 0.984)，不是之前失败的 AHEAD=1(0.83)；预取只取"预测∩非常驻"，不是全预测集。
- **正确性**：Task 2 显式测 no-KeyError + current 不被驱逐；端到端 `exact_match` 与常驻一致。
- **关键风险显式测量**：attention 窗口够不够(ready_on_time 率)、净收益、内存——Task 4 全部实测，不靠假设。
- **类型一致**：`bg.submit` 收 list[int]；`promote_prefetched` 用 `take_ready_layer` 返回的 {e: dict(bf16)}；键名与 `_place_expert` 一致(已在 Task1/2 of 上一计划核对)。
- **不变量**：后台只物化私有 array(规则1)、不写共享池(规则2)、主线程只消费已 eval 交接(规则3)、acquire/promote 不驱逐 current(Task 2)。

## 诚实的预期边界
- 预测够准(0.98@x2)→ 理论上同层预取能把缺失专家的物化藏进 attention 窗口。
- **但窗口很短(~1ms 量级)**：能否藏住取决于"缺失专家数 × 物化时间"vs attention 时间。中等池(缺失少)才有戏；小池(缺失多)窗口可能不够 → Task 4 的 ready_on_time 率会给出判决。
- 即便成立，上限是"接近全常驻吞吐 @ 更低内存",不是超过全常驻。
