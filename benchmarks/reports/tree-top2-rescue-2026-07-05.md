# 最小树 top-2 救回评测报告（2026-07-05，含根因修复）

## 结论一句话

先前"NO-GO / 后端天然数值敏感性"的结论是错的。真根因是 **`demand_dual` 的一个内存布局 bug**：
路由张量 `inds`（`argpartition(...)[..., -k:]` 的切片）是**非连续 strided 视图**，而 `demand_dual`
在 C++ 里按连续内存读 `ids.data<uint32_t>()`，导致 `seq≥2` 时**第 0 个 token 之后的所有 token 读到
错位内存 → 装错专家**。修法：`demand_dual` 入口 `mx::contiguous(inds)` 物化为连续后再读。修后：

- **`seq=K` verify 与 `seq=1` 解码逐位等价**（`_diag_seqk`：pos0/pos1 `max|Δlogit|=0.0000`）。
- **plain spec 与最小树救回均 bit-lossless**（`bench_tree.py`：全 6 prompt `control_mm=0, on_mm=0`）。
- 最小树 top-2 救回裁决从 NO-GO 翻转为 **GO，median +10.8% tok/s**。

> 这印证了"之前我们好像做过这个（lossless）"——它本就该 lossless，是 `demand_dual` 引入的回归。
> 2026-07-04 schemeB 报告把此分歧解释为"良性 N_floor 精确平局"，本次被推翻：`max|Δlogit|` 达数个
> 单位，是"读错专家权重"级别的真错，不是 0.1 级平局。

## 根因：非连续视图被按连续读

`block.py` 里 `inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]`。这个切片在 MLX 里是对完整
`(1, seq, E)` 结果的 **strided 视图**（每行按父数组 stride `E` 偏移、取末 `k` 个）。`demand_dual` 的
`ids.data<uint32_t>()` 假设内存连续、按 `ip[i]=buffer[i]` 顺序读：

- token0（`i=0..k-1`）：偏移恰好落在第 0 行末 `k` 个 → **正确**。
- token1+（`i=k..`）：应读第 1 行末 `k` 个（偏移 `E+(E-k)`），却读成 `(E-k)+k`=连续下一段 → **错位**。

决定性对拍（`_diag_seqk` + 逐位 dump，`seq=2` 第一次 demand 调用）：

| 来源 | token0 | token1 |
|---|---|---|
| Python `inds`（真实路由，scores/gather 用） | `485,474,400,438,433,412,292,21,346,216` | `395,137,231,275,368,228,287,348,391,91` |
| C++ `ip`（`demand_dual` 实际装池用） | `485,474,400,438,433,412,292,21,346,216` ✓ | `379,8,337,199,70,362,164,226,271,39` ✗ |

token0 完全一致，token1 完全不同 → `demand_dual` 把**错的专家**装进池、`local` 指向这些错槽，而
`scores`/combine 用的是真实路由的专家 → seq≥2 的每个 token≥1 全部计算错。

## 收敛过程（systematic debugging）

1. **排除生成机制/双缓冲代**：`_pin_gen` 等假设均未消除发散。
2. **排除侧区**：`FORCE_EMPTY_SIDE=1`（仅真实区）后 `seq=K` 仍发散 → 不是侧区。
3. **排除专家字节错**：`STG_VERIFY=1` 全程 `0 BAD`——真实区每槽字节 == 其**属主**专家真值。池落字节
   本身没错（是"装了错的专家当属主"，不是"字节写坏"）。
4. **新增 `ROUTE_VERIFY`**：校 `pool[local[i]]` 字节是否 == **路由专家** `inds[i]` 真值（STG 只校
   slot↔属主，不校 local 是否指向"该 token 真正路由的专家"）。命中 `seq=2` 的 token1 全 BAD。
5. **`POSTCHK`（C++ 锁内）vs `ROUTE`（Python 锁后）矛盾**：C++ 侧 `local[i]==e2r[ip[i]]` 恒成立
   （0 违例），Python 侧却看到 token1 专家不在 e2r。加 epoch 计数证明**两次读的是同一次调用、同一份
   e2r**——即 C++ 的 `ip` 与 Python 的 `inds` 本就不同 → 锁定入参读取环节。
6. **C++/Python 逐位 dump**：见上表，token1 的 `ip` 与 `inds` 完全不同 → 非连续视图按连续读。
7. **验证修复**：`mx::contiguous(inds)` 后 `ROUTE_VERIFY` 0 BAD、`_diag_seqk` pos0/pos1 均 0.0000。

## 修复后基准（`bench_tree.py`，生产 env，K=3 MAXTOK=64 REPEAT=2）

| 项 | 值 |
|---|---|
| `lossless_all` | **true** |
| `max_control_mm`（tree-off 噪声地板） | **0** |
| `max_on_mm`（tree-on 失配） | **0** |
| `median_delta_pct`（救回净提速） | **+10.8%** |
| `verdict` | **go** |

逐 prompt `control_mm/on_mm` 全 0——**过去所有报告归咎的"后端 run-to-run 噪声地板"实际就是这个 bug**，
修后彻底归零。

## 变更 / 保留

- **核心修复**：`native/ext/pool/demand.cpp::demand_dual` — `mx::array ids = mx::contiguous(inds);`。
- **文档**：`generate.py` 救回块注释、本报告更新为根因+修复定论。
- **清理**：所有诊断脚手架（`ROUTE_VERIFY`/`POSTCHK`/`DCDBG`/`CPPDUMP`/epoch 计数、`FORCE_EMPTY_SIDE`、
  `VERIFY_HOST`）已从源码移除；`demand_dual` 相关单测（11 项）全绿。

## 影响面

`demand_dual` 是 `run_mtp_spec` 生产默认（`POOL_SPEC_SLOTS=32`）。修复前**生产 spec 解码相对 dense
贪婪即有损**（seq≥2 装错专家）；修复后 spec 与最小树救回均恢复 bit-lossless，且保留 schemeB 的
+8% tok/s 基座——无需在 `acquire_gpu`（放弃提速）与 lossless 之间二选一。

## 追加实验：第 2 位（pos1）top-2 救回消融（2026-07-05）

修好 lossless 后追问"树形验证还能不能继续挖接受率"。用 `benchmarks/_probe_topk.py` 跑逐位置
top-1/2/3 覆盖率探针（6 prompt，关树走 plain 单链采集每位置候选）：

| 位置 | top1 | top2 | top3 | gap21（top2 可救回上界） |
|---|---|---|---|---|
| pos0（第 1 草稿） | 0.841 | 0.914 | 0.943 | **0.073** |
| pos1（第 2 草稿） | 0.580 | 0.694 | 0.747 | **0.114** |
| pos2（第 3 草稿） | 0.547 | 0.682 | 0.735 | 0.135 |

探针预测：**pos1 的「首选错次选对」比例（11.4%）比 pos0（7.3%）还大**，是最肥的一块。据此实现
pos1 救回（`drafter.draft_tree` 多返回 chainC 第 2 位次选分支；`generate.py` 在 `matched==1 且
chainC[1]==preds[1]` 时改验 chainC，与 pos0 救回同构、纯重验 → bit-lossless），加 `TREE_TOP2_P1` 开关。

三向消融（`_bench_p1_ablation.py`，生产 env，K=3 MAXTOK=128 REPEAT=4，6 prompt）：

| 配置 | 接受长度 | tok/s（中位均值） | pos0 救回 | pos1 救回 | max_mm |
|---|---|---|---|---|---|
| off | 2.360 | 12.06 | 0 | 0 | 0 |
| pos0（旧） | 2.512 | **12.51** | 28 | 0 | 0 |
| pos0+pos1（新） | **2.580** | 12.12 | 29 | 20 | 0 |

- 接受长度：pos0 vs off **+6.43%**；**pos0+pos1 vs pos0 +2.74%（pos1 净增量）**；合计 vs off +9.35%。
- 全程 `max_mm=0` → pos1 救回 **bit-lossless 确认**。

**结论：pos1 在接受率维度真实有效（+2.74%，无损），但换不来净 tok/s。** 每次 pos1 救回要多跑一次
昂贵的主模型前向（MoE 专家加载），该成本在当前硬件/批量下恰好抵消多接受的 token，`pos0+pos1`
(12.12) 甚至略低于 `pos0`(12.51)。对比 pos0 救回能净赚（第 1 位被拒更常见、一次常多接受 2 token，
性价比高），pos1 边际收益不足以覆盖"多一次前向"的固定成本。

**裁决：机制保留（有测试、lossless），`TREE_TOP2_P1` 默认关**，保护已验证的 pos0 纯路径 tok/s 收益；
留作前向变廉价/批量变大时收益翻正的储备，一行 env 即可开。

## 已知次要 bug（未修，未在生产触发）

- fallback replay 在 `matched==K` 时 replay K+1 个 token，真实模型上会触发专家池
  `inds.size=(K+1)·top_k > cap` 溢出告警。生产恒走直接提交、从不 fallback，故未触发。建议对齐
  直接提交路径：`matched==K` 时只 replay `drafts[:K-1]`、留 `drafts[K-1]` 作 pending。
