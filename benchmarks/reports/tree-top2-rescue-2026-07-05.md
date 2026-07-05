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

## 已知次要 bug（未修，未在生产触发）

- fallback replay 在 `matched==K` 时 replay K+1 个 token，真实模型上会触发专家池
  `inds.size=(K+1)·top_k > cap` 溢出告警。生产恒走直接提交、从不 fallback，故未触发。建议对齐
  直接提交路径：`matched==K` 时只 replay `drafts[:K-1]`、留 `drafts[K-1]` 作 pending。
