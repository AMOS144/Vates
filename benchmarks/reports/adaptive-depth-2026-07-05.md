# 置信度门控动态深度评测报告（2026-07-05）

## 结论一句话

**收益全部来自"向下收缩"，不来自"向上扩展到 K=4"。** τ=0.3、depth_max=3 的纯向下自适应
**+5~6% tok/s、bit-lossless、零额外显存**，是干净的净提速。扩到 K=4 反而更慢，且在生产
`EXPERT_SLOTS=32` 下会撞专家池 cap 溢出致有损——**K=4 已测，判定不值得**。

## 动机与信号

修好 lossless、做完 pos0/pos1 救回后,追问"动态自适应 MTP(P-MTP 置信门控)"能否提速。先用
`benchmarks/_probe_conf.py` 量 MTP 逐位 top-1 softmax 置信度 `p_i` 与实际接受的相关性(包一层
drafter 记录置信度,sync 收到的 replay_in 长度即本步接受长度,零热路径改动):

| p 区间 | 第0位接受率 | 第1位接受率(限第0位已中) |
|---|---|---|
| [0.0,0.2) | 0.200 | 0.103 |
| [0.2,0.4) | 0.486 | 0.341 |
| [0.4,0.6) | 0.551 | 0.548 |
| [0.6,0.8) | 0.700 | 0.667 |
| [0.8,1.0) | 0.974 | 0.972 |

分界(0.5):第0位 低置信 46.2% vs 高置信 88.7%(**+42.5pp**);第1位 27.9% vs 87.0%(**+59.1pp**)。
**置信度是接受率的极佳预测器**——低置信步大概率被拒,给它们抽满 K 是在浪费专家加载。

## 机制:为什么本系统尤其适合"向下收缩"

本系统的主瓶颈是 verify 前向里的**专家加载**(IO-bound),专家并集随草稿 token 数增长。低置信步
若只抽 1 个 token(退化为普通单 token 解码:verify 只喂 `[x]`、恒接受模型真值 1 个),就省掉后续
1~2 个位置的专家加载——而这些位置本来多半被拒,几乎不亏接受长度。与 pos1 救回(**增加**一次前向)
相反,动态深度是**减少**工作量,天然对齐本系统成本结构。

实现:`drafter.draft_adaptive` 逐位贪婪抽,累计置信度 `C_i=∏p` 跌破 τ 即停,最多 depth_max 位;
`generate.py` plain 路径用可变 `step_k=len(drafts)` 替代定长 K(commit/replay 的 verified_len 本就
按参数走,可变深度天然支持)。开关 `MTP_ADAPTIVE_DEPTH` / `MTP_CONF_TAU` / `MTP_DEPTH_MAX`。

## 消融结果（`_bench_adaptive.py`，6 prompt，REPEAT=3，MAXTOK=96）

### EXPERT_SLOTS=32（生产配置）

| config | accept_len | tok/s | vs fixed | max_mm |
|---|---|---|---|---|
| fixed-K3 | 2.381 | 12.91 | — | 0 |
| adaptive τ.3 **d4** | 2.453 | 13.77 | +6.69% | **69（有损！）** |
| adaptive τ.5 d4 | 2.249 | 13.31 | +3.16% | 59（有损） |
| adaptive τ.3 **d3** | 2.301 | 13.70 | **+6.15%** | **0（无损）** |

d4 的 +6.69% 是**假象**:`inds.size=40 > cap=32` 溢出告警,seq=4·top_k=10=40 超过池容量,
逐位装错专家 → 输出损坏。

### EXPERT_SLOTS=48（放大池,让 seq=4 也放得下）

| config | accept_len | tok/s | vs fixed | max_mm |
|---|---|---|---|---|
| fixed-K3 | 2.381 | 15.41 | — | 0 |
| adaptive τ.3 d4 | **2.510** | 16.12 | +4.61% | 0 |
| adaptive τ.5 d4 | 2.321 | 16.02 | +3.95% | 0 |
| adaptive τ.3 **d3** | 2.301 | **16.21** | **+5.16%** | 0 |

放大池后 K=4 无损了,但 **d4(+4.61%)仍慢于纯向下 d3(+5.16%)**——尽管 d4 接受长度最高(2.510),
第 4 位的额外专家加载成本压过了多接受的 token。

## 判定

- **纯向下自适应(τ=0.3, depth_max=3):GO。** +5~6% tok/s,bit-lossless,零额外显存,生产 slots=32
  直接可用。收益来自跳过低置信步的无谓专家加载。
- **向上扩展到 K=4:NO-GO。**(a) 生产 cap 下溢出致有损;(b) 即便放大池到 slots≥40 变无损,也比
  纯向下更慢。第 4 位"多接受的 token"换不回"多加载的专家"。

## 默认与开关

- `MTP_DEPTH_MAX` 默认 **3**(=基础 K,slots=32 安全);要试 K=4 必须同时 `EXPERT_SLOTS≥40`。
- `MTP_ADAPTIVE_DEPTH` 默认 **关**:它是与 `tree_top2` 互斥的可选加速路径(adaptive 仅在非 tree
  的 plain 路径生效)。是否设为生产默认(替换当前 tree_top2 快路径)属产品取舍,留待决定。
- 未来方向:动态深度 + 草稿链救回二者结合(先自适应定深、再对该深度链做 pos0 救回),可能叠加收益。

## 追加实验:动态深度 + pos0 救回能否叠加(合并路径)

动态深度(压每步成本)与树形 pos0 救回(抬接受长度)机理正交,自然会问能否叠加。实现合并路径
`draft_adaptive_tree`(变长 chainA + 对深链 n>=2 抽 pos0 分支 chainB),`MTP_ADAPTIVE_RESCUE` 开关。

消融(`_bench_merged.py`,slots=32,τ=0.3,depth_max=3,6 prompt,REPEAT=3):

| config | accept_len | tok/s | vs fixed | max_mm |
|---|---|---|---|---|
| fixed-K3 | 2.381 | 13.87 | — | 0 |
| tree-pos0(救回单开) | 2.529 | 14.09 | +1.56% | 0 |
| adaptive(动态深度单开) | 2.301 | **14.35** | **+3.46%** | 0 |
| adap+rescue(合并) | 2.369 | 13.58 | **−2.07%** | 0 |

**叠加检验:adap+rescue vs adaptive 单开 = −5.34%(负)。判定:NO-GO,不能叠加。** 两个原因:
1. **机理冲突**:救回确实抬了接受长度(2.301→2.369,resc=14 有触发),但它与动态深度都作用于低置信步、
   方向相反(救回想多花、动态深度想少花),省下的成本又被吐回。
2. **硬成本**:开救回必须保留主 cache 快照(`_skip_snap` 快路径被迫关闭),**每步多付 ~72MiB 深拷贝
   + 同步栅栏**——而纯动态深度正是靠跳过此快照提速。+3% 接受长度盖不住每步快照 + 额外前向的开销。

合并路径(`draft_adaptive_tree` / `MTP_ADAPTIVE_RESCUE`)保留在代码里、默认关,作已验证的死胡同存档。

## 变更

- `drafter.py::draft_adaptive`(新增)、`generate.py`(plain 路径可变 `step_k`)、`config.py`
  (`MTP_ADAPTIVE_DEPTH`/`MTP_CONF_TAU`/`MTP_DEPTH_MAX`)。
- 测试:`test_mtp_generate_adaptive_depth_lossless`(变长深度+拒绝步仍 lossless,25 全绿)。
- 复现:`benchmarks/_probe_conf.py`(置信度探针)、`benchmarks/_bench_adaptive.py`(消融)。
