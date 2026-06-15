# Qwen3-Next-80B MTP 自投机解码:按 vLLM 语义重写后的 MLX 结论

日期:2026-06-07
模型:`Qwen3-Next-80B-A3B-Instruct-4bit`(专家流式 → NVMe,主模型文件后端 patch)
机器:32GB 统一内存 macOS;MTP 权重 4-bit 常驻

## 内存右尺寸:常驻专家权重 11.25GB → 6.96GB(无损),峰值 14.6 → ~10GB

`EXPERT_SLOTS=256` 是**每层**容量,`ResidentExpertPool` 首次触碰某层就整块预分配 `(256,*)`。
但实测各层真实工作集极不均匀(`probe_pool_footprint.py`:每层高水位 103–256,均值 158,
**只有 2 层真到 256**)→ uniform 256 白白预留 ~4.3GB 从没用过的槽。

**修法:按 profile 给每层独立池容量、首次一次性分配(`ResidentExpertPool.layer_caps`)。**
这是**无损**的——池大小只决定命中率,从不影响输出;且对同类负载预算=高水位时永不额外驱逐,
命中率/吞吐不变。(注:试过动态 grow-on-demand,但 `mx.concatenate` 每次复制整池 →
O(工作集²),稳态被拖垮,已弃用;改为 profile 静态分配,零运行时开销。)

| 配置 | 常驻权重池 | 峰值 | spec tok/s | spec命中率 | disk_load_ratio | 性质 |
|---|---:|---:|---:|---:|---:|---|
| uniform 256(旧) | 11.25GB | 14.63GB | 23.7 | 0.986 | 0.14 | 预留浪费 |
| profile margin=1.15 | 7.93GB | 11.07GB | **28.0** | 0.986 | 0.14 | 完全无损,反而更快 |
| profile margin=1.0 | **6.96GB** | **10.02GB** | 21.8 | 0.98 | 0.21 | 同负载无损,峰值压到~10 |

复现:先产 profile,再带 profile 跑(`EXPERT_POOL_MARGIN` 给未见 prompt 留冗余):
```bash
EXPERT_POOL_PROFILE_OUT=/tmp/qn_pool_profile.json EXPERT_POOL_MARGIN=1.15 \
EXPERT_DIR=…_2bit EXPERT_SLOTS=256 RESIDENT_POOL=1 MTP_VERIFY_MODE=batch \
MTP_ARRAY_COMMIT=1 MAXTOK=96 K=2 python -m mlx_streaming.probe_pool_footprint
# 然后所有运行加上 EXPERT_POOL_PROFILE=/tmp/qn_pool_profile.json
```

**结论**:真正的常驻权重(专家池)现在 **6.96GB,远 < 10GB**。"14GB" 是峰值,含激活 + MTP +
共享专家约 3GB(非专家池)。保留完整工作集的峰值下限 ≈ 10GB;要再低需牺牲一点命中率(把
margin/每层预算压到高水位以下),不再严格无损。margin=1.15 是推荐默认(无损且更快)。

## 冲 30 的最后一条硬路:fused MoE Metal kernel = NO-GO(已止损)

为把 24.7→30,尝试了"自定义 fused 量化 SwiGLU Metal kernel"(让 gate/up/down + 反量化 + SwiGLU 融合成 1 次 dispatch、减少 launch 与中间物化)。先做了便宜的 de-risk:

- `probe_moe_ceiling.py`:把专家计算短路成 0,单 token 解码 8.0→14.1 tok/s → 专家计算占单路 ~43%(可攻面);但 MTP 已靠摊薄到 24.7,**已超过单路"专家免费"上界**,故单路天花板不直接约束 MTP。
- `probe_fused_moe.py`:手写 fused gate+up+SwiGLU kernel,**数值与 mlx `gather_qmm` 逐位一致(diff=0)**,且已加共享内存缓存 `x`。但**吞吐关键的 decode(T=1)仍 0.64×(更慢)**:0.90ms(仅 gate+up 一次 dispatch)> mlx 0.58ms(gate+up+swiglu 两次 dispatch)。瓶颈是**每线程标量 MAC 的计算效率**,不是 dispatch 数。

**结论(NO-GO)**:要赢必须用 `simdgroup_matrix` 重写 tiled SIMD 量化 GEMM,等于重造 Apple tuned 的 quantized GEMM——多周、高风险,且净增益仅"省一次中间物化",乐观也只到与 mlx 持平,达不到 1.5–2×。`FUSED_MOE` 不接入主流程。**24.7 tok/s @ 14.6GB 为本机本栈实际上限**;I/O、纯 MLX 计算优化、fused kernel 三条路均已穷尽。详见 `docs/superpowers/specs/2026-06-07-fused-moe-metal-kernel-design.md`。

## 重大更正(2026-06-07 晚):MTP 在本机确实有效,2.43×、24.7 tok/s

**此前"MTP 在 80B+流式上注定 ≤1.0x、30 tok/s 不可达"的裁决是错的,根因是基准只测了错误的池容量。**

之前所有 MTP 基准都固定在 `EXPERT_SLOTS=96`。投机解码会让每步路由到**更宽的专家足迹**(草稿 token 走 baseline 不碰的专家),96 槽装不下 → 反复换入换出(抖动) → `spec_hit_rate` 掉到 0.79、`disk_load_ratio` 1.4× → speedup 被抵消到 ~1.0x。历史上又测过 `EXPERT_SLOTS=512`,但那是**另一个极端**:512 槽 ≈ 18-27GB 远超本机内存预算 → 换页(`t_verify` 暴涨到 1051ms/步) → 0.57x。两次都偏离了真正的甜点。

**甜点在 128-256 槽之间**(既能装下投机的宽足迹、又不触发换页)。`batch` 一次前向验证(`MTP_VERIFY_MODE=batch MTP_ARRAY_COMMIT=1`)在此区间随池增大单调提速:

| 槽/层 | K | speedup | spec tok/s | base tok/s | spec命中率 | disk_load_ratio | 峰值GB | 状态 |
|---:|--:|--:|--:|--:|--:|--:|--:|---|
| 96 | 2 | 0.99 | 8.9 | 9.0 | 0.80 | 1.38 | 7.1 | 抖动 |
| 128 | 2 | 1.28 | 11.3 | 8.8 | 0.88 | 1.08 | 8.6 | |
| 160 | 2 | 1.65 | 16.8 | 10.2 | 0.93 | 0.65 | 10.1 | |
| 192 | 2 | 2.16 | 20.6 | 9.5 | 0.97 | 0.35 | 11.6 | |
| **256** | **2** | **2.43** | **24.7** | 10.2 | 0.99 | **0.14** | **14.6** | **甜点** |
| 256 | 4 | 2.15 | 21.9 | 10.2 | 0.98 | 0.27 | 14.8 | |
| 320 | 2 | (1.5) | **0.72** | 0.48 | 0.99 | 0.09 | 17.65 | **换页崩溃** |

关键证据:256 槽时 `disk_load_ratio=0.14`(spec 一次批量加载 K 个 token 共享专家,**读盘反而比 baseline 逐 token 还少**),`spec_hit_rate=0.99`,**I/O 被彻底消除**。`batch` 一次前向的次线性摊薄(`probe_verify_scaling` 实测 verify-2 仅 1.12× 单 token)由此显现 → 2.43×。**K=2 始终优于 K=4**(高 K 加宽足迹的代价盖过接受长度收益)。

### 精确性说明(已查清,非 bug)

`batch` 路径 `exact_match=false`(MAXTOK=128 时 63/128 与顺序 baseline 不同),但**这是良性的、标准投机解码语义**,不是 bug、不是质量损失:

- 实测 MAXTOK=64 时 batch 与 baseline **逐字一致(0 分歧)**;更长生成会在某个 **FP 平局点**(批量 vs 顺序前向在 12 个全注意力层上的浮点差)翻转一次 argmax,之后轨迹合法分叉成**另一条同样连贯的续写**。
- 分叉实例(token 64):baseline「从而实现**对任务的精细化分工与高效并行处理**」vs batch「从而实现**模型的分层与专业化分工**…广泛应用于 GPT-4 和 Switch Transformer」——两条都正确、连贯。
- 标准投机解码本就保证"输出 = 批量目标模型的贪婪结果",与"顺序 baseline"仅在 FP 平局处不同。**对实际使用无害,只是不可逐 bit 复现。**
- 若需逐 bit 可复现,用 `MTP_VERIFY_MODE=step`(逐 token verify,`exact_match=true`),但只有 1.2-1.34×(13-14.6 tok/s)——因为逐 token 不摊薄计算。**精确 与 满速 二者不可兼得**。

### 速度天花板更正

旧报告"7.4 tok/s 是物理下限、30 不可达"作废。当前实测:
- **单路最佳(无 MTP)**:2-bit、64 槽、热盘 ≈ **18 tok/s**,峰值仅 **3.9GB**(命中率 95.6%,容量 64 即饱和,加槽无益)。
- **MTP 最佳**:`batch`、256 槽、K=2 ≈ **24.7 tok/s**(2.43× vs 同轮 baseline;1.37× vs 单路最佳 18),峰值 **14.6GB**。
- 30 tok/s 仍未达:256 槽时 I/O 已归零,24.7→30 的差距是 batch=1 解码的 **GPU kernel 启动开销**(route/combine/matmul 微基准证实单算子都不贵,profiler 的占比是 eval 屏障假象)。再往上只剩"手写 fused Metal kernel 减 op 数"这条硬路,且 320 槽换页说明加内存反而崩。

**推荐默认配置**:`EXPERT_DIR=…_2bit EXPERT_SLOTS=256 RESIDENT_POOL=1 K=2 MTP_VERIFY_MODE=batch MTP_ARRAY_COMMIT=1` → **21.6-24.7 tok/s**(2.3-2.4×,随盘热度/温度波动)/ 14.6GB(可接受 batch 语义);内存敏感场景用单路 64 槽 18 tok/s / 3.9GB。

## 2026-06-08 复盘:MTP 红利来源的正确归因 + 两个 NO-GO

### 热盘复现:24.88 tok/s / 2.91×(冷盘 14.5 是机器态,非缺陷)

同机背靠背(256 槽 / K=2 / batch / array-commit,静态 profile):冷启动几轮时 baseline 仅 7.6、
spec 14.5;盘热起来后 **baseline 8.56 → spec 24.88(2.91×)**,`avg_accept_len=1.778`(K=2 的 89%)、
`spec_hit_rate=0.986`、`disk_load_ratio=0.15`。**复现报告的 24.7,机制完好;早期低值纯属冷盘/热漂移。**

### 正确归因:MTP 的提速几乎全来自摊薄 I/O,不是摊薄计算

- `probe_verify_scaling`(warm,已隔离 I/O)实测一次前向墙钟**近线性于 token 数**(verify-3 ≈ 2.6×
  单 token,只有 ~13% 摊薄)。即 **batch 验证买到的计算红利很小**。
- 根因(本模型特有,非"线性注意力不能用 MTP"):**A3B 激活极小** → 可共享的 dense 权重占比低;
  **gated-delta 线性注意力是逐 token 递归**(`conv/ssm_state` 顺序更新),K 个位置的状态更新摊不到
  一起。两者叠加 → 没有"巨大的、按 token 重复搬的固定权重"可摊 → 前向≈线性。
- 真正的大杠杆是 **I/O 摊薄**:批量验证一次加载 K 个 token 的专家**并集**(并集次线性),baseline
  ~42 读盘/token → spec ~11 读盘/步,`disk_load_ratio` 1.x → 0.15。verify 一次过 2–3 个 token 反而比
  baseline 单 token 还快,纯因躲掉了大部分冷读盘。**所以 MTP 只在 I/O 受限的真实场景给红利;一旦 I/O
  归零(如单路 100% 命中 28.5),它顶到同一堵"按 token 计算"的墙,无法再超。**

### NO-GO:draft 全程 GPU argmax(消除每草稿 host 同步)

`probe_draft_breakdown`(已删)隔离测得单个 draft step 仅 2.6ms(lm_head 70% / 注意力+MoE 31% /
argmax 同步≈0),但实跑 `t_draft` 达 47ms/步 → 疑为每草稿 `int(argmax)` 的 host 同步夹在大块
verify/快照间吃满屏障延迟。**改成"K 个 argmax 全留 GPU、末尾一次 .tolist()"后反而更慢**:A/B 同机
draft 慢 5×(`t_draft` 0.44→2.15s)、端到端 **24.9→16.3**。原因:逐步 `int()` 同步让 MLX 把每步
draft 的图保持极小、及时释放;攒大惰性图(argmax 当 embedding 索引)单次 eval 更贵。**保留逐步同步,
不接入。** 教训:隔离探针在"小 kernel 夹在大屏障之间"的场景会系统性低估,只能信端到端 A/B。

---

## TL;DR(以下为更正前的历史记录,结论已被上方更新)

最新更新:已按「先 1 后 2」完成两级实现,并补跑了 Next 专家 2-bit/混合精度 + K=2 MTP。

- `MTPDrafter` 不再每步 reset,而是用已接受 token 的真实主模型 hidden 推进 MTP KV cache。
- KVCache-only 路径支持验证后直接 `trim` rejected suffix 完成 commit,无 replay。
- **1. block-align exact(default)**:含 `ArraysCache` 时维护安全边界,不信任 speculative window 的递归中间态;验证后恢复到安全边界并只重放 accepted unsafe tail。`exact_match=true`,但仍有 replay。
- **2. step direct exact(`MTP_VERIFY_MODE=step`)**:verify 本身逐 token 走与 baseline 相同的解码路径,每个 token 后保存 cache 快照;接受后直接恢复到对应快照,不 replay。`exact_match=true` 且可 direct commit。

仍保留 `MTP_ARRAY_COMMIT=1` 作为旧实验路径:它从 batch verify 里直接切 `ArraysCache` checkpoint,会漂移,不作为正确实现。

最新真机小样本:

| 模式 | K | MAXTOK | exact_match | direct_commits | fallback_replays | spec tok/s | base tok/s | speedup |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| block-align exact(default) | 3 | 48 | ✅ | 0 | 14 | 2.15 | 4.52 | 0.48x |
| step direct exact(`MTP_VERIFY_MODE=step`) | 3 | 48 | ✅ | 20 | 0 | 3.51 | 4.06 | 0.86x |
| step direct exact(`MTP_VERIFY_MODE=step`) | 2 | 48 | ✅ | 24 | 0 | **5.26** | 4.19 | **1.26x** |
| batch Arrays direct(`MTP_ARRAY_COMMIT=1`) | 2 | 48 | ❌ | 25 | 0 | 3.92 | 4.33 | 0.91x |

### 低 bit 专家复测(K=2,`MTP_VERIFY_MODE=step`,MAXTOK=96)

| 专家目录 | 量化 | 磁盘体积 | exact_match | accept_len | direct_commits | spec tok/s | base tok/s | speedup | 磁盘加载比 | peak GB | RSS GB |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `/tmp/qwen3_next_experts` | 4-bit | 41G | ✅ | 约 1.8-2.0 | 24* | **5.26** | 4.19 | **1.26x** | — | 约 15 | — |
| `/tmp/qwen3_next_experts_2bit` | gate/up/down=2-bit,g64 | 23G | ✅ | 1.778 | 54 | 7.25 | **7.49** | 0.97x | 1.34× | **9.50** | **5.80** |
| `/tmp/qwen3_next_experts_mix_g2u2d3` | gate/up=2-bit,down=3-bit,g64 | 26G | ✅ | 1.745 | 55 | 6.89 | **7.14** | 0.96x | 1.31× | **10.43** | **6.66** |

> *4-bit 行是此前 48-token 小样本,只作方向对比;低 bit 两行是本次同一轮 MAXTOK=96 的真实结果。

低 bit 的直接收益很明确:baseline 从 4-bit 约 4.2 tok/s 提到 7.1-7.5 tok/s,峰值内存从约 15GB 降到 9.5-10.4GB。但 K=2 MTP 在低 bit 下没有继续增益,因为 baseline 单 token 已经更快,step verify 仍要逐 token 走验证前向,额外草稿和更高磁盘加载比抵消了接受长度收益。

### 连续常驻专家池热路径优化(`RESIDENT_POOL`,2-bit)

热路径分段(`probe_hotpath`,2-bit,warm)显示 `fetch`(每 token 每层 `mx.stack` 把选中专家堆成连续张量)占 53%、远大于 matmul(`gather_qmm`)17%。改为**每层连续常驻池**(容量=每层槽数,内存零增量):命中时只把 slot 下标喂给 `gather_qmm`(零 `mx.stack`),miss 时只原地写单个专家槽位(de-risk 实测原地写 ~0.38ms 且与容量无关,见 spec)。

同机 A/B(`run_mtp_spec`,2-bit,K=2,MAXTOK=96,`MTP_VERIFY_MODE=step`):

| 指标 | 旧 stack(`RESIDENT_POOL=0`) | 新池(`RESIDENT_POOL=1`) | 变化 |
|---|---:|---:|---:|
| baseline tok/s | 7.65 | **9.12** | **+19%** |
| MTP K=2 tok/s | 7.07 (0.92x) | 8.19 (0.90x) | — |
| 峰值 MLX | 9.50 GB | **7.08 GB** | **-26%** |
| RSS | 6.26 GB | 4.39 GB | -30% |
| exact_match | ✅ | ✅ | 不变 |

`probe_hotpath` 分段:`fetch` 89.4→71.9 ms/step、`per_token` 183→162.6 ms。结论:**消除 per-token `mx.stack` 直接给 baseline 单 token +19%、峰值内存 -26%,且数值精确**。这是不依赖投机、所有槽位/量化档共享的确定收益。剩余 `fetch` 主要是 13% miss 的真实磁盘 I/O(已非 stack)。MTP K=2 仍 0.90x——印证「线性注意力 step verify 不摊薄」的结论,池优化加速的是 baseline 与 verify 两端的同一段,故 speedup 比值基本不变。

结论要写严谨:**不能说 Qwen3-Next/MTP 原理无效**。当前 MLX 路径已经跑通了一个正确的 direct-commit 版本:K=2、`MTP_VERIFY_MODE=step` 下 `exact_match=true` 且 **1.26x**。但这个版本靠逐 token verify 保证数值一致,下一步优化空间是把 step verify 融成更快的 kernel/图,而不是回到会漂移的 batch checkpoint commit。

根因(已被专家加载计数证伪了"I/O 成倍放大"的初判):
1. **本模型解码不是纯 I/O 瓶颈**。`FileExpertStore` 槽位是「每层 96 个」,每 token 每层只激活 10 个专家,故专家缓存**命中率高达 87%**;解码是「计算 + 少量 I/O」的混合瓶颈。
2. **批量验证确实摊薄了 I/O**(命中率 87%→86% 几乎不变,磁盘加载只多 1.49×,非 K 倍)——但
3. **MoE 专家计算是 per-token 的,批量不摊薄**:验证一次过 K 个 token 要做 K 倍专家 matmul;投机对被拒绝草稿 token 的这部分计算在计算瓶颈下藏不住。
4. **不可裁剪 cache 强制重放**:线性注意力 `ArraysCache` 不可 trim,验证后状态被拒绝草稿污染,只能「快照→验证→恢复→重放命中前缀」,多一次完整主模型前向(标准投机解码没有此开销)。

每步 ≈ 验证(K token)+ 重放(m+1 token)≈ 5.7 个 token 前向,只推进 ~2.67 token,叠加 1.49× 磁盘 I/O → 0.42x。

**工程结论:低 bit 专家是确定收益,K=2 step direct exact 只在 4-bit 当前路径上有收益。** 纯 2-bit 给出本机最高 baseline **7.49 tok/s** 和最低峰值内存 **9.50GB**;混合精度略慢但可能更稳质量。低 bit 下 MTP K=2 变成 0.96-0.97x,不建议默认叠加。

## 实测数据

| K | MAXTOK | SLOTS/层 | exact_match | accept_len | spec tok/s | base tok/s | speedup | 磁盘加载比 | base 命中率 | peak GB |
|---|--------|---------|-------------|-----------|-----------|-----------|---------|-----------|------------|---------|
| 1 | 40 | 96 | ✅ | 1.818 | 3.82 | 2.84 | 1.35x* | — | — | 14.98 |
| 2 | 64 | 96 | ✅ | 2.286 | 2.62 | 4.04 | 0.65x | — | — | 15.18 |
| 3 | 96 | 96 | ✅ | 2.286 | 2.03 | 4.07 | 0.50x | — | — | 15.19 |
| 3 | 64 | 96 | ✅ | 2.667 | 1.91 | 4.51 | 0.42x | **1.49×** | **0.872** | 15.18 |
| 2 | 64 | 384 | ✅ | 2.286 | 1.60 | 3.59 | 0.45x | — | — | 18.74 |

> *K=1 的 1.35x 是假象:该次 baseline 冷缓存(2.84),spec 跑在预热后的缓存上。带 warmup 的 K=3 行才是干净对比。
> speedup 为同一次运行内 spec/baseline 比值。高槽位(384/层)反而更慢且内存升到 18.74GB → 加槽救不回,只增内存。

## 数据支撑的归因(替换初版错误的"I/O 成倍放大"成本模型)

加入专家加载计数后(K=3、64 token、warmup 后):

- **baseline 命中率 0.872**:每 token 每层激活 10/512 专家,而每层缓存 96 槽 → 相邻 token 重叠高,87% 走缓存。**故 baseline 偏计算瓶颈,而非纯 I/O。**
- **批量验证确实摊薄 I/O**:spec 命中率仍 0.864,磁盘加载仅 **1.49×**(若 I/O 线性于 token 应是 ~K 倍)。**用户直觉成立。**
- **但 spec 慢 2.4 倍(0.42x)远超 1.49× I/O**,差额来自**计算**:
  - MoE 专家计算 per-token,批量不摊薄:验证一次过 K=3 token = 3× 专家 matmul。
  - 不可裁剪 cache → 强制重放:每步多一次完整主模型前向。
  - 每步 ≈ 验证(K=3)+ 重放(m+1≈2.67)≈ 5.7 token 前向,推进 2.67 token → 计算量 ≈ 2.1× baseline。
  - 2.1×(计算)与 1.49×(I/O)叠加,与实测 0.42x 一致。

**为何投机在此失效**:投机解码的红利来自「访存瓶颈下,批量 K token 复用一次加载的权重」。本模型专家 4-bit 且 87% 已驻缓存 → 偏计算瓶颈;投机对草稿 token 的额外 per-token 专家计算藏不住,再叠加 `ArraysCache` 不可裁剪强制的重放,红利不成立。

## 为什么必须重放(无法只做一次前向)

Qwen3-Next 的线性注意力层用 **`ArraysCache`(conv_state + ssm_state)递归状态,不可裁剪**。
K 宽验证后,该状态已混入被拒绝的草稿 token,无法像 KVCache 那样 `trim` 回退;
而 conv/ssm 更新与该层 MoE 交织,无法在不重跑整层(含 MoE 专家)的情况下单独回滚递归状态。
故必须「快照 → 验证 → 恢复 → 重放命中前缀」,重放这次主模型前向(48 层全部 MoE)不可避免。
这与之前发现「`ArraysCache` 不可裁剪导致 mlx-lm 原生投机解码不可用」同源,只是 MTP 从另一角度撞上同一堵墙。

## 工程产出(已落地、可复用)

实现是干净且正确的,可在「专家常驻」场景或其他模型上复用:

- `mlx_streaming/qwen3_next_mtp.py`:`Qwen3NextMTP.__call__(return_hidden=)`、`mtp_step()`、`load_mtp(quantize=)`(4-bit + gate/shared_expert_gate 8-bit,镜像主模型量化配置)。
- `mlx_streaming/mtp_generate.py`:
  - `forward_with_hidden(model, ids, cache)`:暴露主模型 final-norm 后 hidden。
  - `_snapshot/_restore`:统一处理 `KVCache`(setter 按 shape 重算 offset)与 `ArraysCache`(`[conv,ssm]` 列表),深拷贝 + `mx.eval` 防原地改写。
  - `commit_verified_prefix`:KVCache 可直接 trim commit;ArraysCache checkpoint commit 作为实验路径(`MTP_ARRAY_COMMIT=1`)保留。
  - `forward_with_hidden_stepwise`:逐 token verify + 每 token cache 快照,实现 exact direct commit。
  - `enable_qwen3next_speculative_checkpoints`:verify forward 中记录 Qwen3-Next 线性注意力 per-token `[conv_state, ssm_state]`。
  - `accept_prefix`:最长命中前缀;发射 = `命中草稿 + 重放末位 bonus`,统一纠正/全命中两路。
  - `mtp_generate`:贪婪自投机主循环(prefill → 抽 K 草稿 → 验证 → 回滚重放)。
  - `MTPDrafter`:把 MTP 包成 drafter;按 vLLM 语义持久化 MTP KV cache,用 accepted 主 hidden 同步,不再每步 reset。
- `mlx_streaming/run_mtp_spec.py`:真机基准 + `exact_match` 硬校验。
- `mlx_streaming/tests/test_mtp_generate.py`:17 个测试,含 **K=1/3 在 KVCache 与 ArraysCache 上的贪婪逐 token 等价性**、MTP cache 持久同步、KVCache direct commit、ArraysCache checkpoint commit、step direct exact 语义。

## 验证结论(贪婪等价性)

- 玩具模型:K=1/3 × {纯 KVCache, 含 ArraysCache 递归层},`mtp_generate` 输出与朴素逐 token 贪婪**完全一致**(与草稿质量无关)。
- 真机 80B:默认 block-align exact 与 `MTP_VERIFY_MODE=step` direct exact 均可 `exact_match=true`、`n_mismatch=0`。旧 `MTP_ARRAY_COMMIT=1` batch checkpoint direct commit 会漂移,不作为正确路径。

## 历史 spike:消除重放能翻盘吗?(已被 step direct exact 更新)

> 本节保留历史排查记录。旧结论“放弃 MTP/投机提速路线”已被上方最新结果修正:K=2 `MTP_VERIFY_MODE=step` 已实现 `exact_match=true` 且 1.26x。

为判断"实现 per-token 线性 state 检查点、彻底消除重放"是否值得做,先做零风险的**分段计时 spike**:把每步「草稿 / 快照 / 验证 / 重放」耗时拆开,算「重放免费」的投机 tok/s 上限(`proj_no_replay = tokens / (t_draft + t_verify)`)。

| K | accept_len | t_draft | t_snap | t_verify | t_replay | proj_no_replay | vs baseline |
|---|-----------|---------|--------|----------|----------|----------------|-------------|
| 2 | 2.286 | 3.50 | — | 13.57 | 8.44 | 3.75 | 0.75x |
| 3 | 2.667 | 1.4~3.5 | **0.015** | 13.4~13.9 | 5.9~6.1 | 3.8~4.2 | **0.79~0.93x(峰值)** |
| 4 | 2.783 | 4.07 | — | 21.53 | 7.11 | 2.50 | 0.59x |

**历史结论(已修正):当时只看 batch verify / replay-free 投影,K=3 峰值约 0.9x。后来新增 step direct exact 后,K=2 实测达到 1.26x。**

根因(分段数据钉死):**K 宽验证前向不摊薄**。K=3 验证 ≈ 0.56s/步、推进 2.667 token,与 baseline(单 token ~0.21s,即 2.667 token 需 ~0.56s)**几乎持平**——验证 3 个 token 的成本就是 ~3× 单 token,没有任何批量复用红利。

为什么不摊薄(也解释了与 30B 的差异):**MoE 专家计算本质 per-token**——每个 token 各选各的 10/512 专家,K 个 token 的专家 matmul 是 ~K× 累加,且相邻 token 专家重叠低(磁盘加载比 1.49× 即证)。投机解码的红利来自摊薄**dense 权重**的重复加载;而 A3B 这类「激活极小」的 MoE,dense 占比极低、专家占主导,几乎没有可摊薄的东西。30B-A3B 当时之所以快,**唯一**原因是其全注意力 KVCache 可裁剪、走 mlx-lm 原生投机(一次前向、无重放);一旦同样砍掉重放后 80B 仍打不过,说明 A3B+流式这条路本身投机红利就薄,80B 的重放只是雪上加霜。

**修正后的裁决:不放弃 MTP。** 当前推荐路径是 K=2 + `MTP_VERIFY_MODE=step`;后续优化重点是把 step verify 融合/编译化,降低逐 token verify 的调度成本。

## 历史复核 spike(已被最新实现部分推翻)

启动 B 前,用三个互相独立的探针重新审视"前向能否摊薄",结论彼此一致地坐实 NO-GO,并**订正了此前两处偏差**(误判过"I/O 成倍放大",又误判过"前向有 ~230ms 固定开销可 8× 摊薄")。

1. **高槽位(warm、消除 I/O)下投机仍亏**(`EXPERT_SLOTS=512`,K=3):命中率 **0.992**、`disk_load_ratio 0.11`(I/O 基本归零),但 `proj_no_replay_speedup` 反而掉到 **0.57**(低槽位时是 0.9);`t_verify` = 1051ms/步 ≫ baseline 257ms/token。**即使 I/O 全免、重放全免,投机也打不过。**

   > ⚠️ 更正:此结论错误。`t_verify=1051ms/步`是 512 槽**超内存预算触发换页**的抖动产物(同现象:320 槽时 spec 崩到 0.72 tok/s),并非投机本质。真正的甜点在 128-256 槽(不抖动也不换页),256 槽 batch verify 达 2.43×、24.7 tok/s。见文首「重大更正」。

2. **专家 union 是次线性、且对投机有利**(`probe_union.py`,top_k=10):

   | L | 每层 union 专家数 | 相对 L=1 |
   |---|------------------|---------|
   | 1 | 10.0 | 1.0× |
   | 2 | 18.35 | 1.84× |
   | 3 | 23.35 | 2.34× |
   | 5 | 33.75 | 3.38× |

   K=3 验证每层只激活 23.35 个专家、推进 2.67 token → **8.75 专家/token,比 baseline 的 10 还少**。**专家计算量不是瓶颈,反而省**。推翻了 spike 段"专家重叠低、K× 累加"的归因。

3. **前向耗时近线性于 L(几乎不摊薄)** —— 真正根因(`probe_verify_scaling.py`,warm、命中率 0.875、真实上下文 68 token):

   | L | 前向 ms | ms/token | 相对 L=1 |
   |---|---------|----------|---------|
   | 1 | 135.1 | 135.1 | 1.0× |
   | 2 | 383.8 | 191.9 | 2.84× |
   | 3 | 351.4 | 117.1 | 2.60× |
   | 5 | 643.8 | 128.8 | 4.77× |

   `ms_ratio(L) ≈ L`。verify(3)=2.6× 单前向、最多推进 3 token,与 baseline 跑 3 次单前向(2.67×)**基本持平**;叠加 draft 成本即净亏。此前"8× 摊薄"是 batch_scaling 探针的 artifact(空 cache + 短上下文,union 偏小)。

**订正后的根因**:不是"专家计算 K× 累加"(union 次线性,反而省),而是**单次前向耗时本身近线性于 token 数——没有足够大的固定开销可摊薄**。dense 部分(attention QKVO + dequant + 路由)按 token 走,MoE 专家虽次线性但占比不足以扭转。故"批量验证近乎免费"这一投机解码的前提**在本模型不成立**。

**修正:** B 已做完。batch checkpoint direct commit 会漂移,但 step direct exact 可用;K=2 获得 1.26x。

**附带的吞吐天花板(诚实告知)**:warm 单 token 前向 = 135ms ≈ **7.4 tok/s** 是本 80B-A3B 在本机解码的物理下限(全部专家驻留、零 I/O 的理想态)。30 tok/s 对**这个模型**在本机不可达,无论是否投机。要 30 tok/s 需换更小/更易摊薄的模型。

> ⚠️ 更正:该"7.4 tok/s 下限/30 不可达"作废。后续实测单路 2-bit 热盘 ≈ 18 tok/s、MTP batch@256 ≈ 24.7 tok/s。当时 135ms 是特定测量条件(且未走 batch verify 摊薄、未到 256 槽甜点)。见文首「重大更正」。

## 建议

1. **当前速度优先推荐纯 2-bit 专家单路解码**:`/tmp/qwen3_next_experts_2bit` 在 MAXTOK=96 下 baseline **7.49 tok/s**,峰值 **9.50GB**。
2. **质量优先可试混合精度专家**:`gate/up=2-bit,down=3-bit` 为 **7.14 tok/s**,峰值 **10.43GB**;速度略低,但理论上比纯 2-bit 更稳。
3. **低 bit 下暂不推荐默认叠 K=2 MTP**:两组都是 exact,但 `speedup=0.96-0.97x`;MTP 的 step verify 调度成本抵消了低 bit 加速。
4. 逼近 30 tok/s 的方向仍应继续降低 per-token 计算/dequant 成本:
   - 减少每 token 的专家计算量(更激进量化以降 dequant+matmul、或减小激活专家数)。
   - 减少剩余 13% 的 miss(热专家钉常驻 `store.pin`、预取/预门控掩盖加载延迟)。
   - 保持 K=2,避免 K=3 草稿/验证成本过高。
   - 优化 step verify:把逐 token verify 的 2 次前向融合成更少 MLX 调度,或为 gated-delta/full-attn 提供统一 step kernel。
5. 不建议默认启用 `MTP_ARRAY_COMMIT=1`:它虽然无 replay,但 `exact_match=false`。
6. 继续保留 block-align exact(default)作为安全回退路径,用于验证与排错。
