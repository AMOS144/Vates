# 低内存全流式 MoE（blob 格式）实测报告（2026-06-11）

## 结论（TL;DR）
- **实现完成且数值正确**：blob 格式 + `quantized_matmul` 路径与常驻路径**逐 token 等价**（同输入 `n_mismatch` 完全一致；单专家对 safetensors 参考 0 误差）。
- **内存**：峰值 `10.0GB → 5.67GB`（**-43%**）。没到 ~1-2GB，因为剩余被**主模型非专家权重**（注意力/embedding/shared expert/MTP，4bit 80B）占住，专家流式部分已很小。
- **吞吐**：`32.4 → 8.4 tok/s`（同次 A/B），慢 ~3.6×。**瓶颈不是 IO**（blob 已把读做到 6.6 GB/s），而是 **decode 每层的 host 同步（`.tolist`）+ 每次 acquire 的 `mx.stack`/物化编排开销**。
- 关 F_NOCACHE（允许 page cache 复用）只从 8.39 → 8.93，**证明冷读 IO 不是瓶颈**。

## A/B 数据（MAXTOK=64, K=2, batch verify, 2bit-g128 全 48 层）

| 变体 | spec tok/s | mlx_peak_gb | exact_match | n_mismatch |
|---|---|---|---|---|
| resident（常驻） | 32.41 | 10.01 | false | 5 |
| stream_blob（F_NOCACHE） | 8.39 | 5.67 | false | 5 |
| stream_blob（page cache 开） | 8.93 | 5.67 | false | 5 |

（resident 本次 32.4 偏高属运行间方差；早前稳定基线 ~17.5。即便对 17.5，stream_blob 仍慢 ~2×。`exact_match=false` 是 batch verify 近似，两路径一致，与本模式无关。）

## 读带宽（前置实验，已验证）
- SSD 冷顺序 6.84 GB/s；旧 mmap 散读 0.87 GB/s（碎片化软件墙）。
- **blob（每专家 1 连续 blob）+ 8 线程并行 pread = 6.6 GB/s**（7.6×，逼近峰值，K=10/64 稳定）。
- 即每 token 全 48 层 ~425MB / 6.6 GB/s ≈ 64ms，IO 本身能藏进 ~91ms 计算。

## 为什么吞吐没达标（根因）
1. **逐层 host 同步**：blob 路径每层 `inds.reshape(-1).tolist()` 把路由拉回 CPU 才能 pread；常驻的 GPU_REMAP 快路径恰恰消除了这个同步。48 层 × 每 token。
2. **每次 acquire 的 `mx.stack`**：把 K 个专家 9 个张量 stack 成 `[n,...]`，每层每 token 重做，分配 + 编排开销大。
3. 这与早前 native slot pool NO-GO 是**同一类问题**：decode 的逐层编排开销，IO 不是绑定约束。

## 已落地产物
- `mlx_streaming/prep/repack_expert_blobs.py`：`LAYERS=all` 重打包（48 层，stride 884736，16KB 对齐）。
- `mlx_streaming/core/blob_loader.py`：`BlobExpertSource`（并行 pread + F_NOCACHE + 主线程物化 + `acquire` 对齐 ResidentExpertPool + 后台预取 + 滚动窗口）。
- `mlx_streaming/core/streaming_moe.py`：`STREAM_BLOB=1` opt-in 路径 + 跨层预测触发 `prefetch_async`。
- `mlx_streaming/model_builder.py`：`STREAM_BLOB=1` 注入共享 blob 源。
- 测试：`test_repack_blobs / test_blob_loader / test_blob_prefetch / test_stream_blob_equiv`（6 passed），含与常驻路径等价、与 safetensors 0 误差、预取命中、滚动窗口驱逐。

## 追加：blob 接入常驻池 miss-loader（STREAM_BLOB_LOADER，更优解）

诊断"瓶颈是逐层编排而非 IO"后，改用更对的做法：**不另起 STREAM_BLOB 平行路径，而是把 blob 塞进常驻池的 miss-loader（`_raw_load_one`），复用 `acquire_gpu` 的 GPU-remap 快路径**（命中零 host 同步、池数组持久、每层仅 1 个标量同步），用小 `EXPERT_SLOTS` 换低内存。

实测（MAXTOK=64, K=2，单次、方差较大）：

| 变体 | spec tok/s | mlx_peak_gb | spec_hit_rate |
|---|---|---|---|
| 常驻 slots=256 | 30.05 | 10.01 | 0.992 |
| STREAM_BLOB（旧平行路径） | 8.4 | 5.67 | — |
| blob-loader slots=32 | 15.73 | 7.46 | 0.573 |
| blob-loader slots=64 | 10.94 | 5.67 | 0.728 |

要点：
- **blob-loader 比旧 STREAM_BLOB 快 ~2×**（15.7 vs 8.4），坐实"编排是瓶颈、复用 GPU-remap 是对的"。
- 仍不及常驻大池（30），差距来自**小池命中率低**（0.57-0.73）→ miss 层回退 host `acquire`（编排）+ blob 读。
- **内存下降有限（-25%~-43%）**：base 主模型（注意力/embedding/shared/MTP，~5GB）占大头，专家池本就不大；缩 slots 省的是专家那部分。
- tok/s 单次方差大（常驻自身 17~32 波动；slots=64 反比 32 慢属噪声/预取时序），不宜当单调曲线。
- 正确性：所有变体 `n_mismatch` 与常驻一致（数值等价）。

代码：`FileExpertStore._blob_loader`（`_raw_load_one` 命中即用 blob 单专家 pread），`model_builder` 在 `STREAM_BLOB_LOADER=1` 注入；测试 `test_blob_resident_loader`（与 safetensors 逐 key 相等）。

## 追加：C++/零拷贝物化 spike —— 负结果（已回滚）

设想：用 `os.preadv` 把 blob 字节直接散读进 MLX buffer 的可写 numpy 视图（`np.array(mx, copy=False)` 实测可写），省掉 `frombuffer→mx.array` 拷贝。**纯 Python 即可，无需 C++**，也绕开了当年的 `Invalid Resource`。

- **隔离 spike（预分配+复用+批量 eval）**：物化 0.64ms→0.26ms，**2.44×**，对 safetensors 0 误差。看着很好。
- **接进真实流后反而更慢**：slots=32 同配置 **9.86（lazy）→ 6.74（零拷贝）**。已回滚。

根因：要拿到可写 buffer，必须先 `mx.eval` 把 zeros 物化 → **每个专家一次 GPU 同步**；而常驻 loader 是「每专家调用 + lazy place」模式，这些 per-expert eval 同步的代价**高于**省掉的拷贝。旧的 `mx.array(np.frombuffer(...))` 是 lazy 的、批量 eval，反而更契合 MLX 的惰性求值。

教训：**spike 的加速是「预分配+批量 eval」的假象**，没复刻真实调用模式（per-expert + lazy place），所以一接入就翻车。零拷贝物化与 MLX 惰性求值模型相冲突——为写 buffer 而强制物化的同步，比它省的拷贝更贵。

## 追加 2：图内零拷贝 load primitive —— 同样负结果（已验证）

为彻底排掉"零拷贝物化"，实现了 `BlobLoadPrimitive`（C++/nanobind）：`eval_gpu` 里直接 `pread` 进 `mlx::allocator::malloc` 的 MLX 自有 buffer（无 kernel、无拷贝），整个 load 作为惰性图节点、批量 eval —— 这条路**同时避开了**外部 buffer 的 `Invalid Resource`（用 MLX 自有 buffer）**和** per-expert `mx.eval` 同步（load 进图）。

实测（warm, K=10，均 1 次批量 eval）：

| 路径 | 物化耗时 | 正确性 |
|---|---|---|
| `blob_load`（图内零拷贝 pread） | 0.503ms | 0 误差 |
| lazy `frombuffer+mx.array`（现有） | 0.527ms | — |
| → speedup | **1.05×** | |

结论：**零拷贝几乎没有收益(1.05×)**。因为在 MLX 的惰性批量求值下，`frombuffer+mx.array` 的那次拷贝本来就被藏好了、很便宜(~0.5ms/10 专家)。之前"物化 1.3ms"是强制 eval/单大数组的误导测量。**现有 lazy 路径已接近最优，零拷贝这条线关闭。**

`BlobLoadPrimitive` 保留在扩展里（正确、可用），但不值得接入。

## 最终总结（这条探索线）
- **读**：blob 0.87→6.6 GB/s（7.6×）✅ 真实有效。
- **物化/零拷贝**：与 lazy 批量持平 ❌（MLX 惰性批量已把拷贝藏好）。
- **小池吞吐瓶颈**：容量→miss→`.tolist` 编排，与 load/物化无关——真墙在此。
- **吞吐最优**：本机大池常驻 EXPERT_SLOTS=256 ≈ 28-32 tok/s。
- **内存大头**：主模型基座 ~5GB（非专家），要降内存得量化基座。
- **冲 24+ tok/s**：回 MTP 接受率，与 blob/物化无关。

## 追加 3:后台预取池预填(STREAM_BLOB_BG)—— 机制成立但端到端无收益

为攻"每 miss 的主线程编排",实现了后台预取池预填(用户提的多 stream 架构):
- gating 验证(probe_multistream_gate/handoff):**后台线程在独立 stream `s2` 上物化私有 array + eval,与主线程计算重叠、不崩**(我之前"后台不能物化"的判断是错的——只是缺 stream 上下文)。
- 落地:`BackgroundExpertPrefetcher`(s2 物化 + 交接;scales/biases 留 uint16,主线程 take 时再 `.view(bf16)`,因为 `.view` 在后台线程报 no Stream)、`FileExpertStore.promote_prefetched`(主线程把交接 array 写进池槽,**只用空闲容量、绝不驱逐**——否则会驱逐当前请求专家致 KeyError)、`STREAM_BLOB_BG=1` 接入 + 跨层预测 submit。等价测试 <1e-4。

端到端 A/B/C(2bit-g128 全层，MAXTOK=64/K=2):

| 变体 | tok/s | hit_rate | mlx_peak_gb |
|---|---|---|---|
| A 常驻 256 | 30.68 | 0.992 | 10.0 |
| B blob loader 32 | 13.33 | 0.55 | 7.4 |
| C bg 预填 32 | 8.15 | 0.55 | 7.4 |
| B blob loader 96 | 14.34 | 0.804 | 6.6 |
| C bg 预填 96 | 9.5 | 0.802 | 6.7 |

**结论:bg 预填正确但端到端无收益,反而更慢。** 根因:
1. **稳态 LRU 池已把工作集焐热**——demand 加载的专家会留在池里,几个 token 后命中率就收敛(cap=96 无 bg 已 0.804)。预填只是把同样的加载**提前一层**(边际),**抬不高命中率**(C 的 0.802 ≈ B 的 0.804)。
2. **预填只能用空闲容量**(修 KeyError 必须):小池无空闲→几乎不填;大池有空闲→但 LRU 本来就填满了。两头都没增量。
3. 跨层预测(每层 softmax/argpartition/eval)+ 后台线程 GIL/GPU 争用是**净开销** → 一致更慢。

代码保留(opt-in,默认关,正确),但不推荐启用。

## 最终最终结论(整条低内存流式探索)
- **读**:blob 0.87→6.6 GB/s ✅
- **物化/零拷贝**:与 lazy 持平,无收益 ❌
- **后台池预填**:机制成立但无收益(LRU 已焐热 + 开销)❌
- **吞吐最优**:大池常驻 ~30 tok/s
- **真正的墙**:稳态命中率由"池容量 vs 工作集"决定,与预取时机/物化方式无关;低内存=小池=低命中=慢,这是内在取舍。
- **降内存**:得量化主模型基座(~5GB 大头);**提吞吐到 24+**:回 MTP 接受率。两者都与本条线无关。

## 处置与下一步
- **默认关闭**（`STREAM_BLOB=0`）。作为"省内存模式"可选：内存紧张、可接受 ~2-4× 降速时启用。
- 要让流式逼近常驻吞吐，需做一个 **GPU-remap 风格的 blob 路径**：避免逐层 `.tolist` 同步与每次重 stack（把 slot 映射放 GPU、pool 数组复用而非每次 stack）。这是与 native slot pool 相同的硬骨头，单列。
- 内存想再降需量化主模型非专家权重（当前 ~5GB 基座主导）。

## 追加：同层(AHEAD=0)预测预取 —— 攻"窗口够不够"（2026-06-11）

上一版 bg 预填用的是 AHEAD=1(跨层，recall≈0.83) + "只填空闲"。本轮换思路：
- **AHEAD=0 同层预测**：在 decoder 层 `__call__` 开头(attention 前)用 `post_attention_layernorm(x)` 预测**当前层**专家（离线 probe 实测 recall 0.984@x2，远高于 AHEAD=1）。
- **只预取"预测∩非常驻"**（`_submit_missing_prefetch`，去重避免重载常驻、不撑爆窗口）。
- **promote 真正放进池**（去掉"只填空闲"限制；`_choose_victim` 永不驱逐 current 保证不 KeyError；同批就绪专家互保护）。
- **就绪率埋点**：`bg_stats.ready_on_time / not_ready` 直接量化"attention/GDN 窗口能否盖住物化"。

端到端 A(plain) vs C(同层预取)，2bit-g128 全层，MAXTOK=64/K=2：

| cap | A tok/s | C tok/s | C/A | hit_rate(A=C) | ready_rate | mlx_peak_gb |
|---|---|---|---|---|---|---|
| 32 | 11.28 | 9.43 | 0.836 | 0.55 | **0.000** | 7.4 |
| 64 | 14.33 | 10.71 | 0.747 | 0.71 | **0.000** | 5.67 |
| 96 | 18.20 | 18.41 | 1.012 | 0.804 | **0.006** | 6.64 |

（`exact_match=false / n_mismatch=5` 在 A、C 完全相同 → 是与参考的基线数值差异，**非 C 引入**；C 与 A 的 hit_rate/mismatch 逐位一致，证明同层预取数值等价、正确性 OK。）

**结论：同层预取在所有 cap 下都无法把缺失专家藏进 attention 窗口（就绪率 ≈0）。**
- **不是预测不准**（recall 0.984），是 **attention/GDN 窗口（亚毫秒级）远短于物化链路**：提交→后台线程调度→并行 pread→`np.frombuffer`→`mx.array`→`mx.eval` 同步。promote 时几乎从来没就绪（cap=32/64 是 0/6585、0/4341；cap=96 仅 19/3115）。
- **净效果**：小/中池 **净负**（后台线程 + 每层预测 softmax/argpartition/eval + promote 是纯开销，拖慢 16%~25%）；大池 **中性**（cap=96 本就少 miss，bg 几乎不干活，开销可忽略）。
- **量级事实**：48 层循环里，层 L 的就绪条目在后续 2 层内被 window 淘汰，而同一次前向里"层 L 提交→层 L MoE"间隔（attention+GDN）不足以完成 ~4 个专家的物化 → 几乎 100% miss 窗口。

### 追加：native-fused-prefetch（GPU 完成回调，2026-06-12）——零成本机制成功，2bit 净中性

**预取税拆解（实测，cap96/2bit，交错受控）：**
- 完整 Python 跨层预取 = **−32%** vs demand。
- 拆分：`PROBE_PREDICT_ONLY`(只 gate 前向 + eval barrier，不 .tolist/预取) = **−13%**；裸每层同步 = **−9%**。→ 税 ≈ 9% 每层同步 barrier + 4% 额外 gate + ~16% Python 编排(.tolist/线程池)。

**关键洞察**：`acquire_gpu` 第 357 行 `n_miss = int(mx.sum(...))` —— **解码热路径本来每层就有一次 host 同步**。demand 不是零同步，预取的税是**第二次同步**。→ 让预取**搭车这次已有同步**即可零额外成本。

**实现（`prefetch_on_complete` native primitive + `NATIVE_FUSED_PREFETCH=1`）：**
- MoE forward 里用下 AHEAD 层 gate 算预测 inds(lazy)，`prefetch_on_complete` 在**当前 command buffer 挂 Metal 完成回调**；dummy 折进 `inds`(加 0)→ `acquire_gpu` 的 `int(n_miss)` eval 它 → buffer 完成时**回调在 GPU 线程触发**，C++ 读已算好的 id + pread 预热下层专家字节。**主线程零额外同步。**
- de-risk 验证：回调读到的是**正确算出的 id**(非编码期垃圾)；真实跑 `fires=1410`、`exact_match=true`。

**结果（A demand vs NFP，cap96/2bit/MAXTOK=48 交错 ×3）：**

| | tok/s | hit |
|---|---|---|
| A demand | 12.7 / 13.5 / 15.07 (~13.8) | 0.81 |
| NFP | 13.42 / 12.19 / 14.38 (~13.3) | 0.81 |

**判决：零成本机制成功(NFP≈demand，对比 Python 预取 −32%)——难版 GPU 完成回调路线成立。但 2bit 下净中性：**
- NFP 只**预热字节**(page cache)，**不物化进池** → hit 不变(0.81) → 不把 miss 转 hit；
- 2bit 的 miss IO 本就便宜(read ~0.2ms)，预热省的可忽略 → tok/s ≈ demand。
- 要净赢需在回调里**物化进常驻池**(miss→hit)，但物化要建 MLX array、无法在 C++ 完成回调安全做 → **字节预热是这条路的天花板**。
- 何时会赢(外推)：miss IO 昂贵时(更大专家/更高 bit/冷盘 SSD-bound)，提前预热才有实质收益。2bit 暖缓存场景无。

**代码**：`NATIVE_FUSED_PREFETCH=1` opt-in 默认关；`prefetch_on_complete` primitive 已进 native 扩展。

### 追加：完整 miss→hit（GPU 完成回调零拷贝物化进池，2026-06-12）

把字节预热升级为真 miss→hit：de-risk 先证明**回调能把专家字节 pread 进预分配 MLX buffer，主线程 `quantized_matmul` 0 误差读到、不崩**（原始字节逐位一致、matmul max abs diff=0.0）——这正是之前 native backend NO-GO 的 buffer 所有权墙，在"预分配 buffer + 回调写入"受限形态下**可行**。

落地（`prefetch_into_staging` primitive + `NativeStagingManager`）：
- 回调把预测专家 pread 进 per-layer staging buffer + C++ 记录 (expert→row)；
- 目标层 MoE 前主线程读 C++ 记录（纯锁、无 GPU 同步）→ 切片 staging 行（惰性 view）→ 复用 `_place_expert` 写进池槽 + slot 表 → `acquire_gpu` 命中。
- **零主线程 pread / .tolist / np→mx.array**；`exact_match=true`、n_mismatch=0。

**结果（cap96 & cap32，A demand vs S miss→hit，交错）：**

| cap | A hit | S hit | 结论 |
|---|---|---|---|
| 96 | 0.81 | 0.81 | 命中**未升** |
| 32 | 0.543 | 0.54–0.55 | 命中**未升** |

**判决：机制完整成功且正确，但命中/吞吐均无提升——撞上同一堵墙：稳态命中由「池容量 vs 工作集」决定，与预取无关。** 根因:稳态下池满(LRU)，promote 一个预测专家就**驱逐一个同样要用的**→ 只是重排哪些常驻，无法增加常驻**数量** → 命中持平。预取唯一能加命中的前提是「池有空余」，而稳态没有。

**整条预取线最终定论（已穷尽所有形态）**：bg 预取、native 物化、GPU 完成回调字节预热、完整 native miss→hit —— 机制全部能跑通且正确，但**没有一个能超过纯 demand**。瓶颈始终是**池容量(内存)**，不是预取时机/成本/物化方式。要提速只能：① 更大池(更多内存)；② MTP 接受率。预取方向到此彻底结案。`NATIVE_FUSED_PREFETCH=1` 全部 opt-in 默认关。

### 追加：内存铁证 + staging 路径的竞态（2026-06-12，方向终结）

**内存铁证**：cap16 vs cap96 实测 mlx_peak **6.06GB vs 6.63GB**（rss 5.53GB）——把每层 cap 从 96 砍到 16 只省 **0.57GB**。说明：
- **内存被 80B-4bit 基座（attention/GDN/embed/lm_head/共享专家 + KV + MTP）占满，≈5.5GB 是地板**；2bit 流式专家池只占 ~0.5–1GB，**从来不是内存瓶颈**。
- 推论：**"丢 LRU、只留当前层"的滚动 buffer 架构**（用零成本预取支撑）逻辑成立，但**内存收益 ≤~1GB**（省的不是大头）→ 不值得。低内存目标其实任意 cap 流式都已达成（~6GB）；**要再降内存只能量化基座模型**，与预取/池策略无关。

**staging miss→hit 的运行时尝试（已修 norm，仍失败）**：
- 诊断到真 bug：运行时预测用了**错误层的 norm**（`gate_{L+a}` 作用在 `norm_L(x)` 上，而非探针验证的 `norm_{L+a}(未归一化 h)`）。已修（hook 存未归一化输入 + 目标层 norm）。
- 修后仍失败：① **覆盖未改善**（cap32 hit 0.49 ≈ demand 0.51）——预取放置没转成命中（时序/置换）；② **引入正确性竞态**：promote 切的是对 staging 的**惰性 view**，eval 滞后到池 scatter 时，下个 token 的 handler 已覆盖 staging → 读过期字节 → **n_mismatch=28（损坏）**。要修需 promote 时同步拷贝 / 双缓冲 staging。
- 鉴于内存铁证（即便修好且高覆盖也只省 ~1GB），先未修。后续按用户要求修对了（见下）。

### 追加：Bug2 修复（gen-匹配）+ Bug1 诊断（2026-06-12）

**Bug2 根因（证据驱动，排除两个错误假设）**：损坏**只在 GPU_REMAP=1 + 大 budget**；GPU_REMAP=0 或 budget=2 均正确；eval-copy 切片 / eval table+pool 都修不好 → **不是惰性 view 竞态、不是 eval 时序**。真因：**`self._last`(submit 时 Python 记 buffer) 与 `g_stg_ready`(handler 时 C++ 记 expert→row 映射) 解耦**——promote 可能拿"上次 handler 的映射"去切"这次 submit 的 buffer"，buffer↔映射错配 → 把专家 E 当成别的专家字节 → 损坏。
**修复**：handler 在 C++ 里**原子记录 (gen, 映射)**；Python submit 时把 `gen→buffer` 绑定；promote 按 handler 返回的 gen **取回正是它写过的那块 buffer**（取不到就跳过）。再叠加 promote 时批量 eval 切片（拷出 staging）。**实测 exact_match=true、n_mismatch=0**（GPU_REMAP=1 + budget=24），37 测试全过。

**Bug1（覆盖）仍是天花板**：修对正确性后，cap32 hit **0.50 ≈ demand 0.51**，无提升。埋点测得**运行时预测 recall 仅 0.66**（≪ 离线探针 0.95）——MTP draft/verify 路径下 live hidden 与离线快照不同，预测信号变弱。要再提升需**训练式 router predictor**（独立大活），且收益仍被内存地板锁死。

**结论**：native-fused-prefetch + staging miss→hit 现在**机制完整且正确**（zero-host-sync 预取 + GPU 完成回调零拷贝物化进池 + gen-匹配防错配），是一次干净的工程攻坚。但**净收益仍为零**：① 内存被基座锁死（省 ~1GB 不值）；② 运行时 recall 0.66 不足以让命中超过 demand。`NATIVE_FUSED_PREFETCH=1` opt-in 默认关，现已正确可用（但中性）。

### 追加：prefetch 没在 MTP 路径生效的 bug + 修复后净负（2026-06-12，终结）

**关键 bug（用户发现）**：submit 被钉死在 `x.shape[1]==1`（decode 单 token 的 GPU 快路径）。但项目恒开 MTP，verify 前向是 **seq=K+1（多 token），走 host 路径，submit 根本不触发** → 之前所有 "C≈A on spec_tok_per_s" 是假象（预取在 MTP 热路径上没干活；fires 来自 baseline 段）。这也解释了 recall 0.66：埋点测的是 decode 单 token（难口径），而非验证用的 verify 并集（易口径）。

**修复**：submit 提到 seq 分支之前（两条路径都触发）；预测按 seq 维 max 聚合成"K+1 token 并集"近似再取 top-budget（否则 verify 时 pred=(K+1)×budget 超出 staging 行数→越界 segfault）。修后 recall **0.66→0.83**、exact_match=true。

**但真正在 MTP 路径生效后 = 净负**（cap32, MTP, MAXTOK=48, 交错×3）：

| | tok/s(均值) | hit |
|---|---|---|
| A demand | ~17.7 | 0.543 |
| C prefetch(both paths) | ~8.2 (0.46×) | 0.518 |

慢一半、命中反降。两因：① **verify 热路径开销**（每层多 gate 前向+argpartition+promote 放置/驱逐，压在占 87% 的 verify 上 → 2× 慢）；② **容量墙抖动**（cap32 池满，recall 0.83 → ~17% 放错，promote 驱逐了同样要用的专家 → 命中下降）。

**修正（2026-06-12）——"净负"是 budget 调错，不是机制坏**：之前用 budget=24（严重过预取，每层 21MB pread + 24 次池放置）才得出 −2×。budget 扫描（cap32, MTP, MAXTOK=48）纠正：

| budget | tok/s | 备注 |
|---|---|---|
| A demand | 11.59 | — |
| 2 / 4 / 8 | 11.57 / 12.14 / 11.88 | **≈A，开销≈免费** |
| 24 | 8.72 | 过预取，−25% |

**开销随预取量线性；在 budget≈top_k(4–8) 时预取机制几乎免费**（与 seq=1 路径 C≈A 一致）。且 budget=8 下 **hit 0.543→0.606（预取真的转 miss→hit）**，tok/s ≈持平（命中收益 ≈ 小开销，噪声内）。

**预取最终定论（修正后）**：机制全部攻克且正确（zero-sync 回调、gen-匹配、并集预测、两路径触发），在**合理 budget(≈top_k)下开销近乎免费、能提升命中**——但 **tok/s 净收益仍≈0**（命中收益被小开销抵消），且**内存被基座地板锁死**（省 ~1GB 不值）。所以工程上成立、正确、近免费，但**对端到端吞吐/内存仍无实质净赢**。真正杠杆：**MTP 接受率** + **基座量化**。`NATIVE_FUSED_PREFETCH` 默认关，可作"低内存下不掉速地抬命中"的 opt-in。

**最终最终结论**：低内存（~6GB）已达成且被**基座模型**地板锁死；预取/池策略动不了它。两个真实杠杆：**量化基座非专家权重（降内存）**、**MTP 接受率（提速到 24）**——都与本条流式/预取线正交。

---

**最终判决（同层预取）**：方向（高 recall 同层预测）正确，但**被物理窗口卡死**。

### 实测：窗口 vs 物化链路（probe_materialize_latency + WINDOW_PROF，2bit-g128，N=4 专家）

| 链路 | 中位数 | 拆解 |
|---|---|---|
| **attention/GDN 窗口**（submit→promote，n=2160） | **0.070 ms (70 µs)** | decode(seq=1) 单层 attention/GDN |
| **物化链路**（demand load+eval） | **0.34 ms (340 µs)** | read_raw 0.22ms + `frombuffer`/`mx.array`/`mx.eval` 0.12ms |
| 后台 submit→ready（空闲 worker 端到端） | 0.34 ms | 线程调度 + s2 物化 + eval |

**物化(340µs)≈ 窗口(70µs)的 5×。** 物化本身不慢（0.34ms 完全 OK），慢的是它**塞不进 70µs 的同层窗口**——70µs 内后台 worker 甚至可能还没被调度到 → `ready_on_time≈0` 是必然，不是预测不准。

- **窗口由 decode 单层 attention 算力固定（~70µs），同层内无法拉长。**
- 要窗口够只能 **AHEAD=1**（层 L-1 期间预测层 L），窗口 ≈ 整层 ≈ 55ms/48 ≈ **1.1ms ≫ 340µs**，物理上能盖住物化；但 AHEAD=1 之前实测无净收益，**瓶颈不是窗口**而是「稳态命中率由池容量 vs 工作集决定」+ recall 0.83 + 线程开销（见上文 C 段）。

要让物化砍进 70µs 窗口，唯一出路是**真零拷贝 mmap→GPU、免 `mx.eval` 同步**，而这条已在上文验证为与 lazy 持平、无收益。**代码保留为 opt-in（默认关），不推荐启用。**

## 追加：跨-token 大窗口预取的命门测量（2026-06-11，门控失败）

同层预取被 70µs 窗口卡死后，换思路找「大窗口 + 强信号」：用**上一/前几次同层路由**（窗口可达整 token）作预测。但先用离线 probe 判生死——大窗口（历史）信号对真正需要预取的 **miss 子集** 的 recall 到底多少。

probe：`probe_crosstoken_miss_recall.py` + 纯函数 `crosstoken_recall`（`mlx_streaming/core/crosstoken_recall.py`）。跑真实 decode（`ROUTE_TRACE=1`，中池 + LRU），逐层取 `routed` 与 `miss=routed−resident`，用前 `history_n` 次同层 occurrence 的 routed 并集作 `pred`，算 `recall_full` 与 **`recall_miss`**（命门）。2bit-g128 全层，MAXTOK=64/K=2：

| cap | hit_rate | recall_full (n=1/2/3) | **recall_miss (n=1/2/3)** | avg_pred_size | avg_miss/层 |
|---|---|---|---|---|---|
| 64 | 0.674 | 0.47 / 0.55 / 0.62 | **0.003 / 0.005 / 0.009** | 17 / 26 / 33 | 4.0 |
| 96 | 0.725 | 0.47 / 0.55 / 0.62 | **0.000 / 0.000 / 0.000** | 17 / 26 / 33 | 3.1 |

**判决：门控失败（决定性）。** `recall_miss ≈ 0`，远低于 0.8 阈值；cap 越大越精确趋 0。

**这组数字是所有预取方向失败的根因铁证：**
- `recall_full ≈ 0.5`：历史信号能预测**约一半 routed**——但那是**复发专家，本就常驻（hit），不需要预取**。
- `recall_miss ≈ 0`：历史信号几乎**预测不到任何 miss**。miss 按定义是「新需要、最近没用过」的**新颖专家**，与历史正交。
- 即 **可预取的 ⊥ 需预取的**：`pred` 命中的全落在 resident 上，落不到 miss 上。放大 `pred`（n=3、52 个专家）也只把 cap=64 的 recall_miss 抬到 0.009，cap=96 仍是 0。

**两堵墙在此交汇，构成根本性（非偶然）壁垒：**
1. **大窗口 ⟹ 历史信号 ⟹ 预测不到 miss**（本节，recall_miss≈0）。
2. **能预测 miss ⟹ 同层 hidden（0.984）⟹ 窗口仅 70µs，盖不住 340µs 物化**（上一节）。

→ 跨-token（历史）信号不进运行时。**但这并不代表预测死路**——见下节：换成「同 token 上层真实 hidden」信号，对 miss 的 recall 直接翻天。

## 追加：同 token 跨层预测（真实 hidden）—— 命门突破（2026-06-11）

跨-token 用的是**历史**信号（预测不到新颖 miss）。换个信号源：用**当前 token** 第 `L-AHEAD` 层的**真实 hidden**，经第 L 层自己的 `post_attention_layernorm + gate` 预测第 L 层专家。real hidden 携带本 token 内容 → 理应能预测由内容驱动的新颖 miss。

probe：`probe_crosslayer_miss_recall.py`。中池 + LRU，MAXTOK=64/K=2，`recall_miss`（@mult=2，pred≈32 专家/层）：

| AHEAD | 预测窗口 | cap64 recall_miss | cap96 recall_miss | recall_full |
|---|---|---|---|---|
| 1 | ~1.1ms（整层） | **0.953** | 0.953 | 0.96 |
| 2 | ~2.2ms | 0.931 | 0.933 | 0.944 |
| 3 | ~3.3ms | 0.934 | 0.932 | 0.931 |

（mult=1 时 ahead1 recall_miss≈0.84/pred≈17；mult=4 时≈0.98/pred≈62。）

**这是命门突破，三要素首次同时满足：**
- **对 miss 准**：`recall_miss ≈ recall_full ≈ 0.95`（@x2），与跨-token 的 0.00 天壤之别——因为 miss 由当前 token 内容驱动，real hidden 编码了它。
- **窗口够**：AHEAD=1 窗口 ≈ 整层 ≈ **1.1ms ≫ 340µs 物化**；即便 AHEAD=3（窗口 3.3ms）recall_miss 仍 0.93。
- **与 AHEAD=0 同层方案的本质区别**：AHEAD=0 recall 0.984 但窗口仅 70µs（盖不住）；AHEAD≥1 用「上一层 hidden」换来 16×+ 窗口，recall_miss 只从 0.96 降到 0.93–0.95，**完全可接受**。

**结论：值得落运行时。** 推荐配置 **AHEAD=1~2、mult=2**（pred≈32、recall_miss≈0.93–0.95）。运行时方案：在 decoder 层 `L-AHEAD` 处用 `layers[L].mlp.gate(layers[L].post_attention_layernorm(h))` 预测第 L 层专家 → `pred∩非常驻` 异步预取（复用 `BackgroundExpertPrefetcher`/`promote_prefetched`/`_submit_missing_prefetch`，窗口=AHEAD 层）→ 第 L 层 MoE 前 promote。需实测的开放项：每层预取体量（pred∩非常驻可能 ~10–25）能否在 AHEAD 窗口内物化完（`ready_on_time` 率），以及端到端净收益与内存。

### 运行时落地实测（2026-06-11，机制成功 / 净吞吐持平偏负）

落地：抽 `_predict_layer_experts`（用**目标层** `post_attention_layernorm`，匹配验证配置），跑通 `STREAM_BLOB_BG=1 CROSS_LAYER_PREFETCH=1 AHEAD≥1` 的同 token 跨层预取。端到端 A(plain 中池) vs C，MAXTOK=64/K=2：

| 变体 | tok/s | hit_rate | ready_rate | mlx_peak_gb |
|---|---|---|---|---|
| A_plain_64 | 13.80 | 0.710 | – | 5.67 |
| C ahead1 ×2 64 | 9.94 | **0.892** | 0.986 | 5.67 |
| A_plain_96 | 12.40 | 0.804 | – | 6.63 |
| C ahead1 ×2 96 | 9.92 | **0.923** | 0.964 | 6.65 |
| C ahead2 ×2 96 | 10.62 | 0.915 | 0.999 | 6.66 |
| **C ahead2 ×1 BUD12 96** | **12.15** | **0.878** | 1.000 | 6.65 |

**机制完全成功，但净吞吐打平/偏负：**
- **预测+预取彻底奏效**：`ready_rate` 0.96–1.0（窗口完全盖住物化，验证 WINDOW≈4ms ≫ 340µs）；命中率 0.71→0.89、0.80→0.92，**首个真正把 miss 转 hit 的预取方向**。`exact_match` 与 A 逐位一致（数值正确）。
- **但 tok/s 不升反降**。根因：**过预取体量**。为拿 recall 0.95 用 x2 过预测（~32/层），实际只路由 ~10 → 后台多物化 **3× 冗余**专家，这些 GPU 工作与主计算抢占，代价 > 省下的 miss 延迟（2bit 专家 miss 本就只 85µs、极廉）。
- **收紧过预取（mult=1, budget=12）几乎追平**：cap96 达 12.15 tok/s（0.98×A）且 hit 0.878（+7pt），但仍不超过 A。

**最终判决（同 token 跨层预取）**：作为**信号/预测方向是成功的**（强、准、对 miss 准、窗口够），但**对吞吐无净收益**——因为 2bit 专家 miss 太便宜，隐藏它换不回后台 GPU 争用的成本。它真正改善的是**内存-质量权衡**：cap64 内存（5.67GB）下可达 hit 0.89（> cap96-plain 的 0.80），即**更低内存拿到更高命中**，代价是慢 ~15–25%。

**何时会变成净赢**（外推，未实测）：专家更大/更贵（高 bit、更大模型）使 miss 延迟 ≫ bg 争用成本时；或 demand miss 在关键路径上无法被现有重叠吸收时。对当前 2bit 极廉 miss 的设定，是平局。代码全部 opt-in，默认关。

### 追加：8bit 专家实测（2026-06-11，假设证伪：8bit 反而打垮内存+速度）

为验证「专家更大→预取净赢」外推，从 fp16（`/tmp/qwen3_next_80b_fp16`）重量化全 48 层 8bit-g128 专家（`requantize_from_fp16 BITS=8`，74GB），直接打包成 blob（新 `pack_blob_from_experts.py`，免 compute-buffer 中间体，对 blob_loader 数值校验逐位一致）。

| 配置(8bit) | tok/s | hit_rate | mlx_peak_gb | exact_match |
|---|---|---|---|---|
| plain cap24 | 2.35 | 0.356 | 18.45 | true |
| C ahead2 ×2 cap96 | 0.48 | 0.898 | 17.75 | true |
| （参照 2bit plain cap96） | 12.40 | 0.804 | 6.63 | – |

**判决：8bit 是低内存流式的非起点（hypothesis 在"能否跑起来"就失败）。**
- 专家体积 ×4 → 常驻池内存 ×4：cap24 都到 18.45GB、cap96 17.75GB，**远超 8GB 预算并触发换页**。
- **反常信号**：cap96(0.48) 比 cap24(2.35) 还慢、比 2bit 慢 5–25× → 典型的物理内存打爆 → swap。
- 预测/预取机制本身照常工作（ready_rate 1.0、hit 0.898、exact_match true），但被内存墙彻底压垮。
- **根因**：低内存流式的整个前提是「专家足够小」(2bit, 0.88MB/expert, miss 仅 85µs)。8bit(3.1MB/expert) 同时放大内存占用、I/O 体量、MLX buffer 抖动，三重打击。

**含义**：要更高质量又不破内存，方向不是 8bit，而是 **4bit/6bit 这种中间位宽**（2×/3× 2bit 体积），在 8GB 内找 cap-vs-质量 的甜点；或保持 2bit + 主模型基座做功。8bit 这条直接否掉。

### 追加：2/4/6 bit 的「每层 compute vs prefetch」crossover 实测（2026-06-11）

判据：每层 compute > 该层 prefetch(物化)时间 → 预取可被完全藏住 → 该 bit 可用。`ready_rate` 经验上等价该判据（=1 即全藏住）。从 fp16 重量化 2/4/6 bit 全层专家+blob，固定 cap=32（隔离 bit 效应），AHEAD=1/mult=2 实测：

**物化(prefetch)时间/专家**（submit→ready p50，blob 并行 pread+materialize）：

| bit | 字节/专家 | µs/专家 |
|---|---|---|
| 2 | 0.88MB | 79 |
| 4 | 1.67MB | 137 |
| 6 | 2.46MB | 166 |
| 8 | 3.5MB | 208 |

（≈ 线性：`36µs 固定 + 49µs/MB`。）

**端到端（cap=32, AHEAD=1, MAXTOK=16）：**

| bit | rss | mlx_peak | tok/s | **ready_rate** | 窗口(submit→promote) |
|---|---|---|---|---|---|
| 2 | 7.73GB | 8.4 | 10.27 | **0.975** | 4.1ms |
| 4 | 8.04GB | 13.9 | 4.74 | **0.746** | 9.9ms |
| 6 | 9.04GB | 19.3 | 1.81 | **0.638** | 26.9ms |

**关键结论：**
- **纯时间比 compute vs prefetch：prefetch 在所有 bit 都能被藏住。** 每层预取 ~10–14 专家 ×(79–166µs)= 0.8–2.3ms，远小于 compute 窗口（≥4ms）。**所以 compute-vs-prefetch 本身不是瓶颈，即便 6/8bit。**
- **真正的墙是内存+带宽，不是 compute/prefetch 比值。** 反证：4/6bit 的窗口反而**膨胀**到 9.9/26.9ms（系统整体变慢），若纯时间比，更大窗口该更能藏住；但 `ready_rate` 反而掉到 0.75/0.64 → 说明**后台预取被内存压力/带宽争用饿死**（unified memory 上 bg 物化 3× 字节抢主计算带宽 → 主计算也慢 → 双输）。
- **内存**：cap=32 下 rss 2bit 7.73GB（进 8GB）、4bit 8.04GB（临界）、6bit 9.04GB（超）。mlx_peak 更是 8.4/13.9/19.3GB。

**最终判决（max 可用 bit）：**
- **2bit**：唯一同时满足「预取干净藏住(0.975)+ 内存进 8GB(7.7GB)+ 速度保住(10.3 tok/s)」。
- **4bit**：内存临界(8GB 边缘)、预取开始掉(0.75)、速度腰斩(4.7) → **绝对上限/边缘可用**。
- **6bit+**：内存溢出、预取跟不上(0.64)、速度崩(1.8) → 不可用。
- **收益上限 bit = 2（4bit 为边缘）。** 注意：限制并非「compute 追不上 prefetch」（那条到 8bit 都成立），而是**位宽放大内存占用与内存带宽争用**——这才是真正的天花板。
