# MLX 流式 MoE：专家低 bit 重量化（4 → 3/2-bit）实测

日期：2026-06-06
模型：`mlx-community/Qwen3-30B-A3B-4bit`（48 层 MoE，每层 128 专家，top-8，hidden=2048，moe_inter=768，group64）
机器：Apple Silicon 统一内存
目标：用更低 bit 量化**流式专家权重**，缩小 SSD/内存占用、减小每次 miss 的读盘字节，验证对显存与速度的影响。

---

## 1. 动机与方法

模型本身已是 4-bit。要再压只能往 3/2-bit 走。专家是流式 I/O 的主体（≈14GB），所以**只压专家**、dense（注意力/router/norm）保持模型原始 4-bit 常驻——混合精度，把质量损失放在影响最小的地方。

工具 `requantize_experts.py`：对每个 per-expert 文件逐 proj 做 `mx.dequantize(4-bit) → mx.quantize(目标 bit)`，逐专家物化、低内存。`run_streaming` 新增 `EXPERT_BITS`/`EXPERT_GROUP` 覆盖，文件后端的 `QuantizedSwitchLinear` 按目标 bit 解码。

> **严谨性说明（重要）**：源权重已是 4-bit，这里是 `4bit → 反量化 → 2/3bit` 的**二次量化**，质量是「从原始 bf16 直接量化」的**下界**。本报告测的「大小/I/O/速度/显存」完全有效；质量只做粗检，严格评测需从原始 bf16 重量化。

单测（`test_requantize.py`）：低 bit 输出 packed weight 维度减半、结构完整、可反量化；同 bit 重量化近似幂等（MAE≈0.009 ≪ 4-bit 自身量化误差 0.078）。全套 13 个测试通过。

---

## 2. 磁盘/单专家大小

| 精度 | 目录总量 | 单专家文件 | 相对 4-bit |
|---|---|---|---|
| 4-bit（原） | 15G | 2.65MB | 1.00× |
| 3-bit | 12G | 2.07MB | 0.78× |
| 2-bit | 8.5G | 1.48MB | 0.56× |

**注意**：2-bit 不是严格 0.5×。`scales/biases`（每组一个 fp16，与 bit 无关）是定长开销，稀释了低 bit 收益。group64 下这部分占比不小；若要进一步压，可调大 `group_size`（如 128）来摊薄，但会略增量化误差。

---

## 3. 端到端扫描（同 prompt，MAXTOK=128）

### 同槽位（每层 16 槽）

| bit | tok/s | RSS 后(GB) | MLX active(GB) | MLX peak(GB) | 命中率 |
|---|---|---|---|---|---|
| 4 | 6.34 | 3.09 | 3.93 | 7.06 | 0.592 |
| 3 | 6.65 | 2.80 | 3.25 | 5.69 | 0.575 |
| 2 | 7.08 | 2.45 | 2.57 | 4.25 | 0.544 |

### 同槽位（每层 32 槽）

| bit | tok/s | RSS 后(GB) | MLX peak(GB) | 命中率 |
|---|---|---|---|---|
| 4 | 7.57 | 5.04 | 8.98 | 0.786 |
| 3 | 7.20 | 4.14 | 7.18 | 0.768 |
| 2 | 8.11 | 3.59 | 5.31 | 0.771 |

> 命中率随 bit 的微小差异并非缓存机制变化：router 是模型原始 4-bit、不变，但专家不同 → 隐状态/后续生成发散 → 路由序列不同。属正常噪声。

---

## 4. 关键结论

### ① 内存是最实在的收益（对症「省统一内存」这个核心诉求）
2-bit vs 4-bit（16 槽）：MLX 峰值 **7.06 → 4.25GB（−40%）**，active −35%，RSS −21%。这正是用户「哪怕装得下也要省出内存给别的软件」的目标。

### ② 速度只是温和提升，且非线性
2-bit 把专家字节砍 44%，但 tok/s 只 +12%（16 槽 6.34→7.08）。说明专家字节 I/O **不是**步时的绝对大头——还有固定开销（注意力 dense 计算、文件 open、gather kernel 启动、Python ≈每步几十 ms）。所以「字节减半→速度翻倍」的朴素估计不成立。

### ③ 等内存换缓存才是速度的正解
低 bit 的真正用法是「同等内存装更多专家」：

> **2-bit @ 32 槽（峰值 5.31GB，8.11 tok/s，命中 0.77）全面优于 4-bit @ 16 槽（峰值 7.06GB，6.34 tok/s，命中 0.59）**——更省内存、更快、命中更高。

把省下的内存换成更大的 LRU，命中率上去了，miss 次数减少，整体更快。这比单纯降 bit 拿到的速度多。

### ④ 质量（粗检）
即便是 2-bit 二次量化，简单 prompt 的输出仍连贯通顺（见各 run 的 `sample`）。但这只是单条粗检；难任务下 2-bit 二次量化大概率会掉，需用原始 bf16 重量化 + 困惑度/任务评测才能定论。

---

## 4.5 group_size=128 追加扫描（摊薄 scales/biases）

把 `group_size` 64→128，组数减半 → `scales/biases` 元数据减半。维度可整除（hidden=2048、moe_inter=768 都能被 128 整除）。

| 精度 | SSD(g64) | SSD(g128) | 单专家 g64→g128 | MLX峰值(16槽,g128) | tok/s | 质量粗检 |
|---|---|---|---|---|---|---|
| 2-bit | 8.5G | **7.6G** | 1.48→1.33MB(−10%) | 3.92G | 7.5 | 基本连贯，偶发个别乱码字 |
| 3-bit | 12G | 11G | 2.07→1.92MB(−7%) | 5.34G | 7.23 | 连贯 |

要点：
- **节省是定额的**：g64→g128 每专家恒省 ~147KB（就是 scales/biases 减半那部分，与 bit 无关）。所以对越小的文件占比越大（2-bit −10% > 3-bit −7%）。
- **2-bit g128 把 SSD 压到 7.6G ≈ 4-bit 的 0.51×**，已逼近理论 0.5×；相对原始 fp16（~60G）约 **8× 压缩**。峰值显存也进一步降到 3.92G。
- **2-bit g128 质量开始露出毛刺**（输出里偶发单个乱码字），说明 2-bit 下再粗化分组接近可用边界；**3-bit g128 仍干净**，是「省 SSD 又稳」的折中点。

## 4.6 提速：2-bit 槽位曲线与「代码天花板」（目标 ~15 t/s）

全常驻 4-bit 基线实测 **21.11 t/s**——这是流式的速度物理上限（流式只会在常驻基础上叠加 I/O，不可能更快）。2-bit 槽位扫描（MAXTOK=128，同 prompt）：

| 槽位/层 | tok/s | MLX峰值 | 命中率 | 常驻专家 |
|---|---|---|---|---|
| 32 | 8.65 | 5.31G | 0.771 | 1536 |
| 64 | 13.92 | 5.99G | 0.909 | 3069 |
| 96 | 14.81 | 7.14G | 0.925 | 3850 |
| 128 | 15.46 | 7.14G | 0.925 | 3850 |

- **命中率封顶 0.925 = 冷加载下限**：整段只触达 ~3850 个唯一专家，每个首次必 miss（3850/49152≈7.8%）。96 槽以上工作集已全装下，再加槽无效（resident_exp 锁定 3850）。
- **15.46 t/s = 流式代码天花板**（每专家只读一次盘、之后全命中）。它与常驻 21 t/s 的 ~5.5 差距是**纯 Python/GPU 同步开销**：每层每 token 的 `_unique_and_local`（`.tolist()` 强制同步）+ 建 dict + `_update_qsl`，48×128≈6000+ 次同步。非 I/O。
- **甜点（速度 vs 仍 offload）**：**64 槽 = 13.92 t/s，峰值 6.0G**，一半专家 offload，峰值比常驻基线 17.2G 省 11G。
- 想破 15→逼近 21：只能砍流式路径的 Python 同步开销，收益递减但全局受益。30 t/s 在本模型+本机不可达（高于 21 常驻天花板），需投机解码/MTP 或更小模型。

## 4.7 攻「代码天花板」：热路径 profile + 两项尝试（一过一否）

15.46 t/s 的流式代码天花板 vs 21 t/s 常驻，差距 ~5.5。对 128 槽（命中 0.99，几乎无 I/O）做分段 profile（段间插 mx.eval 屏障，看相对占比）：

| 段 | 占比 | 性质 |
|---|---|---|
| forward（MoE 矩阵乘） | 41.5% | 必要计算，常驻也要算 |
| routing（gate+softmax+同步） | 29.6% | 含必要的 gate 计算 + 1 次同步 |
| fetch（重堆叠+同步） | 26.4% | **流式特有**：每 token 把选中 k 专家 mx.stack |
| unique | 2.5% | Python 去重 |

**尝试①：减同步（已采纳）**。原热路径每层 3 处同步（`mx.eval(inds)` + `_unique_and_local` 内 `.tolist()` + `uniq.tolist()`）。改为只在 `inds` 上做一次 `.tolist()`，uniq/local 全在 Python 算。结果在 run-to-run 噪声（±1 t/s）内，严格说**不可证伪为提速**，但代码更干净、同步更少、最坏中性 → 保留。

**尝试②：持久堆叠 buffer 消除重堆叠（已否决）**。把每层缓存做成 `(capacity,O,I)` 常驻 buffer，`gather_qmm` 按行索引、仅 miss 时原地改行，想干掉 fetch 的 26%。实测**全面更差**：

| 引擎 | 32槽 | 64槽 | 96槽 | 128槽 |
|---|---|---|---|---|
| stack（原） | 11.4 / 5.3G | 14.6 / 6.0G | 14.6 / 7.1G | 14.9 / 7.1G |
| buffer | 9.8 / 5.9G | 9.2 / 7.5G | 6.9 / 10.9G | **5.9 / 14.3G** |

否决原因：① buffer 预分配 `capacity×层数` 整块，未填满的槽也占内存（128 槽→14G≈全常驻）；② 从大的 `(capacity,O,I)` 权重 gather 8 行，**比先 stack 出 8 行小张量再 gather 更慢**（大权重 gather 内存局部性差）。即「每 token 重堆叠 8 个专家」其实比「维护大 buffer」更便宜——**原 stack 路径已接近最优**。

**结论**：15→21 的差距主要是**不可约的逐层 MoE 计算 + 一次必要的路由同步**（流式必须把专家 id 同步到 CPU 才能取文件），训练-free 手段已基本到顶。要再快只能改计算（投机解码/MTP，需常驻）或换更小模型。**现实工作点仍是 2-bit @ 64 槽 ≈ 14 t/s、峰值 6G**。

## 4.8 投机解码 + 流式（独立 draft，已验证有效）

此前分析（投机解码对 I/O 流式无益）**被实测推翻**。target=流式 Qwen3-30B(2-bit,64 槽)，draft=常驻 Qwen3-0.6B-4bit(~0.4GB,同 tokenizer)，mlx-lm 原生 `stream_generate(draft_model, num_draft_tokens)`：

| 模式 | tok/s | 草稿接受率 | 命中率 |
|---|---|---|---|
| 不投机 | 12.71 | — | 0.909 |
| 投机 nd=2 | 17.27 (+36%) | 0.531 | 0.93 |
| 投机 nd=3 | **17.37 (+37%)** | 0.594 | 0.93 |
| 投机 nd=4 | 14.86 (+17%) | 0.547 | 0.93 |

- **为什么有效**：专家缓存命中 0.93 → verify pass 要的专家并集基本已在缓存，几乎不增 I/O；batched verify 把每步固定开销摊薄到「接受的多个 token」上。
- **代价极小**：+0.4GB 常驻 draft，仍 offload 一半专家。17.4 t/s 已达全常驻无投机基线(21)的 **83%**。
- **nd 甜点 = 3**；nd=4 起被拒草稿浪费增多、反降。
- 投机是精确解码（temp=0 下输出与不投机逐字一致），不损质量。

**对 MTP 的结论（重要）**：MTP 自投机本质是「完美对齐的微型自 draft」。但独立 0.6B draft 已用 0.4GB 拿到 +37%，MTP 相对它的增量很小（省那 0.4GB + 略高接受率）；而 mlx 上做 MTP 要 ~160GB 原始权重 + 从零实现 MTP 模块/验证循环。**性价比远不如独立 draft → 不建议为 MTP 投入**。

## 4.9 投机 × bits × 槽位 全局扫描（找最优工作点）

固定 draft=Qwen3-0.6B-4bit、nd=3、MAXTOK=128，扫 bits∈{2,3,4} × slots∈{32,64,96}。峰值含 0.4GB 常驻 draft。

| bits | slots | 不投机 t/s | 投机 t/s | 接受率 | 峰值GB |
|---|---|---|---|---|---|
| 2 | 32 | 10.69 | 15.06 | 0.594 | **5.92** |
| 2 | 64 | 15.75 | 19.58 | 0.594 | 8.18 |
| **2** | **96** | 15.97 | **23.77** | 0.594 | 9.33 |
| 3 | 32 | 9.75 | 9.94 | 0.586 | 7.81 |
| 3 | 64 | 13.00 | 15.05 | 0.586 | 10.97 |
| 3 | 96 | 13.74 | 20.13 | 0.586 | 12.59 |
| 4 | 32 | 10.08 | 6.77 | 0.500 | 9.64 |
| 4 | 64 | 12.55 | 10.96 | 0.500 | 13.61 |
| 4 | 96 | 13.53 | 14.41 | 0.500 | 15.51 |

两个关键发现：

1. **全局最优 = 2-bit @ 96 槽 + 投机 = 23.77 t/s @ 9.33GB**。它**反超了全常驻无投机基线(21 t/s)**，却只用约一半显存。原因：2-bit 文件最小、96 槽命中率最高 → verify 批几乎零额外 I/O，投机把固定开销摊薄后净赚。这是「又快又省」的甜点。

2. **投机的收益随「每次 miss 的 I/O 代价」反向**：bits 越低、槽位越多，投机越赚；**4-bit + 投机反而变慢**（6.77 < 10.08）。因为 4-bit 专家文件大、接受率又低(0.50)，verify 批触发的额外 miss 每次读得更多，I/O 吃掉了批处理的收益。**结论：投机只该叠在 2/3-bit 上，4-bit 该么不投机、要么直接全常驻。**

3. 3-bit 在任一轴上都不占优（同槽位比 2-bit 更慢且更吃内存），仅在「2-bit 质量不够」时作为质量挡位（3-bit @ 96 + 投机 = 20.13 t/s @ 12.6GB）。

**推荐工作点（按优先级）**：

| 诉求 | 配置 | tok/s | 峰值 |
|---|---|---|---|
| 极致速度（甚至超全常驻） | 2-bit @ 96 + 投机 | **23.8** | 9.3G |
| 速度/内存均衡 | 2-bit @ 64 + 投机 | 19.6 | 8.2G |
| 内存最省 | 2-bit @ 32 + 投机 | 15.1 | **5.9G** |
| 质量挡位 | 3-bit @ 96 + 投机 | 20.1 | 12.6G |

## 4.10 draft 大小：0.6B vs 1.7B（更大 draft 没换来净收益）

在最优 2-bit 专家上对比两个常驻 draft（同 tokenizer），扫 nd∈{2,3,4}：

| draft | slots | nd | 不投机 | 投机 | 接受率 | 峰值GB |
|---|---|---|---|---|---|---|
| 0.6B | 64 | 3 | 12.83 | 18.64 | 0.594 | 8.20 |
| 1.7B | 64 | 2 | 13.52 | **19.22** | 0.570 | 8.83 |
| 1.7B | 64 | 3 | 13.52 | 18.61 | 0.641 | 8.83 |
| **0.6B** | **96** | **2** | 14.92 | **25.27** | 0.531 | 9.60 |
| 0.6B | 96 | 3 | 14.92 | 24.44 | 0.594 | 9.60 |
| 1.7B | 96 | 2 | 14.98 | 23.28 | 0.570 | 10.27 |
| 1.7B | 96 | 3 | 14.98 | 22.41 | 0.641 | 10.27 |

- **1.7B 接受率确实更高**（nd=3：0.594→**0.641**），猜得更准。
- **但净 tok/s 没赢**：draft 自身贵 ~3×。**96 槽（最优档）1.7B 反而更慢**（23.28 < 0.6B 25.27），target 此时几乎不卡 I/O，draft 前向开销占比放大，接受率提升填不上。64 槽仅微赢（+3%）却多吃 +0.63GB。
- **规律（与 4-bit+投机变慢同源）**：投机净收益 = target 每步省下的成本 − draft 成本。**target 越快，越该用越小的 draft**。甜点配置 target 已够快 → 维持 0.6B。
- 附带：96 槽 nd 甜点是 **nd=2（25.27）** 而非 nd=3。

**结论：维持 0.6B draft，nd=2。**

## 4.11 Hadamard 输入旋转救 2-bit 质量（实测**负面**结论）

动机：2-bit 生成质量差，想用 QuIP#/QuaRot/TurboQuant 的「旋转打散离群再量化」救回来。
实现：离线 `W' = hadamard(W, 1/√n_in)` 沿 input 维预旋转后量化；运行时 `xr = hadamard(x)`、
`ar = hadamard(a)`，`gather_qmm` 数学等价 `W·x`（`RotatedSubGLU`）。

**先证运行时正确**（关键，排除 bug）：

| 配置 | 困惑度 |
|---|---|
| 4bit_plain | 16.875 |
| **4bit_rot** | **16.375**（≤ plain，近无损甚至略好）|

4-bit 下旋转近无损 → **旋转-反旋转管线数学正确**（单测 8-bit rel<2% 亦证）。

**但低比特实测旋转有害**：

| 配置 | 困惑度 | vs 4bit |
|---|---|---|
| 3bit_plain | 21.75 | +4.9 |
| 3bit_rot | 23.125 | **更差** |
| 2bit_plain | 24.625 | +7.8 |
| 2bit_rot | **45.25** | **崩** |

**为什么 MSE 更低却更差**（已查证）：聚合 72 个 proj，旋转后 2-bit **权重重构 MSE 反而更低
（0.93×）**、max/std 从 8.0 降到 4.8（确实压了权重离群）。但端到端更差——因为**只旋转权重，
却把激活离群摊到全部 2048 个通道**：原本激活离群只撞它那一列的量化误差，旋转后撞上 `W'` 全部
列的量化误差，在 2-bit 大误差下被放大。这是 **weight-only 旋转不配激活处理时的已知反噬**。

**根因**：QuIP#/TurboQuant 靠旋转得益，是因为配了**高斯最优的非均匀码本**（E8 格 / Lloyd-Max）。
旋转把分布变高斯，而 MLX 的**均匀仿射量化**在 2-bit（仅 4 个 level）下对高斯最不友好
（min/max 被尾部拉宽、中间挤成 1~2 个 level）。缺的正是那块非均匀码本，而 `mx.gather_qmm`
只支持均匀仿射 → 旋转这条路在当前算子下走不通。

**结论：放弃旋转救 2-bit。** 代码（`rotate_requantize_experts.py` + `RotatedSubGLU` + `EXPERT_ROT`
开关）已留存并通过单测（4-bit 上可用、近无损），但对 2/3-bit 质量是负收益，不投产。
**救质量的现实路径 = 直接上 plain 3-bit**（21.75，明显优于 2-bit 的 24.6），即下方"质量挡位"。

### 4.11.1 非均匀码本（Lloyd-Max）证伪：旋转这条路彻底堵死

承上，怀疑「均匀量化对高斯不友好」是 2bit_rot 崩的主因，于是验证 TurboQuant/QuIP# 真正的做法
——**旋转 + 高斯最优非均匀码本**。`gather_qmm` 跑不了非均匀码本，故用「把 Lloyd-Max-2bit 的*重构值*
存进 8-bit affine 容器、复用旋转 runtime」做零内核的质量证伪（`derisk_lloydmax.py`）。

Lloyd-Max N(0,1) 4-level 质心 = ±0.451 / ±1.51；旋转权重 2-bit 重构 MSE 比 affine **降 27.5%**。

| 配置 | 困惑度 |
|---|---|
| 4bit | 16.875 |
| 3bit_plain | 21.75 |
| **2bit_affine_plain** | **24.625** |
| 2bit_affine_rot（均匀）| 45.25 |
| **2bit_lloydmax_rot（非均匀）** | **32.748** |

- 非均匀码本把 45.25 救回到 32.75，**证实「均匀量化对高斯不友好」确是主因之一**；
- **但 32.75 仍输给最朴素的 plain affine 2-bit（24.6）**，更远不及 3-bit（21.75）。

**根因（结构性，码本无法解决）**：只旋转权重会把**激活离群摊到全部通道**，与权重量化误差相乘放大；
这与权重码本质量无关。best-case 标量码本仍比「不旋转」差 8 个困惑度点，E8 格再强也补不回这个结构损伤。

**最终判定：旋转（QuaRot/QuIP#/TurboQuant 思路）+ 任意码本，在「weight-only、不动激活/残差流」的
自包含改造下，对本模型 2-bit 是净负收益。这条路彻底堵死，不再投入。** 救质量唯一现实选择 = plain 3-bit。

### 4.12 混合精度（逐 proj 不同 bit）：质量/内存的真正甜点 ⭐

旋转救 2-bit 已堵死（§4.11），转而走**原生混合精度**——不是所有权重一样重要，把敏感的留高 bit、其余压 2-bit。

**第一步：逐 proj 敏感度探针**（`sensitivity_probe.py`，内存内全模型，把单个 proj 降到 2-bit、其余保 4-bit，看困惑度涨多少）：

| 把该 proj 降到 2-bit | Δ困惑度 vs 全 4-bit |
|---|---|
| gate_proj | **+0.625**（最不敏感）|
| up_proj | +3.625 |
| down_proj | +3.625 |

**推翻常识**：网传"down 最敏感"在本模型不成立——up 与 down 同样敏感，**gate 才是几乎免费可压 2-bit 的那个**。

**第二步：在流式路径上实测各分配方案**（`validate_mixed.py`，165-token 长文本；注意此处困惑度用的文本与 §4.11 不同，**只在本表内横比**）：

| 方案 | 逐 proj bit | 平均 bit | 磁盘 | 困惑度 | tok/s（96槽，无投机）| 峰值显存 |
|---|---|---|---|---|---|---|
| 全 2bit | 2/2/2 | 2.0 | 8.5G | 16.375 | 11.94 | 7.14G |
| **mixA** | gate2/up3/down2 | 2.33 | 9.6G | **13.562** | 12.04 | 7.72G |
| **mixB** | gate2/up3/down3 | 2.67 | 11G | **12.188** | 11.47 | 9.05G |
| 全 3bit | 3/3/3 | 3.0 | 12G | 12.75 | 11.74 | 9.64G |
| 全 4bit | 4/4/4 | 4.0 | 15G | 11.25 | — | — |

**关键结论：**
- **mixB 在质量、磁盘、显存三项上全面碾压「全 3-bit」**（困惑度 12.19<12.75，磁盘 11G<12G，峰值 9.05G<9.64G，速度持平）→ **没有任何理由再用均匀 3-bit**。
- **mixA 以 9.6G（仅比 2-bit 多 1.1G）把困惑度从 16.375 救到 13.562**，速度还略升 → 极致省内存时的甜点。
- tok/s 各档都在 11.5~12（无投机），**bit 分配几乎不影响速度**：瓶颈在每步开销而非字节数，故混合精度对速度近乎免费。

**叠加投机解码**（mixB + 常驻 0.6B draft，nd=2，96 槽）：**22.2 tok/s @ 峰值 11.9G**（含 draft），接受率 0.52。即「近 3-bit 质量 + 全常驻 4-bit 都达不到的速度 + 比 3-bit 还省的显存」。

**实现（全原生 affine，无新算子）：** `requantize_experts.py` 支持逐 proj bit（`gate=2,up=3,down=3`）；runtime 的 `PersistentSubGLU`/`FileStreamingMoeBlock`/`patch_model_filebacked` 新增 `proj_bits` 参数，按 proj 各建对应 bit 的 `QuantizedSwitchLinear`；`run_streaming.py`/`run_spec.py` 从专家目录 `_split_meta.json` 的 `dims.proj_bits` 自动读取。单测见 `tests/test_mixed_precision.py`。

### 4.13 逐层混合精度：首尾层可压，给出新的帕累托点

在 §4.12（逐 proj）之上再叠**逐层**：不同层给不同 bit。先做逐层敏感度探针（`PROBE=layer`，48 个 MoE 层，首/尾各 4 层为边界）：

| 降到 2-bit 的层 | 困惑度 | Δ vs 4bit |
|---|---|---|
| 只压**中间 40 层** | 22.375 | +5.25 |
| 只压**首尾 8 层** | 16.875 | **−0.25** |

**再次推翻常识**：本模型**首尾 MoE 层对 2-bit 几乎免疫**（压了反而略降，噪声内），中间层才扛敏感度——与"首尾层敏感"的通行说法相反。

据此造 **mixL**：中间 40 层 = mixB(g2u3d3)，首尾 8 层 = 全 2bit。流式路径实测（同 §4.12，165-token）：

| 方案 | 平均bit | 磁盘 | 困惑度 | tok/s（96槽）| 峰值显存 |
|---|---|---|---|---|---|
| 2bit | 2.0 | 8.5G | 16.375 | 11.94 | 7.14G |
| mixA g2u3d2 | 2.33 | 9.6G | 13.562 | 12.04 | 7.72G |
| **mixL** 中mixB/首尾2bit | 2.56 | 10G | 12.938 | 11.38 | **8.30G** |
| mixB g2u3d3 | 2.67 | 11G | 12.188 | 11.47 | 9.05G |
| 3bit | 3.0 | 12G | 12.75 | 11.74 | 9.64G |

**关键结论：**
- **mixL 给出一个真实的新帕累托点**（10G / 困惑度 12.94），介于 mixA 与 mixB 之间，质量约等于 3-bit 但峰值显存少 1.3G（8.30 vs 9.64）。
- 但 **mixL 并不碾压 mixB**：长文本上 mixL（12.94）略逊 mixB（12.19），换来 −1G 内存。短文本探针曾误报"mixL 略胜 mixB"，是 bf16 粗粒度噪声所致——**以长文本流式实测为准**。
- **最终帕累托前沿**：`2bit → mixA → mixL → mixB`，每档约用 ~0.6G 换质量。**均匀 3-bit 被 mixB 和 mixL 双重支配**（质量、显存都更差/更贵），彻底可弃。

**实现**：`requantize_experts.py --layered`（首尾 bnd 层用一种 bit、其余另一种）+ `requantize_dir_layered`/`boundary_scheme`；meta 写 `dims.per_layer_proj_bits`；runtime `patch_model_filebacked(layer_proj_bits=...)` 逐层建对应 bit 的 QSL；`run_streaming`/`run_spec`/`validate_mixed` 自动读取。单测见 `tests/test_mixed_precision.py`。

**逐专家（hot/cold）未做**：`gather_qmm` 要求一次 gather 内专家 bit 一致，热/冷混 bit 须把每层每 token 的 gather 拆成两组矩阵乘 → launch 数翻倍。而前面已证 tok/s 瓶颈在每步 launch 开销而非字节数，**此拆分大概率净伤速度**，故暂不投产（除非未来有「同 batch 多 bit」的 gather 内核）。

## 5. 建议

- **全局最优（4.9 实测）**：**2-bit @ 96 槽 + 独立 draft 投机 = 23.8 t/s @ 9.3GB**，速度反超全常驻基线(21)且显存省一半 → 首选。
- **投机只叠 2/3-bit**：bits 越低、槽位越多，投机越赚；4-bit + 投机反而更慢，别叠。
- **要省内存**：直接上 **2-bit 专家**，峰值显存立省 ~40%，质量在常规生成上可接受 → 最划算。
- **要又省又快**：**2-bit + 更大槽位**，在比 4-bit 更低的显存下拿到更高命中与更高 tok/s。
- **想再压 SSD**：`group_size` 64→128 已实测：2-bit 8.5G→**7.6G**（≈4-bit 的 0.51×），3-bit 12G→11G。但 **2-bit g128 质量开始出毛刺**，建议 SSD 极限走 **3-bit g128（11G，质量稳）** 或接受 2-bit g128 的偶发瑕疵。
- **要严格质量**：从原始 bf16 `Qwen3-30B-A3B` 直接量化到 2/3-bit（而非二次量化），再跑困惑度/任务集对比。
- **2-bit 质量不够：优先混合精度（§4.12），别再用均匀 3-bit。**
  - **质量挡 = mixB（gate2/up3/down3，平均 2.67bit，11G）**：困惑度全面优于 3-bit，且磁盘/显存更省 → 取代均匀 3-bit。
  - **省内存挡 = mixA（gate2/up3/down2，平均 2.33bit，9.6G）**：质量远好于 2-bit，仅多 1.1G。
  - 关键洞察：**gate_proj 几乎免费可压 2-bit，up/down 才敏感**（与"down 最敏感"的常识相反，§4.12 实测）。
  - **逐层再叠（§4.13）= mixL（中间 mixB / 首尾 8 层 2bit，10G）**：新帕累托点，质量≈3bit 但峰值显存少 1.3G；首尾层对 2-bit 几乎免疫（与"首尾敏感"常识相反）。
  - **逐专家（hot/cold）不建议**：`gather_qmm` 同 batch 须同 bit，混 bit 要拆 gather → launch 翻倍、净伤 tok/s（§4.13）。
- **TurboQuant/RotorQuant/旋转类 已证负面（§4.11）**：RotorQuant 压的是 KV cache 与专家权重无关；
  Hadamard 输入旋转 + MLX 均匀量化在 2/3-bit 实测**有害**（2-bit 困惑度 24.6→45.25），因为缺高斯
  最优非均匀码本而 `gather_qmm` 跑不了。**这条路在当前算子下已堵死，不要再尝试**；除非将来自己实现
  非均匀码本 dequant kernel。

## 6. 复现

```bash
# 重量化（4-bit → 3/2-bit），逐专家、低内存
python3 -m mlx_streaming.requantize_experts /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_3bit 3 64
python3 -m mlx_streaming.requantize_experts /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_2bit 2 64

# 扫描（示例：2-bit，32 槽）
EXPERT_DIR=/tmp/mlx_qwen3_experts_2bit EXPERT_BITS=2 EXPERT_GROUP=64 \
  EXPERT_SLOTS=32 MAXTOK=128 python3 -m mlx_streaming.run_streaming

# §4.11 旋转实验（负面结论，复现用）
python3 -m mlx_streaming.rotate_requantize_experts /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_2bit_rot 2 64
python3 -m mlx_streaming.validate_rotation   # 输出 4bit/2bit_plain/2bit_rot 三档困惑度

# §4.12 混合精度（推荐）
python3 -m mlx_streaming.sensitivity_probe   # 逐 proj 敏感度（gate 便宜、up/down 敏感）
python3 -m mlx_streaming.requantize_experts /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_mixA gate=2,up=3,down=2 64
python3 -m mlx_streaming.requantize_experts /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_mixB gate=2,up=3,down=3 64
python3 -m mlx_streaming.validate_mixed      # 2bit/mixA/mixB/3bit/4bit 困惑度（自动读 proj_bits）
# 部署：混合专家 + 投机解码（proj_bits 从目录 meta 自动生效）
EXPERT_DIR=/tmp/mlx_qwen3_experts_mixB EXPERT_SLOTS=96 NDRAFTS=2,3 python3 -m mlx_streaming.run_spec

# §4.13 逐层混合（首尾 4 层 2bit + 中间 mixB）
PROBE=layer BND=4 python3 -m mlx_streaming.sensitivity_probe       # 逐层敏感度（首尾免疫）
PROBE=mixlayer BND=4 python3 -m mlx_streaming.sensitivity_probe    # 逐层+逐proj 组合
python3 -m mlx_streaming.requantize_experts --layered /tmp/mlx_qwen3_experts /tmp/mlx_qwen3_experts_mixL 4 gate=2,up=3,down=3 gate=2,up=2,down=2 64
EXPERT_DIR=/tmp/mlx_qwen3_experts_mixL EXPERT_SLOTS=96 python3 -m mlx_streaming.run_streaming   # per_layer_proj_bits 自动生效
```
