# 低内存全流式 MoE 模式 设计（Spec）

> 2026-06-11。目标读者：想理解"为什么这么做"的工程师。

## 一句话目标
新增一个**可选模式**：每层只临时持有"当前层 + 预取窗口"的专家，其余全部按需从 SSD 流式读取并即时释放。把峰值内存从 ~11GB 砍到 ~1-2GB，吞吐尽量逼近常驻方案（本机 ~17.5 tok/s）。默认仍走常驻，不影响现状。

## 背景与动机
- 当前常驻方案：每层维护 resident pool，把各层热专家常驻内存 → `spec_hit_rate=0.989`、~17.5 tok/s，但峰值 ~11GB（48 层 × 每层热集）。
- 用户诉求：内存大幅下降，且不要明显掉速。
- 关键前提已用实验验证（见下"已验证的事实"），这不是猜想。

## 已验证的事实（实测，2026-06-11）
1. **计算天花板 ~17.5 tok/s**：常驻方案 IO 已基本隐藏（hit_rate 0.989），所以 17.5 ≈ 计算上限。流式只能逼近、不能超过它（要更高得靠 MTP 接受率，另一条线）。
2. **SSD 顺序冷读 6.84 GB/s**；当前 mmap 散读只有 0.87 GB/s（碎片化软件墙，非 SSD 限制）。
3. **blob 格式 + 并行读 = 6.6 GB/s**（K=10/64 两次稳定，7.6× 提升，逼近 SSD 峰值）。
4. **M0 de-risk 通过**：blob 字节 → `np.frombuffer` → `mx.array` →（scales/biases `.view(bfloat16)`）→ `mx.quantized_matmul`，与 safetensors 参考**逐专家 0 误差**、不崩；后台读与计算可重叠（省 ~34%）。
5. **吞吐估算**：全 48 层 K=10 = 425MB/token，6.6 GB/s ≈ 64ms 读，叠在 ~91ms 计算后 → compute-bound，≈ 11-17 tok/s，内存 ~1-2GB。

## 核心设计

### 数据格式：每专家一个连续 blob（按层一个文件）
- 一个专家 = `[gate.w, gate.s, gate.b, up.w, up.s, up.b, down.w, down.s, down.b]` 连续，864KB，正好 16KB 页对齐。
- 按层一个 `layerXX.blob` + `layerXX.blob.index.json`（stride、段表）。
- 读一个专家 = `pread(stride, e*stride)` 一次（而不是当前 9 次散读）。

### 读取：并行 + 即时物化 + 不进 page cache
- 用线程池并行 `pread` K 个专家（提高 SSD 队列深度 → 接近峰值带宽）。
- 每个专家字节 → `np.frombuffer`(零拷贝视图) → `mx.array`(一次拷贝进 MLX) → scales/biases `.view(bfloat16)`。
- 用 `F_NOCACHE`(macOS fcntl) 打开 blob 文件，避免读过的字节堆进 OS page cache（这是真低内存的关键：否则字节虽不在 MLX 里、却堆在页缓存）。

### 计算：复用 MLX `quantized_matmul`（不用 native fused kernel）
- native fused MoE 已判 NO-GO（全模型慢 2.4×）。本模式喂给 MLX 的 `gather_qmm`/`quantized_matmul` 快路径。
- 数值与常驻路径一致（M0 已验证 0 误差）。

### 重叠：跨层预测驱动的后台预取
- 复用现有跨层预测（在算第 L 层时预测 L+1 的专家，已近 100%）。
- 后台线程在算当前层时，并行读下一层预测专家的 blob、物化成 mx.array。
- 算到下一层时直接用；预测命中则零等待，未命中才同步补读（冷 stall）。

### 内存：滚动窗口
- 只保留"当前层 + 预取窗口（如 1-2 层）"的专家 mx.array；层用完即释放。
- 总驻留 ≈ 窗口层数 × 每层 K 专家 ≈ 几十 MB（专家本体），加主模型非专家权重几 GB。

## 正确性底线
任何阶段，本模式输出必须与常驻路径**逐 token 一致**（`exact_match`）。M0 已证明单层 0 误差；集成后需端到端校验。

## 非目标（YAGNI）
- 不追求超过常驻的吞吐（那是 MTP 线的事）。
- 不改 native fused kernel（已 NO-GO）。
- 不追求纯零拷贝（1 次拷贝已够；纠缠 MLX allocator 内部得不偿失）。
- 不做跨层 blob 合并/按共激活排序（收益小、风险高）。

## 风险与已知取舍
- **预测未命中 = 冷 stall**：一次未命中要同步等 ~0.87MB/专家的冷读（~数 ms）。靠高预测精度 + 小兜底缓存缓解。
- **F_NOCACHE 行为**：需实测确认 macOS 上确实不缓存且不反伤带宽。
- **GIL 竞争**：`mx.array` 拷贝持 GIL，与主线程有竞争，重叠非 100%（M0 实测省 34%，可接受）。
- **磁盘占用**：blob 与现有 per-expert/safetensors 重复，全量 ~21GB。可在验证后用 blob 取代旧格式。

## 成功判据
- 端到端 `exact_match=true`。
- 峰值内存显著下降（目标 < 3GB）。
- 吞吐 ≥ 常驻方案的 ~80%（目标 ≥ 14 tok/s）。
