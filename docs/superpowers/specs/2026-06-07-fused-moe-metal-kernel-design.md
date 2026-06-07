# Fused 量化 SwiGLU MoE Metal Kernel — 设计 spec

日期:2026-06-07
目标读者:接手实现的工程师(假设不了解本项目背景)

## ⛔ 最终结论:Phase 0 de-risk = NO-GO(冲 30 失败,已止损)

`mlx_streaming/probe_fused_moe.py` 实测(2-bit affine, hidden=2048, moe_inter=512, k=10):

| 场景 | mlx gather_qmm(gate+up+swiglu) | 手写 fused kernel | 加速比 | 数值 |
|---|---|---|---|---|
| decode T=1(吞吐关键) | 0.58 ms | 0.90 ms | **0.64×(更慢)** | 完全一致 diff=0 |
| verify T=3 | 0.39–1.0 ms(抖动) | 0.90 ms | ~1.1×(噪声内) | 完全一致 |

- 数值**完全正确**(2-bit 解包 + affine 反量化 + SwiGLU 与 mlx 逐位一致,diff=0),已加共享内存缓存 `x`。
- 但**吞吐关键的 decode(T=1)仍比 mlx 慢 36%**:单次 fused dispatch(0.90ms,仅 gate+up)就比 mlx 两次 dispatch(0.58ms)还慢 → 瓶颈是**每线程标量 MAC 的计算效率**,不是 dispatch 数量。
- 要赢必须用 `simdgroup_matrix` 重写 tiled SIMD 量化 GEMM(=重造 Apple 的 quantized GEMM),多周高风险,且净增益仅"省一次中间物化",乐观也只到与 mlx 持平,达不到冲 30 所需的 1.5–2×。
- **决策**:`FUSED_MOE` 不接入主流程;**24.7 tok/s @ 14.6GB 为本机本栈实际上限**。de-risk(~2h)成功避免了多周沉没成本。

下面为原设计内容(留档)。

---

## 一句话目标

为流式 MoE 的专家前向手写一个 **fused Metal kernel**:把"按 slot 收集专家权重 → 2-bit 反量化 → gate/up 矩阵乘 → SwiGLU → down 矩阵乘"**融合成尽量少的 kernel launch**,减少 batch=1 解码下的调度开销与中间张量物化,争取把 MTP 路径从 ~24.7 推到 30 tok/s。

## 为什么做这个(背景与已测数据)

- MTP 投机解码在本机已验证有效:`EXPERT_SLOTS=256, K=2, MTP_VERIFY_MODE=batch` → **~24.7 tok/s(2.43×)**,且 256 槽时 `disk_load_ratio=0.14`,**I/O 已基本消除**。瓶颈从 I/O 转为**计算/调度**。
- de-risk(`probe_moe_ceiling.py`):把专家计算短路成 0,单 token 解码从 8.0 → 14.1 tok/s,即**专家计算占单路前向 ~43%**。这是 fused kernel 能攻的面。
- 微基准已证伪 route/combine 优化(单算子不贵,profiler 占比是 eval 屏障假象);也确认 `_update_qsl` 只重绑引用、不拷贝整池,无纯 MLX 便宜杠杆。
- 结论:要继续提速只剩**减少 MoE 的 kernel launch 数 + 中间物化**这条硬路。当前每层 MoE ≈ 3× `gather_qmm` + SwiGLU + down ≈ 5-6 次 launch,×48 层 ≈ 288 次/token。融合到每层 1-2 次有望显著降调度开销。

## 诚实的风险与上界

- **奖金有限且不确定**:攻 ~43%(单路)/更低(verify 因专家 union 次线性)。乐观把这块做到 2×,MTP 24.7→~30,属于 **50/50 的赌**。
- mlx 的 `gather_qmm` 本身是 tuned Metal;**单个 proj 很难赢**,收益必须来自**跨 proj 融合**(避免 moe_inter 尺寸中间量往返 global memory)。
- 纯 MLX 的 gather_qmm 融合此前试过、反而更慢 → 必须手写 Metal(`mx.fast.metal_kernel`)。
- 因此**强制 de-risk 优先**:先写最小融合原型测真实加速,GO/NO-GO 后再建完整 kernel。

## 数据格式(实测)

每个专家(2-bit affine,group_size=64):
- `gate_proj.weight`/`up_proj.weight`:`(moe_inter=512, 128)` uint32(每行 128 个 uint32 × 每 uint32 16 个 2-bit 值 = 2048 = hidden)
- `gate_proj.scales`/`biases`、`up_proj.scales`/`biases`:`(512, 32)` bf16(32 = 2048/64 组)
- `down_proj.weight`:`(hidden=2048, 32)` uint32(32×16=512=moe_inter)
- `down_proj.scales`/`biases`:`(2048, 8)` bf16(8 = 512/64 组)

常驻池张量:每层 `(capacity, *上面形状)`;`acquire(layer, expert_ids)` 返回 `(pool_arrays, slots)`,`slots[i]` 是第 i 个被选专家在池中的行号。

反量化(affine):`w_float = w_q * scale[group] + bias[group]`,其中 `w_q ∈ {0,1,2,3}`,`group = i // 64`。

decode 输入:`x=(1,1,2048)`,`local/slots` 形状 `(1,1,k)`,k=top_k=10。verify 时 S=K+1。

## 数值等价基准

fused kernel 输出必须与现有 `PersistentSubGLU.forward`(即 `SwitchGLU` + 3×`QuantizedSwitchLinear` + `gather_qmm`)在相同输入上 `allclose(atol=适配 2-bit/bf16 的容差,如 1e-2)`。这是所有 spike/实现步骤的 ground truth。

## 分阶段(de-risk 门控)

- **Phase 0 — spike(本 spec 的核心,先做)**:写最小 fused kernel 只算 **gate+up+SwiGLU → 激活 `(t,k,512)`**(不含 down),对比 mlx 的 gate/up/swiglu 路径:① 数值 allclose;② 实测 ms。**GO 条件:fused ≥ 1.5× 于 mlx 同段**,否则止损、回报告写明 NO-GO。
- **Phase 1**(仅 Phase 0 GO 后):把 down_proj 也融进同一/相邻 kernel,产出最终 `(t,k,2048)`;数值 allclose;端到端 `probe_moe_ceiling` 复测专家段耗时。
- **Phase 2**(仅 Phase 1 GO 后):接进 `FileStreamingMoeBlock`(新增 `FUSED_MOE=1` 开关,默认关),全测试绿 + `run_mtp_spec` 实测 MTP tok/s,确认逼近/达到 30 且数值正确。

## 不做(YAGNI)

- 不支持非 2-bit/非 affine/非 group64(仅针对当前 2-bit 专家;其它档保持原 `gather_qmm` 路径)。
- 不在 Phase 0/1 接入主流程,避免污染默认路径与测试。
- 不追求 prefill(大 S)最优;decode/verify(小 S)是目标场景,大 S 回退原路径。

## 回退

任何阶段 NO-GO 或数值不达标:`FUSED_MOE` 保持默认关闭,默认路径不受影响;在报告中记录 spike 的真实加速数字与结论。
