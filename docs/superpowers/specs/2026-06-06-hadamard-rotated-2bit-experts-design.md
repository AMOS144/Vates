# 设计：Hadamard 旋转 2-bit 专家（救回低比特质量）

日期：2026-06-06
状态：待评审

## 1. 背景与目标

当前流式 MoE 的最优工作点是 2-bit 专家（`/tmp/mlx_qwen3_experts_2bit`，峰值 6~9GB），
但 2-bit 量化后**生成质量明显变差**，无法实用。目标：**在不增加内存/SSD 占用、不破坏
已有流式 + 投机 + 缓存结论的前提下，把 2-bit 的质量救回到可用**。

### 1.1 关键澄清（必须先纠正的概念）

用户最初的提法是「换成 TurboQuant / TQ3 权重」。经查证（arXiv 2504.19874 ICLR 2026、
Google Research 博客、全部社区实现）：

- **TurboQuant 是 KV-cache 压缩算法，不是权重量化方法**，不存在「TurboQuant 权重算子」。
- TurboQuant 里**唯一能迁移到权重侧**的核心思想是**随机旋转**（把张量旋转成近高斯、
  打散离群值后再量化）。这在权重量化的成熟对应就是 **QuIP# / QuaRot 的 Hadamard 旋转**。
- MLX 原生提供 `mx.hadamard_transform`，已实测支持本模型的 2048 与 768 维
  （768 = 2^6 × 12，走 m=12 路径），旋转-反旋转还原误差 ~5e-7。

因此本设计采用 **QuIP#-lite：单边输入维 Hadamard 旋转 + 均匀 2-bit 量化**，
它是 TurboQuant 旋转思想里**精确、自包含、零额外内存**的那部分。

### 1.2 范围

- **做**：离线旋转重量化专家、运行时旋转前向、质量验证。
- **不做**：双边旋转（B 方案，留作 A 质量不足时的加码）、SpinQuant 学习旋转（C 方案，
  YAGNI）、非均匀 Lloyd-Max 码本（`mx.gather_qmm` 只支持均匀仿射量化，无法落地）。

## 2. 数学原理（精确等价）

`mx.hadamard_transform(a, scale)` 计算 `(H_n · a) · scale`，其中 `H_n` 是 ±1 的非归一化
Hadamard 矩阵，满足 `H_n · H_n = n·I`。取 `scale = 1/√n`：

- 旋转两次还原：`(1/n)·H_n·H_n = I`。
- 权重预旋转：令 `W' = hadamard(W, 1/√n)`（沿 input 维），运行时 `xr = hadamard(x, 1/√n)`，
  则 `W'·xr = (H_n W^T/√n)^T · (H_n x/√n) = W·(H_n H_n/n)·x = W·x`。**数学精确**。

量化分组沿 input 维（group_size=64）。旋转把每组的能量打散成近高斯 → 组内动态范围缩小
→ 2-bit 误差大幅下降。这正是 QuIP# 的 incoherence processing。

实测 `W'·xr` 与 `W·x` 的最大误差 ~1e-3（纯 float32 累加重排噪声，非算法误差）。

## 3. 组件设计

### 3.1 离线：`mlx_streaming/rotate_requantize_experts.py`（新文件）

改自 `requantize_experts.py`。对每个 per-expert 文件、每个 proj：

1. `W = mx.dequantize(wq, scales, biases, src_group, src_bits)` 反量化回 float。
2. `Wp = mx.hadamard_transform(W, scale=1/√n_in)` —— `W` 形状 `(out, in)`，Hadamard 作用在
   最后一维（input 维），逐行旋转。
3. `nwq, ns, nb = mx.quantize(Wp, group_size=dst_group, bits=2)` 重新量化。
4. 写出同名文件（键名、形状、文件大小与 plain 2-bit 完全一致）。

各 proj 的 input 维：`gate_proj`/`up_proj` = `hidden`(2048)，`down_proj` = `moe_inter`(768)。

`_split_meta.json` 追加：
```json
"rotated": true,
"rotation": {"type": "hadamard", "scale": "1/sqrt(n_in)",
             "in_dims": {"gate_proj": 2048, "up_proj": 2048, "down_proj": 768}}
```

输出目录：`/tmp/mlx_qwen3_experts_2bit_rot`。

### 3.2 运行时：`mlx_streaming/streaming_moe.py` 新增 `RotatedSubGLU`

镜像 `SwitchGLU.__call__` 的 gate→silu→mul→down，只在两处插 Hadamard：

```python
xr = mx.hadamard_transform(x, scale=hidden ** -0.5)   # gate/up 共用输入旋转
g  = gate_proj(xr, local)
u  = up_proj(xr, local)
a  = silu(g) * u
ar = mx.hadamard_transform(a, scale=moe_inter ** -0.5) # down 前旋转中间激活
out = down_proj(ar, local)
```

- `QuantizedSwitchLinear` 内部 `gather_qmm` 不变，喂旋转权重 + 旋转输入。
- 复用 `PersistentSubGLU` 的对象缓存/原地 update 机制（避免重建 QSL）；旋转版只是替换
  最终的 `self._glu(x, local)` 调用为上面的旋转前向。
- 维度需匹配 `SwitchGLU` 的 `expand_dims` 约定，旋转作用在 hidden / moe_inter 维。

### 3.3 接线开关

`FileStreamingMoeBlock` 增加 `rotated: bool` 参数；`patch_model_filebacked` 透传。
`run_streaming.py` / `run_spec.py` 通过环境变量 `EXPERT_ROT=1`（或读 meta 的 `rotated`）
决定用 `RotatedSubGLU` 还是原 `PersistentSubGLU`。**非旋转路径行为完全不变**。

### 3.4 质量验证：`mlx_streaming/validate_rotation.py`（新文件）

以 **4-bit 专家输出为参照**（本地有，质量最高的现成参照），在同一批真实激活 token 上：

1. **逐块 MoE 输出对比**：plain-2bit vs rotated-2bit，各自相对 4-bit 的 MAE / 余弦相似度。
   断言 `MAE(rotated) < MAE(plain)`。
2. **端到端困惑度**：固定文本上分别用 4-bit / plain-2bit / rotated-2bit 跑前向，算
   token NLL → 困惑度，给出三个数，看 rotated 是否明显靠近 4-bit。

## 4. 测试（TDD，`mlx_streaming/tests/test_rotation.py`）

1. **旋转还原**：`hadamard(hadamard(x, s), s) ≈ x`（MAE < 1e-5），覆盖 2048 与 768。
2. **等价性**：`W'·(Hx) ≈ W·x`（相对误差 < 1e-2），覆盖两个维度。
3. **文件完整性**：旋转重量化输出文件可加载，键名/形状与 plain 2-bit 一致，文件大小相同。
4. **前向无 bug**：用 **4-bit 权重**做「旋转重量化 → RotatedSubGLU 前向」，与原始
   `SwitchGLU` 在同权重上的输出一致（MAE < 1e-3）—— 证明旋转管线本身不引入偏差。
5. **核心质量断言**：旋转 2-bit 的整块 MoE 输出 MAE（相对 4-bit）< 同配置 plain 2-bit。

## 5. 不改动 / 兼容性

- `expert_store.py`、LRU、热专家、投机（`run_spec`）、文件格式、`split_experts.py` 全不动。
- 旋转 = 「换一套专家文件 + 换一个前向」，可与既有结论（2-bit @ 96 槽、nd=2、0.6B draft）
  直接叠加，不需重测拓扑。

## 6. 风险与回退

- **风险**：单边输入旋转可能救不回足够质量（FFN 离群若也分布在输出维）。
  **回退**：升级到 B 方案（down 双边旋转，在专家加权求和后反旋转输出维）。
- **风险**：运行时两次 Hadamard 带来速度回退。
  **缓解**：O(n log n) GPU 算子、x 旋转每 token 仅一次；预期 <5% 回退，验证时实测确认。

## 7. 验收标准

1. `test_rotation.py` 全绿（尤其断言 5）。
2. `validate_rotation.py` 显示 rotated-2bit 的困惑度明显低于 plain-2bit、接近 4-bit。
3. `run_spec` 在 rotated-2bit 上跑通，tok/s 相对 plain-2bit 回退 <5%，内存不变。
