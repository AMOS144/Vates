# 设计：Qwen3-Next-80B MTP 接受率验证 Spike

日期：2026-06-07
状态：待评审

## 1. 背景与目标

流式 Qwen3-Next-80B-A3B-Instruct-4bit 已跑通（峰值 10.45GB、输出连贯），但**单解码速度只有 ~7.5 tok/s**，离 30 tok/s 底线远。两条常规提速路已被证明堵死：

- **独立 draft 投机解码**：mlx-lm 报 `Speculative decoding requires a trimmable prompt cache (got {'ArraysCache'})`——Qwen3-Next 的 gated-delta-net 线性注意力层用递归状态 cache（`ArraysCache.is_trimmable()==False`），拒绝草稿时无法裁剪回滚。
- **调大专家缓存**：实测 96→224 槽（多 5.5GB 内存）只把 6.57→7.51 tok/s，证明稳态是**计算受限**而非 IO 受限，堆缓存无效。

模型自带 **MTP（Multi-Token Prediction）** 是唯一可能的自投机加速手段，但 mlx-community 的 4-bit 移植把 MTP 权重整段删了，mlx-lm 全仓也没有任何 MTP 实现。完整重建 MTP 自投机循环工程量大、且有不确定性。

**本 spike 的目标**：用**最小代价**（只下 3.9GB、不写生成循环、不碰 cache 回滚）**实测 MTP 草稿接受率**，据此做出「是否值得投入完整自投机实现」的决策。

### 1.1 关键事实（已查证）

- 原版 `Qwen/Qwen3-Next-80B-A3B-Instruct`（BF16）**含 MTP 权重**，全部 1553 个张量**仅落在最后一个分片** `model-00041-of-00041.safetensors`（约 3.9GB）。只下这一个分片即可。
- MTP 结构（权重键已确认）：`mtp.pre_fc_norm_embedding`、`mtp.pre_fc_norm_hidden`、`mtp.fc`（2h→h）、`mtp.layers.0`（**全注意力** decoder 层：q/k/v/o_proj + q/k_norm + MoE 512 路由专家 + shared_expert + gate）、`mtp.norm`，logits 复用主模型 `lm_head`（`tie_word_embeddings=False`）。
- MTP 前向图（来自 vLLM / sglang / TensorRT-LLM 三方参考实现，互相印证）：
  ```
  emb = pre_fc_norm_embedding(embed(下一 token))
  hid = pre_fc_norm_hidden(主模型 last-layer hidden)
  x   = fc(concat([emb, hid], dim=-1))     # 拼接顺序：embed 在前、hidden 在后
  x   = MTP 全注意力 decoder 层(x)          # 因果注意力(KVCache 可裁剪) + MoE
  logits = lm_head(mtp.norm(x))
  ```
- MTP 层只有 1 个（`num_nextn_predict_layers=1`）→ 每步预测 1 个 token ahead；迭代喂回自身可抽多 token 草稿。**理论速度天花板约 2-3x**（即便接受率高，乐观把 7.5 顶到 15-22 tok/s，大概率仍不到 30）——此预期已与用户对齐。

### 1.2 范围

- **做**：下载并抽取 MTP 权重、在 MLX 实现 MTP 模块、teacher-forcing 测接受率、出数字与决策建议。
- **不做（YAGNI，全部留到 spike 通过后的完整实现）**：自回归生成循环、ArraysCache 快照/恢复回滚、多 token 迭代草稿、MTP 专家量化与内存优化、与流式 LRU 专家缓存集成。

## 2. 接受率的定义（teacher forcing，无生成循环）

取一段固定样本文本（token 序列 `t_0 … t_{L-1}`），跑一次主模型前向（teacher-forced）拿到每个位置的 last-layer hidden `h_i`。对每个位置 i：

```
emb_i = embed(t_{i+1})                         # 真实下一 token 的 embedding
x_i   = fc(concat(pre_fc_norm_embedding(emb_i), pre_fc_norm_hidden(h_i)))
```

把整段 `x_0 … x_{L-2}` 作为一个序列过 MTP 全注意力层（因果 mask）→ `mtp.norm` → `lm_head` → 每个位置取 argmax 得到对 `t_{i+2}` 的预测 `p_i`。

定义两个指标：
- **MTP→真实文本接受率** = `mean(p_i == t_{i+2})`：MTP 预测命中样本真实 token 的比例。
- **MTP→主模型贪心接受率** = 先用主模型贪心生成出参考序列 `g`，再以 `g` 为样本重跑上式，`mean(p_i == g_{i+2})`。这是**真正的自投机接受率代理**（投机时主模型用贪心/采样，草稿要对齐的是主模型的输出而非语料）。后者是决策主依据。

> 说明：teacher forcing 只验证「MTP 头预测下一个 token 的命中率」，不验证生成时的级联接受长度，但命中率是接受长度的直接决定因素，足以做 go/no-go 决策。

## 3. 组件设计

### 3.1 下载 + 抽取：`mlx_streaming/extract_mtp.py`（新文件）

1. 用现成的鲁棒下载方式（参照 `/tmp/dl_qn.sh` 的 ModelScope 逻辑：无 Range、按 size 校验、限速重试）只下 `model-00041-of-00041.safetensors` 到 `/tmp/qn_mtp_shard/`。
2. 用 `safetensors` 框架读（`safe_open` 惰性按键读取），只取 `key.startswith("mtp.")` 的张量。
3. 把路由专家按 mlx-lm `sanitize` 同样的约定 stack 成 `switch_mlp`：对 `mtp.layers.0.mlp` 的 `gate_proj/up_proj/down_proj`，把 512 个 `experts.{e}.{n}.weight` 堆成 `switch_mlp.{n}.weight`，形状 `(512, ...)`。
4. spike 阶段 MTP 专家保持 **BF16 常驻**（约 3GB，不在乎内存），不量化。
5. 把抽取并整理后的 MTP 权重存成单文件 `/tmp/qn_mtp_weights.safetensors`（含 `mtp.fc`、两个 pre_fc_norm、`mtp.layers.0.*`（switch_mlp 已 stack）、`mtp.norm`）。

### 3.2 MTP 模块：`mlx_streaming/qwen3_next_mtp.py`（新文件）

实现 `Qwen3NextMTP(nn.Module)`，**直接复用 mlx-lm `qwen3_next.py` 里的现成子模块类**（`Qwen3NextDecoderLayer` 的全注意力分支、`Qwen3NextSparseMoeBlock`、attention、RMSNorm），避免重写：

- `pre_fc_norm_embedding`、`pre_fc_norm_hidden`：`nn.RMSNorm(hidden, eps)`
- `fc`：`nn.Linear(2*hidden, hidden, bias=False)`
- `layer`：一个 Qwen3-Next **全注意力** decoder 层（注意力 + `Qwen3NextSparseMoeBlock`，含 shared expert）。从 mlx-lm 复用其类，构造时强制 `is_linear=False`（full attention）。
- `norm`：`nn.RMSNorm(hidden, eps)`
- 前向签名 `__call__(self, hidden, next_ids, lm_head, cache=None)`，按 §2 的图计算，返回 logits。注意力需要 RoPE 与 position（teacher-forcing 用完整因果序列，position 用 `arange`）。

`lm_head` 由调用方（主模型）传入复用。

### 3.3 接受率测量：`mlx_streaming/validate_mtp.py`（新文件）

1. `load(MODEL, lazy=True)` 主 4-bit 模型；**捕获 last-layer hidden**：mlx-lm `Qwen3NextModel.__call__` 末尾返回 `self.norm(hidden)`，需要 norm **之前**的 hidden。实现方式：复制其层循环，跑完所有 `layer(...)` 后**不**过 `self.norm`，直接拿 `hidden_states`（或临时 monkeypatch `model.model.norm` 为 identity 取一次，再还原）。
2. 构造 MTP cache（MTP 层是全注意力 → `KVCache()`）。
3. 按 §2 计算两个接受率指标。
4. 打印 JSON：`{mtp_vs_text_acc, mtp_vs_greedy_acc, n_positions, hidden_variant}`。
5. **hidden 取值消歧**：先用 final-norm **之前** 的 hidden（与 `pre_fc_norm_hidden` 的存在最自洽）。若接受率明显偏低（<30%，疑似接错），再试 final-norm **之后** 的 hidden，二者都报，取高者为准并在产出里标注 `hidden_variant`。

## 4. 测试（TDD，`mlx_streaming/tests/test_mtp_spike.py`）

不依赖 80B 真权重的小型单测（用随机小维度构造），验证管线正确：
1. **抽取约定**：给定伪造的 `mtp.layers.0.mlp.experts.{e}.{n}.weight`（小维度、少量专家），`extract_mtp` 的 stack 结果形状 == `(num_experts, ...)`，键名 == `switch_mlp.{n}.weight`。
2. **fc 拼接顺序**：构造已知 `emb`、`hid`、`fc` 权重，断言 `Qwen3NextMTP` 内部 `concat` 顺序是 `[emb, hid]`（与参考实现一致），即输出等于手算 `fc([emb;hid])`。
3. **前向形状/因果**：小维度 MTP 模块前向，输出 logits 形状 `(B, L, vocab)`；改第 i 位输入不影响 <i 位输出（因果性）。
4. **norm 复用**：`mtp.norm`、两个 `pre_fc_norm` 确实被调用（权重置零时输出随之改变的烟雾测试）。

## 5. 不改动 / 兼容性

- 主流式路径（`streaming_moe.py`、`run_streaming.py`、`expert_store.py`、`split_experts.py`、所有既有测试）**全不动**。
- MTP 模块与测量脚本是**纯新增旁路**，不接入生成主路径，不影响任何现有结论。

## 6. 风险与回退

- **风险**：hidden 取 pre/post final-norm 接错 → 接受率虚低。**缓解**：§3.3 第 5 点两种都试，取高者。
- **风险**：BF16 MTP 专家 + 4-bit 主模型数值口径差异拉低接受率。**判断**：spike 阶段可接受（量化只会进一步小幅降接受率，BF16 是接受率上界，用上界做 go/no-go 更稳妥）。
- **风险**：mlx-lm 子模块类不易脱离 `ModelArgs` 单独构造。**缓解**：构造一份只含必要字段的 `ModelArgs`（hidden/heads/experts/eps/rope 等）喂给复用的类。

## 7. 验收标准（spike 出口）

1. `test_mtp_spike.py` 全绿。
2. `validate_mtp.py` 跑通，打印出 `mtp_vs_greedy_acc` 与 `mtp_vs_text_acc` 两个数字。
3. 给出明确决策建议：
   - `mtp_vs_greedy_acc` ≳ 60% → **值得**做完整自投机（snapshot/restore 回滚 + 迭代草稿 + 流式集成），转 writing-plans。
   - 30%–60% → 边际，结合速度天花板再权衡。
   - < 30% → **不值得**，放弃 MTP 路，回到 35B 轹线或中档模型。
