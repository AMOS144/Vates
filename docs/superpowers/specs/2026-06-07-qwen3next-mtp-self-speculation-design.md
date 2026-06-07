# 设计：Qwen3-Next-80B 完整 MTP 自投机解码

日期：2026-06-07
状态：待评审
前置：spike 已验证 MTP 草稿接受率 ≈ 0.748(见 `benchmarks/reports/qwen3next-mtp-acceptance-2026-06-07.md`)

## 1. 背景与目标

流式 Qwen3-Next-80B-A3B 单解码 ~7.5 tok/s。模型自带 MTP 头(1 层、512 专家),
spike 实测对主模型贪心输出的下一 token 接受率 0.748。本设计实现**完整的 MTP 自投机
解码循环**,把接受率转化为实际加速,目标 **~2–2.7x(≈15–20 tok/s)**,内存增量控制在
~1GB 以内。

### 1.1 已定决策

- **MTP 专家精度**:4-bit 常驻(~0.8GB)。抽取的 BF16 专家在加载时用 `nn.quantize`
  量化(与主模型同方案:group_size=64、bits=4;`mlp.gate` 与 `shared_expert_gate` 8-bit)。
- **解码模式**:只做**贪婪(argmax)**。贪婪下投机解码与非投机**逐 token 等价**,
  这是首要正确性铁律。不做采样(概率拒绝采样)——YAGNI。
- **hidden 变体**:主模型喂给 MTP 的 hidden 取 **final-norm 之后**(spike 实测 0.748 > 0.728)。

### 1.2 范围

- **做**:MTP 4-bit 量化加载、暴露主模型 hidden 的前向、MTP 自回归抽 K 草稿、
  主模型并行验证、cache 快照/恢复回滚、贪婪等价性验证、端到端 tok/s + 内存基准。
- **不做**:采样模式、MTP 专家流式、多 MTP 层(模型只有 1 层)、batch>1、训练。

## 2. MTP 索引约定(防 off-by-one,务必精确)

记主模型在位置 i 的 last-layer hidden(final-norm 后)为 `H_i`,它预测 token `t_{i+1}`。
spike 已验证的 MTP 关系:

```
draft of t_{i+2} = argmax( MTP( H_i, t_{i+1} ) )
```

即 MTP 用「位置 i 的主 hidden」+「该位置主模型预测出的下一个 token」预测**再下一个** token。

**推论**:刚确认 token `x = t_{i+1}`(由 `H_i` 预测得到)后,我们手里同时握有 `H_i` 与 `x`,
即可立刻 `MTP(H_i, x)` 抽出 `t_{i+2}` 的草稿。**不需要未来信息,无循环依赖。**

## 3. 架构与数据流

新增 `mlx_streaming/mtp_generate.py`,核心函数 `mtp_generate(model, mtp, tok, prompt, max_tokens, K)`。

### 3.1 主模型前向(暴露 hidden)

新增辅助 `forward_with_hidden(model, ids, cache) -> (logits, H)`:复刻 mlx-lm
`Qwen3NextModel.__call__` 的层循环,跑完所有层 + `model.model.norm`,返回
`H = norm 后 hidden`(shape `(1, L, hidden)`)与 `logits = model.lm_head(H)`。
(与 `validate_mtp.capture_prenorm_hidden` 的 post_final_norm 分支同源,抽成公共函数。)

### 3.2 MTP 自回归抽 K 草稿

MTP 自带一层全注意力,需维护**自己的 KVCache**,与确认序列同步。抽草稿时:

```
# 入参:H_last(刚确认 token x 的主 hidden,位置 i)、x、mtp_cache(已处理到位置 i-1)
drafts = []
h, cur = H_last, x
for k in range(K):
    logits_mtp, mtp_h = mtp_step(mtp, h, cur, mtp_cache)  # 处理位置 (i+k)
    d = argmax(logits_mtp)
    drafts.append(d)
    h, cur = mtp_h, d        # 用 MTP 自身输出 hidden 作为下一步的 "主 hidden 代理"
return drafts                # 长度 K,mtp_cache 已前进 K
```

- `mtp_step` 返回 (logits, 该位置 MTP 层输出的 norm 后 hidden)。第 2..K 个草稿用 MTP 自身
  hidden 代替主 hidden(这是自投机的近似,acceptance 随 k 衰减,符合预期)。

### 3.3 主模型并行验证

```
# 已确认末 token x(位置 i,主 cache 已处理到 i-1,即 x 尚未入 cache),drafts=d_1..d_K
verify_in = [x, d_1, ..., d_{K-1}]            # K 个 token
snap_main = snapshot(main_cache)              # 见 3.4
logits, H = forward_with_hidden(model, verify_in, main_cache)  # 主 cache 前进 K
# logits[j] 由 verify_in[j] 产生,预测其下一 token
preds = argmax(logits, axis=-1)               # preds[j] = 主模型在 verify_in[j] 后的真实下一 token
# preds[0] 是 x 之后的真实 t_{i+1}? 注意:x=t_{i+1} 已确认,preds[0] 预测 t_{i+2}
```

**接受判定**(贪婪、精确):`verify_in = [x, d_1, …, d_{K-1}]`,主模型 `preds[j]` 是
「`verify_in[j]` 之后的真实下一 token」。求最长命中前缀:

- `preds[0]`(由 x 产生)= 真实 `t_{i+2}`;若 `== d_1` 命中,继续比 `preds[1]` vs `d_2`…
- 设 `m` = 最长命中数(`preds[0..m-1] == d_{1..m}`,`0 ≤ m ≤ K-1`)。命中草稿为 `d_1..d_m`。

**本步新确认序列 = `[d_1..d_m] + x'`**,其中 `x'` 是「重放 `[x, d_1..d_m]` 后主模型末位
argmax」——它恒等于 `preds[m]`(同序列同 cache → argmax 相同),即第一个未命中处的纠正
token,或全命中时的 bonus token。**每步至少新确认 1 个(x'),最多 K 个。**

> 关键:`x'` 始终来自主模型 argmax,这正是贪婪投机与贪婪基线逐 token 等价的根本原因。

### 3.4 cache 回滚(核心难点)

主模型含线性层(`ArraysCache`,**不可裁剪**)与全注意力层(`KVCache`,可裁剪)。
验证前向把所有 cache 前进了 K,但只接受到第 m 个草稿。统一用「快照 / 恢复 / 重放」:

1. **验证前快照**:`snap = _snapshot(main_cache)`,对每个 `c.state` 里的数组做
   `mx.array(a)` 强制拷贝(防止 KVCache 原地增长改掉快照)。同样快照 `mtp_cache`。
2. **恢复**:`_restore(main_cache, snap)`、`_restore(mtp_cache, snap_mtp)`,
   把两者都退回验证前(= 同步到位置 i-1)。
3. **重放**:`logits, H = forward_with_hidden(model, [x] + d_1..d_m, main_cache)`。
   - 这把主 cache 推进到 `d_m`(= 位置 i+m),正确无误。
   - 取 `x' = argmax(logits[:, -1])`、`H_last = H[:, -1]`(`d_m` 处 hidden,预测 `x'`)。
4. **同步 mtp_cache**:用主 hidden 把 mtp_cache 推进相同的 m+1 步——
   逐位置 `mtp_step(mtp, H_repl[:, p], token_{p+1}, mtp_cache)`,token 取 `[x, d_1..d_m]`
   对应的下一个。保证 mtp_cache 与主 cache 严格同步、且基于**真实主 hidden**(非 3.2 的
   自身 hidden 链),避免漂移。

> 实现取舍:首版**主/MTP cache 一律快照-恢复-重放**,不做选择性 trim。重放 m+1 个 token 的
> 额外成本在流式下被专家缓存命中摊薄。下一个生成步直接用 `x'` 作为 `verify_in[0]`、
> `H_last` 抽草稿,`x'` 此刻**不在** cache 中(满足 §3.3 不变量)。

### 3.5 初始化(prefill)

1. `forward_with_hidden(model, prompt, main_cache)` → 拿到 prompt 全部 hidden 与末位 logits;
   `x = argmax(末位 logits)` 为第 1 个生成 token,`H_last = H[:, -1]`。
2. MTP cache 预热:用 prompt 的主 hidden `H[:, :-1]` 与 token `prompt[1:]` 跑一次 MTP,
   把 mtp_cache 推进到 prompt 末位前一格,与主 cache 对齐。

## 4. 组件设计

### 4.1 `mlx_streaming/qwen3_next_mtp.py`(改造现有)

- `Qwen3NextMTP.__call__` 增加可选返回 hidden:`return_hidden=True` 时返回
  `(logits, norm后hidden)`,供自回归与验证复用。
- `mtp_step(mtp, h, token, cache)`:单步,内部走 fc + layer(cache) + norm,返回
  (logits, hidden)。
- `load_mtp(args, weights_path, quantize=True)`:加载 BF16 → 若 `quantize` 则
  `nn.quantize(model, group_size=64, bits=4, class_predicate=<gate/shared_gate→8bit>)`。
  embed_tokens 由调用方共享主模型的(见 spike 修复)。

### 4.2 `mlx_streaming/mtp_generate.py`(新文件)

- `forward_with_hidden(model, ids, cache) -> (logits, H)`。
- `_snapshot(cache)` / `_restore(cache, snap)`:基于 `.state` 的深拷贝快照/恢复。
- `mtp_generate(model, mtp, tok, prompt, max_tokens, K) -> (text, stats)`:主循环,
  stats 含接受 token 总数、验证步数、平均接受长度。

### 4.3 `mlx_streaming/run_mtp_spec.py`(新文件,基准)

仿 `run_streaming.py`:流式 patch 主模型 + 4-bit MTP 常驻,跑 `mtp_generate`,
输出 JSON:tok/s、平均接受长度、mlx_peak_gb、rss、以及**与非投机贪婪输出的逐 token 一致性**。

## 5. 测试(TDD,`mlx_streaming/tests/test_mtp_generate.py`)

小维度(`_tiny_args`,沿用 test_mtp_spike)随机权重即可验证循环逻辑;贪婪等价性是核心:

1. **快照/恢复幂等**:对一个 KVCache + ArraysCache 列表,跑前向 → 快照 → 再前向 → 恢复 →
   状态(`.state` 数组)与快照点逐元素相等(MAE=0)。
2. **接受判定纯函数**:给定 `drafts` 与主模型 `preds`,`accept_prefix(drafts, preds)`
   返回正确的 (新确认 token 列表, 接受数 m):全中、中间断、全不中三种情形。
3. **forward_with_hidden 形状**:返回 logits `(1,L,V)` 与 H `(1,L,hidden)`,
   且 `logits == lm_head(H)`(MAE<1e-4)。
4. **贪婪等价性(关键,可在小模型上构造)**:用 `_tiny_args` 造一个**确定性小主模型 + 小 MTP**
   (固定随机种子),分别用 (a) 朴素逐 token 贪婪 与 (b) `mtp_generate(K=3)` 生成 N token,
   断言两者输出 token 序列**完全一致**。这条覆盖 off-by-one、回滚、接受逻辑的端到端正确性。
   - 为可控,小模型用纯 KVCache(全注意力)版先验证;再加一个含线性层(ArraysCache)的
     变体重复该等价性测试,专门覆盖 ArraysCache 快照/恢复路径。

## 6. 端到端验收

1. `test_mtp_generate.py` 全绿(尤其等价性 4)。
2. `run_mtp_spec.py` 在真实 80B 上:**输出逐 token == 非投机贪婪**(打印 `exact_match: true`)。
3. tok/s 相对非投机基线(~7.5)显著提升;报告平均接受长度与峰值内存(目标增量 <1GB)。
4. 出报告 `benchmarks/reports/qwen3next-mtp-selfspec-2026-06-07.md`。

## 7. 风险与回退

- **off-by-one / 回滚错误** → 输出与基线不符。**缓解**:测试 4 的双变体等价性 + 真机
  `exact_match` 校验;先 K=1(无 MTP 自身 hidden 链)跑通,再开 K>1。
- **4-bit MTP 拉低接受率**:可能从 0.748 略降。**判断**:在 `run_mtp_spec` 里打印实测平均
  接受长度;若掉太多,回退 MTP 注意力/fc 用 BF16、仅专家 4-bit。
- **重放双算开销吃掉收益**:若实测加速 <1.3x,改为 KVCache 选择性 trim + 仅 ArraysCache 重放。
- **mtp_cache 与主 cache 不同步**:严格用主 hidden 重建 mtp_cache(3.4),并由测试 4 兜底。
