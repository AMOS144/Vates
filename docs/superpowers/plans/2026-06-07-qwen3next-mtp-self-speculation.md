# Qwen3-Next-80B 完整 MTP 自投机解码 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Qwen3-Next-80B MTP 自投机贪婪解码循环,把 0.748 接受率转成 ~2x 加速,且输出与非投机贪婪逐 token 等价。

**Architecture:** MTP 4-bit 常驻;每步主模型前向出 hidden→MTP 自回归抽 K 草稿→主模型并行验证→接受最长命中前缀→cache 快照/恢复/重放回滚(统一处理 ArraysCache 与 KVCache)。

**Tech Stack:** MLX / mlx-lm(qwen3_next 复用)、pytest。

参考 spec：`docs/superpowers/specs/2026-06-07-qwen3next-mtp-self-speculation-design.md`

---

## 关键事实(实现前必读)

1. **位置约定**：主 hidden `H_i`(final-norm 后,预测 `t_{i+1}`)+ token `t_{i+1}` → `MTP` 预测 `t_{i+2}`。即确认 `x` 后用「产生 x 的 hidden」+`x` 抽下一个草稿。
2. **cache 由循环持有**：`cache = model.make_cache()` 我们自己传入 `forward(...)`,可在原对象上 `c.state = ...` 原地恢复,不需要模型持有。
3. **KVCache 快照/恢复**：`c.state` getter 返回 `(keys[:offset], values[:offset])`;setter `c.state = v` 会按 `keys.shape[2]` **自动重算 offset**。故快照 = 深拷贝 `c.state`,恢复 = `c.state = snap`(offset 自动)。
4. **ArraysCache 快照/恢复**：解码期 `lengths`/`left_padding` 均为 None;`c.state` = `[conv_state, ssm_state]` 列表;setter `c.state = v` 设 `self.cache = v`。深拷贝/恢复同理。`advance(N)` 在 lengths=None 时为 no-op。
5. **深拷贝防原地改写**：KVCache/ArraysCache 的 `update_and_fetch` 会就地 scatter 进缓冲区,快照必须 `mx.array(a)` 拷贝并 `mx.eval`,否则后续前向会改掉快照。
6. **MTP 4-bit 量化**：先加载 BF16 权重再 `nn.quantize`(`mlx.nn.quantize(module, group_size, bits, class_predicate)`)。predicate 镜像 mlx-lm `qwen3_next.quant_predicate`:路径以 `mlp.gate` 或 `shared_expert_gate` 结尾 → `{"group_size":64,"bits":8}`,其余 4-bit。embed_tokens 量化后由主模型的覆盖。
7. **MTP 层是全注意力**(KVCache),`Qwen3NextDecoderLayer(args, layer_idx=3)`(`(3+1)%full_attention_interval(4)==0`)。
8. **forward 暴露 hidden**：复刻 mlx-lm `Qwen3NextModel.__call__` 层循环(见 `validate_mtp.capture_prenorm_hidden` 的 post_final_norm 分支),跑完层 + `model.model.norm` 得 H,`logits = model.lm_head(H)`。
9. **现有文件**：`mlx_streaming/qwen3_next_mtp.py`(MTP 模块)、`mlx_streaming/validate_mtp.py`(含 `capture_prenorm_hidden` 与 `_build_streaming_model`)。MTP 权重 `/tmp/qn_mtp_weights.safetensors`、专家目录 `/tmp/qwen3_next_experts`、主模型 `/tmp/qwen3_next_80b_4bit`、配置 `/tmp/qn_orig_config.json`。

---

## File Structure

- 改 `mlx_streaming/qwen3_next_mtp.py`：`__call__` 支持 `return_hidden`;新增 `mtp_step`;`load_mtp` 加 `quantize` 参数。
- 新增 `mlx_streaming/mtp_generate.py`：`forward_with_hidden`、`_snapshot`/`_restore`、`accept_prefix`、`mtp_generate`。
- 新增 `mlx_streaming/run_mtp_spec.py`：真实 80B 基准 + 与非投机贪婪一致性校验。
- 新增 `mlx_streaming/tests/test_mtp_generate.py`：纯函数 + 小模型贪婪等价性。

---

## Task 1: MTP 模块增强(return_hidden / mtp_step / 4-bit 量化)

**Files:**
- Modify: `mlx_streaming/qwen3_next_mtp.py`
- Test: `mlx_streaming/tests/test_mtp_generate.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_mtp_generate.py
import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs
from mlx_streaming.qwen3_next_mtp import Qwen3NextMTP, mtp_step


def _tiny_args():
    return ModelArgs(
        model_type="qwen3_next", hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, num_attention_heads=4, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, shared_expert_intermediate_size=64, mlp_only_layers=[],
        moe_intermediate_size=64, rms_norm_eps=1e-6, vocab_size=50,
        num_key_value_heads=2, rope_theta=10000.0, partial_rotary_factor=0.5,
        max_position_embeddings=128, head_dim=8, full_attention_interval=4)


def test_return_hidden_matches_lm_head():
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    head = mx.random.normal((args.hidden_size, args.vocab_size))
    lm = lambda h: h @ head
    hidden = mx.random.normal((1, 4, args.hidden_size))
    next_ids = mx.array([[1, 2, 3, 4]])
    logits, H = mtp(hidden, next_ids, lm, return_hidden=True)
    assert H.shape == (1, 4, args.hidden_size)
    assert float(mx.abs(logits - (H @ head)).max()) < 1e-4


def test_mtp_step_single_token():
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    head = mx.random.normal((args.hidden_size, args.vocab_size))
    lm = lambda h: h @ head
    from mlx_lm.models import cache as kvcache
    c = [kvcache.KVCache()]
    h = mx.random.normal((1, 1, args.hidden_size))
    logits, mh = mtp_step(mtp, h, mx.array([[7]]), lm, c)
    assert logits.shape == (1, args.vocab_size)
    assert mh.shape == (1, 1, args.hidden_size)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -v`
Expected: FAIL（`mtp_step` 不存在 / `return_hidden` 不支持）

- [ ] **Step 3: 改 `Qwen3NextMTP.__call__` 支持 return_hidden,新增 `mtp_step`**

把 `qwen3_next_mtp.py` 的 `__call__` 改为：

```python
    def __call__(self, hidden, next_ids, lm_head, cache=None, return_hidden=False):
        emb = self.pre_fc_norm_embedding(self.embed_tokens(next_ids))
        hid = self.pre_fc_norm_hidden(hidden)
        x = self.fc(mx.concatenate([emb, hid], axis=-1))
        mask = create_attention_mask(x, cache) if cache is not None else "causal"
        x = self.layer(x, mask=mask, cache=cache)
        H = self.norm(x)
        logits = lm_head(H)
        if return_hidden:
            return logits, H
        return logits
```

在文件末尾(`load_mtp` 之前)新增：

```python
def mtp_step(mtp, hidden, token, lm_head, cache):
    """单步:hidden(1,1,H) + token(1,1) -> (logits(1,V), mtp_hidden(1,1,H))。"""
    logits, H = mtp(hidden, token, lm_head, cache=cache, return_hidden=True)
    return logits[:, -1, :], H[:, -1:, :]
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -v`
Expected: 2 passed

- [ ] **Step 5: 给 `load_mtp` 加 4-bit 量化选项**

把 `load_mtp` 改为：

```python
def load_mtp(args, weights_path, quantize=True):
    model = Qwen3NextMTP(args)
    raw = mx.load(weights_path)
    renamed = {}
    for k, v in raw.items():
        nk = k[len("mtp."):] if k.startswith("mtp.") else k
        nk = nk.replace("layers.0.", "layer.", 1)
        renamed[nk] = v
    model.update(tree_unflatten(list(renamed.items())))
    if quantize:
        def pred(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True
        nn.quantize(model, group_size=64, bits=4, class_predicate=pred)
    mx.eval(model.parameters())
    return model
```

文件顶部确保 `import mlx.nn as nn` 已存在(已存在)。

- [ ] **Step 6: 运行测试 + Commit**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py mlx_streaming/tests/test_mtp_spike.py -q`
Expected: all passed

```bash
git add mlx_streaming/qwen3_next_mtp.py mlx_streaming/tests/test_mtp_generate.py
git commit -m "feat: MTP 模块支持 return_hidden/mtp_step 与 4-bit 量化加载"
```

---

## Task 2: forward_with_hidden + cache 快照/恢复

**Files:**
- Create: `mlx_streaming/mtp_generate.py`
- Test: `mlx_streaming/tests/test_mtp_generate.py`（追加）

- [ ] **Step 1: 写失败测试(快照幂等)**

追加：

```python
from mlx_lm.models import cache as kvcache
from mlx_streaming.mtp_generate import _snapshot, _restore


def test_snapshot_restore_kvcache_idempotent():
    mx.random.seed(0)
    c = kvcache.KVCache()
    k0 = mx.random.normal((1, 2, 3, 8))
    c.update_and_fetch(k0, k0)            # offset=3
    snap = _snapshot([c])
    # 再推进,改变状态
    k1 = mx.random.normal((1, 2, 2, 8))
    c.update_and_fetch(k1, k1)            # offset=5
    assert c.offset == 5
    _restore([c], snap)
    assert c.offset == 3
    ks, vs = c.state
    assert ks.shape[2] == 3


def test_snapshot_restore_arrayscache_idempotent():
    c = kvcache.ArraysCache(size=2)
    c.cache = [mx.ones((1, 3, 4)), mx.ones((1, 5))]
    snap = _snapshot([c])
    c.cache = [mx.zeros((1, 3, 4)), mx.zeros((1, 5))]
    _restore([c], snap)
    assert float(c.cache[0].sum()) == 12.0 and float(c.cache[1].sum()) == 5.0
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k snapshot -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 写实现(forward_with_hidden + 快照/恢复)**

```python
# mlx_streaming/mtp_generate.py
"""Qwen3-Next MTP 自投机贪婪解码循环(见 spec 2026-06-07-...-self-speculation)。"""
import time

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from mlx_streaming.qwen3_next_mtp import mtp_step


def forward_with_hidden(model, ids, cache):
    """跑主模型层循环 + 最终 norm,返回 (logits(1,L,V), H(1,L,hidden))。H 为 norm 后。"""
    inner = model.model
    h = inner.embed_tokens(ids)
    layers = inner.layers
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx])
    ssm_mask = create_ssm_mask(h, cache[ssm_idx])
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
    H = inner.norm(h)
    return model.lm_head(H), H


def _snapshot(caches):
    """对每个 cache 深拷贝 state + meta_state(强制 eval,防原地改写)。"""
    snaps = []
    for c in caches:
        st = c.state
        st_copy = tree_map(lambda a: mx.array(a), st) if st is not None else st
        if st_copy is not None:
            mx.eval(st_copy)
        snaps.append((st_copy, c.meta_state))
    return snaps


def _restore(caches, snaps):
    for c, (st, meta) in zip(caches, snaps):
        c.state = st
        c.meta_state = meta
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k snapshot -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add mlx_streaming/mtp_generate.py mlx_streaming/tests/test_mtp_generate.py
git commit -m "feat: forward_with_hidden 与 cache 快照/恢复"
```

---

## Task 3: accept_prefix 接受判定纯函数

**Files:**
- Modify: `mlx_streaming/mtp_generate.py`
- Test: `mlx_streaming/tests/test_mtp_generate.py`（追加）

- [ ] **Step 1: 写失败测试**

追加：

```python
from mlx_streaming.mtp_generate import accept_prefix


def test_accept_prefix_all_hit():
    # drafts=[a,b], preds(由 [x,a] 产生)=[a,b] -> 全中,m=2,新确认 a,b + bonus=preds[2]?
    # 约定:preds 长度 = len(drafts)+? 见实现;这里 preds 对应 verify_in=[x,d1..d_{K-1}]
    # K=3: drafts=[d1,d2,d3], verify_in=[x,d1,d2], preds 长度3
    drafts = [11, 22, 33]
    preds = [11, 22, 99]    # preds[0]==d1, preds[1]==d2, preds[2]!=d3
    matched, new_tokens = accept_prefix(drafts, preds)
    assert matched == 2                      # d1,d2 命中
    assert new_tokens == [11, 22, 99]        # 命中草稿 + 纠正 token preds[2]


def test_accept_prefix_first_miss():
    drafts = [11, 22, 33]
    preds = [99, 22, 33]    # preds[0]!=d1
    matched, new_tokens = accept_prefix(drafts, preds)
    assert matched == 0
    assert new_tokens == [99]                # 仅纠正 token


def test_accept_prefix_all_match_bonus():
    drafts = [11, 22, 33]
    preds = [11, 22, 33]    # 全中 -> matched=3,bonus=preds[2]? 见约定
    matched, new_tokens = accept_prefix(drafts, preds)
    assert matched == 3
    assert new_tokens == [11, 22, 33]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k accept -v`
Expected: FAIL

- [ ] **Step 3: 写实现**

> 约定：`verify_in = [x, d_1..d_{K-1}]`(K 个),`preds[j]` = 主模型在 `verify_in[j]` 后的真实下一 token,共 K 个。
> `preds[j]`(j=0..K-1)预测 `t_{i+2+j}`,与 `drafts[j]`(=d_{j+1})比较。
> `matched` = 最长前缀使 `preds[j]==drafts[j]`;新确认 = `drafts[:matched] + [preds[matched]]`(matched<K),
> 全中(matched==K-1 且 preds[K-1]==drafts[K-1])时 `preds[K-1]` 是 bonus,新确认 = `drafts`(=K-1)+[? ]。
> 为统一:`preds` 总比 `drafts` 多覆盖一位的语义由调用方保证 `len(preds)==len(drafts)`。逐位比对到第一个不等。

在 `mtp_generate.py` 追加：

```python
def accept_prefix(drafts, preds):
    """drafts: MTP 抽的 K 个草稿; preds: 主模型对应位置的真实下一 token(len==K)。
    返回 (matched, new_tokens):matched=命中草稿数;new_tokens=命中草稿 + 1 个纠正/bonus token。
    贪婪下 new_tokens 全部来自主模型 argmax(命中位 preds==drafts),故与非投机等价。
    """
    matched = 0
    for d, p in zip(drafts, preds):
        if int(p) == int(d):
            matched += 1
        else:
            break
    # 命中前缀 + 第一个未命中处的纠正 token(或全中时的最后一位 bonus)
    if matched < len(drafts):
        new_tokens = [int(p) for p in preds[: matched + 1]]
    else:
        new_tokens = [int(p) for p in preds]   # 全中,preds 即全部确认(末位为 bonus)
    return matched, new_tokens
```

> 注:上面三个测试里 `test_accept_prefix_all_match_bonus` 的 `drafts/preds` 长度都为 3 且全等,
> 走 `matched==len(drafts)` 分支,`new_tokens==preds==[11,22,33]`,与断言一致。

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k accept -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add mlx_streaming/mtp_generate.py mlx_streaming/tests/test_mtp_generate.py
git commit -m "feat: accept_prefix 贪婪接受判定纯函数"
```

---

## Task 4: mtp_generate 主循环 + 贪婪等价性(KVCache-only 小模型)

**Files:**
- Modify: `mlx_streaming/mtp_generate.py`
- Test: `mlx_streaming/tests/test_mtp_generate.py`（追加）

> 设计：循环维护 `main_cache`、`mtp_cache`、当前 pending token `x`、其主 hidden `H_last`。
> 每步：MTP 自回归抽 K 草稿 → 快照两 cache → 主模型验证 `[x, d_1..d_{K-1}]` → `accept_prefix`
> → 恢复两 cache → 重放 `[x] + 命中草稿` 取末位 logit/hidden 得新 `x`/`H_last` → MTP cache 用主
> hidden 同步推进。为先跑通正确性,本任务实现并用 **K=1** 验证(K=1 时无 MTP 自回归链)。

- [ ] **Step 1: 写失败测试(KVCache-only 小模型贪婪等价性)**

> 小模型用 mlx-lm 真 `Qwen3NextModel` 不便构造完整(含线性层)。改用一个**仅全注意力**的极简
> 替身模型,接口契合 `forward_with_hidden`(有 `.model.embed_tokens/.layers/.norm`、`.lm_head`、
> `.make_cache()`,层 `.is_linear=False`、`__call__(x,mask,cache)`)。

追加：

```python
import mlx.nn as nn
from mlx_streaming.mtp_generate import mtp_generate, forward_with_hidden


class _FAOnlyLayer(nn.Module):
    is_linear = False
    def __init__(self, h):
        super().__init__()
        self.attn = nn.MultiHeadAttention(h, 2)
        self.norm = nn.RMSNorm(h)
    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        m = nn.MultiHeadAttention.create_additive_causal_mask(L)
        r = self.attn(x, x, x, mask=m)
        return self.norm(x + r)


class _Inner(nn.Module):
    def __init__(self, vocab, h, nl):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.layers = [_FAOnlyLayer(h) for _ in range(nl)]
        self.norm = nn.RMSNorm(h)


class _ToyModel(nn.Module):
    """仅全注意力的玩具自回归模型,接口契合 forward_with_hidden。"""
    def __init__(self, vocab=40, h=32, nl=2):
        super().__init__()
        self.model = _Inner(vocab, h, nl)
        self.lm_head = nn.Linear(h, vocab, bias=False)
    @property
    def layers(self):
        return self.model.layers
    def make_cache(self):
        from mlx_lm.models import cache as kc
        return [kc.KVCache() for _ in self.layers]
    def __call__(self, ids, cache=None):
        logits, _ = forward_with_hidden(self, ids, cache or self.make_cache())
        return logits
```

> 注:`_FAOnlyLayer` 自建因果 mask、不依赖外部 cache(玩具模型不做真正的增量 KV;
> `forward_with_hidden` 每步喂全前缀即可保证等价。为此 `mtp_generate` 在玩具路径下需支持
> 「无增量 cache、每步重算全前缀」模式)。**为避免把玩具复杂化**,改用下方更贴近真实的做法。

实际测试(直接验证 `mtp_generate` 的接受/回滚逻辑,用 mlx-lm 真 KVCache 增量):

```python
def test_mtp_generate_greedy_equiv_kv(monkeypatch):
    """玩具全注意力模型上:mtp_generate(K) 输出 == 朴素逐 token 贪婪输出。"""
    mx.random.seed(0)
    model = _ToyModel()
    mx.eval(model.parameters())
    # MTP 替身:直接复用 model 自身作为"草稿"(草稿==主模型 -> 接受率应≈1,逻辑可验)
    # 这里用一个返回 model 预测的轻量 mtp 包装,确保等价性逻辑被走到。
    prompt = mx.array([[1, 5, 9]])

    def naive_greedy(n):
        cache = model.make_cache()
        ids = prompt
        out = []
        for _ in range(n):
            logits, _ = forward_with_hidden(model, ids, cache)
            nxt = int(mx.argmax(logits[:, -1, :]))
            out.append(nxt)
            ids = mx.array([[nxt]])
        return out

    ref = naive_greedy(12)
    got, stats = mtp_generate(model, _SelfDraft(model), tok=None, prompt=prompt,
                              max_tokens=12, K=1, ids_mode=True)
    assert got == ref
```

> 上面引入 `_SelfDraft`(把主模型当 MTP 草稿源,贪婪下草稿==验证 → matched 总命中,
> 用于隔离验证接受/回滚循环的正确性)。在测试文件内定义:

```python
class _SelfDraft:
    """测试用:用主模型自身产生草稿(贪婪下与验证一致),仅为验证循环骨架。"""
    def __init__(self, model):
        self.model = model
        self.embed_tokens = model.model.embed_tokens
    def draft(self, H_last, x, cache, K):
        # 简化:用主模型对 [x] 前向取 argmax 作为单草稿(K 由调用方限定为 1)
        logits, _ = forward_with_hidden(self.model, x, cache)
        return [int(mx.argmax(logits[:, -1, :]))]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k greedy_equiv_kv -v`
Expected: FAIL（`mtp_generate` 不存在）

- [ ] **Step 3: 写 `mtp_generate`(K 通用,含 K=1 路径)**

在 `mtp_generate.py` 追加。两种草稿源:真实路径用 `Qwen3NextMTP`(经 `mtp_step` 自回归),
测试路径用带 `.draft()` 的对象。为统一,`mtp_generate` 接收一个 `drafter`,其需提供
`drafter.draft(H_last, x_ids, mtp_cache, K) -> list[int]`(长度 K)。真实 MTP 的 drafter
适配见 Task 6;本任务先让 `_SelfDraft` 跑通骨架。

```python
def mtp_generate(model, drafter, tok, prompt, max_tokens, K=3, ids_mode=False):
    """贪婪 MTP 自投机。drafter.draft(H_last, x_ids(1,1), mtp_cache, K)->list[int]。
    ids_mode=True 时 prompt 已是 ids(1,L) 且返回 token id 列表(测试用)。"""
    main_cache = model.make_cache()
    mtp_cache = getattr(drafter, "make_cache", lambda: None)()
    ids = prompt if ids_mode else mx.array([tok.encode(prompt)])

    # prefill:得到第 1 个 pending token x 与其 hidden
    logits, H = forward_with_hidden(model, ids, main_cache)
    x = int(mx.argmax(logits[:, -1, :]))
    H_last = H[:, -1:, :]
    produced = [x]
    t0 = time.perf_counter()
    n_steps = 0

    while len(produced) < max_tokens:
        x_ids = mx.array([[x]])
        drafts = drafter.draft(H_last, x_ids, mtp_cache, K)      # 长度 K
        # 验证输入 [x, d_1..d_{K-1}]
        verify_in = mx.array([[x] + drafts[: K - 1]])
        snap_m = _snapshot(main_cache)
        snap_d = _snapshot(mtp_cache) if mtp_cache else None
        vlogits, _ = forward_with_hidden(model, verify_in, main_cache)
        preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]  # 长度 K
        matched, new_tokens = accept_prefix(drafts, preds)
        # 回滚 + 重放 [x] + 命中草稿
        _restore(main_cache, snap_m)
        if snap_d is not None:
            _restore(mtp_cache, snap_d)
        replay_in = mx.array([[x] + drafts[:matched]])
        rlogits, rH = forward_with_hidden(model, replay_in, main_cache)
        for t in new_tokens:
            produced.append(t)
            if len(produced) >= max_tokens:
                break
        # 下一步 pending：重放末位 argmax(== 最后一个 new_token),hidden 取重放末位
        x = int(mx.argmax(rlogits[:, -1, :]))
        H_last = rH[:, -1:, :]
        # MTP cache 同步:用真实主 hidden 推进(drafter 负责)
        if hasattr(drafter, "sync"):
            drafter.sync(rH, replay_in, mtp_cache)
        n_steps += 1
        mx.eval(x)

    produced = produced[:max_tokens]
    stats = {"steps": n_steps, "tokens": len(produced),
             "avg_accept_len": round(len(produced) / max(n_steps, 1), 3),
             "wall_s": round(time.perf_counter() - t0, 3)}
    if ids_mode:
        return produced, stats
    return tok.decode(produced), stats
```

> 说明：`_SelfDraft` 没有 `make_cache`/`sync`,`mtp_cache=None`,`snap_d=None`,跳过同步。
> K=1 时 `verify_in=[x]`、`replay_in=[x]+drafts[:matched]`(matched∈{0,1})。

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k greedy_equiv_kv -v`
Expected: PASS（`got == ref`）

- [ ] **Step 5: Commit**

```bash
git add mlx_streaming/mtp_generate.py mlx_streaming/tests/test_mtp_generate.py
git commit -m "feat: mtp_generate 自投机主循环(K 通用,KVCache 等价性过)"
```

---

## Task 5: ArraysCache 变体等价性(覆盖线性层回滚)

**Files:**
- Test: `mlx_streaming/tests/test_mtp_generate.py`（追加）

> 真实主模型含线性层(ArraysCache)。本任务用一个**含 ArraysCache 假层**的玩具模型重复等价性,
> 专门覆盖快照/恢复线性递归状态的路径。

- [ ] **Step 1: 写失败/新测试**

追加一个 `_LinearishLayer`:`is_linear=True`,内部维护一个简单递归状态(用 ArraysCache 存
一个累加向量),`__call__` 把 `cache[1]`(状态)累加并参与输出,使「回滚错误」必然导致输出分叉。

```python
class _RecurLayer(nn.Module):
    is_linear = True
    def __init__(self, h):
        super().__init__()
        self.proj = nn.Linear(h, h, bias=False)
    def __call__(self, x, mask=None, cache=None):
        B, L, H = x.shape
        outs = []
        state = None
        if cache is not None and cache[1] is not None:
            state = cache[1]
        if state is None:
            state = mx.zeros((B, H))
        for t in range(L):
            state = state + x[:, t, :]          # 递归累加(不可裁剪语义)
            outs.append(state)
        if cache is not None:
            cache[1] = state
        return self.proj(mx.stack(outs, axis=1))


def test_mtp_generate_greedy_equiv_arrayscache():
    mx.random.seed(1)
    model = _ToyModel(nl=2)
    # 把第 0 层换成递归(ArraysCache)层
    model.model.layers[0] = _RecurLayer(32)
    mx.eval(model.parameters())

    def make_cache():
        from mlx_lm.models import cache as kc
        return [kc.ArraysCache(2) if l.is_linear else kc.KVCache()
                for l in model.layers]
    model.make_cache = make_cache

    prompt = mx.array([[2, 7, 1]])

    def naive_greedy(n):
        cache = model.make_cache()
        ids, out = prompt, []
        for _ in range(n):
            logits, _ = forward_with_hidden(model, ids, cache)
            nxt = int(mx.argmax(logits[:, -1, :])); out.append(nxt)
            ids = mx.array([[nxt]])
        return out

    ref = naive_greedy(10)
    got, _ = mtp_generate(model, _SelfDraft(model), None, prompt, 10, K=1, ids_mode=True)
    assert got == ref
```

- [ ] **Step 2: 运行**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_generate.py -k arrayscache -v`
Expected: PASS（若失败说明 ArraysCache 快照/恢复有 bug,按 spec §3.4 修 `_snapshot/_restore`)

- [ ] **Step 3: 跑全量回归**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/ -q`
Expected: all passed

- [ ] **Step 4: Commit**

```bash
git add mlx_streaming/tests/test_mtp_generate.py
git commit -m "test: ArraysCache 路径的 MTP 自投机贪婪等价性"
```

---

## Task 6: 真实 MTP drafter 适配 + 80B 基准 + 报告

**Files:**
- Modify: `mlx_streaming/mtp_generate.py`（新增 `MTPDrafter`）
- Create: `mlx_streaming/run_mtp_spec.py`

- [ ] **Step 1: 在 `mtp_generate.py` 新增 `MTPDrafter`**

```python
class MTPDrafter:
    """把 Qwen3NextMTP 包成 mtp_generate 需要的 drafter 接口。"""
    def __init__(self, mtp, lm_head):
        self.mtp = mtp
        self.lm_head = lm_head
        self.embed_tokens = mtp.embed_tokens
    def make_cache(self):
        from mlx_lm.models import cache as kc
        return [kc.KVCache()]       # MTP 单层全注意力
    def draft(self, H_last, x_ids, mtp_cache, K):
        drafts = []
        h, cur = H_last, x_ids
        for _ in range(K):
            logits, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0:1])
            d = int(mx.argmax(logits[0]))
            drafts.append(d)
            h, cur = mh, mx.array([[d]])
        return drafts
    def sync(self, rH, replay_in, mtp_cache):
        # 用真实主 hidden 重建 mtp_cache:已在 draft 阶段推进 K;此处不额外动,
        # 由下一步 draft 从正确 H_last 重新起算。简化:每步 draft 前重置 mtp_cache。
        mtp_cache[0] = type(mtp_cache[0])()
```

> 说明:为彻底规避 mtp_cache 与主 cache 的同步漂移风险,首版采用**每步重置 mtp_cache**
> (`sync` 清空),即 MTP 草稿用「当前 H_last + 局部 K 窗口」自注意力,不跨步累积长上下文。
> 这会略降 acceptance(MTP 注意力只看到 K 窗口),但**保证正确性**且实现简单。若实测加速达标即可;
> 不达标再做完整 mtp_cache 同步(spec §3.4)作为优化。

- [ ] **Step 2: 写 `run_mtp_spec.py`**

```python
# mlx_streaming/run_mtp_spec.py
"""真实 80B MTP 自投机基准 + 与非投机贪婪逐 token 一致性校验。"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mem import snapshot, reset_peak
from mlx_streaming.validate_mtp import _build_streaming_model
from mlx_streaming.qwen3_next_mtp import load_mtp
from mlx_streaming.mtp_generate import mtp_generate, MTPDrafter, forward_with_hidden

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "3"))


def _baseline_greedy(model, tok, prompt, n):
    cache = model.make_cache()
    ids = mx.array([tok.encode(prompt)])
    out = []
    for _ in range(n):
        logits, _ = forward_with_hidden(model, ids, cache)
        nxt = int(mx.argmax(logits[:, -1, :])); out.append(nxt)
        ids = mx.array([[nxt]]); mx.eval(ids)
    return out


def main():
    reset_peak()
    model, tok = _build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    base = _baseline_greedy(model, tok, PROMPT, MAXTOK)
    ids, stats = mtp_generate(model, drafter, tok,
                              mx.array([tok.encode(PROMPT)]),
                              MAXTOK, K=K, ids_mode=True)
    after = snapshot()
    print(json.dumps({
        "K": K, "max_tokens": MAXTOK,
        "exact_match": ids == base,
        "n_mismatch": sum(1 for a, b in zip(ids, base) if a != b),
        "avg_accept_len": stats["avg_accept_len"],
        "steps": stats["steps"],
        "tok_per_s": round(stats["tokens"] / stats["wall_s"], 2),
        "mlx_peak_gb": round(after.mlx_peak_bytes / 1e9, 2),
        "rss_gb": round(after.rss_bytes / 1e9, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑真机基准**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && K=3 MAXTOK=96 python -u -m mlx_streaming.run_mtp_spec`
Expected: JSON 含 `"exact_match": true`(必须为 true,否则有 bug),`avg_accept_len > 1`,`tok_per_s` 高于 ~7.5。
排错:
- `exact_match: false` → 回滚/接受逻辑错,先用 Task 4/5 玩具复现;检查 K=1 是否 exact(`K=1` 重跑),逐步加 K。
- `tok_per_s` 不升 → 看 `avg_accept_len`;若接近 1,说明 MTP K-窗口 acceptance 太低,启用完整 mtp_cache 同步。

- [ ] **Step 4: 写报告 + Commit**

把结果写入 `benchmarks/reports/qwen3next-mtp-selfspec-2026-06-07.md`(K、exact_match、avg_accept_len、tok/s、peak_gb 对比 7.5 基线),然后:

```bash
git add mlx_streaming/mtp_generate.py mlx_streaming/run_mtp_spec.py benchmarks/reports/qwen3next-mtp-selfspec-2026-06-07.md
git commit -m "feat: 真实 MTP drafter + 80B 自投机基准与报告"
```

---

## Self-Review(已执行)

- **Spec 覆盖**:4-bit 量化加载(T1)、return_hidden/mtp_step(T1)、forward_with_hidden(T2)、
  快照/恢复(T2)、接受判定(T3)、主循环+回滚(T4)、ArraysCache 路径(T5)、真实 drafter+基准+
  exact_match 校验+报告(T6)。位置约定 §2 落在 T3/T4 的 verify_in/accept_prefix。
- **占位符**:无 TBD;每步含完整代码与命令。`run_mtp_spec` 的报告内容在 T6 Step4 列明字段。
- **类型一致**:`mtp_step(mtp,h,token,lm_head,cache)`、`forward_with_hidden(model,ids,cache)->(logits,H)`、
  `_snapshot(caches)`/`_restore(caches,snaps)`、`accept_prefix(drafts,preds)->(matched,new_tokens)`、
  `mtp_generate(model,drafter,tok,prompt,max_tokens,K,ids_mode)`、`MTPDrafter.draft(H_last,x_ids,mtp_cache,K)`
  跨任务签名一致。
- **已知风险与回退**:mtp_cache 同步首版用「每步重置」降险(T6 Step1 注),exact_match 为硬校验,
  acceptance 不足再上完整同步。
