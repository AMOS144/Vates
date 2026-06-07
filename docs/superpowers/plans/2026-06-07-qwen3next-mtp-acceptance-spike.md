# Qwen3-Next-80B MTP 接受率验证 Spike 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用最小代价(只下 3.9GB、teacher-forcing)实测 Qwen3-Next-80B 自带 MTP 头的草稿接受率,产出 go/no-go 决策依据。

**Architecture:** 从原版 BF16 末分片抽取 MTP 权重 → 在 MLX 用 mlx-lm 现成子模块类搭一个 `Qwen3NextMTP` 模块(BF16 常驻)→ 用 teacher forcing 比对 MTP 对 t+2 的预测命中率(对真实文本、对主模型贪心输出两种)。全程为纯旁路新增,不接生成主路径。

**Tech Stack:** MLX / mlx-lm(qwen3_next 模块复用)、pytest、ModelScope 下载。

参考 spec:`docs/superpowers/specs/2026-06-07-qwen3next-mtp-acceptance-spike-design.md`

---

## 关键事实(实现前必读)

1. **MTP 权重位置**:原版 `Qwen/Qwen3-Next-80B-A3B-Instruct` 的全部 `mtp.*` 张量仅在 `model-00041-of-00041.safetensors`(~3.9GB)。ModelScope 无 HTTP Range,下载用「按 size 校验 + 整文件重下 + 限速重试」(参照已有 `/tmp/dl_qn.sh`)。
2. **zero-centered RMSNorm**:mlx-lm `qwen3_next.py:469-482` 的 `sanitize` 对所有 RMSNorm 权重(`.input_layernorm` / `.post_attention_layernorm` / `model.norm` / `.q_norm` / `.k_norm`,均 1 维)**加 1.0**。MTP 的 norm 被 sanitize 过滤掉,**因此抽取时必须自己对 MTP 的所有 norm 权重 +1.0**(`pre_fc_norm_embedding`、`pre_fc_norm_hidden`、`norm`、`layers.0.input_layernorm`、`layers.0.post_attention_layernorm`、`layers.0.self_attn.q_norm`、`layers.0.self_attn.k_norm`)。漏了会导致数值全错、接受率虚低。`mtp.fc` 是 Linear,**不**加 1。
3. **专家 stack 约定**:mlx-lm 把 `mlp.experts.{e}.{gate,up,down}_proj.weight`(共 `num_experts=512` 个)堆成 `mlp.switch_mlp.{gate,up,down}_proj.weight`,形状 `(512, out, in)`(`mx.stack`,专家序号升序)。
4. **MTP 解码层 = 全注意力 + MoE**:`Qwen3NextDecoderLayer(args, layer_idx=3)` 满足 `(3+1)%full_attention_interval(4)==0` → `is_linear=False`(走 `self_attn`),且 `(3+1)%decoder_sparse_step(1)==0` → `mlp=Qwen3NextSparseMoeBlock`。即用 `layer_idx=3` 构造出与 MTP 同构的层。
5. **MTP 前向图**(vLLM/sglang/trtllm 三方一致):
   ```
   emb = pre_fc_norm_embedding(embed(next_id))
   hid = pre_fc_norm_hidden(主模型 last-layer hidden)
   x   = fc(concat([emb, hid], axis=-1))      # 顺序:emb 在前
   x   = layer(x, mask, cache)                 # mlx Qwen3NextDecoderLayer,内部含残差
   logits = lm_head(norm(x))
   ```
6. **主模型 last-layer hidden**:mlx-lm `Qwen3NextModel.__call__` 末尾返回 `self.norm(hidden)`。MTP 需要 norm **之前** 的 hidden(与 `pre_fc_norm_hidden` 存在自洽)。复制其层循环、不过 `self.norm` 即可拿到。
7. **配置**:原版 config 已下载到 `/tmp/qn_orig_config.json`(`hidden_size=2048`、`num_experts=512`、`num_experts_per_tok=10`、`full_attention_interval=4`、`decoder_sparse_step=1`、`rms_norm_eps`、`vocab_size=151936`、`tie_word_embeddings=False`)。`ModelArgs.from_dict(config)` 可直接构造(自动过滤多余键)。
8. **局部读权重**:用 `mx.load(shard_path)`(mmap,只有被访问/eval 的张量才materialize),过滤 `mtp.` 前缀键。
9. **主模型路径**:已下载的 4-bit 模型在 `/tmp/qwen3_next_80b_4bit`。

---

## File Structure

- 新增 `mlx_streaming/extract_mtp.py`:下载末分片 + 抽取/整理 MTP 权重 → 单文件。含纯函数 `stack_mtp_experts(weights, num_experts)` 与 `bump_mtp_norms(weights)` 便于单测。
- 新增 `mlx_streaming/qwen3_next_mtp.py`:`Qwen3NextMTP(nn.Module)` 模块定义与权重加载辅助 `load_mtp(args, weights_path)`。
- 新增 `mlx_streaming/validate_mtp.py`:主模型 hidden 捕获 + 接受率测量脚本(产出 JSON)。
- 新增 `mlx_streaming/tests/test_mtp_spike.py`:纯函数与小维度模块单测(不依赖 80B 真权重)。

---

## Task 1: 抽取脚本的纯逻辑(stack 专家 + norm +1.0)

**Files:**
- Create: `mlx_streaming/extract_mtp.py`
- Test: `mlx_streaming/tests/test_mtp_spike.py`

- [ ] **Step 1: 写失败测试**

```python
# mlx_streaming/tests/test_mtp_spike.py
import mlx.core as mx
from mlx_streaming.extract_mtp import stack_mtp_experts, bump_mtp_norms


def test_stack_mtp_experts_shapes_and_order():
    # 2 个专家、小维度的伪权重
    w = {
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((4, 3)),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": mx.ones((4, 3)),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((4, 3)),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": mx.ones((4, 3)),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((3, 4)),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": mx.ones((3, 4)),
        "mtp.fc.weight": mx.zeros((3, 6)),
    }
    out = stack_mtp_experts(w, num_experts=2)
    # 专家被堆叠且原 per-expert 键被移除
    g = out["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]
    assert g.shape == (2, 4, 3)
    # 升序:expert 0 全 0、expert 1 全 1
    assert float(g[0].sum()) == 0.0 and float(g[1].sum()) == 12.0
    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" not in out
    # 非专家键原样保留
    assert "mtp.fc.weight" in out


def test_bump_mtp_norms_adds_one_only_to_norms():
    w = {
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((4,)),
        "mtp.pre_fc_norm_embedding.weight": mx.zeros((4,)),
        "mtp.norm.weight": mx.zeros((4,)),
        "mtp.layers.0.input_layernorm.weight": mx.zeros((4,)),
        "mtp.layers.0.post_attention_layernorm.weight": mx.zeros((4,)),
        "mtp.layers.0.self_attn.q_norm.weight": mx.zeros((2,)),
        "mtp.layers.0.self_attn.k_norm.weight": mx.zeros((2,)),
        "mtp.fc.weight": mx.zeros((3, 6)),  # 非 norm,不应被改
    }
    out = bump_mtp_norms(w)
    assert float(out["mtp.norm.weight"][0]) == 1.0
    assert float(out["mtp.pre_fc_norm_hidden.weight"][0]) == 1.0
    assert float(out["mtp.layers.0.self_attn.q_norm.weight"][0]) == 1.0
    assert float(out["mtp.fc.weight"].sum()) == 0.0  # 未被 +1
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_spike.py -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'mlx_streaming.extract_mtp'`

- [ ] **Step 3: 写最小实现(纯函数部分)**

```python
# mlx_streaming/extract_mtp.py
"""下载 Qwen3-Next 原版末分片并抽取/整理 MTP 权重为单文件。

MTP 全部权重在 model-00041-of-00041.safetensors(~3.9GB)。整理规则:
  1) 把 mlp.experts.{e}.{proj}.weight(512 个)stack 成 mlp.switch_mlp.{proj}.weight。
  2) 对所有 RMSNorm 权重 +1.0(Qwen3-Next 用 zero-centered RMSNorm,
     与 mlx-lm sanitize 对主模型的处理一致;MTP 被 sanitize 过滤,故需自行补)。
"""
import os
import mlx.core as mx

# 与 mlx-lm qwen3_next.sanitize 一致的 norm 后缀(去掉 model.norm,因 MTP 用 mtp.norm)
_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def _is_mtp_norm(key: str) -> bool:
    if key.endswith(".pre_fc_norm_hidden.weight") or key.endswith(
        ".pre_fc_norm_embedding.weight"
    ):
        return True
    if key == "mtp.norm.weight":
        return True
    return any(key.endswith(sfx) for sfx in _NORM_SUFFIXES)


def bump_mtp_norms(weights: dict) -> dict:
    out = {}
    for k, v in weights.items():
        if _is_mtp_norm(k) and v.ndim == 1:
            out[k] = v + 1.0
        else:
            out[k] = v
    return out


def stack_mtp_experts(weights: dict, num_experts: int) -> dict:
    out = dict(weights)
    prefix = "mtp.layers.0.mlp"
    for proj in ("gate_proj", "up_proj", "down_proj"):
        keys = [f"{prefix}.experts.{e}.{proj}.weight" for e in range(num_experts)]
        if keys[0] not in out:
            continue
        stacked = mx.stack([out.pop(k) for k in keys])
        out[f"{prefix}.switch_mlp.{proj}.weight"] = stacked
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_spike.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
cd /Users/amos/project/flash-moe/hypura
git add mlx_streaming/extract_mtp.py mlx_streaming/tests/test_mtp_spike.py
git commit -m "feat: MTP 权重整理纯函数(stack 专家 + norm +1.0)"
```

---

## Task 2: 下载末分片 + 抽取主流程(extract_mtp.py 的 IO 部分)

**Files:**
- Modify: `mlx_streaming/extract_mtp.py`

> 说明:这一步是 IO/下载,不写单测(无法在单测内下 3.9GB)。提供可独立运行的 `main()`,在 Task 5 手动执行。

- [ ] **Step 1: 追加下载与抽取主流程**

在 `mlx_streaming/extract_mtp.py` 末尾追加:

```python
import json
import subprocess

SHARD = "model-00041-of-00041.safetensors"
REPO = "Qwen/Qwen3-Next-80B-A3B-Instruct"
SHARD_DIR = os.environ.get("MTP_SHARD_DIR", "/tmp/qn_mtp_shard")
OUT_PATH = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")


def _expected_size() -> int:
    url = (
        f"https://www.modelscope.cn/api/v1/models/{REPO}/repo/files"
        f"?Revision=master&Root=&"
    )
    # 用 index 不含 size;改用 HEAD 拿 content-length(跟随重定向)
    dl = (
        f"https://www.modelscope.cn/api/v1/models/{REPO}/repo"
        f"?Revision=master&FilePath={SHARD}"
    )
    out = subprocess.run(
        ["curl", "-sIL", "--max-time", "60", dl],
        capture_output=True, text=True,
    ).stdout
    size = 0
    for line in out.splitlines():
        low = line.lower()
        if low.startswith("content-length:"):
            size = int(line.split(":", 1)[1].strip())
    return size


def download_shard() -> str:
    os.makedirs(SHARD_DIR, exist_ok=True)
    out = os.path.join(SHARD_DIR, SHARD)
    expect = _expected_size()
    dl = (
        f"https://www.modelscope.cn/api/v1/models/{REPO}/repo"
        f"?Revision=master&FilePath={SHARD}"
    )
    for attempt in range(1, 9):
        cur = os.path.getsize(out) if os.path.exists(out) else 0
        if expect and cur == expect:
            print(f"SKIP shard already complete ({cur}B)")
            return out
        print(f"download attempt {attempt}: have {cur}B / {expect}B")
        subprocess.run(
            ["curl", "-sL", "--max-time", "3600",
             "--speed-limit", "51200", "--speed-time", "30",
             dl, "-o", out],
        )
    cur = os.path.getsize(out) if os.path.exists(out) else 0
    if expect and cur != expect:
        raise RuntimeError(f"download failed: {cur}B != {expect}B")
    return out


def extract(shard_path: str, config_path: str, out_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)
    num_experts = cfg["num_experts"]
    all_w = mx.load(shard_path)
    mtp = {k: v for k, v in all_w.items() if k.startswith("mtp.")}
    print(f"原始 mtp 张量数: {len(mtp)}")
    mtp = stack_mtp_experts(mtp, num_experts)
    mtp = bump_mtp_norms(mtp)
    mx.eval(list(mtp.values()))
    mx.save_safetensors(out_path, mtp)
    print(f"已写出 {len(mtp)} 个张量 -> {out_path}")


def main():
    shard = download_shard()
    extract(shard, CONFIG, OUT_PATH)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 静态检查(import 不报错)**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -c "import mlx_streaming.extract_mtp"`
Expected: 无输出、退出码 0

- [ ] **Step 3: Commit**

```bash
cd /Users/amos/project/flash-moe/hypura
git add mlx_streaming/extract_mtp.py
git commit -m "feat: 下载 Qwen3-Next 末分片并抽取 MTP 权重主流程"
```

---

## Task 3: MTP 模块 Qwen3NextMTP

**Files:**
- Create: `mlx_streaming/qwen3_next_mtp.py`
- Test: `mlx_streaming/tests/test_mtp_spike.py`(追加)

- [ ] **Step 1: 写失败测试(fc 拼接顺序 + 形状 + 因果)**

在 `mlx_streaming/tests/test_mtp_spike.py` 追加:

```python
from mlx_lm.models.qwen3_next import ModelArgs
from mlx_streaming.qwen3_next_mtp import Qwen3NextMTP


def _tiny_args():
    return ModelArgs(
        model_type="qwen3_next",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=64,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=50,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        max_position_embeddings=128,
        head_dim=8,
        full_attention_interval=4,
    )


def test_mtp_forward_shape_and_causal():
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    lm_head = lambda h: h @ mx.random.normal((args.hidden_size, args.vocab_size))
    B, L = 1, 6
    hidden = mx.random.normal((B, L, args.hidden_size))
    next_ids = mx.array([[3, 7, 1, 9, 2, 5]])
    logits = mtp(hidden, next_ids, lm_head)
    assert logits.shape == (B, L, args.vocab_size)
    # 因果:改最后一位输入,前面 logits 不变
    hidden2 = mx.array(hidden)
    hidden2[:, -1, :] = hidden2[:, -1, :] + 10.0
    logits2 = mtp(hidden2, next_ids, lm_head)
    assert float(mx.abs(logits[:, :-1] - logits2[:, :-1]).max()) < 1e-3


def test_mtp_fc_concat_order_is_emb_then_hidden():
    # 用 identity-ish 设置验证 concat 顺序:fc 取前半=emb 贡献、后半=hidden 贡献
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    h = args.hidden_size
    # 关闭两个 pre_fc_norm 的影响:把权重设为 0(RMSNorm weight=0 -> 输出 0)
    mtp.pre_fc_norm_embedding.weight = mx.zeros((h,))
    mtp.pre_fc_norm_hidden.weight = mx.ones((h,))  # 仅 hidden 通过
    # fc 权重设为 [W_emb | W_hidden],令 W_emb=0、W_hidden=I
    W = mx.concatenate([mx.zeros((h, h)), mx.eye(h)], axis=1)  # (h, 2h)
    mtp.fc.weight = W
    mx.eval(mtp.parameters())
    captured = {}
    orig_layer = mtp.layer

    def spy(x, mask=None, cache=None):
        captured["x"] = x
        return x
    mtp.layer = spy
    hidden = mx.random.normal((1, 3, h))
    next_ids = mx.array([[1, 2, 3]])
    mtp(hidden, next_ids, lambda z: z)
    mtp.layer = orig_layer
    # emb 贡献被置零、hidden 经 rms_norm(weight=1) 后过 fc(=I) => 等于 rms_norm(hidden)
    expected = mx.fast.rms_norm(hidden, mx.ones((h,)), args.rms_norm_eps)
    assert float(mx.abs(captured["x"] - expected).max()) < 1e-4
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_spike.py -k mtp -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'mlx_streaming.qwen3_next_mtp'`

- [ ] **Step 3: 写实现**

```python
# mlx_streaming/qwen3_next_mtp.py
"""Qwen3-Next MTP(多 token 预测)模块的 MLX 实现,复用 mlx-lm 现成子模块类。

前向(与 vLLM/sglang/trtllm 一致):
  emb = pre_fc_norm_embedding(embed(next_id))
  hid = pre_fc_norm_hidden(主模型 last-layer hidden, norm 之前)
  x   = fc(concat([emb, hid], axis=-1))     # emb 在前
  x   = layer(x)                             # 全注意力 + MoE 解码层(内部含残差)
  logits = lm_head(norm(x))
"""
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen3_next import ModelArgs, Qwen3NextDecoderLayer


class Qwen3NextMTP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        h = args.hidden_size
        eps = args.rms_norm_eps
        self.embed_tokens = nn.Embedding(args.vocab_size, h)
        self.pre_fc_norm_embedding = nn.RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(h, eps=eps)
        self.fc = nn.Linear(2 * h, h, bias=False)
        # layer_idx=3 配合 full_attention_interval=4 -> 全注意力 + MoE
        self.layer = Qwen3NextDecoderLayer(args, layer_idx=3)
        self.norm = nn.RMSNorm(h, eps=eps)

    def __call__(
        self,
        hidden: mx.array,           # (B, L, H) 主模型 last-layer hidden(norm 前)
        next_ids: mx.array,         # (B, L) 每位置的"下一个" token id
        lm_head: Callable[[mx.array], mx.array],
        cache: Optional[Any] = None,
    ) -> mx.array:
        emb = self.pre_fc_norm_embedding(self.embed_tokens(next_ids))
        hid = self.pre_fc_norm_hidden(hidden)
        x = self.fc(mx.concatenate([emb, hid], axis=-1))
        mask = create_attention_mask(x, cache) if cache is not None else "causal"
        x = self.layer(x, mask=mask, cache=cache)
        return lm_head(self.norm(x))


def load_mtp(args: ModelArgs, weights_path: str) -> Qwen3NextMTP:
    """加载抽取好的 MTP 权重(已 stack 专家、已 norm +1.0)到模块。"""
    model = Qwen3NextMTP(args)
    raw = mx.load(weights_path)
    # 去掉 'mtp.' 前缀,映射到模块属性路径;mtp.layers.0.* -> layer.*
    renamed = {}
    for k, v in raw.items():
        nk = k[len("mtp."):] if k.startswith("mtp.") else k
        nk = nk.replace("layers.0.", "layer.", 1)
        renamed[nk] = v
    model.update(tree_unflatten(list(renamed.items())))
    mx.eval(model.parameters())
    return model
```

> 注意:`create_attention_mask` 在 `cache is None` 时本计划用字符串 `"causal"`(MLX SDPA 接受 `"causal"`)。若 mlx 版本不接受字符串,改为传 `create_attention_mask(x, None)`(在 Task 5 排错时确认)。

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_spike.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /Users/amos/project/flash-moe/hypura
git add mlx_streaming/qwen3_next_mtp.py mlx_streaming/tests/test_mtp_spike.py
git commit -m "feat: Qwen3NextMTP 模块(复用 mlx-lm 子模块 + 权重加载)"
```

---

## Task 4: 主模型 hidden 捕获 + 接受率测量脚本

**Files:**
- Create: `mlx_streaming/validate_mtp.py`
- Test: `mlx_streaming/tests/test_mtp_spike.py`(追加 hidden 捕获辅助的小测)

- [ ] **Step 1: 写失败测试(hidden 捕获辅助返回 norm 前 hidden)**

在 `mlx_streaming/tests/test_mtp_spike.py` 追加:

```python
from mlx_streaming.validate_mtp import capture_prenorm_hidden


class _FakeNorm:
    def __call__(self, x):
        return x * 0.0  # norm 后必为 0,便于区分


class _FakeInner:
    """模拟 Qwen3NextModel:embed -> layers -> norm。"""
    def __init__(self):
        self.norm = _FakeNorm()

    def embed_tokens(self, ids):
        return mx.ones((ids.shape[0], ids.shape[1], 4))

    @property
    def layers(self):
        return []

    def make_cache(self):
        return []


class _FakeModel:
    def __init__(self):
        self.model = _FakeInner()

    def make_cache(self):
        return []


def test_capture_prenorm_hidden_is_before_norm():
    m = _FakeModel()
    ids = mx.array([[1, 2, 3]])
    pre = capture_prenorm_hidden(m, ids)
    # norm 会把它清零;捕获的是 norm 之前 => 全 1
    assert float(pre.sum()) == 12.0  # 1*3*4
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_spike.py -k prenorm -v`
Expected: FAIL,`ModuleNotFoundError: No module named 'mlx_streaming.validate_mtp'`

- [ ] **Step 3: 写实现**

```python
# mlx_streaming/validate_mtp.py
"""teacher-forcing 实测 Qwen3-Next MTP 草稿接受率(spike,无生成循环)。

指标:
  mtp_vs_text_acc   : MTP 对 t_{i+2} 的 argmax 命中真实文本 token 的比例
  mtp_vs_greedy_acc : 先用主模型贪心生成参考序列 g,再以 g 为输入测命中 g_{i+2} 的比例
                      (真正的自投机接受率代理,决策主依据)
环境变量:MODEL / MTP_OUT(MTP 权重) / QN_CONFIG / PROMPT / MAXTOK
"""
import json
import os

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.qwen3_next_mtp import load_mtp

MODEL = os.environ.get("MODEL", "/tmp/qwen3_next_80b_4bit")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))


def capture_prenorm_hidden(model, input_ids: mx.array) -> mx.array:
    """跑主模型层循环但跳过最后的 model.norm,返回 last-layer hidden(norm 前)。"""
    inner = model.model
    h = inner.embed_tokens(input_ids)
    layers = inner.layers
    if not layers:
        return h
    cache = model.make_cache()
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx])
    ssm_mask = create_ssm_mask(h, cache[ssm_idx])
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
    return h


def _greedy(model, input_ids: mx.array, n: int) -> mx.array:
    ids = input_ids
    cache = model.make_cache()
    cur = ids
    out = []
    for _ in range(n):
        logits = model(cur, cache=cache)
        nxt = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        out.append(nxt)
        cur = nxt
        mx.eval(nxt)
    return mx.concatenate([ids] + out, axis=1)


def _acceptance(model, mtp, ids: mx.array) -> float:
    """teacher forcing:对 ids[0..L-1],MTP 预测 t_{i+2},与 ids[i+2] 比。"""
    hidden = capture_prenorm_hidden(model, ids)          # (1, L, H)
    # MTP 输入 next_ids = ids 右移一位(位置 i 喂"下一个" token = ids[i+1])
    next_ids = ids[:, 1:]                                  # (1, L-1)
    hid = hidden[:, :-1, :]                                # 对齐 (1, L-1, H)
    logits = mtp(hid, next_ids, model.lm_head)             # (1, L-1, vocab)
    pred = mx.argmax(logits, axis=-1)                      # 预测 t_{i+2}
    target = ids[:, 2:]                                    # (1, L-2)
    pred = pred[:, : target.shape[1]]
    match = (pred == target).astype(mx.float32)
    return float(match.mean())


def main():
    model, tok = load(MODEL)
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT)

    prompt_ids = mx.array([tok.encode(PROMPT)])
    text_ids = _greedy(model, prompt_ids, MAXTOK)         # 真实=主模型贪心也作"文本"
    mx.eval(text_ids)

    greedy_acc = _acceptance(model, mtp, text_ids)
    # 对"真实文本"指标:用 tokenizer 对一段自然文本编码
    nat = mx.array([tok.encode(PROMPT + "混合专家模型通过门控网络为每个 token 选择少量专家参与计算,从而在巨大参数量下保持较低的激活成本。")])
    text_acc = _acceptance(model, mtp, nat)

    print(json.dumps({
        "mtp_vs_greedy_acc": round(greedy_acc, 4),
        "mtp_vs_text_acc": round(text_acc, 4),
        "n_greedy_positions": int(text_ids.shape[1] - 2),
        "hidden_variant": "pre_final_norm",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m pytest mlx_streaming/tests/test_mtp_spike.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd /Users/amos/project/flash-moe/hypura
git add mlx_streaming/validate_mtp.py mlx_streaming/tests/test_mtp_spike.py
git commit -m "feat: MTP 接受率测量脚本(hidden 捕获 + teacher forcing)"
```

---

## Task 5: 端到端运行 + 决策

**Files:** 无新增(运行 + 记录)

- [ ] **Step 1: 抽取 MTP 权重(下 3.9GB)**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && python -m mlx_streaming.extract_mtp`
Expected: 打印「原始 mtp 张量数: 1553」与「已写出 ... -> /tmp/qn_mtp_weights.safetensors」。
排错:若下载卡死,检查 `_expected_size()` 是否拿到 content-length;ModelScope 偶发 503,重跑即可。

- [ ] **Step 2: 运行接受率测量**

Run: `cd /Users/amos/project/flash-moe/hypura && source mlx_streaming/.venv/bin/activate && MAXTOK=128 python -m mlx_streaming.validate_mtp`
Expected: 打印 JSON,含 `mtp_vs_greedy_acc`。
排错:
- 若 `mtp_vs_greedy_acc < 0.30`(疑似 hidden 接错),把 `capture_prenorm_hidden` 改为返回 `inner.norm(h)`(final-norm 后),再跑一次,两个结果都记录,取高者并把 `hidden_variant` 标为 `post_final_norm`。
- 若 `"causal"` mask 报错,改 `qwen3_next_mtp.py` 的 mask 行为 `create_attention_mask(x, None)`。

- [ ] **Step 3: 记录结果 + 给决策**

把两次(若有)结果写入 `benchmarks/reports/qwen3next-mtp-acceptance-2026-06-07.md`,含:两个接受率、hidden_variant、判据结论:
- `mtp_vs_greedy_acc` ≳ 0.60 → 值得做完整自投机(snapshot/restore 回滚 + 迭代草稿 + 流式集成);
- 0.30–0.60 → 边际,结合 ~2-3x 天花板权衡;
- < 0.30 → 放弃 MTP,回 35B 轹线或中档模型。

- [ ] **Step 4: Commit**

```bash
cd /Users/amos/project/flash-moe/hypura
git add benchmarks/reports/qwen3next-mtp-acceptance-2026-06-07.md
git commit -m "docs: Qwen3-Next MTP 接受率 spike 实测结果与决策"
```

---

## Self-Review(已执行)

- **Spec 覆盖**:下载+抽取(Task 1/2)、MTP 模块(Task 3)、teacher-forcing 接受率(Task 4)、两个指标与决策判据(Task 4/5)、hidden pre/post 消歧(Task 5 排错)、norm +1.0 隐藏坑(Task 1)、不动主路径(全程纯新增)——均有对应任务。
- **占位符**:无 TBD/TODO;每个代码步给了完整代码。
- **类型一致**:`stack_mtp_experts(weights, num_experts)`、`bump_mtp_norms(weights)`、`Qwen3NextMTP(args)`、`Qwen3NextMTP.__call__(hidden, next_ids, lm_head, cache)`、`load_mtp(args, weights_path)`、`capture_prenorm_hidden(model, input_ids)` 在测试与实现间签名一致。
- **已知风险点已在步骤内给出回退**:hidden pre/post、mask 字符串。
