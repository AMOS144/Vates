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
        return_hidden: bool = False,
    ):
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


def mtp_step(mtp, hidden, token, lm_head, cache):
    """单步:hidden(1,1,H) + token(1,1) -> (logits(1,V), mtp_hidden(1,1,H))。"""
    logits, H = mtp(hidden, token, lm_head, cache=cache, return_hidden=True)
    return logits[:, -1, :], H[:, -1:, :]


def mtp_advance(mtp, hidden, token, cache):
    """只推进 MTP cache 并返回 hidden,不计算 lm_head logits。"""
    emb = mtp.pre_fc_norm_embedding(mtp.embed_tokens(token))
    hid = mtp.pre_fc_norm_hidden(hidden)
    x = mtp.fc(mx.concatenate([emb, hid], axis=-1))
    mask = create_attention_mask(x, cache) if cache is not None else "causal"
    x = mtp.layer(x, mask=mask, cache=cache)
    return mtp.norm(x)


def load_mtp(args: ModelArgs, weights_path: str, quantize: bool = True,
             bits: int = 4) -> Qwen3NextMTP:
    """加载抽取好的 MTP 权重(已 stack 专家、已 norm +1.0)到模块。

    quantize=True 时按主模型约定量化:线性层用 `bits`-bit(默认 4),gate/shared_expert_gate
    恒用 8-bit。quantize=False 保持 bf16 全精度——草稿质量最高、接受率最高,但显存更大。
    提高 bits(4→8)或 quantize=False 是直接抬升 MTP 草稿接受率的杠杆。
    """
    model = Qwen3NextMTP(args)
    raw = mx.load(weights_path)
    # 去掉 'mtp.' 前缀,映射到模块属性路径;mtp.layers.0.* -> layer.*
    renamed = {}
    for k, v in raw.items():
        nk = k[len("mtp."):] if k.startswith("mtp.") else k
        nk = nk.replace("layers.0.", "layer.", 1)
        renamed[nk] = v
    model.update(tree_unflatten(list(renamed.items())))
    if quantize:
        def pred(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True
        nn.quantize(model, group_size=64, bits=bits, class_predicate=pred)
    mx.eval(model.parameters())
    return model
