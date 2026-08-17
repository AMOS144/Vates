"""Qwen3-Next MTP(多 token 预测)模块的 MLX 实现,复用 mlx-lm 现成子模块类。

前向(与 vLLM/sglang/trtllm 一致):
  emb = pre_fc_norm_embedding(embed(next_id))
  hid = pre_fc_norm_hidden(主模型 last-layer hidden, norm 之前)
  x   = fc(concat([emb, hid], axis=-1))     # emb 在前
  x   = layer(x)                             # 全注意力 + MoE 解码层(内部含残差)
  logits = lm_head(norm(x))
"""
from typing import Any, Callable, Optional
import os

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


class StreamingMTPMoE(nn.Module):
    """Small file-backed cache for the single MTP routed-expert layer.

    Draft experts are not authoritative model weights: speculative decoding
    always verifies their tokens with the main model.  Keeping only a bounded
    MTP expert working set therefore reduces memory without changing committed
    output semantics.
    """

    def __init__(self, original, root: str, capacity: int, *, bits: int,
                 group_size: int, hidden: int = 2048,
                 intermediate: int = 512):
        super().__init__()
        from mlx_streaming.core.cache.resident_pool import ResidentExpertPool
        from mlx_streaming.core.cache.blob_loader import BlobExpertSource
        from mlx_streaming.core.moe.compute import PersistentSubGLU

        self.gate = original.gate
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate
        self.top_k = int(original.top_k)
        self.norm_topk_prob = bool(original.norm_topk_prob)
        self.root = root
        blob_path = os.path.join(root, "layer100.blob")
        self._blob = (
            BlobExpertSource(
                root, int(hidden), int(intermediate), int(group_size),
                int(bits), 512, workers=8, nocache=False,
            )
            if os.path.exists(blob_path) else None
        )
        loader = (
            (lambda layer, expert: self._blob.load_experts(layer, [expert])[expert])
            if self._blob is not None else self._load_expert
        )
        self._pool = ResidentExpertPool(
            int(capacity), loader=loader,
            batch_loader=(self._blob.load_experts if self._blob is not None else None),
            stacked_batch_loader=(
                self._blob.load_experts_stacked if self._blob is not None else None
            ),
        )
        self._compute = PersistentSubGLU(
            int(hidden), int(intermediate), int(group_size), int(bits),
            layer_idx=100,
        )

    def _load_expert(self, _layer: int, expert: int) -> dict:
        return dict(mx.load(os.path.join(
            self.root, f"expert{int(expert):03d}.safetensors",
        )))

    def __call__(self, x: mx.array) -> mx.array:
        raw = self.gate(x)
        gates = mx.softmax(raw, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        flat = [int(value) for value in inds.reshape(-1).tolist()]
        pool, slots = self._pool.acquire(100, flat)
        active_slots = self._pool.allocated_slots(100)
        local = mx.array(slots, dtype=inds.dtype).reshape(inds.shape)
        y = self._compute.forward(
            pool, active_slots, x, local,
        )
        y = (y * scores[..., None]).sum(axis=-2)
        return y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


def load_mtp(args: ModelArgs, weights_path: str, quantize: bool = True,
             bits: int = 4, group_size: int = 64, stream_experts: bool = False,
             expert_dir: str = "", expert_slots: int = 32) -> Qwen3NextMTP:
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
    if stream_experts:
        # The routed MTP experts already live in the external blob.  Do not
        # materialize and quantize their 512-way BF16 checkpoint tensors just
        # to discard them below: that transient allocation was the remaining
        # >10 GiB startup peak of the otherwise sub-10 GiB runtime path.
        renamed = {
            key: value for key, value in renamed.items()
            if not key.startswith("layer.mlp.switch_mlp.")
        }
        model.layer.mlp.switch_mlp = nn.Identity()
    model.update(tree_unflatten(list(renamed.items())))
    if quantize:
        def pred(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": group_size, "bits": 8}
            return True
        nn.quantize(
            model, group_size=group_size, bits=bits,
            class_predicate=pred,
        )
    mx.eval(model.parameters())
    if stream_experts:
        has_blob = bool(expert_dir) and os.path.exists(
            os.path.join(expert_dir, "layer100.blob"),
        )
        has_files = bool(expert_dir) and os.path.exists(
            os.path.join(expert_dir, "expert000.safetensors"),
        )
        if not (has_blob or has_files):
            raise FileNotFoundError(
                f"MTP streamed expert directory is incomplete: {expert_dir}",
            )
        model.layer.mlp = StreamingMTPMoE(
            model.layer.mlp, expert_dir, expert_slots,
            bits=bits, group_size=group_size,
            hidden=int(args.hidden_size),
            intermediate=int(args.moe_intermediate_size),
        )
        mx.eval(model.parameters())
        mx.clear_cache()
    return model
