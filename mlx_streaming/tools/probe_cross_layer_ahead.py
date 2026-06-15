"""评估 hidden_l 提前预测 layer l+1 / l+2 专家的可行性。"""
import json
import os
from collections import defaultdict

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.qwen3_next import ModelArgs

import mlx_streaming.mtp.generate as mg
from mlx_streaming.core import route_trace
from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "48"))
K = int(os.environ.get("K", "3"))
AHEADS = [int(x) for x in os.environ.get("AHEADS", "1,2").split(",")]
MULTS = [int(x) for x in os.environ.get("AHEAD_MULTS", "1,2,4").split(",")]

STATS = defaultdict(lambda: defaultdict(float))


def _blocks(model) -> dict[int, FileStreamingMoeBlock]:
    out = {}
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FileStreamingMoeBlock):
            out[i] = mlp
    return out


def _predict(blk: FileStreamingMoeBlock, h: mx.array, mult: int) -> set[int]:
    gates = mx.softmax(blk.gate(h), axis=-1, precise=True)
    k = min(gates.shape[-1], blk.top_k * mult)
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(inds)
    return {int(e) for e in inds.reshape(-1).tolist()}


def _score(key: str, pred: set[int], actual: set[int]):
    s = STATS[key]
    hit = len(pred & actual)
    s["layers"] += 1
    s["hit"] += hit
    s["pred"] += len(pred)
    s["actual"] += len(actual)
    s["recall_sum"] += hit / max(1, len(actual))
    s["precision_sum"] += hit / max(1, len(pred))
    s["missed_layers"] += int(hit < len(actual))


def make_forward(model):
    block_by_layer = _blocks(model)

    def instrumented_forward_with_hidden(model, ids, cache):
        inner = model.model
        h = inner.embed_tokens(ids)
        layers = inner.layers
        has_full = any(not l.is_linear for l in layers)
        has_linear = any(l.is_linear for l in layers)
        fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
        ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
        fa_mask = create_attention_mask(h, cache[fa_idx]) if has_full else None
        ssm_mask = create_ssm_mask(h, cache[ssm_idx]) if has_linear else None
        pending: dict[int, list[tuple[str, set[int]]]] = defaultdict(list)
        for i, layer in enumerate(layers):
            # 在执行 layer i 前，用当前 hidden 预测 i+a 的 experts。
            h_proxy = layer.post_attention_layernorm(h)
            for ahead in AHEADS:
                target_idx = i + ahead
                blk = block_by_layer.get(target_idx)
                if blk is None:
                    continue
                for mult in MULTS:
                    key = f"ahead{ahead}_x{mult}"
                    pending[target_idx].append((key, _predict(blk, h_proxy, mult)))

            mask = ssm_mask if layer.is_linear else fa_mask
            route_trace.enable()
            h = layer(h, mask=mask, cache=cache[i])
            mx.eval(h)
            events = route_trace.events()
            route_trace.disable()
            if events and i in pending:
                actual = {int(e) for e in events[-1]["experts"]}
                for key, pred in pending[i]:
                    _score(key, pred, actual)
        H = inner.norm(h)
        return model.lm_head(H), H

    return instrumented_forward_with_hidden


def main():
    os.environ["ROUTE_TRACE"] = "1"
    model, tok, store = build_streaming_model()
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)
    old_forward = mg.forward_with_hidden
    mg.forward_with_hidden = make_forward(model)
    try:
        ids, mtp_stats = mtp_generate(
            model, drafter, tok, mx.array([tok.encode(PROMPT)]),
            MAXTOK, K=K, ids_mode=True, profile=False)
    finally:
        mg.forward_with_hidden = old_forward
    rows = []
    for key, s in sorted(STATS.items()):
        n = max(1, int(s["layers"]))
        rows.append({
            "predictor": key,
            "layers": n,
            "recall": round(s["recall_sum"] / n, 4),
            "precision": round(s["precision_sum"] / n, 4),
            "global_recall": round(s["hit"] / max(1, s["actual"]), 4),
            "global_precision": round(s["hit"] / max(1, s["pred"]), 4),
            "avg_pred_experts": round(s["pred"] / n, 2),
            "avg_actual_experts": round(s["actual"] / n, 2),
            "missed_layers": int(s["missed_layers"]),
        })
    print(json.dumps({
        "K": K,
        "tokens": len(ids),
        "steps": mtp_stats["steps"],
        "avg_accept_len": mtp_stats["avg_accept_len"],
        "store_misses": store.misses,
        "store_hits": store.hits,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
