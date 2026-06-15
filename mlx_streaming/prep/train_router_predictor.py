"""训练 layer-wise router predictor。

输入 `collect_router_pred_data.py` 产出的 NPZ，训练一个小共享 MLP：
  proxy_hidden + layer_embedding -> 512 expert logits

目标是 high-recall candidate set，而不是精确 top-k 分类。
"""
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

DATA = os.environ.get("ROUTER_PRED_DATA", "/tmp/router_pred_data.npz")
OUT = os.environ.get("ROUTER_PRED_OUT", "/tmp/router_predictor.safetensors")
EPOCHS = int(os.environ.get("EPOCHS", "5"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "1e-3"))
HIDDEN = int(os.environ.get("PRED_HIDDEN", "512"))
LAYER_EMB = int(os.environ.get("LAYER_EMB", "64"))
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "8.0"))
SEED = int(os.environ.get("SEED", "0"))
USE_HISTORY = os.environ.get("USE_HISTORY", "1") == "1"
DIRECT_HISTORY = os.environ.get("DIRECT_HISTORY", "0") == "1"
HEAD_MODE = os.environ.get("HEAD_MODE", "shared")  # shared | per_layer
LOSS = os.environ.get("LOSS", "bce")  # bce | bce_rank | bce_pairwise
RANK_WEIGHT = float(os.environ.get("RANK_WEIGHT", "1.0"))
RANK_MARGIN = float(os.environ.get("RANK_MARGIN", "1.0"))
TRAIN_LAYERS = os.environ.get("TRAIN_LAYERS", "")
BEST_TOP = int(os.environ.get("BEST_TOP", "64"))
CANDIDATE_FIELD = os.environ.get("CANDIDATE_FIELD", "")  # 例如 proxy
EXCLUDE_RESIDENT = os.environ.get("EXCLUDE_RESIDENT", "0") == "1"
FUTURE_MISS_STEPS = int(os.environ.get("FUTURE_MISS_STEPS", "0"))


def _parse_layers(s: str) -> set[int] | None:
    if not s:
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _load():
    d = np.load(DATA, allow_pickle=False)
    x = d["x"].astype(np.float32)
    feature_parts = ["x"]
    if USE_HISTORY:
        for name in ("proxy_score", "proxy_sum", "proxy_count", "proxy",
                     "resident", "resident_rank",
                     "freq_score", "prev_same", "hot", "prev_layer", "transition"):
            if name in d:
                x = np.concatenate([x, d[name].astype(np.float32)], axis=1)
                feature_parts.append(name)
    layer = d["layer"].astype(np.int64)
    y = d["y"].astype(np.float32)
    if FUTURE_MISS_STEPS > 0:
        if "prompt_id" not in d or "step" not in d:
            raise ValueError("FUTURE_MISS_STEPS 需要数据包含 prompt_id/step")
        y = _future_union_targets(
            y, layer, d["prompt_id"].astype(np.int64),
            d["step"].astype(np.int64), FUTURE_MISS_STEPS)
    cand = None
    if CANDIDATE_FIELD:
        cand = d[CANDIDATE_FIELD].astype(np.float32)
        if EXCLUDE_RESIDENT and "resident" in d:
            cand = cand * (1.0 - d["resident"].astype(np.float32))
    meta = json.loads(str(d["meta"]))
    keep_layers = _parse_layers(TRAIN_LAYERS)
    if keep_layers is not None:
        mask = np.array([int(l) in keep_layers for l in layer], dtype=bool)
        x, layer, y = x[mask], layer[mask], y[mask]
        if cand is not None:
            cand = cand[mask]
    meta["feature_parts"] = feature_parts
    return x, layer, y, meta, cand


def _future_union_targets(y, layer, prompt_id, step, horizon: int):
    yb = y.astype(np.uint8)
    out = np.zeros_like(yb)
    index = {}
    for i, key in enumerate(zip(prompt_id, layer, step, strict=False)):
        index[key] = i
    for i, (p, l, s) in enumerate(zip(prompt_id, layer, step, strict=False)):
        acc = np.zeros_like(yb[i])
        for dt in range(1, horizon + 1):
            j = index.get((p, l, s + dt))
            if j is not None:
                acc |= yb[j]
        out[i] = acc
    return out.astype(np.float32)


def _init(hidden_in: int, num_layers: int, num_experts: int, hist_parts: int):
    mx.random.seed(SEED)
    scale1 = (hidden_in + LAYER_EMB) ** -0.5
    scale2 = HIDDEN ** -0.5
    params = {
        "layer_emb": mx.random.normal((num_layers, LAYER_EMB)) * 0.02,
        "w1": mx.random.normal((hidden_in + LAYER_EMB, HIDDEN)) * scale1,
        "b1": mx.zeros((HIDDEN,)),
        **({"hist_scale": mx.ones((hist_parts,))} if DIRECT_HISTORY and hist_parts > 0 else {}),
    }
    if HEAD_MODE == "per_layer":
        params["w2_layer"] = mx.random.normal((num_layers, HIDDEN, num_experts)) * scale2
        params["b2_layer"] = mx.zeros((num_layers, num_experts))
    else:
        params["w2"] = mx.random.normal((HIDDEN, num_experts)) * scale2
        params["b2"] = mx.zeros((num_experts,))
    return params


def _forward(params, xb, lb):
    le = params["layer_emb"][lb]
    h = mx.concatenate([xb, le], axis=-1)
    h = mx.maximum(h @ params["w1"] + params["b1"], 0)
    if "w2_layer" in params:
        w = params["w2_layer"][lb]
        logits = mx.sum(h[:, :, None] * w, axis=1) + params["b2_layer"][lb]
        num_experts = int(params["b2_layer"].shape[1])
    else:
        logits = h @ params["w2"] + params["b2"]
        num_experts = int(params["b2"].shape[0])
    if "hist_scale" in params:
        n_parts = int(params["hist_scale"].shape[0])
        if int(xb.shape[1]) >= n_parts * num_experts:
            hist = xb[:, -n_parts * num_experts:]
            parts = mx.split(hist, n_parts, axis=1)
            s = params["hist_scale"]
            for i, part in enumerate(parts):
                logits = logits + s[i] * part
    return logits


def _loss(params, xb, lb, yb, cb=None):
    logits = _forward(params, xb, lb)
    if cb is not None:
        yb = yb * cb
    # 稳定 BCE-with-logits；正例加权，鼓励高召回。
    base = mx.maximum(logits, 0) - logits * yb + mx.log1p(mx.exp(-mx.abs(logits)))
    weight = mx.where(yb > 0, POS_WEIGHT, 1.0)
    if cb is not None:
        denom = mx.maximum(mx.sum(cb), 1.0)
        bce = mx.sum(base * weight * cb) / denom
    else:
        bce = mx.mean(base * weight)
    if LOSS == "bce":
        return bce
    # 排序召回项：要求每个正例 logit 高于该样本最强负例。
    neg_mask = (yb <= 0) if cb is None else ((yb <= 0) & (cb > 0))
    neg_logits = mx.where(neg_mask, logits, -1e9)
    if LOSS == "bce_pairwise":
        # 比 max-negative 更强：用 logsumexp 聚合所有候选负例，
        # 近似优化“每个 miss expert 排在候选 non-miss 前面”。
        neg_ref = mx.logsumexp(neg_logits, axis=1, keepdims=True)
    else:
        neg_ref = mx.max(neg_logits, axis=1, keepdims=True)
    rank = mx.logaddexp(0, neg_ref + RANK_MARGIN - logits) * yb
    rank = mx.sum(rank) / mx.maximum(mx.sum(yb), 1.0)
    return bce + RANK_WEIGHT * rank


def _topk_recall(params, x, layer, y, ks=(32, 48, 64), cand=None):
    xb = mx.array(x)
    lb = mx.array(layer)
    logits = _forward(params, xb, lb)
    mx.eval(logits)
    scores = np.array(logits)
    if cand is not None:
        scores = np.where(cand > 0, scores, -1e9)
    out = {}
    positives = y.sum(axis=1)
    mask = positives > 0
    out["positive_frac"] = float(np.mean(mask)) if len(mask) else 0.0
    if not np.any(mask):
        for k in ks:
            out[f"top{k}_recall"] = 0.0
        return out
    scores = scores[mask]
    y = y[mask]
    positives = positives[mask]
    for k in ks:
        idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        hit = np.take_along_axis(y, idx, axis=1).sum(axis=1)
        out[f"top{k}_recall"] = float(np.mean(hit / np.maximum(positives, 1)))
    return out


def main():
    x, layer, y, meta, cand = _load()
    n, hidden_in = x.shape
    num_layers = int(layer.max()) + 1
    num_experts = y.shape[1]
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    split = max(1, int(n * 0.9))
    tr, va = order[:split], order[split:]
    hist_parts = max(0, len(meta.get("feature_parts", ["x"])) - 1)
    params = _init(hidden_in, num_layers, num_experts, hist_parts)
    loss_and_grad = mx.value_and_grad(_loss)
    best_params = params
    best_score = -1.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        rng.shuffle(tr)
        losses = []
        for start in range(0, len(tr), BATCH):
            ids = tr[start:start + BATCH]
            xb = mx.array(x[ids])
            lb = mx.array(layer[ids])
            yb = mx.array(y[ids])
            cb = mx.array(cand[ids]) if cand is not None else None
            loss, grads = loss_and_grad(params, xb, lb, yb, cb)
            params = {k: params[k] - LR * grads[k] for k in params}
            mx.eval(params)
            losses.append(float(loss))
        val = _topk_recall(
            params, x[va], layer[va], y[va],
            cand=cand[va] if cand is not None and len(va) else None) if len(va) else {}
        score = val.get(f"top{BEST_TOP}_recall", -float(epoch))
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_params = {k: mx.array(v) for k, v in params.items()}
            mx.eval(best_params)
        print(json.dumps({
            "epoch": epoch + 1,
            "train_loss": round(float(np.mean(losses)), 4),
            "best_epoch": best_epoch,
            "best_score": round(float(best_score), 4),
            **{k: round(v, 4) for k, v in val.items()},
        }, ensure_ascii=False), flush=True)

    mx.save_safetensors(OUT, best_params)
    meta_out = {
        **meta,
        "model": "shared_mlp_layer_embedding",
        "head_mode": HEAD_MODE,
        "loss": LOSS,
        "rank_weight": RANK_WEIGHT,
        "rank_margin": RANK_MARGIN,
        "train_layers": sorted(_parse_layers(TRAIN_LAYERS) or []),
        "candidate_field": CANDIDATE_FIELD,
        "exclude_resident": EXCLUDE_RESIDENT,
        "future_miss_steps": FUTURE_MISS_STEPS,
        "best_top": BEST_TOP,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "hidden": hidden_in,
        "use_history": USE_HISTORY,
        "direct_history": DIRECT_HISTORY,
        "feature_parts": meta.get("feature_parts", ["x"]),
        "pred_hidden": HIDDEN,
        "layer_emb": LAYER_EMB,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "pos_weight": POS_WEIGHT,
        "data": DATA,
    }
    Path(OUT + ".json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2))
    print(json.dumps({"out": OUT, "meta": OUT + ".json"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
