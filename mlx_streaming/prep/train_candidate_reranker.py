"""训练 candidate-level miss expert reranker。

和 layer-wise 512 logits 不同，本脚本把每个候选 expert 展开成一条样本：
  (layer, expert, per-expert features) -> score

目标是在 proxy-resident 候选集内，把 miss experts 排到前面。
"""
import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

DATA = os.environ.get("ROUTER_PRED_DATA", "/tmp/router_pred_50p_miss_resident_proxy40_hist_trans.npz")
OUT = os.environ.get("CAND_RERANK_OUT", "/tmp/candidate_reranker.safetensors")
TRAIN_LAYERS = os.environ.get("TRAIN_LAYERS", "0,40-47")
EPOCHS = int(os.environ.get("EPOCHS", "10"))
BATCH = int(os.environ.get("BATCH", "4096"))
GROUP_BATCH = int(os.environ.get("GROUP_BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
HIDDEN = int(os.environ.get("CAND_HIDDEN", "64"))
LAYER_EMB = int(os.environ.get("LAYER_EMB", "16"))
EXPERT_EMB = int(os.environ.get("EXPERT_EMB", "16"))
POS_WEIGHT = float(os.environ.get("POS_WEIGHT", "16.0"))
SEED = int(os.environ.get("SEED", "0"))
TRAIN_MODE = os.environ.get("TRAIN_MODE", "pointwise")  # pointwise | listwise
LISTWISE_BCE_WEIGHT = float(os.environ.get("LISTWISE_BCE_WEIGHT", "0.0"))
TARGET_MINUS_RESIDENT = os.environ.get("TARGET_MINUS_RESIDENT", "0") == "1"


def _parse_layers(s: str) -> set[int]:
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


def _candidate_rows(d):
    layers_keep = _parse_layers(TRAIN_LAYERS)
    layer = d["layer"].astype(np.int16)
    y = d["y"].astype(np.uint8)
    proxy = d["proxy"].astype(np.uint8)
    resident = d["resident"].astype(np.uint8)
    if TARGET_MINUS_RESIDENT:
        y = y & (1 - resident)
    cand = proxy & (1 - resident)
    proxy_score = d["proxy_score"].astype(np.float32) if "proxy_score" in d else proxy.astype(np.float32)
    proxy_sum = d["proxy_sum"].astype(np.float32) if "proxy_sum" in d else proxy_score
    proxy_count = d["proxy_count"].astype(np.float32) if "proxy_count" in d else proxy.astype(np.float32)
    resident_rank = d["resident_rank"].astype(np.float32) if "resident_rank" in d else resident.astype(np.float32)
    freq_score = d["freq_score"].astype(np.float32) if "freq_score" in d else np.zeros_like(proxy_score)
    hot = d["hot"].astype(np.float32) if "hot" in d else np.zeros_like(proxy_score)
    transition = d["transition"].astype(np.float32) if "transition" in d else np.zeros_like(proxy_score)
    prev_same = d["prev_same"].astype(np.float32) if "prev_same" in d else np.zeros_like(proxy_score)
    prev_layer = d["prev_layer"].astype(np.float32) if "prev_layer" in d else np.zeros_like(proxy_score)

    rows_layer, rows_expert, rows_feat, rows_y, rows_group = [], [], [], [], []
    for i in range(layer.shape[0]):
        li = int(layer[i])
        if li not in layers_keep:
            continue
        experts = np.flatnonzero(cand[i])
        if experts.size == 0:
            continue
        feats = np.stack([
            proxy_score[i, experts],
            proxy_sum[i, experts],
            proxy_count[i, experts],
            resident_rank[i, experts],
            freq_score[i, experts],
            hot[i, experts],
            transition[i, experts],
            prev_same[i, experts],
            prev_layer[i, experts],
        ], axis=1)
        rows_layer.append(np.full((experts.size,), li, dtype=np.int16))
        rows_expert.append(experts.astype(np.int16))
        rows_feat.append(feats.astype(np.float32))
        rows_y.append(y[i, experts].astype(np.float32))
        rows_group.append(np.full((experts.size,), i, dtype=np.int32))
    return {
        "layer": np.concatenate(rows_layer),
        "expert": np.concatenate(rows_expert),
        "feat": np.concatenate(rows_feat),
        "y": np.concatenate(rows_y),
        "group": np.concatenate(rows_group),
        "num_experts": y.shape[1],
        "num_layers": int(layer.max()) + 1,
    }


def _init(num_layers: int, num_experts: int, feat_dim: int):
    mx.random.seed(SEED)
    inp = feat_dim + LAYER_EMB + EXPERT_EMB
    return {
        "layer_emb": mx.random.normal((num_layers, LAYER_EMB)) * 0.02,
        "expert_emb": mx.random.normal((num_experts, EXPERT_EMB)) * 0.02,
        "w1": mx.random.normal((inp, HIDDEN)) * (inp ** -0.5),
        "b1": mx.zeros((HIDDEN,)),
        "w2": mx.random.normal((HIDDEN, 1)) * (HIDDEN ** -0.5),
        "b2": mx.zeros((1,)),
    }


def _forward(params, feat, layer, expert):
    x = mx.concatenate([feat, params["layer_emb"][layer], params["expert_emb"][expert]], axis=1)
    h = mx.maximum(x @ params["w1"] + params["b1"], 0)
    return (h @ params["w2"] + params["b2"]).squeeze(-1)


def _loss(params, feat, layer, expert, y):
    logits = _forward(params, feat, layer, expert)
    base = mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits)))
    weight = mx.where(y > 0, POS_WEIGHT, 1.0)
    return mx.mean(base * weight)


def _listwise_loss(params, feat, layer, expert, y, offsets):
    """每个 step-layer 候选集合一个 list，优化正例在该 list 内排名靠前。"""
    logits = _forward(params, feat, layer, expert)
    total = mx.array(0.0)
    n_groups = 0
    for a, b in offsets:
        yy = y[a:b]
        pos = mx.sum(yy)
        if int(pos) <= 0:
            continue
        ll = logits[a:b]
        # 多正例 listwise：-mean(log softmax(pos))。
        total = total + mx.logsumexp(ll) - mx.sum(ll * yy) / pos
        n_groups += 1
    loss = total / max(1, n_groups)
    if LISTWISE_BCE_WEIGHT > 0:
        base = mx.maximum(logits, 0) - logits * y + mx.log1p(mx.exp(-mx.abs(logits)))
        weight = mx.where(y > 0, POS_WEIGHT, 1.0)
        loss = loss + LISTWISE_BCE_WEIGHT * mx.mean(base * weight)
    return loss


def _eval_topk(params, rows, groups, ks=(16, 32, 64)):
    feat = mx.array(rows["feat"])
    layer = mx.array(rows["layer"])
    expert = mx.array(rows["expert"])
    logits = _forward(params, feat, layer, expert)
    mx.eval(logits)
    score = np.array(logits)
    y = rows["y"]
    out = {}
    unique_groups = np.unique(groups)
    for k in ks:
        recs, missed = [], 0
        for g in unique_groups:
            idx = np.flatnonzero(groups == g)
            pos = y[idx].sum()
            if pos <= 0:
                continue
            top = idx[np.argsort(score[idx])[-min(k, idx.size):]]
            hit = y[top].sum()
            recs.append(hit / pos)
            missed += int(hit < pos)
        out[f"top{k}_recall"] = float(np.mean(recs)) if recs else 0.0
        out[f"top{k}_missed_frac"] = missed / max(1, len(recs))
    return out


def _group_offsets(group_ids, rows_group):
    ids = []
    offsets = []
    pos = 0
    for g in group_ids:
        idx = np.flatnonzero(rows_group == g)
        ids.append(idx)
        offsets.append((pos, pos + len(idx)))
        pos += len(idx)
    return np.concatenate(ids), offsets


def main():
    d = np.load(DATA, allow_pickle=False)
    rows = _candidate_rows(d)
    rng = np.random.default_rng(SEED)
    groups = np.unique(rows["group"])
    rng.shuffle(groups)
    split = max(1, int(len(groups) * 0.9))
    train_groups, val_groups = set(groups[:split]), set(groups[split:])
    tr = np.array([g in train_groups for g in rows["group"]])
    va = np.array([g in val_groups for g in rows["group"]])
    params = _init(rows["num_layers"], rows["num_experts"], rows["feat"].shape[1])
    loss_and_grad = mx.value_and_grad(_listwise_loss if TRAIN_MODE == "listwise" else _loss)
    best_params, best_score, best_epoch = params, -1.0, 0

    idx_all = np.flatnonzero(tr)
    train_group_list = np.array(sorted(train_groups))
    for epoch in range(EPOCHS):
        losses = []
        if TRAIN_MODE == "listwise":
            rng.shuffle(train_group_list)
            for start in range(0, len(train_group_list), GROUP_BATCH):
                gids = train_group_list[start:start + GROUP_BATCH]
                ids, offsets = _group_offsets(gids, rows["group"])
                loss, grads = loss_and_grad(
                    params,
                    mx.array(rows["feat"][ids]),
                    mx.array(rows["layer"][ids]),
                    mx.array(rows["expert"][ids]),
                    mx.array(rows["y"][ids]),
                    offsets,
                )
                params = {k: params[k] - LR * grads[k] for k in params}
                mx.eval(params)
                losses.append(float(loss))
        else:
            rng.shuffle(idx_all)
            for start in range(0, len(idx_all), BATCH):
                ids = idx_all[start:start + BATCH]
                loss, grads = loss_and_grad(
                    params,
                    mx.array(rows["feat"][ids]),
                    mx.array(rows["layer"][ids]),
                    mx.array(rows["expert"][ids]),
                    mx.array(rows["y"][ids]),
                )
                params = {k: params[k] - LR * grads[k] for k in params}
                mx.eval(params)
                losses.append(float(loss))
        val_rows = {k: v[va] if isinstance(v, np.ndarray) and v.shape[0] == va.shape[0] else v
                    for k, v in rows.items()}
        val = _eval_topk(params, val_rows, rows["group"][va])
        score = val.get("top64_recall", 0.0)
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            best_params = {k: mx.array(v) for k, v in params.items()}
            mx.eval(best_params)
        print(json.dumps({
            "epoch": epoch + 1,
            "loss": round(float(np.mean(losses)), 4),
            "best_epoch": best_epoch,
            "best_top64": round(best_score, 4),
            **{k: round(v, 4) for k, v in val.items()},
        }, ensure_ascii=False), flush=True)

    mx.save_safetensors(OUT, best_params)
    Path(OUT + ".json").write_text(json.dumps({
        "data": DATA,
        "train_layers": sorted(_parse_layers(TRAIN_LAYERS)),
        "feat_dim": int(rows["feat"].shape[1]),
        "rows": int(rows["y"].shape[0]),
        "positive_rate": float(rows["y"].mean()),
        "train_mode": TRAIN_MODE,
        "group_batch": GROUP_BATCH,
        "listwise_bce_weight": LISTWISE_BCE_WEIGHT,
        "target_minus_resident": TARGET_MINUS_RESIDENT,
        "best_epoch": best_epoch,
        "best_score": best_score,
    }, ensure_ascii=False, indent=2))
    print(json.dumps({"out": OUT, "meta": OUT + ".json"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
