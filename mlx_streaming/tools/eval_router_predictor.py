"""离线评估 layer-wise router predictor 的 top-N recall。"""
import json
import os

import mlx.core as mx
import numpy as np

DATA = os.environ.get("ROUTER_PRED_DATA", "/tmp/router_pred_data.npz")
MODEL = os.environ.get("ROUTER_PRED_MODEL", "/tmp/router_predictor.safetensors")
TOPS = [int(x) for x in os.environ.get("TOPS", "32,48,64,96").split(",")]
BATCH = int(os.environ.get("BATCH", "512"))
USE_HISTORY = os.environ.get("USE_HISTORY", "1") == "1"
EVAL_LAYERS = os.environ.get("EVAL_LAYERS", "")
EVAL_MISS_FROM_ACTUAL = os.environ.get("EVAL_MISS_FROM_ACTUAL", "0") == "1"
CANDIDATE_FIELD = os.environ.get("CANDIDATE_FIELD", "")
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
    return out.astype(np.uint8)


def main():
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
    y = d["y"].astype(np.uint8)
    resident = d["resident"].astype(np.uint8) if "resident" in d else None
    cand = None
    if CANDIDATE_FIELD:
        cand = d[CANDIDATE_FIELD].astype(np.uint8)
        if EXCLUDE_RESIDENT and resident is not None:
            cand = cand & (1 - resident)
    if EVAL_MISS_FROM_ACTUAL:
        if resident is None:
            raise ValueError("EVAL_MISS_FROM_ACTUAL=1 需要数据包含 resident 特征")
        # 目标变成 actual - resident；候选也会在评估时扣掉 resident。
        y = np.clip(y - (y & resident), 0, 1).astype(np.uint8)
    if FUTURE_MISS_STEPS > 0:
        if "prompt_id" not in d or "step" not in d:
            raise ValueError("FUTURE_MISS_STEPS 需要数据包含 prompt_id/step")
        y = _future_union_targets(
            y, layer, d["prompt_id"].astype(np.int64),
            d["step"].astype(np.int64), FUTURE_MISS_STEPS)
    keep_layers = _parse_layers(EVAL_LAYERS)
    if keep_layers is not None:
        mask = np.array([int(l) in keep_layers for l in layer], dtype=bool)
        x, layer, y = x[mask], layer[mask], y[mask]
        if resident is not None:
            resident = resident[mask]
        if cand is not None:
            cand = cand[mask]
    params = mx.load(MODEL)
    positives = y.sum(axis=1)
    pos_mask = positives > 0
    sums = {k: 0.0 for k in TOPS}
    missed = {k: 0 for k in TOPS}
    pos_sums = {k: 0.0 for k in TOPS}
    pos_missed = {k: 0 for k in TOPS}
    n = x.shape[0]
    for start in range(0, n, BATCH):
        end = min(n, start + BATCH)
        logits = _forward(params, mx.array(x[start:end]), mx.array(layer[start:end]))
        mx.eval(logits)
        scores = np.array(logits)
        yy = y[start:end]
        pos = positives[start:end]
        rr = resident[start:end] if (EVAL_MISS_FROM_ACTUAL and resident is not None) else None
        if rr is not None:
            scores = np.where(rr > 0, -1e9, scores)
        cc = cand[start:end] if cand is not None else None
        if cc is not None:
            scores = np.where(cc > 0, scores, -1e9)
        for k in TOPS:
            idx = np.argpartition(scores, -k, axis=1)[:, -k:]
            hit = np.take_along_axis(yy, idx, axis=1).sum(axis=1)
            sums[k] += float(np.sum(hit / np.maximum(pos, 1)))
            missed[k] += int(np.sum(hit < pos))
            local_pos = pos > 0
            if np.any(local_pos):
                pos_sums[k] += float(np.sum(hit[local_pos] / pos[local_pos]))
                pos_missed[k] += int(np.sum(hit[local_pos] < pos[local_pos]))
    rows = []
    for k in TOPS:
        rows.append({
            "top": k,
            "recall": round(sums[k] / max(1, n), 4),
            "avg_pred_experts": k,
            "missed_samples": missed[k],
            "missed_frac": round(missed[k] / max(1, n), 4),
            "positive_recall": round(pos_sums[k] / max(1, int(pos_mask.sum())), 4),
            "positive_missed_samples": pos_missed[k],
            "positive_missed_frac": round(pos_missed[k] / max(1, int(pos_mask.sum())), 4),
        })
    print(json.dumps({
        "data": DATA,
        "model": MODEL,
        "samples": int(n),
        "positive_samples": int(pos_mask.sum()),
        "positive_frac": round(float(np.mean(pos_mask)) if n else 0.0, 4),
        "feature_parts": feature_parts,
        "eval_layers": sorted(keep_layers or []),
        "candidate_field": CANDIDATE_FIELD,
        "exclude_resident": EXCLUDE_RESIDENT,
        "future_miss_steps": FUTURE_MISS_STEPS,
        "rows": rows,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
