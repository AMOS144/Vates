"""离线模拟 budgeted prefetch buffer 的收益。

输入为 router predictor NPZ。候选集通常取 `proxy - resident`，按 `proxy_score`
排序，在每个 (prompt, step) 内用全局预算和每层预算选取少量专家，统计能覆盖多少真实 miss。
"""
import argparse
import json

import mlx.core as mx
import numpy as np


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


def simulate_budgeted_prefetch_arrays(
    prompt_id: np.ndarray,
    step: np.ndarray,
    layer: np.ndarray,
    y: np.ndarray,
    candidate: np.ndarray,
    score: np.ndarray,
    global_budget: int,
    per_layer_budget: int,
) -> dict:
    """按 step 预算选择候选专家，返回覆盖/浪费统计。"""
    miss_total = int(y.sum())
    prefetch_total = hit_total = waste_total = 0
    groups = sorted(set(zip(prompt_id.tolist(), step.tolist(), strict=False)))
    for group in groups:
        idxs = np.flatnonzero((prompt_id == group[0]) & (step == group[1]))
        ranked = []
        for i in idxs:
            cand_experts = np.flatnonzero(candidate[i])
            if cand_experts.size == 0:
                continue
            order = cand_experts[np.argsort(score[i, cand_experts])[::-1]]
            for e in order[:per_layer_budget]:
                ranked.append((float(score[i, e]), int(i), int(e)))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _s, i, e in ranked[:global_budget]:
            prefetch_total += 1
            if y[i, e]:
                hit_total += 1
            else:
                waste_total += 1
    return {
        "miss_total": miss_total,
        "prefetch_total": prefetch_total,
        "hit_total": hit_total,
        "waste_total": waste_total,
        "covered_miss_ratio": round(hit_total / max(1, miss_total), 4),
        "waste_ratio": round(waste_total / max(1, prefetch_total), 4),
        "avg_prefetch_per_step": round(prefetch_total / max(1, len(groups)), 2),
        "steps": len(groups),
    }


def _reranker_scores(d, candidate, model_path: str) -> np.ndarray:
    params = mx.load(model_path)
    layer = d["layer"].astype(np.int16)
    proxy_score = d["proxy_score"].astype(np.float32) if "proxy_score" in d else candidate.astype(np.float32)
    proxy_sum = d["proxy_sum"].astype(np.float32) if "proxy_sum" in d else proxy_score
    proxy_count = d["proxy_count"].astype(np.float32) if "proxy_count" in d else candidate.astype(np.float32)
    resident_rank = d["resident_rank"].astype(np.float32) if "resident_rank" in d else np.zeros_like(proxy_score)
    freq_score = d["freq_score"].astype(np.float32) if "freq_score" in d else np.zeros_like(proxy_score)
    hot = d["hot"].astype(np.float32) if "hot" in d else np.zeros_like(proxy_score)
    transition = d["transition"].astype(np.float32) if "transition" in d else np.zeros_like(proxy_score)
    prev_same = d["prev_same"].astype(np.float32) if "prev_same" in d else np.zeros_like(proxy_score)
    prev_layer = d["prev_layer"].astype(np.float32) if "prev_layer" in d else np.zeros_like(proxy_score)
    score = np.full(candidate.shape, -1e9, dtype=np.float32)
    for i in range(candidate.shape[0]):
        experts = np.flatnonzero(candidate[i])
        if experts.size == 0:
            continue
        feat = np.stack([
            proxy_score[i, experts],
            proxy_sum[i, experts],
            proxy_count[i, experts],
            resident_rank[i, experts],
            freq_score[i, experts],
            hot[i, experts],
            transition[i, experts],
            prev_same[i, experts],
            prev_layer[i, experts],
        ], axis=1).astype(np.float32)
        li = np.full((experts.size,), int(layer[i]), dtype=np.int32)
        ex = experts.astype(np.int32)
        x = mx.concatenate([
            mx.array(feat),
            params["layer_emb"][mx.array(li)],
            params["expert_emb"][mx.array(ex)],
        ], axis=1)
        h = mx.maximum(x @ params["w1"] + params["b1"], 0)
        logits = (h @ params["w2"] + params["b2"]).squeeze(-1)
        mx.eval(logits)
        score[i, experts] = np.array(logits)
    return score


def _load_npz(path: str, layers: set[int] | None, candidate_field: str,
              exclude_resident: bool, target_minus_resident: bool,
              reranker_model: str | None):
    d = np.load(path, allow_pickle=False)
    prompt_id = d["prompt_id"].astype(np.int32)
    step = d["step"].astype(np.int32)
    layer = d["layer"].astype(np.int32)
    y = d["y"].astype(np.uint8)
    resident = d["resident"].astype(np.uint8) if "resident" in d else None
    if target_minus_resident:
        if resident is None:
            raise ValueError("--target-minus-resident 需要数据包含 resident")
        y = y & (1 - resident)
    candidate = d[candidate_field].astype(np.uint8)
    if exclude_resident and resident is not None:
        candidate = candidate & (1 - resident)
    if layers is not None:
        mask = np.array([int(l) in layers for l in layer], dtype=bool)
        prompt_id, step, layer = prompt_id[mask], step[mask], layer[mask]
        y, candidate = y[mask], candidate[mask]
        # 为 reranker 构造一个轻量视图，避免给非目标层打分。
        d = {k: v[mask] if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0] else v
             for k, v in d.items()}
    score = (_reranker_scores(d, candidate, reranker_model)
             if reranker_model else
             (d["proxy_score"].astype(np.float32) if "proxy_score" in d else candidate.astype(np.float32)))
    return prompt_id, step, layer, y, candidate, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--global-budget", type=int, nargs="+", default=[16, 32, 64, 96])
    ap.add_argument("--per-layer-budget", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--layers", default="0,40-47")
    ap.add_argument("--candidate-field", default="proxy")
    ap.add_argument("--include-resident", action="store_true")
    ap.add_argument("--target-minus-resident", action="store_true")
    ap.add_argument("--reranker-model")
    args = ap.parse_args()
    arrays = _load_npz(
        args.data, _parse_layers(args.layers), args.candidate_field,
        exclude_resident=not args.include_resident,
        target_minus_resident=args.target_minus_resident,
        reranker_model=args.reranker_model)
    rows = []
    for gb in args.global_budget:
        for lb in args.per_layer_budget:
            rec = simulate_budgeted_prefetch_arrays(
                *arrays, global_budget=gb, per_layer_budget=lb)
            rec.update({"global_budget": gb, "per_layer_budget": lb})
            rows.append(rec)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
