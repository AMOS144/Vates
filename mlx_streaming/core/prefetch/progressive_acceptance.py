"""Occurrence-local acceptance counters for the production progressive path."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import Lock

import mlx.core as mx


_lock = Lock()
_rows = defaultdict(lambda: {
    "occurrences": 0,
    "actual_routes": 0,
    "raw_top64_hits": 0,
    "selected_hits": 0,
    "selected_width": 0,
    "max_selected_width": 0,
    "width_violations": 0,
    "outside_top64": 0,
})
_training_rows = []


def reset() -> None:
    with _lock:
        _rows.clear()
        _training_rows.clear()


def occurrence_metrics(
    candidate_ids,
    selected_ids,
    online_width,
    actual_ids,
    resident=(),
    proxy_logits=None,
    predictor_hidden=None,
    actual_logits=None,
) -> dict:
    arrays = [candidate_ids, selected_ids, online_width, actual_ids]
    if proxy_logits is not None:
        arrays.append(proxy_logits)
    if predictor_hidden is not None:
        arrays.append(predictor_hidden)
    if actual_logits is not None:
        arrays.append(actual_logits)
    mx.eval(*arrays)
    # Capture candidate IDs from the same device argpartition as production.
    # Recomputing in NumPy may choose a different bfloat16 boundary tie.
    candidates = {
        int(value) for value in candidate_ids.reshape(-1).tolist()
    }
    width = int(online_width.item())
    selected_raw = {
        int(value)
        for value in selected_ids.reshape(-1).tolist()[:width]
    }
    # The contract is occurrence-local *set* coverage.  K3 can route the same
    # expert from several speculative tokens; counting those repeated route
    # positions would silently overweight popular experts and can make a
    # reranker pass while missing too many members of the real union.
    actual = {int(value) for value in actual_ids.reshape(-1).tolist()}
    legal_width = (3 * len(actual)) // 2
    return {
        "actual_routes": len(actual),
        "raw_top64_hits": len(actual & candidates),
        "selected_hits": len(actual & selected_raw),
        "selected_width": len(selected_raw),
        "legal_width": legal_width,
        "width_violation": int(len(selected_raw) > legal_width),
        "outside_top64": len(selected_raw - candidates),
    }


def record(layer: int, **kwargs) -> dict:
    metrics = occurrence_metrics(**kwargs)
    proxy_logits = kwargs.get("proxy_logits")
    predictor_hidden = kwargs.get("predictor_hidden")
    actual_logits = kwargs.get("actual_logits")
    if proxy_logits is not None:
        import numpy as np

        # MLX bfloat16 currently exposes an incompatible PEP-3118 buffer
        # format to NumPy. ``tolist`` performs an explicit numeric conversion
        # after the eval barrier and is used only in opt-in offline capture.
        proxy = np.array(proxy_logits.tolist(), dtype=np.float16).reshape(
            -1, int(proxy_logits.shape[-1]),
        )
        actual = np.array(kwargs["actual_ids"].tolist(), dtype=np.int16).reshape(
            proxy.shape[0], -1,
        )
        hidden = (
            np.array(predictor_hidden.tolist(), dtype=np.float16).reshape(
                proxy.shape[0], -1,
            )
            if predictor_hidden is not None else None
        )
        target_logits = (
            np.array(actual_logits.tolist(), dtype=np.float16).reshape(
                proxy.shape[0], -1,
            )
            if actual_logits is not None else None
        )
        resident_mask = np.zeros((int(proxy.shape[-1]),), dtype=np.bool_)
        resident_ids = [
            int(value) for value in kwargs.get("resident", ())
            if 0 <= int(value) < int(proxy.shape[-1])
        ]
        resident_mask[resident_ids] = True
    with _lock:
        row = _rows[int(layer)]
        row["occurrences"] += 1
        row["actual_routes"] += metrics["actual_routes"]
        row["raw_top64_hits"] += metrics["raw_top64_hits"]
        row["selected_hits"] += metrics["selected_hits"]
        row["selected_width"] += metrics["selected_width"]
        row["max_selected_width"] = max(
            row["max_selected_width"], metrics["selected_width"],
        )
        row["width_violations"] += metrics["width_violation"]
        row["outside_top64"] += metrics["outside_top64"]
        if proxy_logits is not None:
            _training_rows.append(
                (
                    int(layer), proxy, actual, hidden, resident_mask,
                    target_logits,
                )
            )
    return metrics


def export_training_data(path: str) -> dict:
    """Write occurrence-aligned proxy logits and target routes to NPZ."""
    import numpy as np

    with _lock:
        rows = list(_training_rows)
    if not rows:
        return {"path": path, "samples": 0}
    layers = np.concatenate([
        np.full((proxy.shape[0],), layer, dtype=np.int16)
        for layer, proxy, _actual, _hidden, _resident, _target in rows
    ])
    proxy = np.concatenate([item[1] for item in rows], axis=0)
    actual = np.concatenate([item[2] for item in rows], axis=0)
    hidden_rows = [item[3] for item in rows]
    target_rows = [item[5] for item in rows]
    occurrence = np.concatenate([
        np.full((item[1].shape[0],), index, dtype=np.int32)
        for index, item in enumerate(rows)
    ])
    resident_mask = np.stack([item[4] for item in rows], axis=0)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "layer": layers,
        "proxy_logits": proxy,
        "actual": actual,
        "occurrence": occurrence,
        "resident_mask": resident_mask,
    }
    if all(value is not None for value in hidden_rows):
        arrays["predictor_hidden"] = np.concatenate(hidden_rows, axis=0)
    if all(value is not None for value in target_rows):
        arrays["actual_logits"] = np.concatenate(target_rows, axis=0)
    np.savez_compressed(output, **arrays)
    return {
        "path": str(output),
        "samples": int(proxy.shape[0]),
        "layers": int(np.unique(layers).size),
        "num_experts": int(proxy.shape[1]),
        "top_k": int(actual.shape[1]),
        "occurrences": int(len(rows)),
        "hidden": (
            int(arrays["predictor_hidden"].shape[1])
            if "predictor_hidden" in arrays else None
        ),
        "actual_logits": "actual_logits" in arrays,
    }


def report(threshold: float = 0.95) -> dict:
    with _lock:
        snapshot = {layer: dict(values) for layer, values in _rows.items()}
    per_layer = {}
    total_raw = 0
    total_selected = 0
    total_violations = 0
    total_outside = 0
    for layer in sorted(snapshot):
        row = snapshot[layer]
        retention = row["selected_hits"] / max(row["raw_top64_hits"], 1)
        item = {
            **row,
            "raw_top64_coverage": (
                row["raw_top64_hits"] / max(row["actual_routes"], 1)
            ),
            "selected_coverage": (
                row["selected_hits"] / max(row["actual_routes"], 1)
            ),
            "top64_recall_retention": retention,
            "mean_selected_width": (
                row["selected_width"] / max(row["occurrences"], 1)
            ),
            "pass": (
                retention >= float(threshold)
                and row["width_violations"] == 0
                and row["outside_top64"] == 0
            ),
        }
        per_layer[str(layer)] = item
        total_raw += row["raw_top64_hits"]
        total_selected += row["selected_hits"]
        total_violations += row["width_violations"]
        total_outside += row["outside_top64"]
    failing = [int(layer) for layer, row in per_layer.items() if not row["pass"]]
    minimum = min(
        per_layer,
        key=lambda layer: per_layer[layer]["top64_recall_retention"],
        default=None,
    )
    return {
        "schema": "progressive-runtime-acceptance-v1",
        "threshold": float(threshold),
        "per_layer": per_layer,
        "summary": {
            "evaluated_layers": len(per_layer),
            "pass_layers": len(per_layer) - len(failing),
            "failing_layers": failing,
            "minimum_retention": (
                None if minimum is None else {
                    "layer": int(minimum),
                    "value": per_layer[minimum]["top64_recall_retention"],
                }
            ),
            "aggregate_retention": total_selected / max(total_raw, 1),
            "width_violation_samples": total_violations,
            "outside_top64_experts": total_outside,
        },
    }
