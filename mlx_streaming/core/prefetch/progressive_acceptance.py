"""Occurrence-local acceptance counters for the production progressive path."""

from __future__ import annotations

from collections import defaultdict
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


def reset() -> None:
    with _lock:
        _rows.clear()


def occurrence_metrics(
    candidate_ids,
    selected_ids,
    online_width,
    actual_ids,
    resident=(),
) -> dict:
    mx.eval(candidate_ids, selected_ids, online_width, actual_ids)
    # Capture candidate IDs from the same device argpartition as production.
    # Recomputing in NumPy may choose a different bfloat16 boundary tie.
    candidates = {
        int(value) for value in candidate_ids.reshape(-1).tolist()
    }
    resident_set = {int(value) for value in resident}
    width = int(online_width.item())
    selected_raw = {
        int(value)
        for value in selected_ids.reshape(-1).tolist()[:width]
    }
    # Native reservation filters the source-time resident snapshot before any
    # row or I/O is consumed.  Fixed-shape padding can therefore repeat a
    # resident candidate without making it part of the physical rerank set.
    selected = selected_raw - resident_set
    # The contract is occurrence-local *set* coverage.  K3 can route the same
    # expert from several speculative tokens; counting those repeated route
    # positions would silently overweight popular experts and can make a
    # reranker pass while missing too many members of the real union.
    actual = {int(value) for value in actual_ids.reshape(-1).tolist()}
    legal_width = (3 * len(actual)) // 2
    escaped = selected - candidates
    return {
        "actual_routes": len(actual),
        "raw_top64_hits": len(actual & (resident_set | candidates)),
        "selected_hits": len(actual & (resident_set | selected)),
        "selected_width": len(selected),
        "legal_width": legal_width,
        "width_violation": int(len(selected) > legal_width),
        "outside_top64": len(escaped),
    }


def record(layer: int, **kwargs) -> dict:
    metrics = occurrence_metrics(**kwargs)
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
    return metrics


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
