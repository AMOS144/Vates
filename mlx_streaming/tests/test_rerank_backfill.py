import mlx.core as mx

from mlx_streaming.core.prefetch.rerank import rerank_prefetch_candidates


def _selected(logits, *, extra):
    ids, _scores, keep = rerank_prefetch_candidates(
        logits,
        top_k=2,
        max_width=3,
        retained_mass=1.0,
        min_width=2,
        resident=(0, 1),
        width_policy="predicted_route_union",
        ranking_policy="max",
        union_margin=0,
        candidate_width=6,
        backfill_extra=extra,
    )
    mx.eval(ids, keep)
    return [int(value) for value, active in zip(ids.tolist(), keep.tolist()) if active]


def test_resident_backfill_uses_vacated_nonresident_slots():
    logits = mx.array([[[9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]])
    assert _selected(logits, extra=0) == [2]
    assert _selected(logits, extra=1) == [2, 3]
    assert _selected(logits, extra=2) == [2, 3, 4]
