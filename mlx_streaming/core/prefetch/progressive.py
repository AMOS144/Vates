"""Device-side selection for an unchanged early core plus exact late fill."""

from __future__ import annotations

import mlx.core as mx

from mlx_streaming.core.prefetch.rerank import (
    _ranking_scores,
    rerank_prefetch_candidates,
)


def initial_core_ids(
    proxy_logits: mx.array,
    *,
    top_k: int,
    core_width: int,
    resident=(),
    candidate_width: int = 64,
) -> mx.array:
    """Select a fixed nonresident early core from the raw per-token top64."""
    core = int(core_width)
    if not 1 <= core <= 26:
        raise ValueError("progressive early core must be in [1, 26]")
    ids, _scores, _keep = rerank_prefetch_candidates(
        proxy_logits,
        top_k=int(top_k),
        max_width=core,
        retained_mass=1.0,
        min_width=core,
        resident=resident,
        width_policy="mass",
        ranking_policy="topk_union",
        candidate_logits=proxy_logits,
        candidate_width=int(candidate_width),
    )
    return ids


def refined_ids(
    exact_logits: mx.array,
    candidate_logits: mx.array,
    early_ids: mx.array,
    *,
    top_k: int,
    side_capacity: int,
    resident=(),
    candidate_width: int = 64,
    union_margin: int = 0,
) -> "tuple[mx.array, mx.array]":
    """Keep the irreversible early core and fill to exact 1.5x route width.

    ``exact_logits`` may improve the ordering, but both the immutable core and
    late fill remain members of the original per-token top64 union.  This keeps
    the production path a true rerank of the user-defined top64 baseline.

    Returns ``(compact_ids, online_width)``.  ``compact_ids`` has a fixed
    ``side_capacity`` shape and repeats its first ID outside ``online_width``
    so the native callback can deduplicate without a host synchronization.
    """
    capacity = int(side_capacity)
    if capacity <= 0:
        raise ValueError("side capacity must be positive")
    core_width = int(early_ids.size)
    if not 1 <= core_width <= min(26, capacity):
        raise ValueError("invalid progressive early core width")

    num_experts = int(exact_logits.shape[-1])
    exact = mx.where(
        mx.isfinite(exact_logits.reshape(-1, num_experts)),
        exact_logits.reshape(-1, num_experts),
        0.0,
    )
    candidate = mx.where(
        mx.isfinite(candidate_logits.reshape(-1, num_experts)),
        candidate_logits.reshape(-1, num_experts),
        0.0,
    )
    raw_width = max(1, min(num_experts, int(candidate_width)))
    token_candidates = mx.argpartition(
        candidate, kth=-raw_width, axis=-1,
    )[:, -raw_width:]
    expert_axis = mx.arange(num_experts)
    candidate_mask = mx.any(
        token_candidates[..., None] == expert_axis,
        axis=(0, 1),
    )

    resident_set = {
        int(expert) for expert in resident
        if 0 <= int(expert) < num_experts
    }
    available = mx.array(
        [expert not in resident_set for expert in range(num_experts)],
        dtype=mx.bool_,
    )
    early_mask = mx.any(
        early_ids.reshape(-1, 1) == expert_axis.reshape(1, -1),
        axis=0,
    )

    route_ids = mx.argpartition(
        exact, kth=-int(top_k), axis=-1,
    )[:, -int(top_k):]
    route_present = mx.any(
        route_ids[..., None] == expert_axis,
        axis=(0, 1),
    )
    # Exact target logits rank only the frozen raw-top64 candidate membership;
    # they may not turn refinement into an unrestricted second candidate set.
    remaining = candidate_mask & available & ~early_mask

    scores, _ = _ranking_scores(
        exact,
        top_k=int(top_k),
        policy="topk_union",
    )
    eligible_scores = mx.where(remaining, scores, -1e30)
    late_capacity = capacity - core_width
    if late_capacity:
        late = mx.argpartition(
            eligible_scores, kth=-late_capacity,
        )[-late_capacity:]
        late_scores = eligible_scores[late]
        late = late[mx.argsort(late_scores)[::-1]].astype(mx.uint32)
        # The late list is ranked with exact target routes first. Submit those
        # rows before any early-core hole: reserve() skips early rows that are
        # already resident/pending, but a missed stale early candidate must not
        # delay a route-critical exact row at the SSD deadline.
        merged = mx.concatenate([late, early_ids.astype(mx.uint32)])
    else:
        merged = early_ids.astype(mx.uint32)

    exact_union = mx.sum(route_present.astype(mx.int32))
    width_union = mx.maximum(int(top_k), exact_union - int(union_margin))
    online_width = mx.minimum(capacity, (width_union * 3) // 2)
    # A ten-expert route implies a minimum legal width of 15.  A 15-wide
    # early core is therefore always legal; wider multi-token unions still
    # leave exact late-fill slots up to the physical capacity.
    online_width = mx.maximum(core_width, online_width)
    positions = mx.arange(capacity)
    if late_capacity:
        late_keep_count = online_width - core_width
        raw_keep = mx.concatenate([
            mx.arange(late_capacity) < late_keep_count,
            mx.ones((core_width,), dtype=mx.bool_),
        ])
        # Pack the selected exact tail first and the immutable early core
        # second; unselected late capacity becomes a repeated fixed-shape tail.
        pack_key = mx.where(raw_keep, positions, positions + capacity)
        pack_order = mx.argsort(pack_key)
        packed = merged[pack_order]
    else:
        packed = merged
    keep = positions < online_width
    first = mx.broadcast_to(packed[:1], packed.shape)
    return mx.where(keep, packed, first), online_width


def exact_route_union_ids(
    exact_logits: mx.array,
    *,
    top_k: int,
    side_capacity: int,
    resident=(),
) -> "tuple[mx.array, mx.array]":
    """Pack the complete adjacent target route union into device storage.

    The adjacent target attention/gate is the real computation moved to T-1
    and reused by the decoder, so this is not an oracle label.  Existing
    resident routes must remain in the packed set: the native reservation uses
    the full submitted set as its eviction-protection set.  Filtering resident
    routes here could let a missing route evict an expert that this same MoE
    invocation is about to consume.

    GPU-only demand cannot recover from a truncated union.  Require enough
    fixed storage for the worst-case union rather than silently returning a
    partial mapping containing ``-1`` rows.
    """
    capacity = int(side_capacity)
    num_experts = int(exact_logits.shape[-1])
    exact = mx.where(
        mx.isfinite(exact_logits.reshape(-1, num_experts)),
        exact_logits.reshape(-1, num_experts),
        0.0,
    )
    route_ids = mx.argpartition(
        exact, kth=-int(top_k), axis=-1,
    )[:, -int(top_k):]
    max_route_width = int(route_ids.shape[0]) * int(top_k)
    if capacity < max_route_width:
        raise ValueError(
            "exact GPU route storage is too small: "
            f"capacity={capacity}, required={max_route_width}",
        )
    expert_axis = mx.arange(num_experts)
    route_present = mx.any(
        route_ids[..., None] == expert_axis,
        axis=(0, 1),
    )
    # ``resident`` is intentionally not subtracted.  Native reservation skips
    # its I/O but uses every submitted ID to protect current-route rows.
    del resident
    needed = route_present
    scores, _ = _ranking_scores(
        exact, top_k=int(top_k), policy="topk_union",
    )
    ranked_scores = mx.where(needed, scores, -1e30)
    selected = mx.argpartition(
        ranked_scores, kth=-capacity,
    )[-capacity:]
    selected_scores = ranked_scores[selected]
    selected = selected[mx.argsort(selected_scores)[::-1]].astype(mx.uint32)
    width = mx.sum(needed.astype(mx.int32))
    width = mx.minimum(capacity, width)
    keep = mx.arange(capacity) < width
    first = mx.broadcast_to(selected[:1], selected.shape)
    return mx.where(keep, selected, first), width


def exact_candidate_route_ids(
    exact_logits: mx.array,
    candidate_logits: mx.array,
    *,
    top_k: int,
    side_capacity: int,
    resident=(),
    candidate_width: int = 64,
) -> "tuple[mx.array, mx.array]":
    """Select only exact target routes that belong to the frozen raw top64.

    This is the single-callback variant of progressive refinement: it gives up
    the inaccurate early core and spends the adjacent half-layer window only
    on route-critical rows. The resulting set is a strict raw-top64 subset and
    is never wider than the true route union.
    """
    capacity = int(side_capacity)
    num_experts = int(exact_logits.shape[-1])
    exact = exact_logits.reshape(-1, num_experts)
    candidate = candidate_logits.reshape(-1, num_experts)
    route_ids = mx.argpartition(exact, kth=-int(top_k), axis=-1)[:, -int(top_k):]
    raw_width = min(num_experts, int(candidate_width))
    candidate_ids = mx.argpartition(
        candidate, kth=-raw_width, axis=-1,
    )[:, -raw_width:]
    expert_axis = mx.arange(num_experts)
    route_mask = mx.any(route_ids[..., None] == expert_axis, axis=(0, 1))
    candidate_mask = mx.any(
        candidate_ids[..., None] == expert_axis, axis=(0, 1),
    )
    resident_set = {int(value) for value in resident}
    available = mx.array(
        [expert not in resident_set for expert in range(num_experts)],
        dtype=mx.bool_,
    )
    needed = route_mask & candidate_mask & available
    scores, _ = _ranking_scores(exact, top_k=int(top_k), policy="topk_union")
    ranked = mx.where(needed, scores, -1e30)
    ids = mx.argpartition(ranked, kth=-capacity)[-capacity:]
    ids = ids[mx.argsort(ranked[ids])[::-1]].astype(mx.uint32)
    width = mx.minimum(capacity, mx.sum(needed.astype(mx.int32)))
    keep = mx.arange(capacity) < width
    first = mx.broadcast_to(ids[:1], ids.shape)
    return mx.where(keep, ids, first), width
