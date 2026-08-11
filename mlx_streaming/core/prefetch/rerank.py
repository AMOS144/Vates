"""跨层预取候选的无训练重排。"""

import mlx.core as mx


def _ranking_scores(
    token_logits: mx.array,
    *,
    top_k: int,
    policy: str,
    need_noisy_or: bool = True,
    rank_width: int = 64,
    ranked_ids: "mx.array | None" = None,
) -> "tuple[mx.array, mx.array]":
    """返回逐专家排序分数和 Noisy-OR 质量分数。"""
    num_experts = int(token_logits.shape[-1])
    if need_noisy_or or policy == "noisy_or":
        probs = mx.softmax(token_logits, axis=-1, precise=True)
        inclusion = mx.minimum(probs * max(1, int(top_k)), 1.0)
        noisy_or_scores = 1.0 - mx.prod(1.0 - inclusion, axis=0)
    else:
        noisy_or_scores = mx.zeros((num_experts,), dtype=mx.float32)
    if policy == "max":
        scores = mx.max(token_logits, axis=0)
    elif policy == "noisy_or":
        scores = noisy_or_scores
    elif policy == "topk_union":
        order = mx.argsort(-token_logits, axis=-1)
        rank = mx.argsort(order, axis=-1)
        support = mx.sum(rank < min(int(top_k), num_experts), axis=0)
        best_rank = mx.min(rank, axis=0)
        centered = token_logits - mx.mean(token_logits, axis=-1, keepdims=True)
        zmax = mx.max(
            centered * mx.rsqrt(
                mx.mean(mx.square(centered), axis=-1, keepdims=True) + 1e-6,
            ),
            axis=0,
        )
        # 精确编码 lexsort(-support, best_rank, -zmax) 的优先级；zmax
        # 仅在前两项完全相同时打破平局。
        scores = (
            support.astype(mx.float32) * float(num_experts + 1)
            - best_rank.astype(mx.float32)
            + mx.clip(zmax, -16.0, 16.0) * 1e-3
        )
    elif policy == "topk_union_fast":
        # The selectable set is the raw per-token top64 union, so ranking all
        # 512 experts twice is unnecessary on the uncorrected source path.
        # Sort only the same top64 working set and scatter its ranks into a
        # fixed 512-vector.  This preserves exact topk support/best-rank for
        # every selectable expert while reducing the expensive full-axis
        # argsort graph to 64 elements per token.
        ranked_width = min(
            num_experts, max(int(top_k), int(rank_width)),
        )
        ids = ranked_ids
        if ids is None:
            ids = mx.argpartition(
                token_logits, kth=-ranked_width, axis=-1,
            )[:, -ranked_width:]
        elif int(ids.shape[-1]) != ranked_width:
            raise ValueError("ranked_ids width 与 rank_width 不一致")
        values = mx.take_along_axis(token_logits, ids, axis=-1)
        local_order = mx.argsort(-values, axis=-1)
        sorted_ids = mx.take_along_axis(ids, local_order, axis=-1)
        shape = (int(token_logits.shape[0]), num_experts)
        rank = mx.full(shape, num_experts, dtype=mx.int32)
        rows = mx.broadcast_to(
            mx.arange(shape[0]).reshape(-1, 1), sorted_ids.shape,
        )
        local_rank = mx.broadcast_to(
            mx.arange(ranked_width).reshape(1, -1), sorted_ids.shape,
        )
        rank = rank.at[rows, sorted_ids].add(local_rank - num_experts)
        support = mx.sum(rank < min(int(top_k), num_experts), axis=0)
        best_rank = mx.min(rank, axis=0)
        centered = token_logits - mx.mean(token_logits, axis=-1, keepdims=True)
        zmax = mx.max(
            centered * mx.rsqrt(
                mx.mean(mx.square(centered), axis=-1, keepdims=True) + 1e-6,
            ),
            axis=0,
        )
        scores = (
            support.astype(mx.float32) * float(num_experts + 1)
            - best_rank.astype(mx.float32)
            + mx.clip(zmax, -16.0, 16.0) * 1e-3
        )
    else:
        raise ValueError(f"未知 rerank ranking policy: {policy}")
    return scores, noisy_or_scores


def rerank_prefetch_candidates(
    logits: mx.array,
    *,
    top_k: int,
    max_width: int,
    retained_mass: float,
    min_width: int,
    resident=(),
    width_policy: str = "mass",
    ranking_policy: str = "noisy_or",
    union_margin: int = 0,
    candidate_logits: "mx.array | None" = None,
    candidate_ranking_policy: str = "max",
    candidate_width: int = 64,
    return_candidate_ids: bool = False,
) -> "tuple[mx.array, ...]":
    """按专家至少被一个 token 选中的近似概率排序，并压紧无效尾部。

    返回固定长度的专家 ID、对应分数和保留掩码。保留范围外重复首个
    专家 ID，使 native 端现有去重逻辑可在不发生 host 同步的情况下
    获得等价的动态宽度。
    """
    num_experts = int(logits.shape[-1])
    width = max(1, min(num_experts, int(max_width)))
    raw_candidate_width = max(
        1, min(num_experts, int(candidate_width)),
    )
    minimum = max(1, min(width, int(min_width)))
    mass = max(0.0, min(1.0, float(retained_mass)))

    token_logits = logits.reshape(-1, num_experts)
    # 非有限值说明上游数值异常；置零可保证预取退化为有限、可排序的保守分布。
    token_logits = mx.where(mx.isfinite(token_logits), token_logits, 0.0)
    # 候选集和候选内排序是两个独立阶段。每个 token 必须先从原 proxy
    # 独立取 raw top64，再求并集。不能先跨 token 聚合成一个 top64，也不能
    # 用 correction 后的 512 轴补入候选外专家。
    candidate_token_logits = (
        token_logits
        if candidate_logits is None
        else candidate_logits.reshape(-1, num_experts)
    )
    if int(candidate_token_logits.shape[-1]) != num_experts:
        raise ValueError("candidate logits 与 rerank logits 的专家维不一致")
    candidate_token_logits = mx.where(
        mx.isfinite(candidate_token_logits), candidate_token_logits, 0.0,
    )
    # ``candidate_ranking_policy`` used to aggregate tokens before selecting
    # candidates.  Keep validating the public argument, but candidate
    # membership is now deliberately independent of every aggregate policy.
    if candidate_ranking_policy not in {"max", "noisy_or", "topk_union"}:
        raise ValueError(
            f"未知 candidate ranking policy: {candidate_ranking_policy}",
        )
    token_candidate_ids = mx.argpartition(
        candidate_token_logits,
        kth=-raw_candidate_width,
        axis=-1,
    )[:, -raw_candidate_width:]
    same_ranking_source = candidate_logits is None or candidate_logits is logits
    scores, noisy_or_scores = _ranking_scores(
        token_logits,
        top_k=top_k,
        policy=ranking_policy,
        need_noisy_or=width_policy == "mass",
        rank_width=raw_candidate_width,
        ranked_ids=(
            token_candidate_ids
            if ranking_policy == "topk_union_fast" and same_ranking_source
            else None
        ),
    )
    flat_candidate_ids = token_candidate_ids.reshape(-1)
    candidate_counts = mx.zeros((num_experts,), dtype=mx.int32)
    candidate_counts = candidate_counts.at[flat_candidate_ids].add(
        mx.ones(flat_candidate_ids.shape, dtype=mx.int32),
    )
    candidate_mask = candidate_counts > 0


    # Resident filtering happens after both the raw union *and* the logical
    # width selection.  W26 means 26 predicted experts in total, not 26 new
    # SSD reads in addition to whatever is already resident.  Selecting from
    # nonresident candidates here would backfill every resident winner with a
    # lower-ranked nonresident tail and create avoidable side-pool churn.
    resident_set = {int(e) for e in resident if 0 <= int(e) < num_experts}
    if resident_set:
        available = mx.array(
            [False if e in resident_set else True for e in range(num_experts)],
            dtype=mx.bool_,
        )
    else:
        available = mx.ones((num_experts,), dtype=mx.bool_)
    selectable = candidate_mask
    eligible_candidate_scores = mx.where(selectable, scores, -1e30)
    candidate_ids = mx.argpartition(
        eligible_candidate_scores, kth=-width,
    )[-width:]
    rerank_candidate_scores = scores[candidate_ids]
    candidate_valid = selectable[candidate_ids]

    order = mx.argsort(rerank_candidate_scores)[::-1]
    ordered_ids = candidate_ids[order].astype(mx.uint32)
    ordered_valid = candidate_valid[order]
    ordered_nonresident = available[ordered_ids]
    ordered_mass_scores = mx.where(
        ordered_valid, noisy_or_scores[ordered_ids], 0.0,
    )

    positions = mx.arange(width)
    if width_policy == "mass":
        cumulative = mx.cumsum(ordered_mass_scores)
        threshold = mx.sum(ordered_mass_scores) * mass
        keep = ordered_valid & ordered_nonresident & (ordered_mass_scores > 0) & (
            (positions < minimum)
            | ((cumulative - ordered_mass_scores) < threshold)
        )
    elif width_policy == "predicted_route_union":
        # 线上只看当前 proxy logits：逐 token top-k 的预测并集大小减安全
        # margin，再乘 1.5。用广播 membership 保持全程设备侧、固定输出 shape，
        # 不为动态 width 引入 host 同步。
        route_ids = mx.argpartition(
            token_logits, kth=-int(top_k), axis=-1,
        )[:, -int(top_k):]
        expert_axis = mx.arange(num_experts)
        present = mx.any(route_ids[..., None] == expert_axis, axis=(0, 1))
        predicted_union = mx.sum(present.astype(mx.int32))
        reference = mx.maximum(
            int(top_k), predicted_union - max(0, int(union_margin)),
        )
        online_width = mx.minimum(width, (reference * 3) // 2)
        keep = ordered_valid & ordered_nonresident & (positions < online_width)
    else:
        raise ValueError(f"未知 rerank width policy: {width_policy}")

    # Pack submitted rows at the front while preserving their score order.
    # Fixed-shape tail rows repeat the first packed ID, so native de-dup sees
    # exactly the logical set.  With zero submitted rows the first resident ID
    # is repeated, which reserve() filters without issuing a read.
    pack_key = mx.where(keep, positions, positions + width)
    pack_order = mx.argsort(pack_key)
    packed_ids = ordered_ids[pack_order]
    packed_scores = ordered_mass_scores[pack_order]
    kept_count = mx.sum(keep.astype(mx.int32))
    compact_keep = positions < kept_count
    first_id = mx.broadcast_to(packed_ids[:1], packed_ids.shape)
    compact_ids = mx.where(compact_keep, packed_ids, first_id)
    result = (compact_ids, packed_scores, compact_keep)
    if return_candidate_ids:
        # Audit must use the exact device-side raw top-N membership computed by
        # production. Re-running argpartition after generation can disagree at
        # a bfloat16 boundary tie and would make the retention gate ambiguous.
        return (*result, token_candidate_ids)
    return result
