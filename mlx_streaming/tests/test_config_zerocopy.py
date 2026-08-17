import os
from mlx_streaming import config


def test_zerocopy_default_off():
    os.environ.pop("ZEROCOPY_DUAL_SOURCE", None)
    assert config.zerocopy_dual_source() is False


def test_zerocopy_on():
    os.environ["ZEROCOPY_DUAL_SOURCE"] = "1"
    try:
        assert config.zerocopy_dual_source() is True
    finally:
        os.environ.pop("ZEROCOPY_DUAL_SOURCE", None)


def test_pool_spec_slots_default_3():
    os.environ.pop("POOL_SPEC_SLOTS", None)
    assert config.pool_spec_slots() == 3


def test_pool_admission_slots_defaults_to_physical_contribution(monkeypatch):
    monkeypatch.setenv("POOL_SPEC_SLOTS", "56")
    monkeypatch.delenv("POOL_ADMISSION_SLOTS", raising=False)
    assert config.pool_admission_slots() == 56
    monkeypatch.setenv("POOL_ADMISSION_SLOTS", "32")
    assert config.pool_admission_slots() == 32


def test_demand_async_context_override_is_scoped(monkeypatch):
    monkeypatch.setenv("DEMAND_ASYNC", "1")
    assert config.demand_async() is True
    with config.override_demand_async(False):
        assert config.demand_async() is False
    assert config.demand_async() is True


def test_sparse_miss_budget_is_nonnegative(monkeypatch):
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET", "4")
    assert config.demand_sparse_miss_budget() == 4
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET", "-2")
    assert config.demand_sparse_miss_budget() == 0


def test_sparse_miss_budget_layer_overrides(monkeypatch):
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET", "0")
    monkeypatch.setenv(
        "DEMAND_SPARSE_MISS_BUDGET_OVERRIDES", "3-4:1,9:2",
    )
    assert config.demand_sparse_miss_budget_for(2) == 0
    assert config.demand_sparse_miss_budget_for(3) == 1
    assert config.demand_sparse_miss_budget_for(4) == 1
    assert config.demand_sparse_miss_budget_for(9) == 2
    assert config.demand_sparse_enabled() is True


def test_sparse_miss_budget_sequence_overrides(monkeypatch):
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET", "17")
    monkeypatch.setenv(
        "DEMAND_SPARSE_MISS_BUDGET_BY_SEQ", "1:9,2:12,3:17",
    )
    assert config.demand_sparse_miss_budget_for(3, 1) == 9
    assert config.demand_sparse_miss_budget_for(3, 2) == 12
    assert config.demand_sparse_miss_budget_for(3, 3) == 17
    assert config.demand_sparse_miss_budget_for(3, 4) == 17


def test_sparse_sequence_budget_keeps_explicit_layer_floor(monkeypatch):
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET", "10")
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET_BY_SEQ", "1:4,2:6,3:10")
    monkeypatch.setenv("DEMAND_SPARSE_MISS_BUDGET_OVERRIDES", "2:18,6:13")
    monkeypatch.setenv("DEMAND_SPARSE_SEQ_LAYER_MAX", "1")
    assert config.demand_sparse_miss_budget_for(1, 1) == 4
    assert config.demand_sparse_miss_budget_for(2, 1) == 18
    assert config.demand_sparse_miss_budget_for(6, 2) == 13


def test_sparse_local_correction_default_off(monkeypatch):
    monkeypatch.delenv("DEMAND_SPARSE_LOCAL_CORRECTION", raising=False)
    assert config.demand_sparse_local_correction() is False
    monkeypatch.setenv("DEMAND_SPARSE_LOCAL_CORRECTION", "1")
    assert config.demand_sparse_local_correction() is True


def test_rerank_residual_decay_is_clamped(monkeypatch):
    monkeypatch.setenv("PREFETCH_RERANK_RESIDUAL_DECAY", "0.5")
    assert config.prefetch_rerank_residual_decay() == 0.5
    monkeypatch.setenv("PREFETCH_RERANK_RESIDUAL_DECAY", "2")
    assert config.prefetch_rerank_residual_decay() == 0.999
def test_rerank_backfill_layer_filter(monkeypatch):
    from mlx_streaming import config

    monkeypatch.setenv("PREFETCH_RERANK_BACKFILL_EXTRA", "2")
    monkeypatch.setenv("PREFETCH_RERANK_BACKFILL_LAYERS", "7-10,22")
    assert config.prefetch_rerank_backfill_extra_for(7) == 2
    assert config.prefetch_rerank_backfill_extra_for(10) == 2
    assert config.prefetch_rerank_backfill_extra_for(22) == 2
    assert config.prefetch_rerank_backfill_extra_for(11) == 0


def test_rerank_max_width_layer_overrides(monkeypatch):
    monkeypatch.setenv("PREFETCH_RERANK_MAX_WIDTH", "26")
    monkeypatch.setenv(
        "PREFETCH_RERANK_MAX_WIDTH_OVERRIDES", "3-5:22,46:21",
    )
    assert config.prefetch_rerank_max_width_for(2, 26) == 26
    assert config.prefetch_rerank_max_width_for(3, 26) == 22
    assert config.prefetch_rerank_max_width_for(5, 26) == 22
    assert config.prefetch_rerank_max_width_for(46, 26) == 21


def test_post_moe_prefetch_is_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_POST_MOE", raising=False)
    assert config.prefetch_post_moe() is False
    monkeypatch.setenv("PREFETCH_POST_MOE", "1")
    assert config.prefetch_post_moe() is True


def test_post_moe_refinement_is_independently_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_POST_MOE_REFINEMENT", raising=False)
    assert config.prefetch_post_moe_refinement() is False
    monkeypatch.setenv("PREFETCH_POST_MOE_REFINEMENT", "1")
    assert config.prefetch_post_moe_refinement() is True
    monkeypatch.delenv("PREFETCH_POST_MOE_REFINE_WIDTH", raising=False)
    assert config.prefetch_post_moe_refine_width() == 1
    monkeypatch.setenv("PREFETCH_POST_MOE_REFINE_WIDTH", "2")
    assert config.prefetch_post_moe_refine_width() == 2
    monkeypatch.delenv("PREFETCH_POST_MOE_REFINEMENT_LAYERS", raising=False)
    assert config.prefetch_post_moe_refinement_layers() is None
    monkeypatch.setenv("PREFETCH_POST_MOE_REFINEMENT_LAYERS", "6,18,30-32")
    assert config.prefetch_post_moe_refinement_layers() == {6, 18, 30, 31, 32}


def test_post_moe_replacement_layers_are_optional(monkeypatch):
    monkeypatch.delenv("PREFETCH_POST_MOE_REPLACEMENT_LAYERS", raising=False)
    assert config.prefetch_post_moe_replacement_layers() is None
    monkeypatch.setenv("PREFETCH_POST_MOE_REPLACEMENT_LAYERS", "6,18,30-32")
    assert config.prefetch_post_moe_replacement_layers() == {6, 18, 30, 31, 32}


def test_sparse_hit_aux_stream_is_opt_in(monkeypatch):
    monkeypatch.delenv("DEMAND_SPARSE_HIT_AUX_STREAM", raising=False)
    assert config.demand_sparse_hit_aux_stream() is False
    monkeypatch.setenv("DEMAND_SPARSE_HIT_AUX_STREAM", "1")
    assert config.demand_sparse_hit_aux_stream() is True


def test_late_candidate_rerank_is_bounded_and_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_LATE_CANDIDATE_RERANK", raising=False)
    monkeypatch.delenv("PREFETCH_LATE_CANDIDATE_WIDTH", raising=False)
    monkeypatch.delenv("PREFETCH_LATE_CANDIDATE_LAYERS", raising=False)
    assert config.prefetch_late_candidate_rerank() is False
    assert config.prefetch_late_candidate_width() == 2
    assert config.prefetch_late_candidate_layers() is None
    monkeypatch.setenv("PREFETCH_LATE_CANDIDATE_RERANK", "1")
    monkeypatch.setenv("PREFETCH_LATE_CANDIDATE_WIDTH", "3")
    monkeypatch.setenv("PREFETCH_LATE_CANDIDATE_LAYERS", "6,18-19")
    assert config.prefetch_late_candidate_rerank() is True
    assert config.prefetch_late_candidate_width() == 3
    assert config.prefetch_late_candidate_layers() == {6, 18, 19}


def test_partial_demand_tail_is_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_PARTIAL_DEMAND_TAIL", raising=False)
    assert config.prefetch_partial_demand_tail() is False
    monkeypatch.setenv("PREFETCH_PARTIAL_DEMAND_TAIL", "1")
    assert config.prefetch_partial_demand_tail() is True


def test_async_logical_protection_is_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_PROTECT_LOGICAL", raising=False)
    assert config.prefetch_protect_logical() is False
    monkeypatch.setenv("PREFETCH_PROTECT_LOGICAL", "1")
    assert config.prefetch_protect_logical() is True


def test_multistage_early_is_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_MULTISTAGE_EARLY", raising=False)
    monkeypatch.delenv("PREFETCH_MULTISTAGE_EARLY_AHEAD", raising=False)
    monkeypatch.delenv("PREFETCH_MULTISTAGE_EARLY_LAYERS", raising=False)
    assert config.prefetch_multistage_early() is False
    assert config.prefetch_multistage_early_ahead() == 3
    assert config.prefetch_multistage_early_layers() is None
    monkeypatch.setenv("PREFETCH_MULTISTAGE_EARLY", "1")
    monkeypatch.setenv("PREFETCH_MULTISTAGE_EARLY_LAYERS", "6,18")
    assert config.prefetch_multistage_early() is True
    assert config.prefetch_multistage_early_layers() == {6, 18}


def test_multistage_history_is_opt_in(monkeypatch):
    monkeypatch.delenv("PREFETCH_MULTISTAGE_HISTORY", raising=False)
    monkeypatch.delenv("PREFETCH_MULTISTAGE_HISTORY_WIDTH", raising=False)
    assert config.prefetch_multistage_history() is False
    assert config.prefetch_multistage_history_width() == 10
    monkeypatch.setenv("PREFETCH_MULTISTAGE_HISTORY", "1")
    monkeypatch.setenv("PREFETCH_MULTISTAGE_HISTORY_WIDTH", "6")
    assert config.prefetch_multistage_history() is True
    assert config.prefetch_multistage_history_width() == 6
