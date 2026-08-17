import os

from mlx_streaming.runtime.run_mtp_spec import _canonical_baseline_config


def test_canonical_baseline_disables_and_restores_prefetch(monkeypatch):
    monkeypatch.setenv("PREFETCH_PROGRESSIVE", "1")
    monkeypatch.setenv("NATIVE_FUSED_PREFETCH", "1")
    monkeypatch.setenv("CROSS_LAYER_PREFETCH", "1")
    monkeypatch.setenv("DEMAND_ASYNC", "1")
    monkeypatch.setenv("SHARED_EXPERT_OVERLAP", "1")

    with _canonical_baseline_config():
        assert os.environ["PREFETCH_PROGRESSIVE"] == "0"
        assert os.environ["NATIVE_FUSED_PREFETCH"] == "0"
        assert os.environ["CROSS_LAYER_PREFETCH"] == "0"
        assert os.environ["DEMAND_ASYNC"] == "0"
        assert os.environ["SHARED_EXPERT_OVERLAP"] == "0"

    assert os.environ["PREFETCH_PROGRESSIVE"] == "1"
    assert os.environ["NATIVE_FUSED_PREFETCH"] == "1"
    assert os.environ["CROSS_LAYER_PREFETCH"] == "1"
    assert os.environ["DEMAND_ASYNC"] == "1"
    assert os.environ["SHARED_EXPERT_OVERLAP"] == "1"


def test_noncanonical_baseline_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("BASELINE_ALLOW_PREFETCH", "1")
    monkeypatch.setenv("NATIVE_FUSED_PREFETCH", "1")
    with _canonical_baseline_config():
        assert os.environ["NATIVE_FUSED_PREFETCH"] == "1"
