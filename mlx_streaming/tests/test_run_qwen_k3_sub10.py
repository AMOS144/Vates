import os
import pytest

from mlx_streaming.runtime.run_qwen_k3_sub10 import (
    FINAL_DEFAULTS,
    _chat_argv,
    configure,
)


def test_final_profile_defaults(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    configure()
    assert FINAL_DEFAULTS.items() <= os.environ.items()
    assert FINAL_DEFAULTS["MTP_BITS"] == "4"
    assert FINAL_DEFAULTS["MTP_EXPERT_DIR"].endswith("vates-runtime/mtp/experts")
    assert FINAL_DEFAULTS["NATIVE_NO_SUBMIT"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_PHYSICAL_READ_BUDGET"] == "3"
    assert FINAL_DEFAULTS["MTP_CONF_TAU"] == "0.3"
    assert FINAL_DEFAULTS["PREFETCH_ISOLATED_SIDE"] == "0"
    assert FINAL_DEFAULTS["SIDEREGION_ROW_LEASES"] == "1"
    assert FINAL_DEFAULTS["PREFETCH_TARGET_LAYERS"] == (
        "1-7,10-16,18-21,23,25-26,28-36,38-41,43-47"
    )
    assert FINAL_DEFAULTS["MTP_EXPERT_SLOTS"] == "256"
    assert FINAL_DEFAULTS["PREFETCH_ADAPTIVE"] == "0"
    assert FINAL_DEFAULTS["EXPERT_POOL_PROFILE"].endswith(
        "qwen_k3_prefetch_wait_rebalanced_same10g.json"
    )
    assert FINAL_DEFAULTS["DEMAND_ASYNC"] == "0"
    assert FINAL_DEFAULTS["SPEC_SPLIT_DEMAND_AFTER_PREFILL"] == "1"
    assert FINAL_DEFAULTS["DEMAND_ASYNC_PY_SUBMIT"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_ASYNC_PREDICT"] == "1"
    assert FINAL_DEFAULTS["SHARED_EXPERT_OVERLAP"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_PROGRESSIVE"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_ADAPTIVE_COOLDOWN"] == "32"
    assert FINAL_DEFAULTS["PREFETCH_RERANK_RANKING_POLICY"] == "topk_union_fast"
    assert FINAL_DEFAULTS["MTP_VERIFY_MODE"] == "batch"
    assert FINAL_DEFAULTS["TREE_VERIFY"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_OPTIMISTIC_VERIFY"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_ONLINE_TRANSITION"] == "0"


def test_runtime_root_relocates_mtp_experts(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "environ", {"VATES_RUNTIME_DIR": str(tmp_path)})
    configure()
    assert os.environ["MTP_EXPERT_DIR"] == str(tmp_path / "mtp" / "experts")


def test_final_profile_overrides_stale_performance_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {"EXPERT_SLOTS": "999"})
    configure()
    assert os.environ["EXPERT_SLOTS"] == "152"


def test_final_profile_short_chat_entry():
    assert _chat_argv(["--chat", "--stats", "--plain"]) == [
        "chat", "--expert-slots", "152", "--spec-slots", "0",
        "--stats", "--plain",
    ]
    assert _chat_argv([
        "--chat", "--expert-slots", "152", "--stats",
    ]) == [
        "chat", "--expert-slots", "152", "--spec-slots", "0", "--stats",
    ]
    with pytest.raises(ValueError, match="fixed at 152"):
        _chat_argv(["--chat", "--expert-slots", "64"])
    with pytest.raises(ValueError, match="fixed at 152"):
        _chat_argv(["--chat", "--expert-slots=64"])
    with pytest.raises(ValueError, match="fixed at 3"):
        _chat_argv(["--chat", "-k4"])
    assert _chat_argv(["--chat", "--k=3", "--stats"])[-1] == "--stats"
    assert _chat_argv([]) is None
