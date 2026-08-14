import os

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
    assert FINAL_DEFAULTS["MTP_EXPERT_DIR"].endswith("_4bit_g64")
    assert FINAL_DEFAULTS["NATIVE_NO_SUBMIT"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_ASYNC_PREDICT"] == "1"
    assert FINAL_DEFAULTS["PREFETCH_PROGRESSIVE"] == "0"
    assert FINAL_DEFAULTS["PREFETCH_RERANK_RANKING_POLICY"] == "topk_union_fast"


def test_final_profile_preserves_explicit_overrides(monkeypatch):
    monkeypatch.setattr(os, "environ", {"EXPERT_SLOTS": "999"})
    configure()
    assert os.environ["EXPERT_SLOTS"] == "999"


def test_final_profile_short_chat_entry():
    assert _chat_argv(["--chat", "--stats", "--plain"]) == [
        "chat", "--expert-slots", "152", "--spec-slots", "0",
        "--stats", "--plain",
    ]
    assert _chat_argv([]) is None
