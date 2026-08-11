from mlx_streaming.runtime.run_qwen_k3_sub10 import (
    FINAL_DEFAULTS,
    _chat_argv,
    configure,
)


def test_final_profile_defaults(monkeypatch):
    for name in FINAL_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    configure()
    assert FINAL_DEFAULTS.items() <= __import__("os").environ.items()


def test_final_profile_preserves_explicit_overrides(monkeypatch):
    monkeypatch.setenv("EXPERT_SLOTS", "999")
    configure()
    assert __import__("os").environ["EXPERT_SLOTS"] == "999"


def test_final_profile_short_chat_entry():
    assert _chat_argv(["--chat", "--stats", "--plain"]) == [
        "chat", "--expert-slots", "152", "--spec-slots", "0",
        "--stats", "--plain",
    ]
    assert _chat_argv([]) is None
