from mlx_streaming.runtime.run_qwen_k3_sub10 import FINAL_DEFAULTS, configure


def test_final_profile_defaults(monkeypatch):
    for name in FINAL_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    configure()
    assert FINAL_DEFAULTS.items() <= __import__("os").environ.items()


def test_final_profile_preserves_explicit_overrides(monkeypatch):
    monkeypatch.setenv("EXPERT_SLOTS", "999")
    configure()
    assert __import__("os").environ["EXPERT_SLOTS"] == "999"
