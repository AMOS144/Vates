import os
import sys
from types import SimpleNamespace

import pytest

from mlx_streaming.entrypoint import _public_chat_argv, main


def test_public_cli_uses_fixed_pool_profile():
    assert _public_chat_argv([]) == [
        "chat", "--expert-slots", "152", "--spec-slots", "0",
    ]
    assert _public_chat_argv(["--stats", "--plain"]) == [
        "chat", "--expert-slots", "152", "--spec-slots", "0",
        "--stats", "--plain",
    ]


def test_public_cli_accepts_optional_chat_word():
    assert _public_chat_argv(["chat", "--demo"])[-1] == "--demo"


def test_public_cli_rejects_non_profile_pool_size():
    with pytest.raises(ValueError, match="fixed at 152"):
        _public_chat_argv(["--expert-slots", "64"])
    with pytest.raises(ValueError, match="fixed at 152"):
        _public_chat_argv(["--expert-slots=64"])
    with pytest.raises(ValueError, match="fixed at 3"):
        _public_chat_argv(["--k=4"])


def test_public_cli_configures_profile_before_loading_chat(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    seen = {}

    def fake_chat(argv):
        seen["argv"] = argv
        seen["slots"] = os.environ["EXPERT_SLOTS"]
        return 0

    monkeypatch.setitem(
        sys.modules,
        "mlx_streaming.cli",
        SimpleNamespace(main=fake_chat),
    )
    assert main(["--demo"]) == 0
    assert seen == {
        "argv": [
            "chat", "--expert-slots", "152", "--spec-slots", "0", "--demo",
        ],
        "slots": "152",
    }


def test_public_cli_reports_profile_override_cleanly(capsys):
    assert main(["--k=4"]) == 2
    assert "vates: error: --k is fixed at 3" in capsys.readouterr().err
