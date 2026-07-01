import importlib
import mlx_streaming.config as config


def test_sideregion_lfu_default_off(monkeypatch):
    monkeypatch.delenv("SIDEREGION_LFU", raising=False)
    importlib.reload(config)
    assert config.sideregion_lfu() is False


def test_sideregion_lfu_on(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    importlib.reload(config)
    assert config.sideregion_lfu() is True
