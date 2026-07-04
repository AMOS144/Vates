import importlib
import mlx_streaming.config as config


def test_sideregion_lfu_default_on(monkeypatch):
    # 默认 on:持久 LFU 单缓冲=生产路径(与 cli 默认对齐)。
    monkeypatch.delenv("SIDEREGION_LFU", raising=False)
    importlib.reload(config)
    assert config.sideregion_lfu() is True


def test_sideregion_lfu_explicit_off(monkeypatch):
    # 仅显式 SIDEREGION_LFU=0 才回退 legacy 双缓冲。
    monkeypatch.setenv("SIDEREGION_LFU", "0")
    importlib.reload(config)
    assert config.sideregion_lfu() is False


def test_sideregion_lfu_on(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    importlib.reload(config)
    assert config.sideregion_lfu() is True
