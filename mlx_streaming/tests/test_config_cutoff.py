import mlx_streaming.config as config


def test_cutoff_defaults_and_env(monkeypatch):
    monkeypatch.delenv("CROSS_LAYER_CUTOFF", raising=False)
    monkeypatch.delenv("CROSS_LAYER_AHEAD_LO", raising=False)
    monkeypatch.delenv("CROSS_LAYER_AHEAD_HI", raising=False)
    assert config.cross_layer_cutoff() == 6
    assert config.cross_layer_ahead_lo() == 1
    assert config.cross_layer_ahead_hi() == 3
    monkeypatch.setenv("CROSS_LAYER_CUTOFF", "30")
    monkeypatch.setenv("CROSS_LAYER_AHEAD_LO", "2")
    monkeypatch.setenv("CROSS_LAYER_AHEAD_HI", "6")
    assert config.cross_layer_cutoff() == 30
    assert config.cross_layer_ahead_lo() == 2
    assert config.cross_layer_ahead_hi() == 6
