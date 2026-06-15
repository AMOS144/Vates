from mlx_streaming.core.mem import MemSnapshot, snapshot, rss_bytes


def test_rss_is_positive():
    assert rss_bytes() > 0


def test_snapshot_has_fields():
    s = snapshot()
    assert isinstance(s, MemSnapshot)
    assert s.rss_bytes > 0
    assert s.mlx_active_bytes >= 0
    assert s.mlx_peak_bytes >= 0
