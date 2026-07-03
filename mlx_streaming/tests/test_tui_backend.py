"""TUI 后端抽象:FakeBackend 的加载/流式/中断行为,及 banner 常量存在。"""
from mlx_streaming.tui.backend import FakeBackend, GenResult
from mlx_streaming.tui.banner import LOGO


def test_logo_is_nonempty_str():
    assert isinstance(LOGO, str) and LOGO.strip()


def test_fake_backend_load_reports_status():
    b = FakeBackend(status_msgs=["a", "b"])
    seen = []
    b.load(seen.append)
    assert seen == ["a", "b"]


def test_fake_backend_streams_full_text_incrementally():
    b = FakeBackend(reply="你好世界")
    fulls = []

    def on_text(full):
        fulls.append(full)
        return False

    res = b.generate([{"role": "user", "content": "hi"}], on_text)
    assert isinstance(res, GenResult)
    assert res.text == "你好世界"
    assert res.stopped is False
    assert fulls[-1] == "你好世界"
    assert fulls == ["你", "你好", "你好世", "你好世界"]


def test_fake_backend_stop_via_callback():
    b = FakeBackend(reply="abcdef")

    def on_text(full):
        return len(full) >= 2  # 收到 2 个字符后请求中断

    res = b.generate([{"role": "user", "content": "hi"}], on_text)
    assert res.stopped is True
    assert res.text == "ab"


def test_fake_backend_records_seen_messages():
    b = FakeBackend(reply="x")
    msgs = [{"role": "user", "content": "问题"}]
    b.generate(msgs, lambda full: False)
    assert b.seen_messages[-1] == [{"role": "user", "content": "问题"}]
