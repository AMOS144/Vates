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
    ns = []

    def on_text(full, n):
        fulls.append(full)
        ns.append(n)
        return False

    res = b.generate([{"role": "user", "content": "hi"}], on_text)
    assert isinstance(res, GenResult)
    assert res.text == "你好世界"
    assert res.stopped is False
    assert fulls == ["你", "你好", "你好世", "你好世界"]
    assert ns == [1, 2, 3, 4]   # 每步回传已生成 token 数(假后端用字符数近似)


def test_fake_backend_stop_via_callback():
    b = FakeBackend(reply="abcdef")

    def on_text(full, n):
        return len(full) >= 2  # 收到 2 个字符后请求中断

    res = b.generate([{"role": "user", "content": "hi"}], on_text)
    assert res.stopped is True
    assert res.text == "ab"


def test_fake_backend_records_seen_messages():
    b = FakeBackend(reply="x")
    msgs = [{"role": "user", "content": "问题"}]
    b.generate(msgs, lambda full, n: False)
    assert b.seen_messages[-1] == [{"role": "user", "content": "问题"}]


def test_mlx_backend_stops_generation_on_eos(monkeypatch):
    """MLXBackend 命中 EOS 应提前停止,不空跑到 max_tokens;EOS 属正常完成而非用户中断。"""
    import types

    import mlx_streaming.mtp.generate as gen_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        # _eos_set 会读取这两个属性;此处 EOS 定为 99
        eos_token_ids = None
        eos_token_id = 99
        chat_template = None

        def encode(self, s):
            return [1, 2, 3]

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    fed = []

    def fake_mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                          ids_mode=False, profile=False, on_tokens=None):
        # 序列第 3 个是 EOS(99);正确实现应在此停止,后面的 12/13 不应再被喂出
        produced = []
        for t in [10, 11, 99, 12, 13]:
            produced.append(t)
            fed.append(t)
            if on_tokens is not None and on_tokens([t]):
                break
        return produced, {}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)

    args = types.SimpleNamespace(model="m", k=1, max_tokens=100, system=None)
    b = MLXBackend(args)
    b._tok = _Tok()
    b._model = object()
    b._drafter = object()

    res = b.generate([{"role": "user", "content": "hi"}], lambda full, n: False)

    assert fed == [10, 11, 99]      # 命中 EOS 即止,未继续喂 12/13
    assert res.stopped is False     # EOS 是正常完成,不算用户中断
    assert res.text == "10,11"      # 截断掉 EOS 及其后
