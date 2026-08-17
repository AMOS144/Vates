"""TUI 后端抽象:FakeBackend 的加载/流式/中断行为,及 banner 常量存在。"""
import pytest

from mlx_streaming.tui.backend import (
    FakeBackend, GenResult, _common_prefix_len, _reuse_prefix_len)
from mlx_streaming.tui.banner import LOGO


@pytest.fixture(autouse=True)
def _deterministic_decode_stream(monkeypatch):
    """Unit-test tokenizer doubles use the same single streaming contract."""
    class _Stream:
        def __init__(self, **kwargs):
            self.first = True

        def step(self, tokenizer, token):
            prefix = "" if self.first else ","
            self.first = False
            return f"{prefix}{int(token)}"

    import tokenizers.decoders
    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", _Stream)


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
                          ids_mode=False, profile=False, on_tokens=None,
                          main_cache=None, cached_len=0,
                          on_prefill_complete=None):
        if on_prefill_complete is not None:
            on_prefill_complete()
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
    b._model = types.SimpleNamespace(make_cache=lambda: object())
    b._drafter = object()

    res = b.generate([{"role": "user", "content": "hi"}], lambda full, n: False)

    assert fed == [10, 11, 99]      # 命中 EOS 即止,未继续喂 12/13
    assert res.stopped is False     # EOS 是正常完成,不算用户中断
    assert res.text == "10,11"      # 截断掉 EOS 及其后


def test_mlx_backend_detokenizes_on_text_worker(monkeypatch):
    """A tight inference loop may coalesce frames without losing final text."""
    import types

    import mlx_streaming.cli as cli_mod
    import mlx_streaming.mtp.generate as gen_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        eos_token_ids = None
        eos_token_id = -1
        chat_template = None

        def __init__(self):
            self.decode_calls = 0

        def decode(self, ids):
            self.decode_calls += 1
            return ",".join(str(i) for i in ids)

    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: [1, 2, 3])
    def fake_mtp_generate(*args, on_prefill_complete=None, on_tokens=None,
                          **kwargs):
        if on_prefill_complete:
            on_prefill_complete()
        produced = []
        for token in range(10, 20):
            produced.append(token)
            assert on_tokens is not None
            on_tokens([token])
        return produced, {
            "resident_tokens": 12,
            "wall_s": 1.0,
            "avg_accept_len": 2.5,
        }

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)
    tok = _Tok()
    backend = MLXBackend(types.SimpleNamespace(
        model="m", k=3, max_tokens=10, system=None,
    ))
    backend._tok = tok
    backend._model = types.SimpleNamespace(make_cache=lambda: object())
    backend._drafter = object()
    callbacks = []
    result = backend.generate(
        [{"role": "user", "content": "hi"}],
        lambda text, n: callbacks.append((text, n)) or False,
    )

    assert callbacks
    assert callbacks[-1] == ("10,11,12,13,14,15,16,17,18,19", 10)
    # Streaming uses DecodeStream; only the authoritative final result calls
    # the tokenizer's full-prefix decode once.
    assert tok.decode_calls == 1
    assert result.text.endswith("19")
    assert result.avg_accept_len == 2.5


def test_async_text_publisher_never_waits_for_slow_detokenizer(monkeypatch):
    """Inference submits stay non-blocking while the text worker is stalled."""
    import threading
    import time

    from mlx_streaming.tui.backend import _AsyncTextPublisher

    entered = threading.Event()
    release = threading.Event()

    class _SlowStream:
        def __init__(self, **kwargs):
            pass

        def step(self, tokenizer, token):
            entered.set()
            release.wait(timeout=2)
            return chr(0x4E00 + int(token))

    import tokenizers.decoders
    monkeypatch.setattr(tokenizers.decoders, "DecodeStream", _SlowStream)

    class _Tok:
        backend_tokenizer = object()

    seen = []
    publisher = _AsyncTextPublisher(
        _Tok(), set(), lambda text, n: seen.append((text, n)) or False,
    )
    assert publisher.submit([1]) is False
    assert entered.wait(timeout=1)
    started = time.perf_counter()
    for token in range(2, 100):
        assert publisher.submit([token]) is False
    submit_s = time.perf_counter() - started
    assert submit_s < 0.1
    assert len(publisher._signals) <= 1
    release.set()
    publisher.close()
    assert seen[-1][1] == 99


def test_incremental_detokenizer_uses_native_decode_stream():
    from mlx_streaming.tui.backend import _IncrementalDetokenizer

    class _Stream:
        def __init__(self, **kwargs):
            self.seen = []

        def step(self, tokenizer, token):
            self.seen.append((tokenizer, token))
            return {1: "hello", 2: " world"}[token]

    class _Tok:
        _tokenizer = object()

        def decode(self, ids):
            raise AssertionError("native stream must avoid prefix decode")

    import tokenizers.decoders
    original = tokenizers.decoders.DecodeStream
    tokenizers.decoders.DecodeStream = _Stream
    try:
        decoder = _IncrementalDetokenizer(_Tok())
        assert decoder.push([1]) == "hello"
        assert decoder.push([2]) == "hello world"
        assert decoder.finish() == "hello world"
        assert [token for _, token in decoder._native_stream.seen] == [1, 2]
    finally:
        tokenizers.decoders.DecodeStream = original


def test_incremental_detokenizer_unwraps_mlx_tokenizer_backend():
    from mlx_streaming.tui.backend import _IncrementalDetokenizer

    class _Backend:
        pass

    backend = _Backend()

    class _Wrapped:
        backend_tokenizer = backend

    class _Tok:
        _tokenizer = _Wrapped()

    decoder = _IncrementalDetokenizer(_Tok())
    assert decoder._native_tokenizer is backend


def test_mlx_backend_load_warms_up(monkeypatch):
    """加载后应做一次预热(把首轮 kernel 编译/专家池开销前移),并上报预热状态。"""
    import types

    import mlx_streaming.cli as cli_mod
    from mlx_streaming.tui.backend import MLXBackend

    monkeypatch.setattr(cli_mod, "_build_engine",
                        lambda args, on_status=None: ("M", "T", "D"))
    warmed = []
    monkeypatch.setattr(
        cli_mod, "_warmup",
        lambda model, tok, drafter, args: warmed.append((model, tok, drafter)))

    b = MLXBackend(types.SimpleNamespace(model="m", k=3, max_tokens=8, system=None))
    seen = []
    b.load(seen.append)

    assert (b._model, b._tok, b._drafter) == ("M", "T", "D")
    assert warmed == [("M", "T", "D")]           # 预热用加载好的引擎跑了一次
    assert any("预热" in s for s in seen)          # 有预热状态提示


def test_mlx_backend_splits_sync_prefill_and_async_decode(monkeypatch):
    """Throughput profile must enable async demand at the decode boundary."""
    import os
    import types

    import mlx_streaming.cli as cli_mod
    import mlx_streaming.mtp.generate as gen_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        eos_token_ids = None
        eos_token_id = -1
        chat_template = None

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: [1, 2, 3])
    monkeypatch.setenv("SPEC_SPLIT_DEMAND_AFTER_PREFILL", "1")
    monkeypatch.setenv("DEMAND_ASYNC", "0")
    seen = []

    def fake_mtp_generate(
        model, drafter, tok, prompt, max_tokens, K=3, ids_mode=False,
        profile=False, on_tokens=None, main_cache=None, cached_len=0,
        on_prefill_complete=None,
    ):
        seen.append(("prefill", os.environ["DEMAND_ASYNC"]))
        assert on_prefill_complete is not None
        on_prefill_complete()
        seen.append(("decode", os.environ["DEMAND_ASYNC"]))
        if on_tokens is not None:
            on_tokens([10])
        return [10], {
            "resident_tokens": prompt.shape[1],
            "wall_s": 0.25,
        }

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)
    args = types.SimpleNamespace(model="m", k=3, max_tokens=8, system=None)
    backend = MLXBackend(args)
    backend._tok = _Tok()
    backend._model = types.SimpleNamespace(make_cache=lambda: object())
    backend._drafter = object()

    prefill = []
    result = backend.generate(
        [{"role": "user", "content": "hi"}],
        lambda full, n: False,
        on_prefill=lambda tokens, seconds, tps: prefill.append(
            (tokens, seconds, tps)
        ),
    )

    assert seen == [("prefill", "0"), ("decode", "1")]
    assert os.environ["DEMAND_ASYNC"] == "0"
    assert len(prefill) == 1
    assert prefill[0][0] == 3
    assert prefill[0][1] > 0
    assert prefill[0][2] > 0
    assert result.prefill_tokens == 3
    assert result.prefill_s > 0
    assert result.prefill_tok_per_s > 0
    assert result.tok_per_s == 4.0


def test_mlx_backend_reports_prefill_without_split_demand(monkeypatch):
    """Prefill telemetry is public UI behavior, not a split-demand feature."""
    import types

    import mlx_streaming.cli as cli_mod
    import mlx_streaming.mtp.generate as gen_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        eos_token_ids = None
        eos_token_id = -1
        chat_template = None

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: [1, 2, 3])
    monkeypatch.delenv("SPEC_SPLIT_DEMAND_AFTER_PREFILL", raising=False)

    def fake_mtp_generate(*args, on_prefill_complete=None, on_tokens=None,
                          **kwargs):
        assert on_prefill_complete is not None
        on_prefill_complete()
        if on_tokens is not None:
            on_tokens([10])
        return [10], {"resident_tokens": 3, "wall_s": 0.25}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)
    backend = MLXBackend(types.SimpleNamespace(
        model="m", k=3, max_tokens=8, system=None,
    ))
    backend._tok = _Tok()
    backend._model = types.SimpleNamespace(make_cache=lambda: object())
    backend._drafter = object()

    seen = []
    result = backend.generate(
        [{"role": "user", "content": "hi"}],
        lambda full, n: False,
        on_prefill=lambda tokens, seconds, tps: seen.append(
            (tokens, seconds, tps)
        ),
    )
    assert len(seen) == 1 and seen[0][0] == 3
    assert result.prefill_tokens == 3
    assert result.prefill_s > 0
    assert result.prefill_tok_per_s > 0


def test_common_prefix_len():
    assert _common_prefix_len([], [1, 2]) == 0
    assert _common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert _common_prefix_len([1, 2], [1, 2, 3, 4]) == 2
    assert _common_prefix_len([1, 2, 3], [1, 2, 3]) == 3


def test_reuse_prefix_len_only_on_strict_prefix_extension():
    # 旧 cache 是新序列严格前缀且新序列更长 → 复用整段前缀
    assert _reuse_prefix_len([1, 2, 3, 10], [1, 2, 3, 10, 11, 4]) == 4
    # 中途分叉(如 /reset、编辑历史、retokenize 不一致)→ 不复用,全量重建
    assert _reuse_prefix_len([1, 2, 3, 10], [1, 2, 9, 10, 11]) == 0
    # 新序列不比旧长(无新增可 prefill)→ 不复用
    assert _reuse_prefix_len([1, 2, 3], [1, 2, 3]) == 0
    assert _reuse_prefix_len([1, 2, 3], [1, 2]) == 0
    # 无历史 cache → 不复用
    assert _reuse_prefix_len([], [1, 2, 3]) == 0


def test_mlx_backend_reuses_cache_on_strict_prefix_second_turn(monkeypatch):
    """第二轮 prompt 是首轮(prompt+生成)的严格延伸时,应复用同一 main_cache 且只 prefill 后缀。"""
    import types

    import mlx_streaming.mtp.generate as gen_mod
    import mlx_streaming.cli as cli_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        eos_token_ids = None
        eos_token_id = -1        # 无 EOS 干扰
        chat_template = None

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    # 两轮的完整编码:turn2 是 turn1(prompt[1,2,3] + 生成[10,11] → cache 记 [1,2,3,10])的严格延伸
    encoded = iter([[1, 2, 3], [1, 2, 3, 10, 11, 4, 5]])
    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: next(encoded))

    calls = []

    def fake_mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                          ids_mode=False, profile=False, on_tokens=None,
                          main_cache=None, cached_len=0,
                          on_prefill_complete=None):
        if on_prefill_complete is not None:
            on_prefill_complete()
        calls.append({"cached_len": cached_len, "cache": main_cache})
        produced = [10, 11]      # 生成两个 token;不变式 → cache 记 prompt + [10]
        if on_tokens is not None:
            on_tokens(produced)
        # 无 over-commit:resident 恰为 len(prompt)+len(produced)-1
        resident = prompt.shape[1] + len(produced) - 1
        return produced, {"resident_tokens": resident}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)

    class _Model:
        def __init__(self):
            self.n = 0

        def make_cache(self):
            self.n += 1
            return f"cache{self.n}"

    args = types.SimpleNamespace(model="m", k=1, max_tokens=100, system=None)
    b = MLXBackend(args)
    b._tok = _Tok()
    b._model = _Model()
    b._drafter = object()

    b.generate([{"role": "user", "content": "u1"}], lambda full, n: False)
    b.generate([{"role": "user", "content": "u2"}], lambda full, n: False)

    # 首轮:无历史 → 全量重建(cached_len=0,新建 cache1)
    assert calls[0]["cached_len"] == 0
    assert calls[0]["cache"] == "cache1"
    # 次轮:严格前缀 → 复用 cache1,只 prefill 后缀(cached_len=4 = len([1,2,3,10]))
    assert calls[1]["cached_len"] == 4
    assert calls[1]["cache"] == "cache1"       # 同一 cache 对象,未重新 make_cache
    assert b._model.n == 1                       # make_cache 只调用了一次


def test_mlx_backend_disables_reuse_after_overcommit(monkeypatch):
    """末步跨 max_tokens 的 over-commit(cache 领先于 produced)后,应禁用复用、下轮全量重建。"""
    import types

    import mlx_streaming.mtp.generate as gen_mod
    import mlx_streaming.cli as cli_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        eos_token_ids = None
        eos_token_id = -1
        chat_template = None

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    encoded = iter([[1, 2, 3], [1, 2, 3, 10, 11, 4]])
    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: next(encoded))

    calls = []

    def fake_mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                          ids_mode=False, profile=False, on_tokens=None,
                          main_cache=None, cached_len=0,
                          on_prefill_complete=None):
        if on_prefill_complete is not None:
            on_prefill_complete()
        calls.append({"cached_len": cached_len})
        produced = [10, 11]
        # 模拟 over-commit:resident 比不变式预期多 1(cache 领先)
        resident = prompt.shape[1] + len(produced)
        return produced, {"resident_tokens": resident}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)

    class _Model:
        def __init__(self):
            self.n = 0

        def make_cache(self):
            self.n += 1
            return f"cache{self.n}"

    args = types.SimpleNamespace(model="m", k=1, max_tokens=100, system=None)
    b = MLXBackend(args)
    b._tok = _Tok()
    b._model = _Model()
    b._drafter = object()

    b.generate([{"role": "user", "content": "u1"}], lambda full, n: False)
    assert b._main_cache is None and b._cached_ids == []   # over-commit → 不记录 cache
    b.generate([{"role": "user", "content": "u2"}], lambda full, n: False)
    assert calls[1]["cached_len"] == 0                     # 下轮全量重建
    assert b._model.n == 2


def test_mlx_backend_rebuilds_cache_when_history_diverges(monkeypatch):
    """历史分叉(如 /reset 或编辑)时不复用旧 cache,应全量重建新 cache。"""
    import types

    import mlx_streaming.mtp.generate as gen_mod
    import mlx_streaming.cli as cli_mod
    from mlx_streaming.tui.backend import MLXBackend

    class _Tok:
        eos_token_ids = None
        eos_token_id = -1
        chat_template = None

        def decode(self, ids):
            return ",".join(str(i) for i in ids)

    # turn2 在位置 2 就与 turn1 的 cache([1,2,3,10])分叉 → 不可复用
    encoded = iter([[1, 2, 3], [1, 2, 99, 88]])
    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: next(encoded))

    calls = []

    def fake_mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                          ids_mode=False, profile=False, on_tokens=None,
                          main_cache=None, cached_len=0,
                          on_prefill_complete=None):
        if on_prefill_complete is not None:
            on_prefill_complete()
        calls.append({"cached_len": cached_len, "cache": main_cache})
        produced = [10, 11]
        resident = prompt.shape[1] + len(produced) - 1   # 无 over-commit,首轮正常记录 cache
        return produced, {"resident_tokens": resident}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)

    class _Model:
        def __init__(self):
            self.n = 0

        def make_cache(self):
            self.n += 1
            return f"cache{self.n}"

    args = types.SimpleNamespace(model="m", k=1, max_tokens=100, system=None)
    b = MLXBackend(args)
    b._tok = _Tok()
    b._model = _Model()
    b._drafter = object()

    b.generate([{"role": "user", "content": "u1"}], lambda full, n: False)
    b.generate([{"role": "user", "content": "u2"}], lambda full, n: False)

    assert calls[1]["cached_len"] == 0           # 分叉 → 不复用
    assert calls[1]["cache"] == "cache2"         # 新建了第二个 cache
    assert b._model.n == 2
