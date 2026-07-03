"""_chat_repl(--plain 纯文本 REPL)的跨轮 KV cache 复用与预热接线。

用假的 mtp_generate / _encode_chat / input 驱动两轮对话,验证第二轮命中严格前缀时复用同一 cache
且只 prefill 后缀;不加载真实模型。
"""
import builtins
import types

import mlx_streaming.cli as cli_mod
import mlx_streaming.mtp.generate as gen_mod
from mlx_streaming.tests.test_mtp_stream_hook import _kv_toy_k3


class _Tok:
    eos_token_ids = None
    eos_token_id = -1        # 无 EOS 干扰
    chat_template = None

    def decode(self, ids):
        return ",".join(str(i) for i in ids)


class _Model:
    def __init__(self):
        self.n = 0

    def make_cache(self):
        self.n += 1
        return f"cache{self.n}"


def _args(**kw):
    d = dict(model="m", k=3, max_tokens=100, system=None, stats=False)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_chat_repl_reuses_cache_across_turns(monkeypatch):
    model = _Model()
    monkeypatch.setattr(cli_mod, "_build_engine", lambda args: (model, _Tok(), object()))
    monkeypatch.setattr(cli_mod, "_warmup", lambda *a, **k: None)

    # 两轮编码:turn2 是 turn1(prompt[1,2,3] + 生成入 cache 的 [10])的严格延伸
    encoded = iter([[1, 2, 3], [1, 2, 3, 10, 11, 4, 5]])
    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: next(encoded))

    calls = []

    def fake_mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                          ids_mode=False, profile=False, on_tokens=None,
                          main_cache=None, cached_len=0):
        calls.append({"cached_len": cached_len, "cache": main_cache})
        produced = [10, 11]
        if on_tokens is not None:
            on_tokens(produced)
        resident = prompt.shape[1] + len(produced) - 1     # 无 over-commit
        return produced, {"resident_tokens": resident, "avg_accept_len": 1.0}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)

    # 驱动输入:两轮对话后退出
    inputs = iter(["问题一", "问题二", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    rc = cli_mod._chat_repl(_args())
    assert rc == 0

    # 首轮全量重建(cached_len=0,新建 cache1);次轮严格前缀复用 cache1(cached_len=4)
    assert calls[0]["cached_len"] == 0 and calls[0]["cache"] == "cache1"
    assert calls[1]["cached_len"] == 4 and calls[1]["cache"] == "cache1"
    assert model.n == 1          # make_cache 只调用一次(第二轮复用,未新建)


def test_warmup_drives_long_diverse_prompt(monkeypatch):
    """预热应用一段较长、token id 分散的合成 prompt 跑一次生成(覆盖多专家 + 多块 prefill)。"""
    captured = {}

    def spy(model, drafter, tok, prompt, max_tokens, **kw):
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        return [0], {"resident_tokens": 0}

    monkeypatch.setattr(gen_mod, "mtp_generate", spy)

    model = _kv_toy_k3()          # vocab=40
    cli_mod._warmup(model, None, object(), types.SimpleNamespace(k=3))

    p = captured["prompt"]
    assert p.shape[1] >= 16       # 远长于旧的极短 prompt
    distinct = len(set(int(x) for x in p[0].tolist()))
    assert distinct >= 8          # id 分散,能路由到更多专家
    assert captured["max_tokens"] >= 4


def test_warmup_end_to_end_on_toy_model():
    """不打桩:合成 prompt 走真实 mtp_generate 在玩具模型上应无异常跑通。"""
    from mlx_streaming.tests.test_mtp_generate import _RandDraft
    model = _kv_toy_k3()          # vocab=40
    cli_mod._warmup(model, None, _RandDraft(40), types.SimpleNamespace(k=3))


def test_warmup_swallows_errors(monkeypatch):
    """预热失败不应抛出(不能因预热问题中断启动)。"""
    def boom(*a, **k):
        raise RuntimeError("kernel compile failed")

    monkeypatch.setattr(gen_mod, "mtp_generate", boom)
    model = _kv_toy_k3()
    cli_mod._warmup(model, None, object(), types.SimpleNamespace(k=3))   # 不抛异常即通过


def test_chat_repl_reset_drops_cache(monkeypatch):
    """/reset 后应弃用旧 cache:即使编码恰是旧 cache 的延伸,也全量重建。"""
    model = _Model()
    monkeypatch.setattr(cli_mod, "_build_engine", lambda args: (model, _Tok(), object()))
    monkeypatch.setattr(cli_mod, "_warmup", lambda *a, **k: None)

    encoded = iter([[1, 2, 3], [1, 2, 3, 10, 11]])
    monkeypatch.setattr(cli_mod, "_encode_chat", lambda tok, msgs: next(encoded))

    calls = []

    def fake_mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                          ids_mode=False, profile=False, on_tokens=None,
                          main_cache=None, cached_len=0):
        calls.append({"cached_len": cached_len})
        produced = [10, 11]
        if on_tokens is not None:
            on_tokens(produced)
        return produced, {"resident_tokens": prompt.shape[1] + len(produced) - 1}

    monkeypatch.setattr(gen_mod, "mtp_generate", fake_mtp_generate)

    inputs = iter(["问题一", "/reset", "问题二", "/exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    cli_mod._chat_repl(_args())

    assert calls[0]["cached_len"] == 0     # 首轮
    assert calls[1]["cached_len"] == 0     # /reset 后弃用旧 cache,全量重建
    assert model.n == 2
