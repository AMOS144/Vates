"""VatesApp 交互测试:用 FakeBackend + Textual run_test,不加载真实模型。"""
import types

import pytest

from mlx_streaming.tui.app import ChatMessage, VatesApp
from mlx_streaming.tui.backend import FakeBackend


def _args(**kw):
    d = dict(model="models/qwen3_next_80b_4bit", k=3, max_tokens=512, system=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


@pytest.mark.asyncio
async def test_loads_and_becomes_ready():
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        from textual.widgets import Input
        assert app.query_one("#prompt", Input).disabled is False
        # textual 8.x 用 Static.content 取代旧的 .renderable
        assert "就绪" in str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_send_message_streams_and_finalizes():
    backend = FakeBackend(reply="快排是一种分治排序。")
    app = VatesApp(backend, _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        inp.focus()
        inp.value = "讲讲快排"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        msgs = list(app.query(ChatMessage))
        assert msgs[-2].role == "user" and msgs[-2].text == "讲讲快排"
        assert msgs[-1].role == "assistant"
        assert msgs[-1].text == "快排是一种分治排序。"
        assert msgs[-1].final is True
        assert backend.seen_messages[-1][-1] == {"role": "user", "content": "讲讲快排"}


@pytest.mark.asyncio
async def test_reset_clears_history():
    backend = FakeBackend(reply="ok")
    app = VatesApp(backend, _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        inp.focus()
        inp.value = "第一轮"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        inp.value = "/reset"
        await pilot.press("enter")
        await pilot.pause()
        inp.value = "第二轮"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        contents = [m["content"] for m in backend.seen_messages[-1]]
        assert "第一轮" not in contents
        assert "第二轮" in contents


@pytest.mark.asyncio
async def test_reset_ignored_while_busy():
    """生成中执行 /reset 应被拒绝且不清历史(避免卸载正在流式的组件)。"""
    app = VatesApp(FakeBackend(reply="ok"), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._busy = True                      # 模拟生成中
        before = len(app._messages)
        app._command("/reset")
        assert len(app._messages) == before   # 历史未被清空


@pytest.mark.asyncio
async def test_interrupt_sets_stop_flag():
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._busy = True
        app._stop = False
        app.action_interrupt()
        assert app._stop is True


def test_cmd_chat_launches_tui_by_default(monkeypatch):
    """默认(非 --plain)应构造 MLXBackend 并调用 run_tui,不进旧 REPL。"""
    import types as _t

    from mlx_streaming import cli

    called = {}

    def fake_run_tui(backend, args):
        called["backend_type"] = type(backend).__name__
        called["plain"] = getattr(args, "plain", None)
        return 0

    monkeypatch.setattr("mlx_streaming.tui.run_tui", fake_run_tui)
    args = _t.SimpleNamespace(model="m", k=3, max_tokens=512, system=None,
                              plain=False, expert_dir="e", mtp_out="o",
                              qn_config="q", expert_slots=32, spec_slots=None,
                              stats=False)
    rc = cli.cmd_chat(args)
    assert rc == 0
    assert called["backend_type"] == "MLXBackend"
    assert called["plain"] is False


@pytest.mark.asyncio
async def test_on_done_stopped_shows_interrupted_status():
    """被中断的一轮:finalize 已生成文本,状态栏出现「已中断」,输入框重新可用。"""
    from mlx_streaming.tui.backend import GenResult
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._cur = app._add("assistant", "", final=False)
        app._busy = True
        app._on_done(GenResult(text="ab", n_tokens=2, tok_per_s=0.0, stopped=True))
        await pilot.pause()
        from textual.widgets import Input
        assert app._busy is False
        assert app.query_one("#prompt", Input).disabled is False
        status = str(app.query_one("#status").content)
        assert "已中断" in status
        msgs = list(app.query(ChatMessage))
        assert msgs[-1].text == "ab" and msgs[-1].final is True


@pytest.mark.asyncio
async def test_load_failed_shows_error_and_status():
    """加载失败:对话区出现错误消息,状态栏为「加载失败」。"""
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._on_load_failed("模型文件缺失")
        await pilot.pause()
        status = str(app.query_one("#status").content)
        assert "加载失败" in status
        msgs = list(app.query(ChatMessage))
        assert any("模型文件缺失" in m.text for m in msgs)


def test_window_tps_uses_recent_samples_only():
    """滑动窗口瞬时速度:只按最近 ~窗口 内的采样算,旧采样被丢弃。"""
    app = VatesApp(FakeBackend(), _args())
    app._tps_window = 1.0
    # 前 1 秒 20 token → 窗口内 (100.0,0)→(101.0,20),瞬时 20 tok/s
    app._record_sample(100.0, 0)
    app._record_sample(100.5, 10)
    app._record_sample(101.0, 20)
    assert app._window_tps(101.0) == 20.0
    # 之后一段变慢:now=102.5,窗口(1s)内应丢掉 100.0/100.5,基准变成 (101.0,20)
    app._record_sample(102.5, 40)
    assert app._samples[0] == (101.0, 20)
    # (40-20)/(102.5-101.0) = 20/1.5 ≈ 13.33,反映当前速度而非历史平均
    assert abs(app._window_tps(102.5) - (20 / 1.5)) < 1e-6


def test_window_tps_needs_two_samples():
    """采样不足两点时不出速度(避免除零/瞎猜)。"""
    app = VatesApp(FakeBackend(), _args())
    app._record_sample(100.0, 5)
    assert app._window_tps(100.0) is None


@pytest.mark.asyncio
async def test_stream_status_shows_tokens_then_windowed_speed():
    """生成中状态栏:首次回调只显示 token 数,拿到第二个采样后显示瞬时 tok/s。"""
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._cur = app._add("assistant", "", final=False)
        app._gen_t0 = 0.0
        app._n0 = 0
        app._samples.clear()
        # 首次回调:只记基准,尚无法算窗口速度
        app._on_stream("你好世界呀", 5)
        assert app._gen_t0 != 0.0 and app._n0 == 5
        s1 = str(app.query_one("#status").content)
        assert "5 tok" in s1
        assert "tok/s" not in s1
        # 制造一个时间差,第二次回调应显示滑动窗口瞬时速度
        app._samples[0] = (app._samples[0][0] - 1.0, 5)
        app._on_stream("你好世界呀又三字", 8)
        s2 = str(app.query_one("#status").content)
        assert "8 tok" in s2
        assert "tok/s" in s2


@pytest.mark.asyncio
async def test_status_keeps_prefill_and_decode_speeds():
    """Prefill 完成后立即显示，流式生成和最终状态都不丢失。"""
    from mlx_streaming.tui.backend import GenResult

    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._cur = app._add("assistant", "", final=False)
        app._on_prefill(256, 0.5, 512.0)
        prefill_status = str(app.query_one("#status").content)
        assert "prefill 512.0 tok/s (256 tok)" in prefill_status

        app._on_stream("你好", 2)
        stream_status = str(app.query_one("#status").content)
        assert "prefill 512.0 tok/s (256 tok)" in stream_status
        assert "decode 2 tok" in stream_status

        app._on_done(GenResult(
            text="你好",
            n_tokens=2,
            tok_per_s=30.0,
            stopped=False,
            prefill_tokens=256,
            prefill_s=0.5,
            prefill_tok_per_s=512.0,
        ))
        done_status = str(app.query_one("#status").content)
        assert "prefill 512.0 tok/s (256 tok)" in done_status
        assert "decode 30.0 tok/s (2 tok)" in done_status


@pytest.mark.asyncio
async def test_final_status_uses_backend_decode_clock():
    """结束数字使用引擎 decode-only wall_s，不混入 UI 重绘时间。"""
    import time
    from mlx_streaming.tui.backend import GenResult
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        app._cur = app._add("assistant", "", final=False)
        # 首 token 在 ~2 秒前到达,基准 2 token;本轮共产出 22 token
        app._gen_t0 = time.monotonic() - 2.0
        app._n0 = 2
        app._on_done(GenResult(text="回答", n_tokens=22, tok_per_s=10.0, stopped=False))
        status = str(app.query_one("#status").content)
        assert "就绪" in status
        assert "22 tok" in status
        assert "decode 10.0 tok/s" in status
