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
