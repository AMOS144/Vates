"""vates 全屏 TUI(Textual 实现,视觉对标 opencode)。

界面只依赖 ChatBackend 接口。加载与每轮生成均在 worker 线程执行,
通过 call_from_thread 把状态/流式文本安全回推到 UI 线程。
"""
from __future__ import annotations

import os
from typing import Optional

from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from mlx_streaming.tui.backend import ChatBackend, GenResult
from mlx_streaming.tui.banner import LOGO

_ACCENT = "#2dd4bf"

_HELP = (
    "可用命令:\n"
    "  /help    显示本帮助\n"
    "  /reset   清空对话历史(保留 system)\n"
    "  /clear   清空对话区显示\n"
    "  /exit    退出\n\n"
    "快捷键:Enter 发送 · Esc 中断生成 · Ctrl+C 退出"
)


def _short_model(path: str) -> str:
    # 只取模型路径末段,避免顶栏/状态栏过长
    return os.path.basename(path.rstrip("/")) or path


class ChatMessage(Static):
    """一条对话消息。role ∈ {'user','assistant'};助手消息支持流式更新与 Markdown 收尾。"""

    def __init__(self, role: str, text: str = "", *, final: bool = True):
        super().__init__(classes=f"msg {role}")
        self.role = role
        self.text = text
        self.final = final
        self._refresh_content()

    def stream(self, text: str) -> None:
        # 流式过程中用纯文本渲染,避免半截 Markdown 抖动
        self.text = text
        self.final = False
        self._refresh_content()

    def finalize(self, text: str) -> None:
        # 收尾时切换为 Markdown 渲染
        self.text = text
        self.final = True
        self._refresh_content()

    def _refresh_content(self) -> None:
        if self.role == "user":
            header = Text("› 你", style="bold")
            body = Text(self.text)
        else:
            header = Text("⏺ vates", style=f"bold {_ACCENT}")
            if not self.text:
                body = Text("正在思考…", style="dim italic")
            elif self.final:
                body = RichMarkdown(self.text)
            else:
                body = Text(self.text)
        self.update(Group(header, body))


class VatesApp(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("escape", "interrupt", "中断"),
        ("ctrl+c", "quit", "退出"),
    ]

    def __init__(self, backend: ChatBackend, args):
        super().__init__()
        self.backend = backend
        self.args = args
        self._messages: list[dict] = []
        # 保留 system 消息,/reset 时不清空这部分
        if getattr(args, "system", None):
            self._messages.append({"role": "system", "content": args.system})
        self._base_len = len(self._messages)
        self._busy = False
        self._stop = False
        self._cur: Optional[ChatMessage] = None

    def compose(self) -> ComposeResult:
        yield Static(self._top_text(), id="top")
        yield VerticalScroll(id="chat")
        yield Input(placeholder="输入消息，回车发送，/help 查看命令", id="prompt")
        yield Static("", id="status")

    def on_mount(self) -> None:
        # 加载完成前禁用输入框,避免用户在模型就绪前发消息
        self.query_one("#prompt", Input).disabled = True
        self._set_status("加载模型中…")
        self._load()

    def _top_text(self) -> Text:
        t = Text()
        t.append(LOGO, style=f"bold {_ACCENT}")
        t.append(f"   {_short_model(self.args.model)} · k={self.args.k} · "
                 f"{self.args.max_tokens} tok", style="dim")
        return t

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(
            f" {_short_model(self.args.model)} · {text}")

    @work(thread=True, exclusive=True, group="load")
    def _load(self) -> None:
        # 在 worker 线程执行阻塞加载,回调需经 call_from_thread 回到 UI 线程
        def on_status(msg: str):
            # 应用已退出时不再回推,避免 call_from_thread 抛错
            if self.is_running:
                self.call_from_thread(self._set_status, f"加载中 · {msg}")

        try:
            self.backend.load(on_status)
        except Exception as e:  # noqa: BLE001
            if self.is_running:
                self.call_from_thread(self._on_load_failed, str(e))
            return
        if self.is_running:
            self.call_from_thread(self._on_load_done)

    def _enable_input(self) -> None:
        """重新启用并聚焦输入框(加载完成/一轮生成结束后统一调用)。"""
        inp = self.query_one("#prompt", Input)
        inp.disabled = False
        inp.focus()

    def _on_load_done(self) -> None:
        self._set_status("就绪")
        self._enable_input()

    def _on_load_failed(self, err: str) -> None:
        self._add("assistant", f"模型加载失败:{err}\n\n请检查路径后用 /exit 退出重试。")
        self._set_status("加载失败")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._command(text)
            return
        if self._busy:
            return
        self._start(text)

    def _command(self, cmd: str) -> None:
        if cmd in ("/exit", "/quit"):
            self.exit()
        elif cmd == "/help":
            self._add("assistant", _HELP)
        elif cmd in ("/reset", "/clear") and self._busy:
            # 生成中改动历史/移除正在流式的 _cur 会让 _on_stream/_on_done 操作已卸载组件,故拒绝
            self._add("assistant", "生成中,请等本轮结束或按 Esc 中断后再执行该命令。")
        elif cmd == "/reset":
            # 只清空 system 之后的历史
            del self._messages[self._base_len:]
            self._add("assistant", "对话历史已清空。")
        elif cmd == "/clear":
            self.query_one("#chat", VerticalScroll).remove_children()
        else:
            self._add("assistant", f"未知命令:{cmd}(/help 查看可用命令)")

    def _start(self, user_text: str) -> None:
        self._messages.append({"role": "user", "content": user_text})
        self._add("user", user_text)
        self._cur = self._add("assistant", "", final=False)
        self._busy = True
        self._stop = False
        self.query_one("#prompt", Input).disabled = True
        self._set_status("思考中…")
        self._generate(list(self._messages))

    @work(thread=True, exclusive=True, group="gen")
    def _generate(self, messages: list[dict]) -> None:
        # 生成在 worker 线程运行;on_text 返回 self._stop 让后端可提前中断
        def on_text(full: str) -> bool:
            # 应用已退出时,call_from_thread 会抛错;此时直接请求停止,避免 worker 线程未捕获异常。
            if not self.is_running:
                return True
            try:
                self.call_from_thread(self._on_stream, full)
            except Exception:  # noqa: BLE001
                return True
            return self._stop

        try:
            result = self.backend.generate(messages, on_text)
        except Exception as e:  # noqa: BLE001
            if self.is_running:
                self.call_from_thread(self._on_error, str(e))
            return
        if self.is_running:
            self.call_from_thread(self._on_done, result)

    def _on_stream(self, full: str) -> None:
        if self._cur is not None:
            self._cur.stream(full)
            self._scroll_end()

    def _on_done(self, result: GenResult) -> None:
        if self._cur is not None:
            self._cur.finalize(result.text)
        self._messages.append({"role": "assistant", "content": result.text})
        self._busy = False
        self._cur = None
        suffix = " · 已中断" if result.stopped else ""
        self._set_status(
            f"就绪 · {result.n_tokens} tok · {result.tok_per_s:.1f} tok/s{suffix}")
        self._enable_input()

    def _on_error(self, err: str) -> None:
        if self._cur is not None:
            self._cur.finalize(f"生成出错:{err}")
        self._busy = False
        self._cur = None
        self._set_status("就绪(上一轮出错)")
        self._enable_input()

    def action_interrupt(self) -> None:
        # 仅在生成中时置位中断标志,由 on_text 闭包读取
        if self._busy:
            self._stop = True
            self._set_status("正在中断…")

    def _add(self, role: str, text: str, *, final: bool = True) -> ChatMessage:
        msg = ChatMessage(role, text, final=final)
        self.query_one("#chat", VerticalScroll).mount(msg)
        self._scroll_end()
        return msg

    def _scroll_end(self) -> None:
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
