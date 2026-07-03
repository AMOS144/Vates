"""vates 全屏 TUI(Textual 实现,视觉对标 opencode)。

界面只依赖 ChatBackend 接口。加载与每轮生成均在 worker 线程执行,
通过 call_from_thread 把状态/流式文本安全回推到 UI 线程。
"""
from __future__ import annotations

import os
import time
from collections import deque
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

# 生成中状态栏用滑动窗口算「瞬时」tok/s 的时间窗(秒);越小越灵敏、越大越平滑。
_TPS_WINDOW = 1.0

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
        # 首 token 到达时刻与基准 token 数,用于结束时计算「累计解码平均」tok/s
        # (排除 prefill)。_gen_t0 为 0.0 表示本轮尚未收到首 token。
        self._gen_t0 = 0.0
        self._n0 = 0
        # 生成中「瞬时」tok/s 的滑动窗口采样:每项为 (时刻, 累计 token 数)。
        self._tps_window = _TPS_WINDOW
        self._samples: deque[tuple[float, int]] = deque()

    def compose(self) -> ComposeResult:
        yield Static(self._top_text(), id="top")
        yield VerticalScroll(id="chat")
        # 提示放到边框标题里,不用文本区的 placeholder:
        # 长占位符在部分终端增量重绘时不会被擦除,打字后会残留「后面还有字」,
        # 直到全量重绘(Enter/截图/resize)才消失;边框标题在边框上,不受此影响。
        yield Input(id="prompt")
        yield Static("", id="status")

    def on_mount(self) -> None:
        inp = self.query_one("#prompt", Input)
        inp.border_title = "输入消息 · Enter 发送 · Esc 中断 · /help"
        # 加载完成前禁用输入框,避免用户在模型就绪前发消息
        inp.disabled = True
        self._set_status("加载模型中…")
        self._load()

    def on_input_changed(self, event: Input.Changed) -> None:
        # 兜底:强制整屏重绘,清除个别终端增量重绘遗留的输入残影
        # (等价于截图/resize 触发的全量刷新)。
        self.refresh()

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
        """重新启用并聚焦输入框(加载完成/一轮生成结束后统一调用)。

        刚把 disabled 置 False 时 can_focus 状态可能还没刷新,直接 focus 偶发无效
        (表现为输入框失焦时占位提示不消失);故延到下一次刷新后再聚焦,确保稳定拿到焦点。
        """
        inp = self.query_one("#prompt", Input)
        inp.disabled = False
        self.call_after_refresh(inp.focus)

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
        # 置 0 表示还没收到首 token;真正的计时起点推迟到第一次 _on_stream。
        self._gen_t0 = 0.0
        self._n0 = 0
        self._samples.clear()
        self.query_one("#prompt", Input).disabled = True
        self._set_status("思考中…")
        self._generate(list(self._messages))

    @work(thread=True, exclusive=True, group="gen")
    def _generate(self, messages: list[dict]) -> None:
        # 生成在 worker 线程运行;on_text 返回 self._stop 让后端可提前中断
        def on_text(full: str, n_tokens: int) -> bool:
            # 应用已退出时,call_from_thread 会抛错;此时直接请求停止,避免 worker 线程未捕获异常。
            if not self.is_running:
                return True
            try:
                self.call_from_thread(self._on_stream, full, n_tokens)
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

    def _record_sample(self, now: float, n_tokens: int) -> None:
        """记录一次采样,并丢弃早于滑动窗口的旧点(保留跨越窗口边界的那一个)。"""
        self._samples.append((now, n_tokens))
        w = self._tps_window
        # 当第二个点仍比窗口还老时,第一个点已冗余,可丢弃;
        # 循环后 _samples[0] 恰好是刚跨过窗口边界的采样,窗口长度 ≈ w。
        while len(self._samples) >= 2 and now - self._samples[1][0] >= w:
            self._samples.popleft()

    def _window_tps(self, now: float) -> Optional[float]:
        """按滑动窗口算瞬时 tok/s;样本不足(不够两点或时间差为 0)时返回 None。"""
        if len(self._samples) < 2:
            return None
        t0, n0 = self._samples[0]
        dt = now - t0
        if dt <= 0:
            return None
        return (self._samples[-1][1] - n0) / dt

    def _on_stream(self, full: str, n_tokens: int) -> None:
        if self._cur is not None:
            self._cur.stream(full)
            self._scroll_end()
        now = time.monotonic()
        # 首次回调:prefill 刚结束,记下计时起点与基准 token 数,供结束时算累计解码平均。
        if self._gen_t0 == 0.0:
            self._gen_t0 = now
            self._n0 = n_tokens
        # 生成中显示滑动窗口「瞬时」速度:一直在动,能反映后期变慢,不被历史平均拖住。
        self._record_sample(now, n_tokens)
        tps = self._window_tps(now)
        if tps is None:
            self._set_status(f"思考中 · {n_tokens} tok")
        else:
            self._set_status(f"思考中 · {n_tokens} tok · {tps:.1f} tok/s")

    def _on_done(self, result: GenResult) -> None:
        if self._cur is not None:
            self._cur.finalize(result.text)
        self._messages.append({"role": "assistant", "content": result.text})
        self._busy = False
        self._cur = None
        suffix = " · 已中断" if result.stopped else ""
        # 与流式状态栏同口径:从首 token 起、按解码 token 数算,避免结束瞬间数字回落。
        # 若本轮没触发过流式(_gen_t0 仍为 0),退回后端上报的 tok/s。
        dt = time.monotonic() - self._gen_t0
        if self._gen_t0 > 0.0 and dt > 0:
            tps = (result.n_tokens - self._n0) / dt
        else:
            tps = result.tok_per_s
        self._set_status(
            f"就绪 · {result.n_tokens} tok · {tps:.1f} tok/s{suffix}")
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
