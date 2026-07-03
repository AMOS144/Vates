# vates 全屏 TUI(仿 opencode)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development(推荐)或 superpowers:executing-plans 逐任务实现本计划。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 把 `vates chat` 从裸 `print`/`input` REPL 升级为一个用 Textual 写的全屏 TUI(顶栏 logo + 模型信息、可滚动对话区、带边框输入框、底部状态栏),助手回答逐字流式渲染、完成后按 Markdown 排版,支持 `Esc` 中断,并保留 `--plain` 走旧 REPL 作兜底。

**Architecture:** 界面与推理引擎彻底解耦——UI 只依赖 `ChatBackend` 抽象(不 import 任何 MLX 符号),真实推理由 `MLXBackend` 封装 `_build_engine`/`mtp_generate`,测试用 `FakeBackend`(不加载模型)。生成/加载均跑在 Textual worker 线程,通过 `call_from_thread` 回推 UI。对核心生成代码仅加一个向后兼容的 `on_tokens` 钩子,同时承载「流式吐字」+「Esc 中断」。

**Tech Stack:** Python 3.11+、Textual(TUI)、Rich(Markdown 渲染)、pytest(含 Textual `run_test`/`Pilot`)、MLX(现有推理栈,测试不触碰)。

**关联 spec:** `docs/superpowers/specs/2026-07-03-vates-tui-opencode-style-design.md`

> 说明:spec 第 3.1 节原写「只发新增后缀」;实现改为**每步回调传完整累计文本**(`on_text(full)->bool`),UI 直接整段刷新。这样对 BPE 多字节/合并改写更稳健,几百 token 的整段刷新在 Textual 里开销可忽略。本计划以此为准。

---

## 文件结构

- 修改:`mlx_streaming/mtp/generate.py` — 新增可选 `on_tokens` 钩子(两处 append 点 + break)。
- 修改:`mlx_streaming/cli.py` — `_build_engine` 加 `on_status`;`cmd_chat` 分流 TUI/`--plain`;旧 REPL 抽成 `_chat_repl`;`_add_chat_args` 加 `--plain`。
- 修改:`pyproject.toml` — dependencies 增加 `textual`。
- 新建:`mlx_streaming/tui/__init__.py` — 导出 `run_tui(backend, args)`。
- 新建:`mlx_streaming/tui/backend.py` — `GenResult`、`ChatBackend` 协议、`FakeBackend`、`MLXBackend`。
- 新建:`mlx_streaming/tui/banner.py` — `LOGO` 常量。
- 新建:`mlx_streaming/tui/app.py` — `VatesApp`、`ChatMessage` 组件、worker 线程、事件、斜杠命令。
- 新建:`mlx_streaming/tui/styles.tcss` — 配色/边框/间距(teal 主题)。
- 新建测试:`mlx_streaming/tests/test_mtp_stream_hook.py`、`test_tui_backend.py`、`test_tui_app.py`。

依赖顺序:Task 1(钩子)→ Task 2(banner + backend)→ Task 3(app + tui 包 + 样式)→ Task 4(cli 接线 + 依赖 + README)。

---

## Task 1: 给 mtp_generate 加流式/中断钩子

**Files:**
- Modify: `mlx_streaming/mtp/generate.py`(函数签名约 line 187;两处 `produced.append` 约 line 250-253 与 line 374-377)
- Test: `mlx_streaming/tests/test_mtp_stream_hook.py`

- [ ] **Step 1: 写失败测试**

新建 `mlx_streaming/tests/test_mtp_stream_hook.py`:

```python
"""mtp_generate 的 on_tokens 钩子:流式回调 + 中断,且默认行为不变。"""
import mlx.core as mx
from mlx_lm.models import cache as kvcache

from mlx_streaming.mtp.generate import mtp_generate
# 复用现有测试里的玩具模型与草稿器(均为模块级类,import 无副作用)
from mlx_streaming.tests.test_mtp_generate import _ToyModel, _SelfDraft


def _kv_toy():
    mx.random.seed(0)
    model = _ToyModel(nl=2)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    return model


def test_on_tokens_receives_all_produced_in_order():
    """所有回调收到的 token 拼起来应等于最终 produced。"""
    model = _kv_toy()
    prompt = mx.array([[1, 5, 9]])
    seen = []

    def on_tokens(new_ids):
        seen.extend(new_ids)
        return False

    produced, _ = mtp_generate(model, _SelfDraft(model), None, prompt, 12,
                               K=1, ids_mode=True, on_tokens=on_tokens)
    assert seen == produced
    assert len(produced) == 12


def test_on_tokens_true_requests_stop():
    """回调返回 True 后应尽快停止,产出远少于 max_tokens。"""
    model = _kv_toy()
    prompt = mx.array([[1, 5, 9]])
    calls = {"n": 0}

    def on_tokens(new_ids):
        calls["n"] += 1
        return True  # 第一次即请求中断

    produced, _ = mtp_generate(model, _SelfDraft(model), None, prompt, 100,
                               K=1, ids_mode=True, on_tokens=on_tokens)
    assert calls["n"] == 1
    assert len(produced) < 100


def test_default_none_unchanged():
    """不传 on_tokens 时行为与之前一致(能正常产出 max_tokens 个)。"""
    model = _kv_toy()
    prompt = mx.array([[1, 5, 9]])
    produced, _ = mtp_generate(model, _SelfDraft(model), None, prompt, 12,
                               K=1, ids_mode=True)
    assert len(produced) == 12
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_mtp_stream_hook.py -v`
Expected: FAIL — `mtp_generate() got an unexpected keyword argument 'on_tokens'`

- [ ] **Step 3: 加钩子参数与调用点**

在 `mlx_streaming/mtp/generate.py` 修改签名(约 line 187):

```python
def mtp_generate(model, drafter, tok, prompt, max_tokens, K=3, ids_mode=False,
                 profile=False, on_tokens=None):
```

在**树验证路径**的 append 之后(约 line 250-253),把:

```python
            for t in new_tokens:
                produced.append(t)
                if len(produced) >= max_tokens:
                    break
```

改为:

```python
            _stop = False
            for t in new_tokens:
                produced.append(t)
                if len(produced) >= max_tokens:
                    break
            if on_tokens is not None and on_tokens(list(new_tokens)):
                _stop = True
            H_last = rH[:, -1:, :]
            if mtp_cache is not None and hasattr(drafter, "sync"):
                drafter.sync(prev_H_last, rH, accepted_in, mtp_cache)
            if profile:
                t_sync += time.perf_counter() - _tic
            n_steps += 1
            mx.eval(x, H_last)
            if _stop:
                break
            continue
```

> 注意:树验证分支原本以 `continue` 结束(其 append 之后还有 `H_last=... / drafter.sync / n_steps / mx.eval / continue`)。上面把「产出后的收尾」整体保留,只在 `mx.eval` 后、`continue` 前插入 `if _stop: break`。实现者请对照原文件把该分支的收尾语句原样保留,仅新增 `on_tokens` 调用与 `_stop` 判断。

在**主路径**的 append 之后(约 line 374-377),把:

```python
        for t in new_tokens:
            produced.append(t)
            if len(produced) >= max_tokens:
                break

        H_last = rH[:, -1:, :]
```

改为:

```python
        for t in new_tokens:
            produced.append(t)
            if len(produced) >= max_tokens:
                break
        _stop = on_tokens is not None and on_tokens(list(new_tokens))

        H_last = rH[:, -1:, :]
```

并在主循环体末尾(约 line 386,`mx.eval(x, H_last)` 与 `if profile: t_finalize...` 之后、`while` 回到条件之前)加入:

```python
        if profile:
            t_finalize += time.perf_counter() - _tic
        if _stop:
            break
```

> `_stop` 在主路径每次迭代都会被赋值(append 之后),故循环末尾读取安全。树路径已在自身分支内 break,不会走到这里。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_mtp_stream_hook.py -v`
Expected: PASS(3 passed)

回归:`cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_mtp_generate.py -q`
Expected: 与改动前一致(全绿)。

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/mtp/generate.py mlx_streaming/tests/test_mtp_stream_hook.py
git commit -m "feat(mtp): mtp_generate 新增可选 on_tokens 钩子(流式+中断,默认行为不变)"
```

---

## Task 2: TUI 后端抽象(banner + backend)

**Files:**
- Create: `mlx_streaming/tui/__init__.py`(本任务先建空包文件,占位)
- Create: `mlx_streaming/tui/banner.py`
- Create: `mlx_streaming/tui/backend.py`
- Test: `mlx_streaming/tests/test_tui_backend.py`

- [ ] **Step 1: 写失败测试**

新建 `mlx_streaming/tests/test_tui_backend.py`:

```python
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
    # 每步都是累计文本,单调加长,最后一步等于完整回答
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_streaming.tui'`

- [ ] **Step 3: 建包与实现**

新建 `mlx_streaming/tui/__init__.py`:

```python
"""vates 全屏 TUI 包。run_tui 延迟 import app(避免无谓加载 textual)。"""


def run_tui(backend, args) -> int:
    """启动全屏 TUI。backend 为 ChatBackend 实现,args 为解析后的命令行参数。"""
    from mlx_streaming.tui.app import VatesApp

    VatesApp(backend, args).run()
    return 0
```

新建 `mlx_streaming/tui/banner.py`:

```python
"""vates 顶栏 logo。保持单行,便于与模型信息拼在同一行。"""

LOGO = "▚ vates"
```

新建 `mlx_streaming/tui/backend.py`:

```python
"""TUI 后端抽象:把界面与 MLX 推理引擎解耦。

界面只依赖 ChatBackend 接口,不 import 任何 MLX 符号,从而能用 FakeBackend 做无模型测试。
load / generate 都是阻塞调用,由 UI 层放到 worker 线程执行。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class GenResult:
    """一轮生成的汇总。"""

    text: str          # 完整回答(已截断 EOS)
    n_tokens: int      # 新生成 token 数
    tok_per_s: float   # 吞吐
    stopped: bool      # 是否被用户中断


class ChatBackend(Protocol):
    """聊天后端接口。所有方法阻塞,调用方负责放 worker 线程。"""

    def load(self, on_status: Callable[[str], None]) -> None:
        """加载模型/权重;通过 on_status(msg) 上报进度。"""
        ...

    def generate(
        self,
        messages: list[dict],
        on_text: Callable[[str], bool],
    ) -> GenResult:
        """跑一轮生成。每步把「累计完整文本」传给 on_text;on_text 返回 True 表示请求中断。"""
        ...


@dataclass
class FakeBackend:
    """测试/演示用假后端:不加载模型,把预设回答按字符流式吐出。"""

    reply: str = "你好，这是一个测试回答。"
    status_msgs: list[str] = field(default_factory=lambda: ["加载中(模拟)…"])
    seen_messages: list[list[dict]] = field(default_factory=list)

    def load(self, on_status: Callable[[str], None]) -> None:
        for m in self.status_msgs:
            on_status(m)

    def generate(self, messages, on_text) -> GenResult:
        self.seen_messages.append([dict(m) for m in messages])
        acc = ""
        for ch in self.reply:
            acc += ch
            if on_text(acc):
                return GenResult(acc, len(acc), 0.0, stopped=True)
        return GenResult(acc, len(self.reply), 0.0, stopped=False)


class MLXBackend:
    """真实后端:封装 _build_engine + mtp_generate。MLX 相关 import 全部延迟到方法内。"""

    def __init__(self, args):
        self.args = args
        self._model = None
        self._tok = None
        self._drafter = None

    def load(self, on_status: Callable[[str], None]) -> None:
        from mlx_streaming.cli import _build_engine

        self._model, self._tok, self._drafter = _build_engine(
            self.args, on_status=on_status)

    def generate(self, messages, on_text) -> GenResult:
        import time

        import mlx.core as mx

        from mlx_streaming.cli import _encode_chat, _eos_set, _truncate_eos
        from mlx_streaming.mtp.generate import mtp_generate

        tok = self._tok
        eos = _eos_set(tok)
        ids = _encode_chat(tok, messages)

        produced_all: list[int] = []
        stopped = {"v": False}

        def on_tokens(new_ids):
            produced_all.extend(new_ids)
            text = tok.decode(_truncate_eos(produced_all, eos))
            if on_text(text):
                stopped["v"] = True
                return True
            return False

        t0 = time.perf_counter()
        produced, _stats = mtp_generate(
            self._model, self._drafter, tok, mx.array([ids]),
            self.args.max_tokens, K=self.args.k, ids_mode=True,
            profile=False, on_tokens=on_tokens)
        dt = time.perf_counter() - t0

        out_ids = _truncate_eos(produced, eos)
        text = tok.decode(out_ids)
        tps = len(out_ids) / dt if dt > 0 else 0.0
        return GenResult(text, len(out_ids), tps, stopped=stopped["v"])
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_backend.py -v`
Expected: PASS(5 passed)

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/tui/__init__.py mlx_streaming/tui/banner.py mlx_streaming/tui/backend.py mlx_streaming/tests/test_tui_backend.py
git commit -m "feat(tui): 后端抽象 ChatBackend + FakeBackend/MLXBackend + banner"
```

---

## Task 3: Textual 应用(VatesApp + 样式)

**Files:**
- Create: `mlx_streaming/tui/app.py`
- Create: `mlx_streaming/tui/styles.tcss`
- Test: `mlx_streaming/tests/test_tui_app.py`

> 前置:`pyproject.toml` 尚未加 textual(Task 4 才加),所以本任务先手动装 textual 以便跑测试:
> `cd mlx-streaming-moe && .venv/bin/pip install "textual>=0.80"`(Task 4 再固化到依赖)。

- [ ] **Step 1: 写失败测试**

新建 `mlx_streaming/tests/test_tui_app.py`:

```python
"""VatesApp 交互测试:用 FakeBackend + Textual run_test,不加载真实模型。"""
import types

import pytest

from mlx_streaming.tui.app import ChatMessage, VatesApp
from mlx_streaming.tui.backend import FakeBackend


def _args(**kw):
    d = dict(model="models/qwen3_next_80b_4bit", k=3, max_tokens=512, system=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


async def _wait_ready(app):
    await app.workers.wait_for_complete()


@pytest.mark.asyncio
async def test_loads_and_becomes_ready():
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await _wait_ready(app)
        await pilot.pause()
        from textual.widgets import Input
        assert app.query_one("#prompt", Input).disabled is False
        assert "就绪" in str(app.query_one("#status").renderable)


@pytest.mark.asyncio
async def test_send_message_streams_and_finalizes():
    backend = FakeBackend(reply="快排是一种分治排序。")
    app = VatesApp(backend, _args())
    async with app.run_test() as pilot:
        await _wait_ready(app)
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
        await _wait_ready(app)
        await pilot.pause()
        from textual.widgets import Input
        inp = app.query_one("#prompt", Input)
        inp.focus()
        inp.value = "第一轮"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        # /reset 清历史
        inp.value = "/reset"
        await pilot.press("enter")
        await pilot.pause()
        # 第二轮:后端收到的 messages 不应再含「第一轮」
        inp.value = "第二轮"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        contents = [m["content"] for m in backend.seen_messages[-1]]
        assert "第一轮" not in contents
        assert "第二轮" in contents


@pytest.mark.asyncio
async def test_interrupt_sets_stop_flag():
    app = VatesApp(FakeBackend(), _args())
    async with app.run_test() as pilot:
        await _wait_ready(app)
        await pilot.pause()
        app._busy = True          # 模拟生成中
        app._stop = False
        app.action_interrupt()
        assert app._stop is True
```

> 说明:测试需要 `pytest-asyncio`。若未安装:`cd mlx-streaming-moe && .venv/bin/pip install pytest-asyncio`,并在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 加 `asyncio_mode = "auto"`(Task 4 一并固化)。本任务先本地装上以便跑测试。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_app.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_streaming.tui.app'`

- [ ] **Step 3: 实现 app 与样式**

新建 `mlx_streaming/tui/styles.tcss`:

```css
Screen {
    background: #0b0e14;
}

#top {
    height: auto;
    padding: 1 2;
    background: #11151f;
    color: #e6e6e6;
    border-bottom: solid #2a2f3a;
}

#chat {
    padding: 1 2;
    height: 1fr;
}

.msg {
    margin: 1 0;
    padding: 0 1;
    height: auto;
}

.msg.assistant {
    border-left: solid #2dd4bf;
    padding-left: 2;
}

#prompt {
    margin: 0 2;
    border: round #2dd4bf;
    background: #11151f;
}

#status {
    height: 1;
    padding: 0 2;
    background: #11151f;
    color: #8b93a7;
}
```

新建 `mlx_streaming/tui/app.py`:

```python
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
        self.text = text
        self.final = False
        self._refresh_content()

    def finalize(self, text: str) -> None:
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
        self.query_one("#prompt", Input).disabled = True
        self._set_status("加载模型中…")
        self._load()

    # ---------- 顶栏/状态栏 ----------
    def _top_text(self) -> Text:
        t = Text()
        t.append(LOGO, style=f"bold {_ACCENT}")
        t.append(f"   {_short_model(self.args.model)} · k={self.args.k} · "
                 f"{self.args.max_tokens} tok", style="dim")
        return t

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(
            f" {_short_model(self.args.model)} · {text}")

    # ---------- 加载 ----------
    @work(thread=True, exclusive=True, group="load")
    def _load(self) -> None:
        def on_status(msg: str):
            self.call_from_thread(self._set_status, f"加载中 · {msg}")

        try:
            self.backend.load(on_status)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._on_load_failed, str(e))
            return
        self.call_from_thread(self._on_load_done)

    def _on_load_done(self) -> None:
        self._set_status("就绪")
        inp = self.query_one("#prompt", Input)
        inp.disabled = False
        inp.focus()

    def _on_load_failed(self, err: str) -> None:
        self._add("assistant", f"模型加载失败:{err}\n\n请检查路径后用 /exit 退出重试。")
        self._set_status("加载失败")

    # ---------- 输入 ----------
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
        elif cmd == "/reset":
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

    # ---------- 生成 ----------
    @work(thread=True, exclusive=True, group="gen")
    def _generate(self, messages: list[dict]) -> None:
        def on_text(full: str) -> bool:
            self.call_from_thread(self._on_stream, full)
            return self._stop

        try:
            result = self.backend.generate(messages, on_text)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._on_error, str(e))
            return
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
        inp = self.query_one("#prompt", Input)
        inp.disabled = False
        inp.focus()

    def _on_error(self, err: str) -> None:
        if self._cur is not None:
            self._cur.finalize(f"生成出错:{err}")
        self._busy = False
        self._cur = None
        self._set_status("就绪(上一轮出错)")
        inp = self.query_one("#prompt", Input)
        inp.disabled = False
        inp.focus()

    def action_interrupt(self) -> None:
        if self._busy:
            self._stop = True
            self._set_status("正在中断…")

    # ---------- 工具 ----------
    def _add(self, role: str, text: str, *, final: bool = True) -> ChatMessage:
        msg = ChatMessage(role, text, final=final)
        self.query_one("#chat", VerticalScroll).mount(msg)
        self._scroll_end()
        return msg

    def _scroll_end(self) -> None:
        self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_app.py -v`
Expected: PASS(4 passed)

> 若 `test_send_message_streams_and_finalizes` 因 worker 时序偶发失败,在断言前加一次 `await app.workers.wait_for_complete(); await pilot.pause()` 再取一次即可(call_from_thread 在 worker 结束前已全部落地,理论上 wait_for_complete 后即稳定)。

- [ ] **Step 5: 提交**

```bash
git add mlx_streaming/tui/app.py mlx_streaming/tui/styles.tcss mlx_streaming/tests/test_tui_app.py
git commit -m "feat(tui): VatesApp 全屏界面(流式渲染/Esc 中断/斜杠命令)+ teal 主题"
```

---

## Task 4: CLI 接线 + 依赖 + README

**Files:**
- Modify: `mlx_streaming/cli.py`(`_build_engine` 约 line 34-69;`cmd_chat` 约 line 114-169;`_add_chat_args` 约 line 172-189)
- Modify: `pyproject.toml`(dependencies 约 line 7-11;`[tool.pytest.ini_options]` 约 line 36-38)
- Modify: `README.md`
- Test: 复用 `mlx_streaming/tests/test_tui_backend.py` / `test_tui_app.py`;新增一个 cli 分流的轻量测试到 `test_tui_app.py` 末尾。

- [ ] **Step 1: 写失败测试(cli 分流)**

在 `mlx_streaming/tests/test_tui_app.py` 末尾追加:

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_app.py::test_cmd_chat_launches_tui_by_default -v`
Expected: FAIL — `cmd_chat` 目前直接进 REPL(会尝试构建引擎),不会调用 `run_tui`。

- [ ] **Step 3: 改 cli.py**

在 `mlx_streaming/cli.py` 顶部 `_build_engine` 内,把两处 `print(..., file=sys.stderr, flush=True)` 改走回调。签名与前两行改为:

```python
def _build_engine(args, on_status=None):
    """按 MTP 快路径装配 model / tokenizer / drafter。

    on_status:可选进度回调;为 None 时进度打到 stderr(保持旧行为)。
    """
    def _emit(msg):
        if on_status is not None:
            on_status(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

    import os
```

然后把函数体内:
- `print("正在加载主模型 + 专家(流式)...", file=sys.stderr, flush=True)` → `_emit("正在加载主模型 + 专家(流式)...")`
- `print("正在加载 MTP drafter...", file=sys.stderr, flush=True)` → `_emit("正在加载 MTP drafter...")`

把现有 `def cmd_chat(args):` 整个函数**改名**为 `def _chat_repl(args):`(函数体一字不改,保持旧 REPL 行为),然后在其上方新增新的 `cmd_chat`:

```python
def cmd_chat(args):
    """默认启动全屏 TUI;--plain 走纯文本 REPL。"""
    if getattr(args, "plain", False):
        return _chat_repl(args)
    from mlx_streaming.tui import run_tui
    from mlx_streaming.tui.backend import MLXBackend
    return run_tui(MLXBackend(args), args)
```

在 `_add_chat_args(p)` 中新增(放在 `--stats` 之后、`p.set_defaults` 之前):

```python
    p.add_argument("--plain", action="store_true",
                   help="用纯文本 REPL,不启动全屏 TUI(终端不兼容/调试时用)")
```

> 注意:`_chat_repl` 仍调用 `_build_engine(args)`(不传 on_status),因此旧 REPL 的 stderr 进度打印行为完全不变。

- [ ] **Step 4: 运行 cli 分流测试确认通过**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_app.py::test_cmd_chat_launches_tui_by_default -v`
Expected: PASS

- [ ] **Step 5: 固化依赖与 pytest 配置**

改 `pyproject.toml` 的 `dependencies`(约 line 7-11),追加一行:

```toml
dependencies = [
    "mlx>=0.31",
    "mlx-lm>=0.31",
    "numpy>=2.0",
    "textual>=0.80",
]
```

在 `[dependency-groups]` 的 `dev` 里加入 `pytest-asyncio`:

```toml
[dependency-groups]
dev = [
    "nanobind>=2.12.0",
    "pytest>=8",
    "pytest-asyncio>=0.23",
]
```

在 `[tool.pytest.ini_options]`(约 line 36-38)加 `asyncio_mode`:

```toml
[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["mlx_streaming/tests"]
asyncio_mode = "auto"
```

同步安装:`cd mlx-streaming-moe && uv sync`(或 `.venv/bin/pip install "textual>=0.80" "pytest-asyncio>=0.23"`)。

- [ ] **Step 6: 更新 README**

在 `README.md` 里描述 `vates chat` 用法处,补充一段:

```markdown
## 交互式对话(TUI)

默认进入全屏 TUI(仿 opencode):

    uv run vates chat

- 回车发送,`Esc` 中断当前生成,`Ctrl+C` 退出。
- 斜杠命令:`/help`、`/reset`(清历史)、`/clear`(清屏)、`/exit`。
- 终端不兼容或需要纯文本时:`uv run vates chat --plain`(走旧版逐行 REPL)。
```

- [ ] **Step 7: 全量回归**

Run: `cd mlx-streaming-moe && .venv/bin/python -m pytest mlx_streaming/tests/test_tui_backend.py mlx_streaming/tests/test_tui_app.py mlx_streaming/tests/test_mtp_stream_hook.py mlx_streaming/tests/test_mtp_generate.py -v`
Expected: 全部 PASS。

- [ ] **Step 8: 提交**

```bash
git add mlx_streaming/cli.py pyproject.toml README.md mlx_streaming/tests/test_tui_app.py
git commit -m "feat(cli): chat 默认启动 TUI(--plain 兜底)+ textual 依赖 + README"
```

---

## 自查(Self-Review 结果)

- **Spec 覆盖**:第 3.1 后端抽象→Task 2;第 3.2 on_tokens 钩子→Task 1;第 3.3 `_build_engine` on_status→Task 4 Step 3;第 4 布局→Task 3(app+tcss);第 5 快捷键/斜杠命令→Task 3;第 6 线程模型→Task 3(`@work(thread=True)`+`call_from_thread`);第 7 主题→Task 3(styles.tcss);第 8 CLI 接线/`--plain`→Task 4;第 9 错误处理→Task 3(`_on_load_failed`/`_on_error`)+Task 1(中断);第 10 测试→各 Task 的测试;第 11 依赖→Task 4;第 12 交付清单全覆盖。
- **占位符扫描**:无 TBD/TODO;每个代码步骤含完整代码。
- **类型/命名一致性**:`ChatBackend.generate(messages, on_text)`、`on_text(full)->bool`、`GenResult(text,n_tokens,tok_per_s,stopped)`、`on_tokens(new_ids)->bool` 在 Task 1/2/3 全程一致;`ChatMessage.stream/finalize/text/role/final`、`VatesApp._busy/_stop/_cur` 在 app 与测试中一致。
- **范围**:聚焦单一实现,无跨子系统。
