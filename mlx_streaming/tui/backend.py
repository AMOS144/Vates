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

    text: str  # 完整回答(已截断 EOS)
    n_tokens: int  # 新生成 token 数
    tok_per_s: float  # 吞吐
    stopped: bool  # 是否被用户中断


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
            self.args, on_status=on_status
        )

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
            truncated = _truncate_eos(produced_all, eos)
            text = tok.decode(truncated)
            if on_text(text):          # 用户按 Esc 请求中断
                stopped["v"] = True
                return True
            # 命中 EOS(截断后短于累计产出):完整回答已生成,提前停止,
            # 避免引擎空跑到 max_tokens 让界面长时间卡在「思考中」。EOS 属正常完成,不算中断。
            if len(truncated) < len(produced_all):
                return True
            return False

        t0 = time.perf_counter()
        produced, _stats = mtp_generate(
            self._model,
            self._drafter,
            tok,
            mx.array([ids]),
            self.args.max_tokens,
            K=self.args.k,
            ids_mode=True,
            profile=False,
            on_tokens=on_tokens,
        )
        dt = time.perf_counter() - t0

        out_ids = _truncate_eos(produced, eos)
        text = tok.decode(out_ids)
        tps = len(out_ids) / dt if dt > 0 else 0.0
        return GenResult(text, len(out_ids), tps, stopped=stopped["v"])
