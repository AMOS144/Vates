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
    prefill_tokens: int = 0  # 本轮实际 prefill token 数（扣除已复用前缀）
    prefill_s: float = 0.0
    prefill_tok_per_s: float = 0.0


class ChatBackend(Protocol):
    """聊天后端接口。所有方法阻塞,调用方负责放 worker 线程。"""

    def load(self, on_status: Callable[[str], None]) -> None:
        """加载模型/权重;通过 on_status(msg) 上报进度。"""
        ...

    def generate(
        self,
        messages: list[dict],
        on_text: Callable[[str, int], bool],
        on_prefill: Callable[[int, float, float], None] | None = None,
    ) -> GenResult:
        """跑一轮生成。on_prefill 在 prompt ingestion 完成时上报 token/耗时/速度。"""
        ...


@dataclass
class FakeBackend:
    """测试/演示用假后端:不加载模型,把预设回答按字符流式吐出。

    delay > 0 时每字符间 sleep,用于 --demo 模式模拟真实吐字节奏;测试默认 0(不拖慢)。
    """

    reply: str = "你好，这是一个测试回答。"
    status_msgs: list[str] = field(default_factory=lambda: ["加载中(模拟)…"])
    delay: float = 0.0
    seen_messages: list[list[dict]] = field(default_factory=list)

    def load(self, on_status: Callable[[str], None]) -> None:
        for m in self.status_msgs:
            on_status(m)

    def generate(self, messages, on_text, on_prefill=None) -> GenResult:
        import time

        self.seen_messages.append([dict(m) for m in messages])
        acc = ""
        for ch in self.reply:
            acc += ch
            if self.delay:
                time.sleep(self.delay)
            # 用字符数近似 token 数(假后端无真实分词)
            if on_text(acc, len(acc)):
                return GenResult(acc, len(acc), 0.0, stopped=True)
        return GenResult(acc, len(self.reply), 0.0, stopped=False)


def _common_prefix_len(a, b) -> int:
    """返回两个 token id 序列的最长公共前缀长度。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _reuse_prefix_len(cached_ids, new_ids) -> int:
    """可跨轮复用的前缀长度:仅当旧 cache 的 token 是新序列的**严格前缀**且新序列更长时,
    返回该前缀长度(= len(cached_ids));否则返回 0 表示需全量重建。

    只在严格前缀时复用,是为了永远「只延续、不回退」cache——Qwen3-Next 的线性注意力递归态
    无法裁剪回任意历史位置;而 detokenize→retokenize 不一致、/reset、编辑历史等都会让公共前缀
    短于旧长度,此时回退整段重建,绝不基于错位的 cache 续算。
    """
    if not cached_ids or len(new_ids) <= len(cached_ids):
        return 0
    c = _common_prefix_len(cached_ids, new_ids)
    return c if c == len(cached_ids) else 0


class MLXBackend:
    """真实后端:封装 _build_engine + mtp_generate。MLX 相关 import 全部延迟到方法内。"""

    def __init__(self, args):
        self.args = args
        self._model = None
        self._tok = None
        self._drafter = None
        # 跨轮复用:持久化上一轮的 main_cache 及其对应的 token 序列(prompt + 已入 cache 的生成 token)。
        self._main_cache = None
        self._cached_ids: list[int] = []

    def load(self, on_status: Callable[[str], None]) -> None:
        from mlx_streaming.cli import _build_engine, _warmup

        self._model, self._tok, self._drafter = _build_engine(
            self.args, on_status=on_status
        )
        # 预热:把首轮的 kernel 编译 + 专家池填充开销移到加载阶段,避免第一条消息莫名卡很久。
        on_status("预热中(编译 kernel + 填专家池)…")
        _warmup(self._model, self._tok, self._drafter, self.args)

    def generate(self, messages, on_text, on_prefill=None) -> GenResult:
        import os
        import time

        import mlx.core as mx

        from mlx_streaming.cli import _encode_chat, _eos_set, _truncate_eos
        from mlx_streaming.mtp.generate import mtp_generate

        tok = self._tok
        eos = _eos_set(tok)
        ids = _encode_chat(tok, messages)

        # 跨轮复用 KV/递归态:旧 cache 是本轮 prompt 的严格前缀时,只 prefill 新增后缀,
        # 不重算整段历史(prefill 从 ∝历史长度 降到 ∝新消息长度)。否则全量重建。
        cached_len = (_reuse_prefix_len(self._cached_ids, ids)
                      if self._main_cache is not None else 0)
        main_cache = self._main_cache if cached_len else self._model.make_cache()

        produced_all: list[int] = []
        stopped = {"v": False}

        def on_tokens(new_ids):
            produced_all.extend(new_ids)
            truncated = _truncate_eos(produced_all, eos)
            text = tok.decode(truncated)
            if on_text(text, len(truncated)):   # 用户按 Esc 请求中断
                stopped["v"] = True
                return True
            # 命中 EOS(截断后短于累计产出):完整回答已生成,提前停止,
            # 避免引擎空跑到 max_tokens 让界面长时间卡在「思考中」。EOS 属正常完成,不算中断。
            if len(truncated) < len(produced_all):
                return True
            return False

        # The throughput profile keeps wide prompt ingestion synchronous, then
        # enables overlapped demand exactly at the prefill/decode boundary.
        # Keep the TUI on the same path as the plain CLI and benchmark runner;
        # otherwise DEMAND_ASYNC remains 0 for the whole decode and throughput
        # falls back to the synchronous ~24 tok/s path.
        split_demand = os.environ.get(
            "SPEC_SPLIT_DEMAND_AFTER_PREFILL",
        ) == "1"
        saved_demand_async = os.environ.get("DEMAND_ASYNC")
        if split_demand:
            os.environ["DEMAND_ASYNC"] = "0"

        prefill_tokens = max(0, len(ids) - cached_len)
        prefill_s = 0.0

        def on_prefill_complete():
            nonlocal prefill_s
            prefill_s = max(0.0, time.perf_counter() - t0)
            if split_demand:
                os.environ["DEMAND_ASYNC"] = "1"
            if on_prefill is not None:
                prefill_tps = (
                    prefill_tokens / prefill_s if prefill_s > 0 else 0.0
                )
                on_prefill(prefill_tokens, prefill_s, prefill_tps)

        t0 = time.perf_counter()
        try:
            generate_kwargs = {}
            if split_demand:
                generate_kwargs["on_prefill_complete"] = on_prefill_complete
            produced, stats = mtp_generate(
                self._model,
                self._drafter,
                tok,
                mx.array([ids]),
                self.args.max_tokens,
                K=self.args.k,
                ids_mode=True,
                profile=False,
                on_tokens=on_tokens,
                main_cache=main_cache,
                cached_len=cached_len,
                **generate_kwargs,
            )
        finally:
            if split_demand:
                if saved_demand_async is None:
                    os.environ.pop("DEMAND_ASYNC", None)
                else:
                    os.environ["DEMAND_ASYNC"] = saved_demand_async
        dt = time.perf_counter() - t0

        # 持久化本轮 cache 供下轮复用。正常情况下 main_cache 恰好持有 `ids + produced[:-1]`
        # (produced[-1] 为 pending 未入 cache)。但末步多 token 跨 max_tokens 会 over-commit:
        # cache 领先于 produced,无法用已知 token 精确表述——此时禁用复用,下轮全量重建,绝不错算。
        resident = stats.get("resident_tokens")
        expected = len(ids) + len(produced) - 1
        if resident == expected:
            self._main_cache = main_cache
            self._cached_ids = list(ids) + list(produced[:-1])
        else:
            self._main_cache = None
            self._cached_ids = []

        out_ids = _truncate_eos(produced, eos)
        text = tok.decode(out_ids)
        # mtp_generate.wall_s 从 prefill/decode 边界开始，是引擎权威的
        # decode-only 时钟。dt 包含 prompt ingestion，不能用来标 decode 吞吐。
        decode_s = float(stats.get("wall_s") or 0.0)
        if decode_s <= 0:
            decode_s = max(0.0, dt - prefill_s)
        tps = len(out_ids) / decode_s if decode_s > 0 else 0.0
        prefill_tps = prefill_tokens / prefill_s if prefill_s > 0 else 0.0
        return GenResult(
            text,
            len(out_ids),
            tps,
            stopped=stopped["v"],
            prefill_tokens=prefill_tokens,
            prefill_s=prefill_s,
            prefill_tok_per_s=prefill_tps,
        )
