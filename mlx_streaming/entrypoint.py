"""vates 公开命令入口：默认加载当前固定生产配置。"""

from __future__ import annotations

import sys

from mlx_streaming.runtime.run_qwen_k3_sub10 import _chat_argv, configure


_HELP = """usage: vates [command] [options]

commands:
  prepare   下载并生成约 44 GB 的紧凑运行目录
  chat      启动交互式对话（默认命令）

运行 `vates prepare --help` 或 `vates chat --help` 查看子命令参数。
"""


def _public_chat_argv(argv: list[str]) -> list[str]:
    """将稳定的 vates 用户参数转为固定配置的 chat 参数。"""
    rest = list(argv)
    if rest and rest[0] in ("chat", "--chat"):
        rest = rest[1:]
    normalized = _chat_argv(["chat", *rest])
    if normalized is None:  # pragma: no cover - 前缀在上行固定
        raise RuntimeError("无法构造 vates chat 参数")
    return normalized


def main(argv: list[str] | None = None) -> int:
    public_argv = list(sys.argv[1:] if argv is None else argv)
    if public_argv in (["-h"], ["--help"]):
        print(_HELP, end="")
        return 0
    # prepare 是独立的离线流程，不能先导入/安装聊天运行时配置。
    if public_argv and public_argv[0] == "prepare":
        from mlx_streaming.prep.runtime_bundle import main as prepare

        return int(prepare(public_argv[1:]) or 0)
    # 多个运行时模块在 import 时读取配置，必须先安装固定配置。
    configure()
    try:
        chat_argv = _public_chat_argv(public_argv)
    except ValueError as error:
        print(f"vates: error: {error}", file=sys.stderr)
        return 2
    from mlx_streaming.cli import main as chat

    return int(chat(chat_argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
