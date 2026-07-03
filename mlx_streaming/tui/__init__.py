"""vates 全屏 TUI 包。run_tui 延迟 import app(避免无谓加载 textual)。"""


def run_tui(backend, args) -> int:
    """启动全屏 TUI。backend 为 ChatBackend 实现,args 为解析后的命令行参数。"""
    from mlx_streaming.tui.app import VatesApp

    VatesApp(backend, args).run()
    return 0
