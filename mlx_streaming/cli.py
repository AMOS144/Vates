"""vates 命令行:面向用户的交互式多轮对话(MTP 自投机快路径)。

用法示例:
    vates                       # 直接进入交互式对话(默认子命令 chat)
    vates chat                  # 同上
    vates -k 4 -n 800 --stats   # 调宽投机、加长生成、每轮打印吞吐
    vates --system "你是一个简洁的助手"
    vates --model models/qwen3_next_80b_4bit --expert-slots 32

只做「生成」一件事:走 MTP 自投机 + 零拷贝双源侧区快路径。关键参数做成命令行 flag,
其余调优项仍从环境变量读取(见 mlx_streaming/config.py)。

交互期间可用命令:
    /exit 或 /quit   退出
    /reset           清空对话历史(保留 system)
    /help            打印帮助
"""
import argparse
import sys
import time

from mlx_streaming import config

# MTP 快路径环境变量兜底配方(benchmark 验证过的最优组合)。
# 用 setdefault 兜底:用户显式导出的环境变量优先级更高,不会被覆盖。
_FASTPATH_ENV = {
    "STREAM_BLOB_LOADER": "1",
    "NATIVE_FUSED_PREFETCH": "1",
    "ZEROCOPY_DUAL_SOURCE": "1",
    "SIDEREGION_LFU": "1",
}


def _build_engine(args, on_status=None):
    """按 MTP 快路径装配 model / tokenizer / drafter。

    注意:model_builder 在 import 时就读 MODEL/EXPERT_DIR 等环境变量,所以必须
    先把命令行参数写进 os.environ,再 import build_streaming_model。

    on_status:可选进度回调;为 None 时进度打到 stderr(保持旧行为)。
    """
    def _emit(msg):
        if on_status is not None:
            on_status(msg)
        else:
            print(msg, file=sys.stderr, flush=True)

    import os

    os.environ["MODEL"] = args.model
    os.environ["QN_CONFIG"] = args.qn_config
    os.environ["MTP_OUT"] = args.mtp_out
    os.environ["EXPERT_DIR"] = args.expert_dir
    os.environ["EXPERT_SLOTS"] = str(args.expert_slots)
    spec = args.spec_slots if args.spec_slots is not None else args.expert_slots
    os.environ["POOL_SPEC_SLOTS"] = str(spec)
    for k, v in _FASTPATH_ENV.items():
        os.environ.setdefault(k, v)

    import json

    import mlx.core as mx  # noqa: F401  确保 MLX 已就绪
    from mlx_lm.models.qwen3_next import ModelArgs

    from mlx_streaming.model_builder import build_streaming_model
    from mlx_streaming.mtp.drafter import MTPDrafter
    from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

    _emit("正在加载主模型 + 专家(流式)...")
    model, tok, _store = build_streaming_model()
    _emit("正在加载 MTP drafter...")
    with open(args.qn_config) as f:
        margs = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(margs, args.mtp_out, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens          # 共享主模型 embedding
    drafter = MTPDrafter(mtp, model.lm_head)
    return model, tok, drafter


def _encode_chat(tok, messages):
    """把多轮对话按聊天模板编码成 token id 列表。"""
    tmpl = getattr(tok, "chat_template", None)
    if tmpl:
        out = tok.apply_chat_template(messages, add_generation_prompt=True)
        # mlx_lm 通常直接返回 list[int];老版本可能返回字符串。
        return tok.encode(out) if isinstance(out, str) else list(out)
    # 无聊天模板:朴素拼接兜底。
    text = ""
    for m in messages:
        text += f"{m['role']}: {m['content']}\n"
    text += "assistant: "
    return tok.encode(text)


def _eos_set(tok):
    """收集所有可能的结束符 token id。"""
    eos = set()
    ids = getattr(tok, "eos_token_ids", None)
    if ids:
        eos |= set(ids)
    one = getattr(tok, "eos_token_id", None)
    if one is not None:
        eos.add(one)
    return eos


def _truncate_eos(produced, eos):
    """遇到第一个结束符即截断(不含结束符本身)。"""
    for i, t in enumerate(produced):
        if t in eos:
            return produced[:i]
    return produced


_HELP = """可用命令:
  /exit, /quit   退出
  /reset         清空对话历史(保留 system)
  /help          显示本帮助
直接输入文本即可对话。"""


def cmd_chat(args):
    """默认启动全屏 TUI;--plain 走纯文本 REPL。"""
    if getattr(args, "plain", False):
        return _chat_repl(args)
    from mlx_streaming.tui import run_tui
    from mlx_streaming.tui.backend import MLXBackend
    return run_tui(MLXBackend(args), args)


def _chat_repl(args):
    model, tok, drafter = _build_engine(args)

    import mlx.core as mx

    from mlx_streaming.mtp.generate import mtp_generate

    eos = _eos_set(tok)
    base_messages = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    messages = list(base_messages)

    print("\n模型已就绪。首轮生成会因编译 Metal kernel + 补专家池而明显偏慢,属正常现象。",
          file=sys.stderr, flush=True)
    print("输入 /help 查看命令,/exit 退出。", file=sys.stderr, flush=True)

    while True:
        try:
            user = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            messages = list(base_messages)
            print("对话历史已清空。", file=sys.stderr)
            continue
        if user == "/help":
            print(_HELP, file=sys.stderr)
            continue

        messages.append({"role": "user", "content": user})
        ids = _encode_chat(tok, messages)

        t0 = time.perf_counter()
        produced, stats = mtp_generate(
            model, drafter, tok, mx.array([ids]),
            args.max_tokens, K=args.k, ids_mode=True, profile=args.stats)
        dt = time.perf_counter() - t0

        out_ids = _truncate_eos(produced, eos)
        text = tok.decode(out_ids)
        print(f"\n助手 > {text}")
        messages.append({"role": "assistant", "content": text})

        if args.stats:
            tps = len(out_ids) / dt if dt > 0 else 0.0
            print(f"[{len(out_ids)} tok, {tps:.1f} tok/s, "
                  f"accept_len={stats.get('avg_accept_len')}]", file=sys.stderr)

    print("再见。", file=sys.stderr)
    return 0


def _add_chat_args(p):
    p.add_argument("--model", default=config.model_path(), help="主模型路径(4-bit MLX)")
    p.add_argument("--expert-dir", default=config.expert_dir(),
                   help="拆分后的 per-expert 目录")
    p.add_argument("--mtp-out", default=config.mtp_out(), help="MTP 权重文件")
    p.add_argument("--qn-config", default=config.qn_config(),
                   help="Qwen3-Next 配置 JSON")
    p.add_argument("-k", "--k", type=int, default=3, help="MTP 投机宽度(默认 3)")
    p.add_argument("-n", "--max-tokens", type=int, default=512,
                   help="每轮最多生成的新 token 数(默认 512)")
    p.add_argument("--expert-slots", type=int, default=32,
                   help="常驻专家池容量(默认 32,同时作为侧区行数默认)")
    p.add_argument("--spec-slots", type=int, default=None,
                   help="侧区行数 POOL_SPEC_SLOTS(默认跟随 --expert-slots)")
    p.add_argument("--system", default=None, help="可选 system 提示词")
    p.add_argument("--stats", action="store_true",
                   help="每轮结束在 stderr 打印 token 数 / tok·s / 接受长度")
    p.add_argument("--plain", action="store_true",
                   help="用纯文本 REPL,不启动全屏 TUI(终端不兼容/调试时用)")
    p.set_defaults(func=cmd_chat)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="vates",
        description="vates:Apple Silicon 上的流式 MoE + Qwen3-Next MTP 自投机推理")
    sub = parser.add_subparsers(dest="cmd")
    chat = sub.add_parser("chat", help="进入交互式多轮对话(MTP 自投机快路径)")
    _add_chat_args(chat)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    subcmds = {"chat"}
    # 让 chat 成为默认子命令:不带子命令(或首参是 flag)时自动补上 chat;
    # 但保留顶层 -h/--help 直接显示总帮助。
    if not argv or (argv[0] not in subcmds and argv[0] not in ("-h", "--help")):
        argv = ["chat"] + argv
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
