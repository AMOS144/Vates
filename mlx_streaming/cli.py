"""vates 命令行:面向用户的交互式多轮对话(MTP 自投机快路径)。

用法示例:
    vates                       # 直接进入交互式对话(默认子命令 chat)
    vates chat                  # 同上
    vates -n 800 --stats        # 加长生成、每轮分别打印 prefill/decode 吞吐
    vates --system "你是一个简洁的助手"
    vates --model models/qwen3_next_80b_4bit

只做「生成」一件事。公开 vates 入口会在导入本模块前安装固定生产配置；
模型路径、生成长度和交互方式仍作为命令行参数。

交互期间可用命令:
    /exit 或 /quit   退出
    /reset           清空对话历史(保留 system)
    /help            打印帮助
"""
import argparse
import os
import sys
import time

from mlx_streaming import config

# MTP 快路径环境变量兜底配方(benchmark 验证过的最优组合)。
# 用 setdefault 兜底:用户显式导出的环境变量优先级更高,不会被覆盖。
# MTP_ADAPTIVE_DEPTH:置信度门控动态深度。逐位累计置信度跌破 tau 即停,低置信步抽浅省专家加载
# (本系统 IO 瓶颈)。消融(reports/adaptive-depth-2026-07-05)证 τ=0.3、depth_max=3 纯向下收缩
# +5~6% tok/s 且 bit-lossless、零额外显存,稳定优于最小树 pos0 救回 → 设为用户主路径默认。
# depth_max=3 与基础 K 一致,在生产 EXPERT_SLOTS=32 下 seq·top_k 不溢出 cap(扩到 4 须 slots>=40)。
# 注:动态深度与 TREE_TOP2 互斥(adaptive 仅在非 tree 的 plain 路径生效),故此处不开 TREE_TOP2。
# KV_QUANT:IsoQuant K4/V3 + SO(4) 块旋转,仅作用于 12 个全注意力层(线性层递归态不动)。
# 极致压缩长上下文 KV:128k 3.0→~0.68 GiB;短会话收益小但无害。K4/V3/旋转均取默认值,
# 开 KV_QUANT=1 即整套生效。质量验收:token 一致率≥95% + logits cosine≥0.99。
_FASTPATH_ENV = {
    "STREAM_BLOB_LOADER": "1",
    "NATIVE_FUSED_PREFETCH": "1",
    "ZEROCOPY_DUAL_SOURCE": "1",
    "SIDEREGION_LFU": "1",
    "KV_QUANT": "1",
    "MTP_ADAPTIVE_DEPTH": "1",
    "MTP_CONF_TAU": "0.3",
    "MTP_DEPTH_MAX": "3",
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

    from mlx_streaming.core.mem import setup_memory_hygiene
    from mlx_streaming.model_builder import build_streaming_model
    from mlx_streaming.mtp.drafter import MTPDrafter
    from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

    # 长会话内存防御:封顶 MLX 可回收缓冲(默认 1GB),防长对话里缓冲缓存膨胀把常驻推过墙 /
    # 触发 macOS 压缩器抖动。TUI 是典型长会话场景,故在用户主路径启动时就设上。
    _applied = setup_memory_hygiene(cache_gb=config.mlx_cache_limit_gb(),
                                    wired_gb=config.mlx_wired_limit_gb())
    if _applied:
        _emit(f"内存防御: {_applied}")

    _emit("正在加载主模型 + 专家(流式)...")
    model, tok, _store = build_streaming_model()
    _emit("正在加载 MTP drafter...")
    with open(args.qn_config) as f:
        margs = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(
        margs, args.mtp_out, quantize=True, bits=config.mtp_bits(),
        group_size=config.mtp_group_size(),
        stream_experts=config.mtp_stream_experts(),
        expert_dir=config.mtp_expert_dir(),
        expert_slots=config.mtp_expert_slots(),
    )
    mtp.embed_tokens = model.model.embed_tokens          # 共享主模型 embedding
    drafter = MTPDrafter(mtp, model.lm_head)
    return model, tok, drafter


def _warmup(model, tok, drafter, args):
    """跑一次生成做预热:首轮的明显卡顿主要来自现编译 Metal kernel + 填 MoE 专家 resident 池,
    提前把这部分一次性开销移到加载阶段。

    覆盖增强:用一段**较长、token id 跨大跨度词表分散**的合成 prompt——MoE 路由依赖 token 内容,
    分散的 id 会命中更多专家、更充分地预填专家池;较长 prompt 又能走通多块分块 prefill。
    合成 id 直接走 ids_mode(不依赖 tokenizer),分块 prefill(chunk=2)保证长 prompt 也不抬高显存峰值。
    预热失败不致命(直接吞掉异常),不影响后续真实生成。
    """
    import mlx.core as mx

    from mlx_streaming.mtp.generate import mtp_generate

    try:
        vocab = int(model.model.embed_tokens.weight.shape[0])
        n = min(64, vocab)                       # 预热 prompt 长度(兼顾覆盖与耗时)
        step = max(1, vocab // n)                 # 在词表内均匀取样,最大化专家覆盖
        ids = [(1 + i * step) % vocab for i in range(n)]
        # The measured 37 tok/s profile used a 64-token warmup.  Eight tokens
        # compile kernels but do not stabilize the expert pool, and previously
        # ran entirely with synchronous demand even though interactive decode
        # switches to async demand at the prefill boundary.
        warmup_tokens = max(1, int(os.environ.get("CLI_WARMUP_TOK", "64")))
        split_demand = os.environ.get(
            "SPEC_SPLIT_DEMAND_AFTER_PREFILL",
        ) == "1"
        saved_demand_async = os.environ.get("DEMAND_ASYNC")
        if split_demand:
            os.environ["DEMAND_ASYNC"] = "0"

        def on_prefill_complete():
            if split_demand:
                os.environ["DEMAND_ASYNC"] = "1"

        try:
            mtp_generate(
                model, drafter, tok, mx.array([ids]), warmup_tokens,
                K=args.k, ids_mode=True,
                on_prefill_complete=on_prefill_complete,
            )
        finally:
            if split_demand:
                if saved_demand_async is None:
                    os.environ.pop("DEMAND_ASYNC", None)
                else:
                    os.environ["DEMAND_ASYNC"] = saved_demand_async
    except Exception:  # noqa: BLE001  预热仅为压首轮延迟,失败不应中断启动
        pass


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
    """默认启动全屏 TUI;--plain 走纯文本 REPL;--demo 用假后端免模型预览界面。"""
    if getattr(args, "plain", False):
        return _chat_repl(args)
    from mlx_streaming.tui import run_tui
    if getattr(args, "demo", False):
        # 免模型预览:秒开 TUI,假流式回答,用于验证界面/占位符/状态栏
        from mlx_streaming.tui.backend import FakeBackend
        demo = FakeBackend(
            reply="这是 --demo 演示回答:界面、逐字流式、状态栏(token 数 / tok·s)"
                  "均为模拟,不加载模型。按 Esc 可中断,/help 看命令。",
            delay=0.03)
        return run_tui(demo, args)
    from mlx_streaming.tui.backend import MLXBackend
    return run_tui(MLXBackend(args), args)


def _chat_repl(args):
    model, tok, drafter = _build_engine(args)

    import mlx.core as mx

    from mlx_streaming.mtp.generate import mtp_generate
    from mlx_streaming.tui.backend import _reuse_prefix_len

    eos = _eos_set(tok)
    base_messages = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})
    messages = list(base_messages)
    # 跨轮复用:持久化上一轮 main_cache 及其对应的 token 序列(与 MLXBackend 同机制)。
    main_cache = None
    cached_ids: list[int] = []

    print("\n正在预热(编译 kernel + 填专家池)…", file=sys.stderr, flush=True)
    _warmup(model, tok, drafter, args)
    print("模型已就绪。输入 /help 查看命令,/exit 退出。", file=sys.stderr, flush=True)

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
            main_cache, cached_ids = None, []       # 清历史同时弃用旧 cache
            print("对话历史已清空。", file=sys.stderr)
            continue
        if user == "/help":
            print(_HELP, file=sys.stderr)
            continue

        messages.append({"role": "user", "content": user})
        ids = _encode_chat(tok, messages)

        # 旧 cache 是本轮 prompt 的严格前缀时只 prefill 新增后缀,否则全量重建。
        cached_len = (_reuse_prefix_len(cached_ids, ids)
                      if main_cache is not None else 0)
        cur_cache = main_cache if cached_len else model.make_cache()

        # EOS 提前停止:否则会空跑到 max_tokens,且 produced 会含 EOS 后垃圾 token,
        # 使 cached_ids 与下轮编码前缀断裂、复用永不触发。
        produced_all: list[int] = []

        def _on_tokens(new_ids):
            produced_all.extend(new_ids)
            truncated = _truncate_eos(produced_all, eos)
            return len(truncated) < len(produced_all)

        t0 = time.perf_counter()
        split_demand = os.environ.get(
            "SPEC_SPLIT_DEMAND_AFTER_PREFILL",
        ) == "1"
        saved_demand_async = os.environ.get("DEMAND_ASYNC")
        if split_demand:
            os.environ["DEMAND_ASYNC"] = "0"

        def _on_prefill_complete():
            if split_demand:
                os.environ["DEMAND_ASYNC"] = "1"
            if args.stats and split_demand:
                from mlx_streaming import native_moe_ext as native
                native.demand_async_stats_reset()

        try:
            produced, stats = mtp_generate(
                model, drafter, tok, mx.array([ids]),
                args.max_tokens, K=args.k, ids_mode=True, profile=args.stats,
                on_tokens=_on_tokens,
                on_prefill_complete=_on_prefill_complete,
                main_cache=cur_cache, cached_len=cached_len)
        finally:
            if split_demand:
                if saved_demand_async is None:
                    os.environ.pop("DEMAND_ASYNC", None)
                else:
                    os.environ["DEMAND_ASYNC"] = saved_demand_async
        dt = time.perf_counter() - t0

        # offset 对账:仅无 over-commit 时记录 cache 供下轮复用(见 MLXBackend.generate)。
        if stats.get("resident_tokens") == len(ids) + len(produced) - 1:
            main_cache = cur_cache
            cached_ids = list(ids) + list(produced[:-1])
        else:
            main_cache, cached_ids = None, []

        out_ids = _truncate_eos(produced, eos)
        text = tok.decode(out_ids)
        print(f"\n助手 > {text}")
        messages.append({"role": "assistant", "content": text})

        if args.stats:
            # ``dt`` starts before prompt prefill, while mtp_generate's
            # ``wall_s`` starts at the exact prefill/decode boundary.  A
            # single undifferentiated tok/s number made short replies look
            # much slower than long replies and was not comparable with the
            # decode-only benchmark.  Report both clocks explicitly.
            e2e_tps = len(out_ids) / dt if dt > 0 else 0.0
            decode_s = float(stats.get("wall_s") or 0.0)
            decode_tps = len(out_ids) / decode_s if decode_s > 0 else 0.0
            prefill_s = max(0.0, dt - decode_s)
            print(
                f"[{len(out_ids)} tok, decode={decode_tps:.1f} tok/s, "
                f"e2e={e2e_tps:.1f} tok/s, prefill={prefill_s:.2f}s, "
                f"accept_len={stats.get('avg_accept_len')}]",
                file=sys.stderr,
            )
            if split_demand:
                from mlx_streaming import native_moe_ext as native
                values = list(native.demand_async_stats())
                if len(values) >= 10:
                    positions = max(int(values[5]), 1)
                    waits_ms = (int(values[8]) + int(values[9])) / 1000.0
                    print(
                        f"[entry_hit={int(values[4]) / positions:.2%}, "
                        f"disk_loads={int(values[3])}, "
                        f"ssd_wait={waits_ms:.1f}ms]",
                        file=sys.stderr,
                    )
                    layer_values = list(native.demand_async_layer_stats())
                    layer_rows = []
                    for layer in range(len(layer_values) // 7):
                        row = layer_values[layer * 7:layer * 7 + 7]
                        calls, _, loads, misses, _, fallback_us, pending_us = row
                        wait_ms = (int(fallback_us) + int(pending_us)) / 1000.0
                        if calls and (wait_ms > 0.0 or misses or loads):
                            layer_rows.append(
                                (wait_ms, layer, int(misses), int(loads)),
                            )
                    layer_rows.sort(reverse=True)
                    slow = ", ".join(
                        f"L{layer}:{wait_ms:.1f}ms/"
                        f"miss={misses}/load={loads}"
                        for wait_ms, layer, misses, loads in layer_rows[:8]
                    )
                    if slow:
                        print(f"[slow_layers={slow}]", file=sys.stderr)

    print("再见。", file=sys.stderr)
    return 0


def _add_chat_args(p):
    p.add_argument("--model", default=config.model_path(), help="主模型路径(4-bit MLX)")
    p.add_argument("--expert-dir", default=config.expert_dir(),
                   help="拆分后的 per-expert 目录")
    p.add_argument("--mtp-out", default=config.mtp_out(), help="MTP 权重文件")
    p.add_argument("--qn-config", default=config.qn_config(),
                   help="Qwen3-Next 配置 JSON")
    p.add_argument("-k", "--k", type=int, default=3,
                   help="MTP 投机宽度(公开 vates 固定为 3)")
    p.add_argument("-n", "--max-tokens", type=int, default=4096,
                   help="每轮最多生成的新 token 数(默认 4096)")
    p.add_argument("--expert-slots", type=int, default=32,
                   help="主模型专家池容量(公开 vates 固定为 152)")
    p.add_argument("--spec-slots", type=int, default=None,
                   help="旧侧区行数(公开 vates 固定为 0)")
    p.add_argument("--system", default=None, help="可选 system 提示词")
    p.add_argument("--stats", action="store_true",
                   help="每轮结束在 stderr 打印 token 数 / tok·s / 接受长度")
    p.add_argument("--plain", action="store_true",
                   help="用纯文本 REPL,不启动全屏 TUI(终端不兼容/调试时用)")
    p.add_argument("--demo", action="store_true",
                   help="免模型预览:用假后端秒开 TUI,验证界面/流式/状态栏")
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
