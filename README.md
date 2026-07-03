# vates

Apple Silicon 上的显存外置流式 MoE 推理 + Qwen3-Next MTP 自投机解码(基于 MLX)。

把 Qwen3-Next-80B-A3B 的专家权重放在磁盘,按需流式加载 + 预测式预取,配合 MTP
自投机解码,在 32GB 内存的 Mac 上跑起原本装不下的 80B MoE 模型。

## 安装

```bash
uv pip install -e .
```

## 命令行:交互式对话

装好后直接用 `vates` 命令进入多轮对话(默认走 MTP 自投机快路径):

```bash
vates                       # 进入交互式对话
vates -k 4 -n 800 --stats   # 调宽投机 / 加长生成 / 每轮打印吞吐
vates --system "你是一个简洁的助手"
```

未安装入口脚本时,也可用模块方式运行:

```bash
.venv/bin/python -m mlx_streaming.cli
```

常用参数(其余调优项走环境变量,见 `mlx_streaming/config.py`):

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--model` | 主模型路径(4-bit MLX) | `models/qwen3_next_80b_4bit` |
| `--expert-dir` | 拆分后的 per-expert 目录 | `models/qwen3_next_experts_4bit_g64` |
| `--mtp-out` | MTP 权重文件 | `models/qn_mtp_weights.safetensors` |
| `-k, --k` | MTP 投机宽度 | `3` |
| `-n, --max-tokens` | 每轮最多生成 token 数 | `512` |
| `--expert-slots` | 常驻专家池容量(同时作侧区行数默认) | `32` |
| `--system` | 可选 system 提示词 | 无 |
| `--stats` | 每轮打印 token 数 / tok·s / 接受长度 | 关 |

交互期间命令:`/exit` 退出、`/reset` 清空历史、`/help` 帮助。

## 交互式对话(TUI)

默认进入全屏 TUI(仿 opencode):

    uv run vates chat

- 回车发送,`Esc` 中断当前生成,`Ctrl+C` 退出。
- 斜杠命令:`/help`、`/reset`(清历史)、`/clear`(清屏)、`/exit`。
- 终端不兼容或需要纯文本时:`uv run vates chat --plain`(走旧版逐行 REPL)。

## 基准测试

benchmark 脚本(环境变量驱动)见 `mlx_streaming/runtime/`,例如:

```bash
STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 \
  SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 K=3 \
  .venv/bin/python -m mlx_streaming.runtime.run_mtp_spec
```
