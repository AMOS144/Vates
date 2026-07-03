# vates 全屏 TUI(仿 opencode 风格)设计

日期：2026-07-03
状态：设计草案(已与用户对齐方向:全屏 TUI + 流式输出 + teal 强调色 + 保留 --plain 兜底 + Esc 中断)
关联：`mlx_streaming/cli.py`(现有简单 REPL)、`mlx_streaming/mtp/generate.py`(生成循环)

---

## 0. 一句话

把现在「一问一答、纯 `print`」的 `vates chat` 命令行,升级为一个**用 Textual 写的全屏交互式 TUI**:
顶栏 logo + 模型信息、可滚动对话区(用户/助手气泡)、带边框输入框、底部状态栏,
助手回答**逐字流式渲染**、完成后按 Markdown 排版,支持 `Esc` 中断生成。视觉与交互对标 opencode。

为把风险压到最低,**界面与推理引擎彻底解耦**(界面只依赖一个抽象后端),对核心生成代码仅做
**一处向后兼容的钩子改动**,并保留 `--plain` 走旧 REPL 作为兜底。

---

## 1. 背景与问题

现状(`mlx_streaming/cli.py` 的 `cmd_chat`):

- 交互形态是 `input("你 > ")` + `print("助手 > ...")` 的裸终端循环,样式单一。
- 助手回答是**整轮生成完才一次性 `print`**,生成期间用户看不到任何进度(首轮还很慢)。
- 模型加载进度用 `print(..., file=sys.stderr)` 直接打屏。
- 没有中断、没有 Markdown 排版、没有状态信息(tok/s、token 数)可视化。

目标:做出接近 opencode 的观感与交互,同时不破坏现有生成路径的数值行为与 `--plain` 可用性。

---

## 2. 非目标(YAGNI)

- 不做多标签/多会话管理、不做会话持久化到磁盘。
- 不做主题切换 UI(配色集中在一个 tcss 文件里,改色靠改文件,不做运行时切换器)。
- 不改动 MTP 生成算法本身(接受率、投机逻辑一律不动)。
- 不做鼠标复杂交互;以键盘为主。

---

## 3. 架构

新增 `mlx_streaming/tui/` 包,职责单一、可独立测试:

```
mlx_streaming/
  tui/
    __init__.py     # 导出 run_tui(args)
    backend.py      # ChatBackend 抽象接口 + MLXBackend 实现 + FakeBackend(测试用)
    app.py          # VatesApp:Textual 应用,布局/事件/worker 线程/渲染
    banner.py       # vates ASCII logo 文本
    styles.tcss     # Textual CSS,集中定义配色/边框/间距(teal 主题)
  cli.py            # chat 子命令:默认启动 TUI;--plain 走旧 REPL
  mtp/generate.py   # 仅新增可选钩子 on_tokens(向后兼容)
```

### 3.1 后端抽象(解耦关键)

`backend.py` 定义一个窄接口,让 UI 完全不 import MLX:

```python
class ChatBackend(Protocol):
    def load(self, on_status: Callable[[str], None]) -> None:
        """加载模型/权重;通过 on_status(msg) 上报进度。阻塞、在 worker 线程调用。"""

    def generate(
        self,
        messages: list[dict],           # [{"role","content"}, ...]
        on_token: Callable[[str], bool],# 收到新增文本片段;返回 True 表示请求中断
    ) -> GenResult:                     # GenResult(text, n_tokens, tok_per_s, accept_len)
        """跑一轮生成。阻塞、在 worker 线程调用。"""
```

- `MLXBackend`:封装现有 `_build_engine` / `mtp_generate`。它负责:
  - `load()`:调用重构后的 `_build_engine(args, on_status=...)`。
  - `generate()`:编码消息 → 调 `mtp_generate(..., on_tokens=cb)`,内部把新增 token 增量解码成文本片段回调给 UI,并统计 tok/s。
- `FakeBackend`:`load()` 空转、`generate()` 把一段预设文本按字符/词分片回调,用于测试与 `--plain` 无关的 UI 冒烟。

> 增量解码策略:维护累计 `produced` token 列表,每步用 `tok.decode(produced)` 得到完整文本,
> 与上一次已发出的文本做**前缀 diff**,只把新增后缀通过 `on_token` 发出。对话长度(几百 token)下
> O(n) 解码开销可忽略,且对 BPE/多字节字符稳健(避免半个字符乱码)。

### 3.2 对 `mtp/generate.py` 的改动(最小、向后兼容)

新增**一个**可选参数 `on_tokens=None`。默认 `None` 时行为与现在**逐字节一致**:

```python
def mtp_generate(model, drafter, tok, prompt, max_tokens, K=3,
                 ids_mode=False, profile=False, on_tokens=None):
    ...
    # 在两处 produced.append(new_tokens...) 之后各加一段:
    #   树验证路径(约 line 250-253)与主路径(约 line 374-377)
    if on_tokens is not None:
        if on_tokens(new_tokens_this_step):   # 回调返回 True => 用户请求中断
            break
    ...
```

- 一个钩子同时承载「流式吐字」+「Esc 中断」两个需求,不引入其它参数。
- `new_tokens_this_step` 是该步刚 append 的 token 列表(树路径可能多个)。
- 已有 `max_tokens` 截断逻辑不变;`break` 后走原有的收尾/return 路径,返回值结构不变。

### 3.3 对 `_build_engine` 的改动

签名加可选 `on_status: Callable[[str], None] | None = None`,把原来的三处
`print("正在加载...", file=sys.stderr)` 改为:有回调走回调,无回调仍走 stderr(保持 `--plain` 行为不变)。

---

## 4. 界面布局(对应 opencode 四段式)

```
┌────────────────────────────────────────────────────────┐
│  ▚ vates   qwen3-next-80b · k=3 · 512 tok      [顶栏]   │
├────────────────────────────────────────────────────────┤
│                                                          │
│   你   你好，帮我写个快排                    [用户气泡]  │  ← VerticalScroll
│                                                          │
│   ⣾ vates  <流式逐字渲染中…>                 [助手气泡]  │
│                                                          │
├────────────────────────────────────────────────────────┤
│ › 输入消息，回车发送，/help 查看命令          [输入框]   │
├────────────────────────────────────────────────────────┤
│  qwen3-next-80b · 就绪 · 128 tok · 11.6 tok/s  [状态栏] │
└────────────────────────────────────────────────────────┘
```

Textual 组件映射:

| 区域   | 组件                              | 说明 |
|--------|-----------------------------------|------|
| 顶栏   | 自定义 `Static`                   | banner logo + `模型 · k · max_tokens` |
| 对话区 | `VerticalScroll` 容器             | 内含若干消息组件,自动滚到底 |
| 用户消息 | `Static`(角色标签 + 文本)       | teal 标签、右侧留白区分 |
| 助手消息 | `Static`(流式)→ `Markdown`(完成) | 生成中显 spinner + 累计纯文本;完成后整段 Markdown 渲染 |
| 输入框 | `Input`                           | 圆角边框 + 占位提示 + 聚焦高亮 |
| 状态栏 | 自定义 `Static`                   | 模型名 · 状态 · 本轮 token 数 · tok/s |

助手消息渲染策略:流式阶段用 `Static` 显示「累计纯文本 + 光标」,**完成后**用 Markdown 渲染整段
(避免每 token 重排 Markdown 的开销与抖动)。

---

## 5. 交互 / 快捷键

- `Enter`:发送当前输入。空输入忽略。
- 斜杠命令(在 `Input.Submitted` 里拦截):
  - `/help`:弹出帮助(用一条系统消息或 Modal 展示可用命令)。
  - `/reset`:清空对话历史(保留 system),对话区插入一条「历史已清空」提示。
  - `/clear`:仅清空对话区显示(历史保留)。
  - `/exit`、`/quit`:退出应用。
- `Esc`:中断当前生成。置位 `self._stop = True`;`on_token` 检测到即返回 `True`,生成循环 break;
  已生成的部分保留为完整助手消息。
- `Ctrl+C` / `Ctrl+D`:退出应用。

生成进行中禁用输入框(或忽略新提交),状态栏显示「思考中 ⣾」,避免并发生成。

---

## 6. 线程模型(关键正确性)

Textual 是 asyncio 事件循环;`load()` 与 `generate()` 都是**阻塞重计算**,必须放 worker 线程:

- 用 `@work(thread=True)` 或 `run_worker(..., thread=True)` 跑 `load()` 和每轮 `generate()`。
- `on_status` / `on_token` 回调在 **worker 线程**触发。更新 UI 必须经
  `self.app.call_from_thread(self._append_delta, text)`,由 UI 线程实际改组件,保证线程安全。
- `Esc` 在 UI 线程置 `self._stop`;`on_token`(worker 线程读)返回 `True` 通知生成循环停止。
  用简单布尔标志即可(GIL 下读写单个 bool 原子,且只有「停止」这一单向翻转,无竞态风险)。

状态机:`LOADING → READY → GENERATING → READY`(或 `GENERATING → READY`(被 Esc 中断))。
`LOADING` 期间输入框禁用并显示加载进度(逐条 `on_status` 消息 + spinner)。

---

## 7. 配色主题(opencode 风格,集中在 styles.tcss)

- 深色底(近黑),中性灰边框,单一 **teal 强调色**(用户角色标签、输入框聚焦边、状态栏高亮)。
- 圆角边框(Textual `border: round`)、适度 padding,呈现 opencode 的「卡片感」。
- 所有颜色/边框/间距集中在 `styles.tcss`,改色只改这一个文件。

---

## 8. CLI 接线(cli.py)

- `cmd_chat(args)`:
  - 默认:构造 `MLXBackend(args)`,调用 `mlx_streaming.tui.run_tui(backend, args)` 启动 TUI。
  - `args.plain` 为真:走**现有**的旧 REPL 代码(原样保留,作为兜底/调试路径)。
- `_add_chat_args` 新增 `--plain`(`action="store_true"`,help:「用纯文本 REPL,不启动全屏 TUI」)。
- 其余参数(`--model/-k/-n/--stats/--system` 等)语义不变;TUI 与 REPL 共用同一套 args。

---

## 9. 错误处理

- **加载失败**(模型路径不存在 / 权重缺失):worker 捕获异常,`call_from_thread` 在对话区显示
  错误消息并把状态切到「加载失败」,输入框保持禁用,提示用 `/exit` 退出。不崩溃退出。
- **生成中异常**:worker 捕获,当前助手消息标记为「生成出错:<摘要>」,状态回到 `READY`,可继续下一轮。
- **终端不支持 TUI**(极少数):Textual 自身会报错;文档提示可用 `vates chat --plain` 兜底。
- **Esc 中断**:视为正常路径,非错误;保留已生成内容。

---

## 10. 测试

因界面/引擎解耦,可在**不加载 80B 模型**的前提下测 UI:

- `FakeBackend`:`load()` 立即成功并发几条 `on_status`;`generate()` 把预设文本按片回调,
  可注入「片数」「是否响应 stop」用于测中断。
- 用 Textual `App.run_test()` + `Pilot`:
  1. 启动后进入 `READY`,顶栏显示模型信息。
  2. 输入文本 + 回车 → 对话区出现用户气泡 + 助手气泡,助手文本随片增长(流式)。
  3. 生成中按 `Esc` → 生成停止,保留已生成部分,状态回 `READY`。
  4. `/reset` → 对话区出现「已清空」提示,后续生成不带旧历史(通过 FakeBackend 记录收到的 messages 断言)。
  5. `/exit` → 应用退出。
- 生成钩子单元测试:直接对 `mtp_generate` 传一个 `on_tokens`(用 `FakeBackend` 层面或小规模 stub),
  验证「默认 None 行为不变」与「返回 True 提前 break」。核心数值路径不需真实模型的部分用现有测试保障。

测试全部秒级完成,不触碰真实 MLX 推理。

---

## 11. 依赖变更

- `pyproject.toml` 的 `dependencies` 增加 `textual>=0.80`(会带上 `rich`)。
- `dev` 组已有 `pytest`;Textual 测试用其自带的 `run_test`,无需额外测试依赖。

---

## 12. 交付清单

1. `mtp/generate.py`:加 `on_tokens` 钩子(两处 append 点 + break),默认行为不变。
2. `cli.py`:`_build_engine` 加 `on_status`;`cmd_chat` 分流 TUI / `--plain`;`_add_chat_args` 加 `--plain`。
3. `mlx_streaming/tui/`:`backend.py`、`app.py`、`banner.py`、`styles.tcss`、`__init__.py`。
4. `pyproject.toml`:加 `textual` 依赖。
5. 测试:`mlx_streaming/tests/` 下新增 TUI 交互测试与钩子测试。
6. README:补一句 TUI 用法与 `--plain` 兜底说明。
