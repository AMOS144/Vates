# AGENTS.md — mlx-streaming-moe

本项目:Apple Silicon 上基于 MLX 的显存外置流式 MoE + Qwen3-Next MTP 自投机解码。

## 约定

- **代码注释一律用中文**;回答也用中文。
- 包名为 `mlx_streaming`,按职责分子包(见 `README.md` 目录结构):`config.py`(集中 env 常量)、`core/`(运行时核心:`moe/`+`cache/`+`prefetch/`+`profiling`)、`mtp/`(自投机)、`model_builder.py`(共享装配层)、`prep/`(离线数据准备)、`runtime/`(生产推理入口 run_*)、`tools/`(诊断/实验脚本 probe_*/validate_*/simulate_*)。项目用 **uv** 管理:
  - 安装/同步:`uv sync`
  - 跑测试:`uv run pytest`
  - 跑入口:生产入口 `uv run python -m mlx_streaming.runtime.<name>`,诊断工具 `uv run python -m mlx_streaming.tools.<name>`(数据准备用 `mlx_streaming.prep.<name>`)
- 实验结论写入 `benchmarks/reports/`;设计写 `docs/superpowers/specs/`,实现计划写 `docs/superpowers/plans/`(均中文,spec 易懂、plan 严谨)。

## 工程纪律(本项目踩坑总结)

- **de-risk 优先**:动手多天/高风险改动前,先写最便宜的探针脚本实测可行性与上界,再决定是否深入(例:fused Metal kernel 用 ~2h 探针证伪后及时止损)。一次性探针出结论后把结论写进 `benchmarks/reports/`,脚本本身可删(历史可在 git 找回),不要长期堆在包里。
- **MLX 惰性求值坑**:`mx.concatenate` 等会复制整块且产惰性图;动态增长池被证伪,改用按 profile 一次性静态分配。
- 改核心数据路径(`core/expert_store.py` / `core/streaming_moe.py`)务必配单测并跑全绿。

## 关键环境变量

见 `README.md`。常用:`EXPERT_DIR / EXPERT_SLOTS / RESIDENT_POOL / EXPERT_POOL_PROFILE / MTP_VERIFY_MODE / K`。
