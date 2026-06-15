"""热路径计时与诊断埋点（从 streaming_moe 抽出，集中管理）。

- PROF：细粒度分段计时（STREAM_PROF=1），probe 读取。
- WINDOW_PROF：同层 submit→promote 的 attention/GDN 窗口时间（WINDOW_PROF=1）。
- PREDICT_RECALL_PROF：运行时预测对实际路由的覆盖率（分诊预取覆盖问题，PREDICT_RECALL_PROF=1）。
这些 dict 为共享可变对象：各模块 import 后原地累加，外部（run_mtp_spec）读同一对象。
"""
import time

import mlx.core as mx

from mlx_streaming import config

_PROF_ON = config.stream_prof()
PROF = {"route": 0.0, "pyremap": 0.0, "fetch": 0.0, "matmul": 0.0, "combine": 0.0,
        "n_calls": 0}
WINDOW_PROF = {"sum_s": 0.0, "n": 0}
PREDICT_RECALL_PROF = {"hit": 0, "routed": 0, "n": 0}


def prof_reset():
    for k in PROF:
        PROF[k] = 0.0


def _tick(seg, t0):
    mx.eval  # noqa  (占位,避免误用)
    PROF[seg] += time.perf_counter() - t0
