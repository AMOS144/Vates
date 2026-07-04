"""字节真值 oracle：跑一段 decode，断言每个占用槽的池字节 == 其属主专家磁盘真值（0 BAD）。

实现说明（相对原计划的修正）：
- 默认 ZEROCOPY_DUAL_SOURCE 路径的字节真值开关是 **DUAL_VERIFY**，不是 STG_VERIFY——实测
  STG_VERIFY 在该路径 calls=0（不触发），用它会 0 检查假绿。故本 oracle 用 DUAL_VERIFY，
  并解析 run_mtp_spec 的 VERIFY_SUMMARY，同时断言 bad==0 且 ok>0（确保确实校验过）。
- 用独立子进程跑（避免 native 全局状态跨用例泄漏，且与集成验收同口径）。
需真实 80B 模型；无则 skip。
"""
import json
import os
import subprocess
import sys

import pytest

from mlx_streaming import config as _cfg

_HAS_MODEL = os.path.exists(_cfg.qn_config()) and os.path.exists(_cfg.mtp_out())
pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="需真实 80B 模型权重")

_DV_KEY = "DUAL_VERIFY.resident(_verify_side_bytes)"


def test_pool_bytes_are_disk_truth():
    env = dict(os.environ)
    env.update(
        DUAL_VERIFY="1",
        STREAM_BLOB_LOADER="1",
        NATIVE_FUSED_PREFETCH="1",
        ZEROCOPY_DUAL_SOURCE="1",
        SIDEREGION_LFU="1",
        EXPERT_SLOTS="48",
        POOL_SPEC_SLOTS="32",
        K="3",
        MAXTOK="32",
        WARMUP_TOK="0",
        REPEAT="1",
    )
    r = subprocess.run(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert "[DUAL_VERIFY] BAD" not in r.stdout, f"检测到池槽字节污染:\n{r.stdout[-3000:]}"

    summary = None
    for line in r.stdout.splitlines():
        if line.startswith("VERIFY_SUMMARY "):
            summary = json.loads(line[len("VERIFY_SUMMARY "):])
    assert summary is not None, (
        f"未捕获 VERIFY_SUMMARY (rc={r.returncode})\n"
        f"stdout尾:\n{r.stdout[-2000:]}\nstderr尾:\n{r.stderr[-2000:]}"
    )
    dv = summary.get(_DV_KEY, {})
    assert dv.get("bad", -1) == 0, f"DUAL_VERIFY 检出坏字节：{dv}"
    assert dv.get("ok", 0) > 0, f"DUAL_VERIFY 未实际校验（ok=0），oracle 失效：{dv}"
