"""字节真值 oracle：跑一段 decode，断言每个占用槽的池字节 == 其属主专家磁盘真值（0 BAD）。

生产默认路径是 demand_dual（C++ 真实区唯一权威），其字节落池校验开关是 **STG_VERIFY**
（`_verify_native_bytes`：逐槽比对 g_real 属主专家 blob 真值）。解析 run_mtp_spec 的
VERIFY_SUMMARY，断言 bad==0 且 ok>0（确保确实校验过，非 calls=0 空结论）。
用独立子进程跑（避免 native 全局状态跨用例泄漏，且与集成验收同口径）。需真实 80B 模型；无则 skip。
"""
import json
import os
import subprocess
import sys

import pytest

from mlx_streaming import config as _cfg

_HAS_MODEL = os.path.exists(_cfg.qn_config()) and os.path.exists(_cfg.mtp_out())
pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="需真实 80B 模型权重")

_SV_KEY = "STG_VERIFY.virtual(_verify_native_bytes)"


def _run(env_extra, maxtok):
    env = dict(os.environ)
    env.update(
        STREAM_BLOB_LOADER="1",
        NATIVE_FUSED_PREFETCH="1",
        ZEROCOPY_DUAL_SOURCE="1",
        SIDEREGION_LFU="1",
        EXPERT_SLOTS="48",
        POOL_SPEC_SLOTS="32",
        K="3",
        MAXTOK=str(maxtok),
        WARMUP_TOK="0",
        REPEAT="1",
    )
    env.update(env_extra)
    r = subprocess.run(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    summary = None
    for line in r.stdout.splitlines():
        if line.startswith("VERIFY_SUMMARY "):
            summary = json.loads(line[len("VERIFY_SUMMARY "):])
    assert summary is not None, (
        f"未捕获 VERIFY_SUMMARY (rc={r.returncode})\n"
        f"stdout尾:\n{r.stdout[-2000:]}\nstderr尾:\n{r.stderr[-2000:]}"
    )
    return r, summary


def test_demand_dual_bytes_are_disk_truth():
    # 生产默认路径(demand_dual)：STG_VERIFY 校验 C++ 真实区每槽字节 == 属主专家 blob 真值。
    # 逐槽 pread 较慢,故用较短 MAXTOK(仍足以覆盖多轮驱逐/落池)。
    r, summary = _run({"STG_VERIFY": "1"}, maxtok=16)
    assert "[STG_VERIFY-DUAL] BAD" not in r.stdout, f"检测到 demand_dual 落池字节错:\n{r.stdout[-3000:]}"
    sv = summary.get(_SV_KEY, {})
    assert sv.get("bad", -1) == 0, f"STG_VERIFY 检出坏字节：{sv}"
    assert sv.get("ok", 0) > 0, f"STG_VERIFY 未实际校验（ok=0），oracle 失效：{sv}"
