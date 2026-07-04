"""容量不变性 oracle：正确的流式缓存对 cap 不变——同一 prompt greedy 在不同 cap 下
token 序列必须逐位一致。cap 只影响命中率/速度，绝不影响输出。

实现说明（相对原计划的修正）：
- `model_builder.EXPERT_SLOTS` 是模块级常量（import 时读一次），同进程内改环境变量再重建
  模型无法切换 cap（第二次 import 命中 sys.modules 缓存，仍用首次 cap）→ 会假绿。
- 故本 oracle 用**独立子进程**各跑一个 cap（贴合真实部署=每次全新进程），比对 run_mtp_spec 的
  DUMP_BASE_IDS token 序列。这与手工验证（cap32≡cap48 逐字节一致）同口径。
需真实 80B 模型（QN_CONFIG/MTP_OUT 就位）；无模型环境自动 skip。
"""
import json
import os
import subprocess
import sys

import pytest

from mlx_streaming import config as _cfg

_HAS_MODEL = os.path.exists(_cfg.qn_config()) and os.path.exists(_cfg.mtp_out())
pytestmark = pytest.mark.skipif(not _HAS_MODEL, reason="需真实 80B 模型权重")

_MAXTOK = 32


def _base_ids_at_cap(cap: int) -> list:
    """在独立子进程里以 EXPERT_SLOTS=cap 跑非投机 greedy，返回 DUMP_BASE_IDS token 序列。"""
    env = dict(os.environ)
    env.update(
        DUMP_IDS="1",
        STREAM_BLOB_LOADER="1",
        NATIVE_FUSED_PREFETCH="1",
        ZEROCOPY_DUAL_SOURCE="1",
        SIDEREGION_LFU="1",
        EXPERT_SLOTS=str(cap),   # 真实区容量：被测的 cap 变量
        POOL_SPEC_SLOTS="32",    # 侧区固定，只变真实区 cap
        K="3",
        MAXTOK=str(_MAXTOK),
        WARMUP_TOK="0",
        REPEAT="1",
    )
    r = subprocess.run(
        [sys.executable, "-m", "mlx_streaming.runtime.run_mtp_spec"],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    for line in r.stdout.splitlines():
        if line.startswith("DUMP_BASE_IDS "):
            return json.loads(line[len("DUMP_BASE_IDS "):])
    raise AssertionError(
        f"未捕获 DUMP_BASE_IDS (rc={r.returncode})\n"
        f"stdout尾:\n{r.stdout[-2000:]}\nstderr尾:\n{r.stderr[-2000:]}"
    )


def test_greedy_capacity_invariant():
    # 两个 cap 都 ≥ Phase 0 实测 U_max=30（32/48 满足）。
    toks_a = _base_ids_at_cap(32)
    toks_b = _base_ids_at_cap(48)
    assert len(toks_a) == _MAXTOK, f"cap32 token 数异常：{len(toks_a)}"
    assert toks_a == toks_b, (
        f"容量不变性被破坏：cap32 与 cap48 token 序列不同\n{toks_a}\n{toks_b}"
    )
