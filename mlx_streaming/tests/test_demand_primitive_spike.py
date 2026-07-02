"""Phase 2 方案B 地基机制探针（可复现证据）。

结论（见 benchmarks/reports/virtualpool-phase2-spike-2026-07-02.md）：
- 「eval_gpu body 读 GPU 算出的 inds」不可靠（inds kernel 未执行 → 读到脏值）。
- 「完成回调写 output」对同前向下游不可见。
- ⇒ 同前向「零主线程同步」demand 不可行；拿 inds 值必须先同步（inds.eval）。
- 一旦 inds 已同步，C++ 在 eval 里读 inds 值是正确的（这是「1 次同步 + C++ 落池」可行路径的地基）。

本文件把「已同步后 C++ 读 inds 正确」钉成可复现单测；未编译 native 则 skip。
"""
import pytest
import mlx.core as mx

try:
    import mlx_streaming.native_moe_ext as N
    _HAS_NATIVE = hasattr(N, "demand_probe")
except Exception:
    _HAS_NATIVE = False

pytestmark = pytest.mark.skipif(not _HAS_NATIVE, reason="native_moe_ext demand_probe 未编译")


def test_demand_probe_correct_after_inds_evaluated():
    # 已同步（eval）后，C++ 在 primitive 里读 inds 值正确 → 「1 次同步 + C++ 落池」地基成立。
    OFF = 1000
    for _ in range(50):
        inds = (mx.arange(64, dtype=mx.uint32) * 7 + 3) % 64
        mx.eval(inds)                                  # 关键：先同步 inds（1 次同步）
        local = N.demand_probe(inds, OFF)
        mx.eval(local)
        exp = [(i * 7 + 3) % 64 + OFF for i in range(64)]
        assert local.tolist() == exp


def test_materialize_spike_positive_control():
    # 正对照：eval_gpu body 写 output(常量) → 下游 gather 可见（机制的另一半成立）。
    src = mx.arange(8, dtype=mx.uint32) + 1000
    mx.eval(src)
    out = N.materialize_spike(src, 777)
    g = mx.take(out, mx.array([0, 1, 2, 3], dtype=mx.uint32))
    mx.eval(g)
    # 偶数行=777，奇数行保留 1000+row
    assert g.tolist() == [777, 1001, 777, 1003]
