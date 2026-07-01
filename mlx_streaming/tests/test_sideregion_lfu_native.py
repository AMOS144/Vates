"""侧区持久 LFU:∉P 不清、free 空才淘 freq 最小;off 时回退旧'∉P 全弃'。"""
import struct
import tempfile
import time

import mlx.core as mx

import mlx_streaming.native_moe_ext as N

W, S = 16, 8
SEG = [W * 4, S * 1]
STRIDE = sum(SEG)
NE = 16


def _blob(path):
    with open(path, "wb") as f:
        for e in range(NE):
            f.write(struct.pack(f"<{W}I", *([e + 1] * W)))
            f.write(bytes([(e + 100) & 0xFF] * S))


def _pool(cap, spec):
    w = mx.zeros((cap + spec, W), dtype=mx.uint32)
    sc = mx.zeros((cap + spec, S), dtype=mx.uint8)
    mx.eval(w, sc)
    return [w, sc]


def _fill(pool, experts, layer, cap, spec):
    d = N.prefetch_pool_sideregion(
        pool, SEG, mx.array(experts, dtype=mx.uint32), layer, _PATH, STRIDE, [], spec, cap, gen=0)
    mx.eval(d)


def _contents(layer):
    flat = N.sideregion_contents(layer, 0)
    return {flat[i]: flat[i + 1] for i in range(0, len(flat), 2)}


def _wait_set(layer, expected, timeout=2.0):
    # 异步发布(bg 线程)：轮询到 e2r 键集 == 期望集合再返回,避免读到过渡态。
    t = time.time() + timeout
    exp = set(expected)
    while time.time() < t:
        if set(_contents(layer).keys()) == exp:
            break
        time.sleep(0.01)
    return _contents(layer)


_PATH = tempfile.NamedTemporaryFile(suffix=".blob", delete=False).name
_blob(_PATH)


def test_lfu_persist_keeps_old(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    cap, spec = 4, 4          # 4 侧区行,留空
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 0, cap, spec); _wait_set(0, {5, 6})
    _fill(pool, [7, 8], 0, cap, spec)       # 与 fill1 完全不重叠
    m = _wait_set(0, {5, 6, 7, 8})
    assert set(m.keys()) == {5, 6, 7, 8}    # 旧专家 5,6 未被清(持久)


def test_lfu_evicts_min_freq(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    cap, spec = 4, 2          # 只有 2 侧区行 → 满
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 1, cap, spec); _wait_set(1, {5, 6})
    _fill(pool, [5], 1, cap, spec)          # 再预测 5 → freq[5] 升(reserve 内 +1)
    _fill(pool, [5], 1, cap, spec)
    time.sleep(0.2)                          # 让上面两次 reserve(freq++) 先落地
    _fill(pool, [7], 1, cap, spec)          # free 空 → 淘 freq 最小且 ∉P 的 6
    m = _wait_set(1, {5, 7})
    assert set(m.keys()) == {5, 7}          # 保留高频 5,淘汰低频 6


def test_sideregion_kv_matches_contents(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    cap, spec = 4, 4
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 3, cap, spec); _wait_set(3, {5, 6})
    keys, vals = N.sideregion_kv(3, 0)          # C++ 直接吐 device 版 (uint32 keys, int32 vals)
    assert keys.dtype == mx.uint32 and vals.dtype == mx.int32
    kv = {int(k): int(v) for k, v in zip(keys.tolist(), vals.tolist())}
    assert kv == _contents(3)                   # 与 contents 语义一致


def test_sideregion_kv_empty(monkeypatch):
    monkeypatch.setenv("SIDEREGION_LFU", "1")
    N.sideregion_reset()
    keys, vals = N.sideregion_kv(99, 0)         # 空层 → size 0,不崩
    assert int(keys.size) == 0 and int(vals.size) == 0


def test_lfu_off_is_legacy(monkeypatch):
    monkeypatch.delenv("SIDEREGION_LFU", raising=False)
    N.sideregion_reset()
    cap, spec = 4, 4
    pool = _pool(cap, spec)
    _fill(pool, [5, 6], 2, cap, spec); _wait_set(2, {5, 6})
    _fill(pool, [7, 8], 2, cap, spec)       # 旧行为:∉P 全弃
    m = _wait_set(2, {7, 8})
    assert set(m.keys()) == {7, 8}          # 5,6 被清(回退旧语义)
