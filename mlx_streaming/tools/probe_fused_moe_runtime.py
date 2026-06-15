"""用真实 expert 权重验证 fused MoE runtime helper。

不加载完整模型，只比较单层单次 MoE expert 计算：
- MLX QuantizedSwitchLinear/SwitchGLU 参考路径
- 现有三段 custom qproj 路径
- 新 fused gate/up/SwiGLU/down 路径
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.moe.compute import PersistentSubGLU
from mlx_streaming.core.moe.custom_kernel import _custom_fused_moe_indexed, _custom_qlinear_indexed

EXPERT_DIR = os.environ.get(
    "EXPERT_DIR",
    "/Users/amos/project/flash-moe/mlx-streaming-moe/models/qwen3_next_experts_bnd12_l43_l47_6_g128",
)
LAYER = int(os.environ.get("LAYER", "43"))
ACTIVE = int(os.environ.get("ACTIVE", "16"))
SEQ = int(os.environ.get("SEQ", "1"))
K = int(os.environ.get("K", "8"))
HIDDEN = int(os.environ.get("HIDDEN", "2048"))
INTER = int(os.environ.get("INTER", "512"))
GROUP = int(os.environ.get("GROUP", "128"))
BITS = int(os.environ.get("BITS", "6"))
REPEAT = int(os.environ.get("REPEAT", "30"))
TILE = int(os.environ.get("CUSTOM_QPROJ_TILE", "4"))


def _with_env_disabled(fn):
    old_fused = os.environ.get("CUSTOM_FUSED_MOE")
    old_qproj = os.environ.get("CUSTOM_QPROJ")
    os.environ["CUSTOM_FUSED_MOE"] = "0"
    os.environ["CUSTOM_QPROJ"] = "0"
    try:
        return fn()
    finally:
        if old_fused is None:
            os.environ.pop("CUSTOM_FUSED_MOE", None)
        else:
            os.environ["CUSTOM_FUSED_MOE"] = old_fused
        if old_qproj is None:
            os.environ.pop("CUSTOM_QPROJ", None)
        else:
            os.environ["CUSTOM_QPROJ"] = old_qproj


def _flatten_inputs(x: mx.array, local: mx.array):
    k = int(local.shape[-1])
    x_flat = mx.broadcast_to(mx.expand_dims(x, -2), x.shape[:-1] + (k, x.shape[-1]))
    return x_flat.reshape(-1, HIDDEN).astype(mx.float32), local.reshape(-1).astype(mx.uint32)


def _qproj_chain(sub: PersistentSubGLU, x: mx.array, local: mx.array) -> mx.array:
    x_flat, idx = _flatten_inputs(x, local)
    up = _custom_qlinear_indexed(
        x_flat, idx,
        sub._glu.up_proj["weight"], sub._glu.up_proj["scales"], sub._glu.up_proj["biases"],
        INTER, HIDDEN, GROUP, BITS, TILE,
    )
    gate = _custom_qlinear_indexed(
        x_flat, idx,
        sub._glu.gate_proj["weight"], sub._glu.gate_proj["scales"], sub._glu.gate_proj["biases"],
        INTER, HIDDEN, GROUP, BITS, TILE,
    )
    act = gate * mx.sigmoid(gate) * up
    y = _custom_qlinear_indexed(
        act, idx,
        sub._glu.down_proj["weight"], sub._glu.down_proj["scales"], sub._glu.down_proj["biases"],
        HIDDEN, INTER, GROUP, BITS, TILE,
    )
    return y.reshape(local.shape + (HIDDEN,))


def _fused(sub: PersistentSubGLU, x: mx.array, local: mx.array) -> mx.array:
    x_flat, idx = _flatten_inputs(x, local)
    y = _custom_fused_moe_indexed(
        x_flat, idx,
        sub._glu.gate_proj["weight"], sub._glu.gate_proj["scales"], sub._glu.gate_proj["biases"],
        sub._glu.up_proj["weight"], sub._glu.up_proj["scales"], sub._glu.up_proj["biases"],
        sub._glu.down_proj["weight"], sub._glu.down_proj["scales"], sub._glu.down_proj["biases"],
        HIDDEN, INTER, GROUP, BITS,
    )
    return y.reshape(local.shape + (HIDDEN,))


def _bench(fn) -> float:
    y = fn()
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        y = fn()
        mx.eval(y)
    return (time.perf_counter() - t0) * 1000 / REPEAT


def main():
    active = max(ACTIVE, K)
    experts = list(range(active))
    store = FileExpertStore(EXPERT_DIR, capacity=max(512, active))
    fetched = store.fetch(LAYER, experts)
    sub = PersistentSubGLU(
        HIDDEN, INTER, GROUP, BITS,
        proj_bits={"gate_proj": BITS, "up_proj": BITS, "down_proj": BITS},
        layer_idx=LAYER,
    )
    sub._ensure(active)
    ref = _with_env_disabled(lambda: sub.forward(fetched, active, _make_x(), _make_local(active)))
    mx.eval(ref)

    x = _make_x()
    local = _make_local(active)
    # 先更新一次权重，之后三条路径复用同一组 QSL 参数。
    _with_env_disabled(lambda: sub.forward(fetched, active, x, local))

    ref = _with_env_disabled(lambda: sub._glu(x, local))
    qproj = _qproj_chain(sub, x, local)
    fused = _fused(sub, x, local)
    mx.eval(ref, qproj, fused)
    qproj_err = float(mx.max(mx.abs(ref - qproj)))
    fused_err = float(mx.max(mx.abs(ref - fused)))
    ref_ms = _bench(lambda: _with_env_disabled(lambda: sub._glu(x, local)))
    qproj_ms = _bench(lambda: _qproj_chain(sub, x, local))
    fused_ms = _bench(lambda: _fused(sub, x, local))
    print(json.dumps({
        "expert_dir": EXPERT_DIR,
        "layer": LAYER,
        "active": active,
        "seq": SEQ,
        "k": K,
        "pairs": int(local.size),
        "bits": BITS,
        "group": GROUP,
        "fused_block": int(os.environ.get("CUSTOM_FUSED_MOE_BLOCK", "256")),
        "fused_lanes": int(os.environ.get("CUSTOM_FUSED_MOE_LANES", "8")),
        "repeat": REPEAT,
        "ref_ms": round(ref_ms, 4),
        "qproj_ms": round(qproj_ms, 4),
        "fused_ms": round(fused_ms, 4),
        "fused_vs_ref": round(ref_ms / max(fused_ms, 1e-9), 4),
        "fused_vs_qproj": round(qproj_ms / max(fused_ms, 1e-9), 4),
        "qproj_max_abs": qproj_err,
        "fused_max_abs": fused_err,
        "pass_error": fused_err < 1e-3,
    }, ensure_ascii=False, indent=2))


def _make_x() -> mx.array:
    mx.random.seed(0)
    return (mx.random.normal((1, SEQ, HIDDEN)).astype(mx.float32) * 0.1)


def _make_local(active: int) -> mx.array:
    vals = [(i % active) for i in range(SEQ * K)]
    return mx.array(vals, dtype=mx.uint32).reshape(1, SEQ, K)


if __name__ == "__main__":
    main()
