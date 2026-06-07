"""测流式前向耗时随序列长度的变化，估计投机解码的「验证摊薄」上限。

投机解码的 verify = 把 K 个 token 放进一次前向并行算。若每次前向调用的固定开销大，
则 per-position 耗时随 L 增大而下降，spec 摊薄收益大；若 per-position 基本不变，
说明开销是按 token 线性的，spec 只能靠接受率/批内专家复用获益。

输出：不同 L 下的单次前向耗时与 per-position 耗时（暖缓存，尽量排除 I/O 抖动）。
"""
import os
import time
import json

import mlx.core as mx
from mlx_lm import load

from mlx_streaming.expert_store import FileExpertStore
from mlx_streaming.streaming_moe import patch_model_filebacked

MODEL = os.environ.get("MODEL", "/tmp/qwen3moe")
EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts")
SLOTS = int(os.environ.get("EXPERT_SLOTS", "32"))
LENS = [int(x) for x in os.environ.get("LENS", "1,4,8,16").split(",")]
REPEAT = int(os.environ.get("REPEAT", "5"))


def _dims(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            gp = mlp.switch_mlp.gate_proj
            return gp.input_dims, gp.output_dims, gp.group_size, gp.bits


def time_forward(model, L):
    inp = mx.array([[1 + (i % 100) for i in range(L)]])
    # 暖：先跑一次把这批 token 的专家载入缓存
    y = model(inp); mx.eval(y)
    best = 1e9
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        y = model(inp); mx.eval(y)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    model, tok = load(MODEL, lazy=True)
    h, mi, gs, bits = _dims(model)
    store = FileExpertStore(EXPERT_DIR, capacity=SLOTS)
    patch_model_filebacked(model, store, h, mi, gs, bits)

    rows = []
    for L in LENS:
        t = time_forward(model, L)
        rows.append({"L": L, "fwd_ms": round(t * 1000, 1),
                     "per_pos_ms": round(t * 1000 / L, 1)})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    p1 = next(r["per_pos_ms"] for r in rows if r["L"] == 1)
    pmax = rows[-1]["per_pos_ms"]
    print(f"\nper-position：L=1 {p1}ms → L={rows[-1]['L']} {pmax}ms，"
          f"摊薄比 ≈ {round(p1 / pmax, 2)}×（spec verify 的理想加速上限近似此值）")


if __name__ == "__main__":
    main()
