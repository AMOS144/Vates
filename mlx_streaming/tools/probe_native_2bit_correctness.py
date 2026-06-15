"""验证 2bit compute buffer 下 native fused MoE 与 MLX quantized_matmul 参考一致。

只有数值一致，后面 native slot pool 的 A/B 才有意义（之前的 kernel 测试都是 6bit）。
"""
import os

import mlx.core as mx

from mlx_streaming.core.moe import native_moe

HIDDEN = 2048
INTER = 512
GROUP = 128
BITS = 2
NUM_EXPERTS = 512
COMPUTE_DIR = os.environ.get("COMPUTE_BUFFER_DIR", "/tmp/cb_2bit_g128")
EXPERT_DIR = os.environ.get(
    "EXPERT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen3_next_experts_2bit_g128"),
)
LAYER = int(os.environ.get("LAYER", "0"))
EXPERT = int(os.environ.get("EXPERT", "0"))


def _reference(x: mx.array, w: dict) -> mx.array:
    """纯 MLX quantized_matmul 参考实现：gate/up + SwiGLU + down。"""
    gate = mx.quantized_matmul(
        x, w["gate_proj.weight"], w["gate_proj.scales"], w["gate_proj.biases"],
        transpose=True, group_size=GROUP, bits=BITS)
    up = mx.quantized_matmul(
        x, w["up_proj.weight"], w["up_proj.scales"], w["up_proj.biases"],
        transpose=True, group_size=GROUP, bits=BITS)
    act = gate * mx.sigmoid(gate) * up
    out = mx.quantized_matmul(
        act, w["down_proj.weight"], w["down_proj.scales"], w["down_proj.biases"],
        transpose=True, group_size=GROUP, bits=BITS)
    return out


def main():
    os.environ["COMPUTE_BUFFER_DIR"] = COMPUTE_DIR
    mx.random.seed(0)
    x = mx.random.normal((1, HIDDEN)).astype(mx.float32) * 0.1

    expert_path = os.path.join(EXPERT_DIR, f"layer{LAYER:02d}_expert{EXPERT:03d}.safetensors")
    w = mx.load(expert_path)
    ref = _reference(x, w)
    mx.eval(ref)

    ext = native_moe._load_ext()
    pool = native_moe._slot_pool(COMPUTE_DIR, LAYER, HIDDEN, INTER, GROUP, BITS, NUM_EXPERTS)
    local, pool_arrays = pool.acquire([EXPERT])
    local_arr = mx.array(local, dtype=mx.uint32)
    y = ext.fused_moe_slots(
        x, local_arr, mx.array([[1.0]], dtype=mx.float32),
        *pool_arrays, HIDDEN, INTER, GROUP, BITS)
    mx.eval(y)

    diff = mx.abs(y - ref)
    denom = mx.maximum(mx.abs(ref), mx.array(1e-6))
    rel = float(mx.max(diff / denom))
    max_abs = float(mx.max(diff))
    ref_scale = float(mx.max(mx.abs(ref)))
    # 余弦相似度（方向一致性）
    cos = float(mx.sum(y * ref) / (mx.sqrt(mx.sum(y * y)) * mx.sqrt(mx.sum(ref * ref)) + 1e-9))
    print({
        "layer": LAYER, "expert": EXPERT,
        "ref_max_abs": round(ref_scale, 5),
        "native_max_abs": round(float(mx.max(mx.abs(y))), 5),
        "max_abs_diff": round(max_abs, 6),
        "max_rel_diff": round(rel, 4),
        "cosine": round(cos, 6),
        "pass": bool(cos > 0.999 and max_abs < 0.02 * ref_scale + 1e-4),
    })


if __name__ == "__main__":
    main()
