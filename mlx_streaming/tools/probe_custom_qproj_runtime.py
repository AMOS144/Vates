"""用真实专家权重验证 custom qlinear projection 是否有 runtime 收益。

只做 projection 级 de-risk，不接完整 MoE runtime。
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.core.cache.expert_store import FileExpertStore

EXPERT_DIR = os.environ.get(
    "EXPERT_DIR",
    "/Users/amos/project/flash-moe/mlx-streaming-moe/models/qwen3_next_experts_bnd12_l43_l47_6_g128",
)
LAYER = int(os.environ.get("LAYER", "43"))
PROJ = os.environ.get("PROJ", "down_proj")
EXPERTS = [int(x) for x in os.environ.get("EXPERTS", "0,1,2,3,4,5,6,7").split(",")]
BITS = int(os.environ.get("BITS", "6"))
GROUP = int(os.environ.get("GROUP", "128"))
REPEAT = int(os.environ.get("REPEAT", "50"))
VARIANT = os.environ.get("VARIANT", "tile4")  # row | tile2 | tile4 | tile8


def _kernel_tiled(x, w, scales, biases, experts, out_dim, in_dim, bits, group, tile):
    source = r"""
        constexpr int lanes_per_row = 256 / rows_per_group;
        constexpr int block_size = rows_per_group * lanes_per_row;
        uint tid = thread_position_in_threadgroup.x;
        uint group_id = thread_position_in_grid.x / block_size;
        uint local_row = tid / lanes_per_row;
        uint lane = tid % lanes_per_row;
        uint global_row = group_id * rows_per_group + local_row;
        if (global_row >= experts * out_dim) return;
        uint expert = global_row / out_dim;
        uint out_row = global_row % out_dim;
        constexpr uint mask = (1u << bits) - 1u;
        constexpr int words_per_row = (in_dim * bits) / 32;
        constexpr int groups_per_row = in_dim / group_size;
        threadgroup float partial[block_size];
        float acc = 0.0f;
        for (int col = int(lane); col < in_dim; col += lanes_per_row) {
            int bit_offset = col * bits;
            int word_idx = bit_offset / 32;
            int shift = bit_offset % 32;
            uint base = (expert * out_dim + out_row) * words_per_row;
            uint word = w[base + word_idx];
            uint q = (word >> shift);
            if (shift + bits > 32) {
                uint next_word = w[base + word_idx + 1];
                q |= (next_word << (32 - shift));
            }
            q = q & mask;
            int g = col / group_size;
            uint sb = (expert * out_dim + out_row) * groups_per_row + g;
            float weight = float(q) * scales[sb] + biases[sb];
            acc += weight * x[expert * in_dim + col];
        }
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            y[global_row] = partial[tid];
        }
    """
    kernel = mx.fast.metal_kernel(
        name="qlinear_runtime_tiled",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=source,
    )
    groups_grid = (experts * out_dim + tile - 1) // tile
    (y,) = kernel(
        inputs=[x, w, scales, biases],
        output_shapes=[(experts, out_dim)],
        output_dtypes=[mx.float32],
        grid=(groups_grid * 256, 1, 1),
        threadgroup=(256, 1, 1),
        template=[
            ("experts", experts),
            ("out_dim", out_dim),
            ("in_dim", in_dim),
            ("group_size", group),
            ("bits", bits),
            ("rows_per_group", tile),
        ],
    )
    return y


def _custom_qlinear(x, w, scales, biases):
    experts, out_dim, _words = w.shape
    in_dim = x.shape[-1]
    tile = {"row": 1, "tile2": 2, "tile4": 4, "tile8": 8}[VARIANT]
    return _kernel_tiled(x, w, scales, biases, experts, out_dim, in_dim, BITS, GROUP, tile)


def _mlx_loop(x, w, scales, biases):
    ys = []
    for i in range(w.shape[0]):
        y = mx.quantized_matmul(
            x[i:i + 1], w[i], scales[i], biases[i],
            transpose=True, group_size=GROUP, bits=BITS, mode="affine")
        ys.append(y)
    return mx.concatenate(ys, axis=0)


def main():
    store = FileExpertStore(EXPERT_DIR, capacity=max(512, len(EXPERTS)))
    rec = store.fetch(LAYER, EXPERTS)
    w = rec[f"{PROJ}.weight"]
    scales = rec[f"{PROJ}.scales"]
    biases = rec[f"{PROJ}.biases"]
    in_dim = (GROUP * scales.shape[-1])
    x = mx.random.normal((len(EXPERTS), in_dim)).astype(mx.float32) * 0.1
    y_ref = _mlx_loop(x, w, scales, biases)
    y_custom = _custom_qlinear(x, w, scales, biases)
    mx.eval(y_ref, y_custom)
    max_abs = float(mx.max(mx.abs(y_ref - y_custom)))

    t0 = time.perf_counter()
    for _ in range(REPEAT):
        y = _mlx_loop(x, w, scales, biases)
        mx.eval(y)
    t1 = time.perf_counter()
    for _ in range(REPEAT):
        y = _custom_qlinear(x, w, scales, biases)
        mx.eval(y)
    t2 = time.perf_counter()
    mlx_ms = (t1 - t0) * 1000 / REPEAT
    custom_ms = (t2 - t1) * 1000 / REPEAT
    print(json.dumps({
        "expert_dir": EXPERT_DIR,
        "layer": LAYER,
        "proj": PROJ,
        "experts": len(EXPERTS),
        "variant": VARIANT,
        "bits": BITS,
        "in_dim": in_dim,
        "out_dim": int(w.shape[1]),
        "repeat": REPEAT,
        "mlx_ms": round(mlx_ms, 4),
        "custom_ms": round(custom_ms, 4),
        "custom_vs_mlx": round(mlx_ms / max(custom_ms, 1e-9), 4),
        "max_abs": max_abs,
        "pass_70pct": custom_ms <= mlx_ms / 0.70,
        "pass_error": max_abs < 1e-4,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
