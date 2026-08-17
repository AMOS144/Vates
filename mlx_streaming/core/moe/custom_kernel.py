"""自定义 Metal 算子：indexed 量化线性 + fused MoE expert（实验性加速路径）。

这些 kernel 走 `mx.fast.metal_kernel`，按 `indices[pair]` 选择专家权重，
直接在 packed 量化权重上算 SwiGLU/down，避免反量化中间张量。仅在对应
CUSTOM_* 环境开关命中且 bit 宽匹配时启用，默认关闭（不改变模型数值）。
"""
from typing import Tuple  # noqa: F401  (保留以兼容历史 import)

import mlx.core as mx

from mlx_streaming import config
from mlx_streaming.config import parse_layers_env as _parse_layers_env

# kernel 编译缓存：按 (tile) / (hidden,moe_inter,...) 复用已编译 kernel。
_CUSTOM_QKERNEL_CACHE = {}
_CUSTOM_FUSED_MOE_CACHE = {}
_EXPERT_MAJOR_MOE_CACHE = {}


def _expert_major_fused_moe(
    x_sorted: mx.array, offsets: mx.array,
    gate_w: mx.array, gate_s: mx.array, gate_b: mx.array,
    up_w: mx.array, up_s: mx.array, up_b: mx.array,
    down_w: mx.array, down_s: mx.array, down_b: mx.array,
    hidden: int, moe_inter: int, group_size: int, bits: int,
    shards_per_expert: int = 8,
) -> mx.array:
    """Expert-major fused GEMM: each threadgroup owns one expert shard.

    Routes must be sorted by local expert and ``offsets`` contains the
    exclusive assignment ranges. Unlike the decode pair-major kernel, a
    threadgroup keeps one expert's packed rows hot while processing several
    tokens.
    """
    if bits != 4 or group_size != 64:
        raise ValueError("expert-major Metal GEMM currently requires affine Q4/G64")
    experts = int(offsets.size) - 1
    pairs = int(x_sorted.shape[0])
    shards = max(1, int(shards_per_expert))
    key = (hidden, moe_inter, experts, shards)
    kernel = _EXPERT_MAJOR_MOE_CACHE.get(key)
    if kernel is None:
        source = r"""
            constexpr uint block = 256;
            constexpr uint lanes = 8;
            constexpr uint rows_step = block / lanes;
            uint tid = thread_position_in_threadgroup.x;
            uint task = thread_position_in_grid.x / block;
            uint expert = task / shards;
            uint shard = task - expert * shards;
            if (expert >= experts) return;
            constexpr uint mask = 15;
            constexpr int gu_words = hidden * 4 / 32;
            constexpr int gu_groups = hidden / 64;
            constexpr int down_words = moe_inter * 4 / 32;
            constexpr int down_groups = moe_inter / 64;
            uint local_row = tid / lanes;
            uint lane = tid - local_row * lanes;
            threadgroup float act[512];
            threadgroup float gate_part[block];
            threadgroup float up_part[block];

            for (uint pair = offsets[expert] + shard;
                 pair < offsets[expert + 1]; pair += shards) {
              for (uint base_row = 0; base_row < moe_inter; base_row += rows_step) {
                uint row = base_row + local_row;
                float ga = 0.0f, ua = 0.0f;
                if (row < moe_inter) {
                  uint wbase = (expert * moe_inter + row) * gu_words;
                  uint sbase = (expert * moe_inter + row) * gu_groups;
                  for (uint col = lane; col < hidden; col += lanes) {
                    uint bit = col * 4;
                    uint word = bit >> 5;
                    uint shift = bit & 31;
                    uint qg = (gate_w[wbase + word] >> shift) & mask;
                    uint qu = (up_w[wbase + word] >> shift) & mask;
                    uint sb = sbase + col / 64;
                    float xv = x_sorted[pair * hidden + col];
                    ga += (float(qg) * gate_s[sb] + gate_b[sb]) * xv;
                    ua += (float(qu) * up_s[sb] + up_b[sb]) * xv;
                  }
                }
                gate_part[tid] = ga;
                up_part[tid] = ua;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = lanes / 2; stride; stride >>= 1) {
                  if (lane < stride) {
                    gate_part[tid] += gate_part[tid + stride];
                    up_part[tid] += up_part[tid + stride];
                  }
                  threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0 && row < moe_inter) {
                  float g = gate_part[tid];
                  act[row] = g / (1.0f + exp(-g)) * up_part[tid];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }

              for (uint base_row = 0; base_row < hidden; base_row += rows_step) {
                uint row = base_row + local_row;
                float acc = 0.0f;
                if (row < hidden) {
                  uint wbase = (expert * hidden + row) * down_words;
                  uint sbase = (expert * hidden + row) * down_groups;
                  for (uint col = lane; col < moe_inter; col += lanes) {
                    uint bit = col * 4;
                    uint word = bit >> 5;
                    uint shift = bit & 31;
                    uint q = (down_w[wbase + word] >> shift) & mask;
                    uint sb = sbase + col / 64;
                    acc += (float(q) * down_s[sb] + down_b[sb]) * act[col];
                  }
                }
                gate_part[tid] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = lanes / 2; stride; stride >>= 1) {
                  if (lane < stride) gate_part[tid] += gate_part[tid + stride];
                  threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0 && row < hidden)
                  y[pair * hidden + row] = gate_part[tid];
                threadgroup_barrier(mem_flags::mem_threadgroup);
              }
            }
        """
        kernel = mx.fast.metal_kernel(
            name=f"expert_major_moe_h{hidden}_i{moe_inter}_e{experts}_s{shards}",
            input_names=["x_sorted", "offsets", "gate_w", "gate_s", "gate_b",
                         "up_w", "up_s", "up_b", "down_w", "down_s", "down_b"],
            output_names=["y"], source=source,
        )
        _EXPERT_MAJOR_MOE_CACHE[key] = kernel
    (out,) = kernel(
        inputs=[x_sorted, offsets.astype(mx.uint32), gate_w, gate_s, gate_b,
                up_w, up_s, up_b, down_w, down_s, down_b],
        output_shapes=[(pairs, hidden)], output_dtypes=[mx.float32],
        grid=(experts * shards * 256, 1, 1), threadgroup=(256, 1, 1),
        template=[("experts", experts), ("shards", shards),
                  ("hidden", hidden), ("moe_inter", moe_inter)],
    )
    return out


def _custom_qproj_enabled(layer_idx: int, bits: int) -> bool:
    if not config.custom_qproj():
        return False
    if bits != config.custom_qproj_bits():
        return False
    layers = _parse_layers_env("CUSTOM_QPROJ_LAYERS")
    return layers is None or int(layer_idx) in layers


def _custom_fused_moe_enabled(layer_idx: int, proj_bits: dict) -> bool:
    if not config.custom_fused_moe():
        return False
    bits = config.custom_fused_moe_bits()
    if any(int(proj_bits[name]) != bits for name in ("gate_proj", "up_proj", "down_proj")):
        return False
    layers = _parse_layers_env("CUSTOM_FUSED_MOE_LAYERS")
    return layers is None or int(layer_idx) in layers


def _custom_qproj_targets() -> set[str]:
    spec = config.custom_qproj_targets()
    return {x.strip() for x in spec.split(",") if x.strip()}


def _custom_qlinear_indexed(x: mx.array, indices: mx.array, weight: mx.array,
                            scales: mx.array, biases: mx.array,
                            out_dim: int, in_dim: int, group_size: int,
                            bits: int, tile: int = 4) -> mx.array:
    """indexed custom qlinear: x[p,in] 使用 indices[p] 选择专家权重，输出 [p,out]。"""
    key = tile
    kernel = _CUSTOM_QKERNEL_CACHE.get(key)
    if kernel is None:
        source = r"""
            constexpr int lanes_per_row = 256 / rows_per_group;
            constexpr int block_size = rows_per_group * lanes_per_row;
            uint tid = thread_position_in_threadgroup.x;
            uint group_id = thread_position_in_grid.x / block_size;
            uint local_row = tid / lanes_per_row;
            uint lane = tid % lanes_per_row;
            uint global_row = group_id * rows_per_group + local_row;
            if (global_row >= pairs * out_dim) return;
            uint pair = global_row / out_dim;
            uint out_row = global_row % out_dim;
            uint expert = indices[pair];
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
                uint word = weight[base + word_idx];
                uint q = (word >> shift);
                if (shift + bits > 32) {
                    uint next_word = weight[base + word_idx + 1];
                    q |= (next_word << (32 - shift));
                }
                q = q & mask;
                int g = col / group_size;
                uint sb = (expert * out_dim + out_row) * groups_per_row + g;
                float wv = float(q) * scales[sb] + biases[sb];
                acc += wv * x[pair * in_dim + col];
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
                y[global_row] = partial[0 + (local_row * lanes_per_row)];
            }
        """
        kernel = mx.fast.metal_kernel(
            name=f"custom_qlinear_indexed_tile{tile}",
            input_names=["x", "indices", "weight", "scales", "biases"],
            output_names=["y"],
            source=source,
        )
        _CUSTOM_QKERNEL_CACHE[key] = kernel
    pairs = int(indices.size)
    grid_groups = (pairs * out_dim + tile - 1) // tile
    (y,) = kernel(
        inputs=[x, indices.astype(mx.uint32), weight, scales, biases],
        output_shapes=[(pairs, out_dim)],
        output_dtypes=[mx.float32],
        grid=(grid_groups * 256, 1, 1),
        threadgroup=(256, 1, 1),
        template=[
            ("pairs", pairs),
            ("out_dim", out_dim),
            ("in_dim", in_dim),
            ("group_size", group_size),
            ("bits", bits),
            ("rows_per_group", tile),
        ],
    )
    return y


def _custom_fused_moe_indexed(x: mx.array, indices: mx.array,
                              gate_w: mx.array, gate_s: mx.array, gate_b: mx.array,
                              up_w: mx.array, up_s: mx.array, up_b: mx.array,
                              down_w: mx.array, down_s: mx.array, down_b: mx.array,
                              hidden: int, moe_inter: int, group_size: int,
                              bits: int,
                              active_mask: "mx.array | None" = None) -> mx.array:
    """fused MoE expert: x[p] 用 indices[p] 选择专家，输出 down 后的 hidden。"""
    lanes_per_row = config.custom_fused_moe_lanes()
    block_size = config.custom_fused_moe_block()
    key = (hidden, moe_inter, group_size, bits, lanes_per_row, block_size)
    kernel = _CUSTOM_FUSED_MOE_CACHE.get(key)
    if kernel is None:
        source = r"""
            uint tid = thread_position_in_threadgroup.x;
            uint pair = thread_position_in_grid.x / block_size;
            if (pair >= pairs) return;
            if (active_mask[pair] == 0) {
                for (uint row = tid; row < hidden; row += block_size) {
                    y[pair * hidden + row] = 0.0f;
                }
                return;
            }
            uint expert = indices[pair];
            constexpr uint rows_per_step = block_size / lanes_per_row;
            uint local_row = tid / lanes_per_row;
            uint row_lane = tid % lanes_per_row;
            constexpr uint mask = (1u << bits) - 1u;
            constexpr int gu_words_per_row = (hidden * bits) / 32;
            constexpr int gu_groups_per_row = hidden / group_size;
            constexpr int down_words_per_row = (moe_inter * bits) / 32;
            constexpr int down_groups_per_row = moe_inter / group_size;
            threadgroup float act[1024];
            threadgroup float gate_part[block_size];
            threadgroup float up_part[block_size];

            // 多个 lane 协作计算同一 row，避免每个 dot 完全串行。
            for (uint row_base = 0; row_base < moe_inter; row_base += rows_per_step) {
                uint row = row_base + local_row;
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                if (row < moe_inter) {
                    for (uint col = row_lane; col < hidden; col += lanes_per_row) {
                        int bit_offset = int(col * bits);
                        int word_idx = bit_offset / 32;
                        int shift = bit_offset % 32;
                        uint gu_base = (expert * moe_inter + row) * gu_words_per_row;
                        uint qg = gate_w[gu_base + word_idx] >> shift;
                        uint qu = up_w[gu_base + word_idx] >> shift;
                        if (shift + bits > 32) {
                            qg |= gate_w[gu_base + word_idx + 1] << (32 - shift);
                            qu |= up_w[gu_base + word_idx + 1] << (32 - shift);
                        }
                        qg &= mask;
                        qu &= mask;
                        uint g = col / group_size;
                        uint sb = (expert * moe_inter + row) * gu_groups_per_row + g;
                        float xv = x[pair * hidden + col];
                        gate_acc += (float(qg) * gate_s[sb] + gate_b[sb]) * xv;
                        up_acc += (float(qu) * up_s[sb] + up_b[sb]) * xv;
                    }
                }
                gate_part[tid] = gate_acc;
                up_part[tid] = up_acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
                    if (row_lane < stride) {
                        gate_part[tid] += gate_part[tid + stride];
                        up_part[tid] += up_part[tid + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (row_lane == 0 && row < moe_inter) {
                    float gate_v = gate_part[tid];
                    float up_v = up_part[tid];
                    float sig = 1.0f / (1.0f + exp(-gate_v));
                    act[row] = gate_v * sig * up_v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // down projection 输出最终 hidden。
            for (uint row_base = 0; row_base < hidden; row_base += rows_per_step) {
                uint row = row_base + local_row;
                float acc = 0.0f;
                if (row < hidden) {
                    for (uint col = row_lane; col < moe_inter; col += lanes_per_row) {
                        int bit_offset = int(col * bits);
                        int word_idx = bit_offset / 32;
                        int shift = bit_offset % 32;
                        uint base = (expert * hidden + row) * down_words_per_row;
                        uint q = down_w[base + word_idx] >> shift;
                        if (shift + bits > 32) {
                            q |= down_w[base + word_idx + 1] << (32 - shift);
                        }
                        q &= mask;
                        uint g = col / group_size;
                        uint sb = (expert * hidden + row) * down_groups_per_row + g;
                        acc += (float(q) * down_s[sb] + down_b[sb]) * act[col];
                    }
                }
                gate_part[tid] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
                    if (row_lane < stride) {
                        gate_part[tid] += gate_part[tid + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (row_lane == 0 && row < hidden) {
                    y[pair * hidden + row] = gate_part[tid];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        """
        kernel = mx.fast.metal_kernel(
            name=f"custom_fused_moe_h{hidden}_i{moe_inter}_b{bits}_l{lanes_per_row}",
            input_names=[
                "x", "indices",
                "active_mask",
                "gate_w", "gate_s", "gate_b",
                "up_w", "up_s", "up_b",
                "down_w", "down_s", "down_b",
            ],
            output_names=["y"],
            source=source,
        )
        _CUSTOM_FUSED_MOE_CACHE[key] = kernel
    pairs = int(indices.size)
    if active_mask is None:
        active_mask = mx.ones((pairs,), dtype=mx.uint8)
    else:
        active_mask = active_mask.reshape(-1).astype(mx.uint8)
        if int(active_mask.size) != pairs:
            raise ValueError("active_mask must have one value per route pair")
    (y,) = kernel(
        inputs=[
            x, indices.astype(mx.uint32),
            active_mask,
            gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            down_w, down_s, down_b,
        ],
        output_shapes=[(pairs, hidden)],
        output_dtypes=[mx.float32],
        grid=(pairs * block_size, 1, 1),
        threadgroup=(block_size, 1, 1),
        template=[
            ("pairs", pairs),
            ("hidden", hidden),
            ("moe_inter", moe_inter),
            ("group_size", group_size),
            ("bits", bits),
            ("lanes_per_row", lanes_per_row),
            ("block_size", block_size),
        ],
    )
    return y
