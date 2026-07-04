#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

using namespace mlx::core;

static int arg_int(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return std::stoi(argv[i + 1]);
    }
    return fallback;
}

static std::string arg_str(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return argv[i + 1];
    }
    return fallback;
}

static std::vector<float> make_random_float(size_t n, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

static void quantize_affine(
    const std::vector<float>& w,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    std::vector<uint32_t>& packed,
    std::vector<float>& scales,
    std::vector<float>& biases) {
    int groups = in_dim / group;
    int words = (in_dim * bits) / 32;
    int qmax = (1 << bits) - 1;
    packed.assign(out_dim * words, 0);
    scales.assign(out_dim * groups, 0);
    biases.assign(out_dim * groups, 0);
    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups; ++g) {
            float mn = w[o * in_dim + g * group];
            float mx = mn;
            for (int i = 0; i < group; ++i) {
                float v = w[o * in_dim + g * group + i];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            float scale = (mx - mn) / std::max(1, qmax);
            if (scale == 0) scale = 1.0f;
            scales[o * groups + g] = scale;
            biases[o * groups + g] = mn;
            for (int i = 0; i < group; ++i) {
                int col = g * group + i;
                int q = static_cast<int>(std::round((w[o * in_dim + col] - mn) / scale));
                q = std::max(0, std::min(qmax, q));
                int bit_offset = col * bits;
                int word = bit_offset / 32;
                int shift = bit_offset % 32;
                uint32_t uq = static_cast<uint32_t>(q);
                packed[o * words + word] |= (uq << shift);
                if (shift + bits > 32) {
                    packed[o * words + word + 1] |= (uq >> (32 - shift));
                }
            }
        }
    }
}

static array custom_qlinear(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int out_dim,
    int in_dim,
    int group,
    int bits) {
    std::string source = R"(
        constexpr int block_size = 256;
        uint row = thread_position_in_grid.x / block_size;
        uint tid = thread_position_in_threadgroup.x;
        if (row >= out_dim) return;
        constexpr uint mask = (1u << bits) - 1u;
        constexpr int words_per_row = (in_dim * bits) / 32;
        threadgroup float partial[block_size];
        float acc = 0.0f;
        for (int col = int(tid); col < in_dim; col += block_size) {
            int bit_offset = col * bits;
            int word_idx = bit_offset / 32;
            int shift = bit_offset % 32;
            uint word = w[row * words_per_row + word_idx];
            uint q = (word >> shift);
            if (shift + bits > 32) {
                uint next_word = w[row * words_per_row + word_idx + 1];
                q |= (next_word << (32 - shift));
            }
            q = q & mask;
            int g = col / group_size;
            float weight = float(q) * scales[row * (in_dim / group_size) + g] + biases[row * (in_dim / group_size) + g];
            acc += weight * x[col];
        }
        partial[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            y[row] = partial[0];
        }
    )";
    auto kernel = fast::metal_kernel(
        "qlinear_one_expert",
        {"x", "w", "scales", "biases"},
        {"y"},
        source);
    auto outs = kernel(
        {x, w, scales, biases},
        {Shape{out_dim}},
        {float32},
        {static_cast<int>(out_dim * 256), 1, 1},
        {256, 1, 1},
        {
            {"out_dim", out_dim},
            {"in_dim", in_dim},
            {"group_size", group},
            {"bits", bits},
        },
        std::nullopt,
        false,
        {});
    return outs[0];
}

static array custom_gather_qlinear(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int experts,
    int out_dim,
    int in_dim,
    int group,
    int bits) {
    std::string source = R"(
        constexpr int block_size = 256;
        uint row = thread_position_in_grid.x / block_size;
        uint tid = thread_position_in_threadgroup.x;
        if (row >= experts * out_dim) return;
        uint expert = row / out_dim;
        uint out_row = row % out_dim;
        constexpr uint mask = (1u << bits) - 1u;
        constexpr int words_per_row = (in_dim * bits) / 32;
        constexpr int groups_per_row = in_dim / group_size;
        threadgroup float partial[block_size];
        float acc = 0.0f;
        for (int col = int(tid); col < in_dim; col += block_size) {
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
        for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            y[row] = partial[0];
        }
    )";
    auto kernel = fast::metal_kernel(
        "qlinear_multi_expert",
        {"x", "w", "scales", "biases"},
        {"y"},
        source);
    auto outs = kernel(
        {x, w, scales, biases},
        {Shape{experts, out_dim}},
        {float32},
        {static_cast<int>(experts * out_dim * 256), 1, 1},
        {256, 1, 1},
        {
            {"experts", experts},
            {"out_dim", out_dim},
            {"in_dim", in_dim},
            {"group_size", group},
            {"bits", bits},
        },
        std::nullopt,
        false,
        {});
    return outs[0];
}

static array custom_gather_qlinear_tiled(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    int experts,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    int tile) {
    std::string source = R"(
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
    )";
    auto kernel = fast::metal_kernel(
        "qlinear_multi_expert_tiled",
        {"x", "w", "scales", "biases"},
        {"y"},
        source);
    int groups_grid = (experts * out_dim + tile - 1) / tile;
    auto outs = kernel(
        {x, w, scales, biases},
        {Shape{experts, out_dim}},
        {float32},
        {groups_grid * 256, 1, 1},
        {256, 1, 1},
        {
            {"experts", experts},
            {"out_dim", out_dim},
            {"in_dim", in_dim},
            {"group_size", group},
            {"bits", bits},
            {"rows_per_group", tile},
        },
        std::nullopt,
        false,
        {});
    return outs[0];
}

int main(int argc, char** argv) {
    int in_dim = arg_int(argc, argv, "--in", 2048);
    int out_dim = arg_int(argc, argv, "--out", 512);
    int group = arg_int(argc, argv, "--group", 128);
    int bits = arg_int(argc, argv, "--bits", 4);
    int repeat = arg_int(argc, argv, "--repeat", 50);
    int experts = arg_int(argc, argv, "--experts", 1);
    auto variant = arg_str(argc, argv, "--variant", "row");
    auto x_host = make_random_float(static_cast<size_t>(experts) * in_dim, 0.1f, 1);
    auto w_full = make_random_float(static_cast<size_t>(experts) * out_dim * in_dim, 0.02f, 2);
    std::vector<uint32_t> wq;
    std::vector<float> sc, bs;
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    wq.resize(static_cast<size_t>(experts) * out_dim * words);
    sc.resize(static_cast<size_t>(experts) * out_dim * groups);
    bs.resize(static_cast<size_t>(experts) * out_dim * groups);
    for (int e = 0; e < experts; ++e) {
        std::vector<float> one_w(
            w_full.begin() + static_cast<size_t>(e) * out_dim * in_dim,
            w_full.begin() + static_cast<size_t>(e + 1) * out_dim * in_dim);
        std::vector<uint32_t> one_q;
        std::vector<float> one_s, one_b;
        quantize_affine(one_w, out_dim, in_dim, group, bits, one_q, one_s, one_b);
        std::copy(one_q.begin(), one_q.end(), wq.begin() + static_cast<size_t>(e) * out_dim * words);
        std::copy(one_s.begin(), one_s.end(), sc.begin() + static_cast<size_t>(e) * out_dim * groups);
        std::copy(one_b.begin(), one_b.end(), bs.begin() + static_cast<size_t>(e) * out_dim * groups);
    }
    array x(x_host.data(), Shape{experts, in_dim}, float32);
    array w(wq.data(), Shape{experts, out_dim, words}, uint32);
    array scales(sc.data(), Shape{experts, out_dim, groups}, float32);
    array biases(bs.data(), Shape{experts, out_dim, groups}, float32);
    std::vector<array> x_one, w_one, s_one, b_one;
    x_one.reserve(experts);
    w_one.reserve(experts);
    s_one.reserve(experts);
    b_one.reserve(experts);
    for (int e = 0; e < experts; ++e) {
        x_one.emplace_back(x_host.data() + static_cast<size_t>(e) * in_dim, Shape{1, in_dim}, float32);
        w_one.emplace_back(wq.data() + static_cast<size_t>(e) * out_dim * words, Shape{out_dim, words}, uint32);
        s_one.emplace_back(sc.data() + static_cast<size_t>(e) * out_dim * groups, Shape{out_dim, groups}, float32);
        b_one.emplace_back(bs.data() + static_cast<size_t>(e) * out_dim * groups, Shape{out_dim, groups}, float32);
    }
    std::vector<array> y_parts;
    for (int e = 0; e < experts; ++e) {
        y_parts.push_back(quantized_matmul(
            x_one[e], w_one[e], s_one[e], b_one[e], true, group, bits, "affine"));
    }
    auto y_mlx = concatenate(y_parts, 0);
    y_mlx.eval();
    int tile = 1;
    if (variant == "tile2") tile = 2;
    else if (variant == "tile4") tile = 4;
    else if (variant == "tile8") tile = 8;
    auto y_custom = (tile > 1)
        ? custom_gather_qlinear_tiled(x, w, scales, biases, experts, out_dim, in_dim, group, bits, tile)
        : custom_gather_qlinear(x, w, scales, biases, experts, out_dim, in_dim, group, bits);
    y_custom.eval();
    auto* a = y_mlx.data<float>();
    auto* b = y_custom.data<float>();
    double max_abs = 0.0;
    for (int i = 0; i < experts * out_dim; ++i) {
        max_abs = std::max(max_abs, static_cast<double>(std::abs(a[i] - b[i])));
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        std::vector<array> ys;
        for (int e = 0; e < experts; ++e) {
            ys.push_back(quantized_matmul(
                x_one[e], w_one[e], s_one[e], b_one[e], true, group, bits, "affine"));
        }
        auto y = concatenate(ys, 0);
        y.eval();
    }
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i) {
        auto y = (tile > 1)
            ? custom_gather_qlinear_tiled(x, w, scales, biases, experts, out_dim, in_dim, group, bits, tile)
            : custom_gather_qlinear(x, w, scales, biases, experts, out_dim, in_dim, group, bits);
        y.eval();
    }
    auto t2 = std::chrono::steady_clock::now();
    double mlx_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
    double custom_ms = std::chrono::duration<double, std::milli>(t2 - t1).count() / repeat;
    std::cout << "{"
              << "\"in\":" << in_dim
              << ",\"out\":" << out_dim
              << ",\"experts\":" << experts
              << ",\"variant\":\"" << variant << "\""
              << ",\"group\":" << group
              << ",\"bits\":" << bits
              << ",\"repeat\":" << repeat
              << ",\"mlx_ms\":" << mlx_ms
              << ",\"custom_ms\":" << custom_ms
              << ",\"custom_vs_mlx\":" << (mlx_ms / custom_ms)
              << ",\"max_abs\":" << max_abs
              << "}\n";
    return 0;
}
