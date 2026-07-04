#include <chrono>
#include <cstring>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "mlx/array.h"
#include "mlx/fast.h"
#include "mlx/ops.h"

using namespace mlx::core;

static int arg_int(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return std::stoi(argv[i + 1]);
    return fallback;
}

static std::string arg_str(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
    return fallback;
    }

static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t n = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<char> out(n);
    f.read(out.data(), n);
    return out;
}

static bool has_flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) if (argv[i] == name) return true;
    return false;
}

static int dtype_size(const std::string& s) {
    if (s == "U32" || s == "F32") return 4;
    if (s == "F16" || s == "BF16") return 2;
    std::cerr << "unsupported dtype size " << s << "\n";
    std::exit(3);
}

static void read_exact_at(int fd, char* dst, size_t nbytes, uint64_t offset) {
    size_t done = 0;
    while (done < nbytes) {
        ssize_t n = pread(fd, dst + done, nbytes - done, static_cast<off_t>(offset + done));
        if (n <= 0) {
            std::perror("pread");
            std::exit(4);
        }
        done += static_cast<size_t>(n);
    }
}

struct OpenFiles {
    int wfd{-1};
    int sfd{-1};
    int bfd{-1};
};

static OpenFiles open_buffers(const std::string& weight_path,
                              const std::string& scales_path,
                              const std::string& biases_path) {
    OpenFiles f;
    f.wfd = open(weight_path.c_str(), O_RDONLY);
    f.sfd = open(scales_path.c_str(), O_RDONLY);
    f.bfd = open(biases_path.c_str(), O_RDONLY);
    if (f.wfd < 0 || f.sfd < 0 || f.bfd < 0) {
        std::perror("open compact buffers");
        std::exit(5);
    }
    return f;
}

static void close_buffers(OpenFiles& f) {
    if (f.wfd >= 0) close(f.wfd);
    if (f.sfd >= 0) close(f.sfd);
    if (f.bfd >= 0) close(f.bfd);
}

struct MappedFiles {
    void* w{MAP_FAILED};
    void* s{MAP_FAILED};
    void* b{MAP_FAILED};
    size_t wn{0}, sn{0}, bn{0};
};

static size_t file_size_fd(int fd) {
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        std::perror("fstat");
        std::exit(6);
    }
    return static_cast<size_t>(st.st_size);
}

static MappedFiles mmap_buffers(OpenFiles& f) {
    MappedFiles m;
    m.wn = file_size_fd(f.wfd);
    m.sn = file_size_fd(f.sfd);
    m.bn = file_size_fd(f.bfd);
    m.w = mmap(nullptr, m.wn, PROT_READ, MAP_PRIVATE, f.wfd, 0);
    m.s = mmap(nullptr, m.sn, PROT_READ, MAP_PRIVATE, f.sfd, 0);
    m.b = mmap(nullptr, m.bn, PROT_READ, MAP_PRIVATE, f.bfd, 0);
    if (m.w == MAP_FAILED || m.s == MAP_FAILED || m.b == MAP_FAILED) {
        std::perror("mmap");
        std::exit(7);
    }
    return m;
}

static void unmap_buffers(MappedFiles& m) {
    if (m.w != MAP_FAILED) munmap(m.w, m.wn);
    if (m.s != MAP_FAILED) munmap(m.s, m.sn);
    if (m.b != MAP_FAILED) munmap(m.b, m.bn);
}

static void resize_compact_buffers(
    int active,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    int scale_dtype_size,
    std::vector<char>& wb,
    std::vector<char>& sb,
    std::vector<char>& bb) {
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    size_t weight_bytes = static_cast<size_t>(out_dim) * words * 4;
    size_t scale_bytes = static_cast<size_t>(out_dim) * groups * scale_dtype_size;
    wb.resize(static_cast<size_t>(active) * weight_bytes);
    sb.resize(static_cast<size_t>(active) * scale_bytes);
    bb.resize(static_cast<size_t>(active) * scale_bytes);
}

static void read_compact_buffers_merged(
    const OpenFiles& f,
    int active,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    int scale_dtype_size,
    std::vector<char>& wb,
    std::vector<char>& sb,
    std::vector<char>& bb) {
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    size_t weight_bytes = static_cast<size_t>(out_dim) * words * 4;
    size_t scale_bytes = static_cast<size_t>(out_dim) * groups * scale_dtype_size;
    // 当前 bench 使用 active experts = 0..active-1，因此可以合并成每个文件一次读取。
    read_exact_at(f.wfd, wb.data(), static_cast<size_t>(active) * weight_bytes, 0);
    read_exact_at(f.sfd, sb.data(), static_cast<size_t>(active) * scale_bytes, 0);
    read_exact_at(f.bfd, bb.data(), static_cast<size_t>(active) * scale_bytes, 0);
}

static void copy_compact_buffers_mmap(
    const MappedFiles& m,
    int active,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    int scale_dtype_size,
    std::vector<char>& wb,
    std::vector<char>& sb,
    std::vector<char>& bb) {
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    size_t weight_bytes = static_cast<size_t>(active) * out_dim * words * 4;
    size_t scale_bytes = static_cast<size_t>(active) * out_dim * groups * scale_dtype_size;
    std::memcpy(wb.data(), m.w, weight_bytes);
    std::memcpy(sb.data(), m.s, scale_bytes);
    std::memcpy(bb.data(), m.b, scale_bytes);
}

static std::vector<float> random_x(size_t n) {
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

static Dtype dtype_of(const std::string& s) {
    if (s == "U32") return uint32;
    if (s == "F32") return float32;
    if (s == "F16") return float16;
    if (s == "BF16") return bfloat16;
    std::cerr << "unsupported dtype " << s << "\n";
    std::exit(2);
}

static array custom_gather_qlinear_tiled(
    const array& x, const array& w, const array& scales, const array& biases,
    int experts, int out_dim, int in_dim, int group, int bits, int tile) {
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
            if (lane < stride) partial[tid] += partial[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) y[global_row] = partial[tid];
    )";
    auto kernel = fast::metal_kernel(
        "compute_buffer_qlinear_tiled", {"x", "w", "scales", "biases"}, {"y"}, source);
    int groups_grid = (experts * out_dim + tile - 1) / tile;
    auto outs = kernel(
        {x, w, scales, biases}, {Shape{experts, out_dim}}, {float32},
        {groups_grid * 256, 1, 1}, {256, 1, 1},
        {{"experts", experts}, {"out_dim", out_dim}, {"in_dim", in_dim},
         {"group_size", group}, {"bits", bits}, {"rows_per_group", tile}},
        std::nullopt, false, {});
    return outs[0];
}

int main(int argc, char** argv) {
    auto weight_path = arg_str(argc, argv, "--weight", "");
    auto scales_path = arg_str(argc, argv, "--scales", "");
    auto biases_path = arg_str(argc, argv, "--biases", "");
    auto scales_dtype = arg_str(argc, argv, "--scales-dtype", "BF16");
    int all_experts = arg_int(argc, argv, "--all-experts", 512);
    int active = arg_int(argc, argv, "--active", 16);
    int in_dim = arg_int(argc, argv, "--in", 2048);
    int out_dim = arg_int(argc, argv, "--out", 512);
    int group = arg_int(argc, argv, "--group", 128);
    int bits = arg_int(argc, argv, "--bits", 6);
    int repeat = arg_int(argc, argv, "--repeat", 30);
    int tile = arg_int(argc, argv, "--tile", 4);
    bool compact = has_flag(argc, argv, "--compact");
    bool use_mmap = has_flag(argc, argv, "--mmap");
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    auto xh = random_x(static_cast<size_t>(active) * in_dim);
    array x(xh.data(), Shape{active, in_dim}, float32);
    auto t0 = std::chrono::steady_clock::now();
    if (compact) {
        int sd = dtype_size(scales_dtype);
        std::vector<char> wb, sb, bb;
        resize_compact_buffers(active, out_dim, in_dim, group, bits, sd, wb, sb, bb);
        OpenFiles files = open_buffers(weight_path, scales_path, biases_path);
        MappedFiles mapped;
        if (use_mmap) mapped = mmap_buffers(files);
        for (int i = 0; i < repeat; ++i) {
            if (use_mmap) {
                copy_compact_buffers_mmap(mapped, active, out_dim, in_dim, group, bits, sd, wb, sb, bb);
            } else {
                read_compact_buffers_merged(files, active, out_dim, in_dim, group, bits, sd, wb, sb, bb);
            }
            array w(wb.data(), Shape{active, out_dim, words}, uint32);
            array scales(sb.data(), Shape{active, out_dim, groups}, dtype_of(scales_dtype));
            array biases(bb.data(), Shape{active, out_dim, groups}, dtype_of(scales_dtype));
            auto yy = custom_gather_qlinear_tiled(x, w, scales, biases, active, out_dim, in_dim, group, bits, tile);
            yy.eval();
        }
        if (use_mmap) unmap_buffers(mapped);
        close_buffers(files);
    } else {
        auto wb = read_file(weight_path);
        auto sb = read_file(scales_path);
        auto bb = read_file(biases_path);
        array w(wb.data(), Shape{all_experts, out_dim, words}, uint32);
        array scales(sb.data(), Shape{all_experts, out_dim, groups}, dtype_of(scales_dtype));
        array biases(bb.data(), Shape{all_experts, out_dim, groups}, dtype_of(scales_dtype));
        auto y = custom_gather_qlinear_tiled(x, w, scales, biases, active, out_dim, in_dim, group, bits, tile);
        y.eval();
        for (int i = 0; i < repeat; ++i) {
            auto yy = custom_gather_qlinear_tiled(x, w, scales, biases, active, out_dim, in_dim, group, bits, tile);
            yy.eval();
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double custom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
    std::cout << "{"
              << "\"all_experts\":" << all_experts
              << ",\"active\":" << active
              << ",\"in\":" << in_dim
              << ",\"out\":" << out_dim
              << ",\"bits\":" << bits
              << ",\"tile\":" << tile
              << ",\"compact\":" << (compact ? "true" : "false")
              << ",\"mmap\":" << (use_mmap ? "true" : "false")
              << ",\"custom_ms\":" << custom_ms
              << "}\n";
}
