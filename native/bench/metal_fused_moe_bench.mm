#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct FusedParams {
    int experts;
    int hidden;
    int inter;
    int group_size;
    int bits;
};

struct MappedProj {
    void* w{MAP_FAILED};
    void* s{MAP_FAILED};
    void* b{MAP_FAILED};
    size_t wn{0}, sn{0}, bn{0};
    int wfd{-1}, sfd{-1}, bfd{-1};
};

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

static size_t file_size(int fd) {
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        std::exit(2);
    }
    return static_cast<size_t>(st.st_size);
}

static void* map_file(const std::string& path, size_t& nbytes, int& fd) {
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(path.c_str());
        std::exit(3);
    }
    nbytes = file_size(fd);
    void* p = mmap(nullptr, nbytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        std::exit(4);
    }
    return p;
}

static MappedProj map_proj(const std::string& dir, int layer, const std::string& proj) {
    MappedProj m;
    std::string layer_name = "layer" + std::string(layer < 10 ? "0" : "") + std::to_string(layer);
    std::string base = dir + "/" + layer_name + "." + proj;
    m.w = map_file(base + ".weight.bin", m.wn, m.wfd);
    m.s = map_file(base + ".scales.bin", m.sn, m.sfd);
    m.b = map_file(base + ".biases.bin", m.bn, m.bfd);
    return m;
}

static void unmap_proj(MappedProj& m) {
    if (m.w != MAP_FAILED) munmap(m.w, m.wn);
    if (m.s != MAP_FAILED) munmap(m.s, m.sn);
    if (m.b != MAP_FAILED) munmap(m.b, m.bn);
    if (m.wfd >= 0) close(m.wfd);
    if (m.sfd >= 0) close(m.sfd);
    if (m.bfd >= 0) close(m.bfd);
}

static void copy_projection(
    const MappedProj& src,
    int active,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    id<MTLBuffer> wbuf,
    id<MTLBuffer> sbuf,
    id<MTLBuffer> bbuf) {
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    size_t weight_per = static_cast<size_t>(out_dim) * words * sizeof(uint32_t);
    size_t scale_per = static_cast<size_t>(out_dim) * groups * sizeof(uint16_t);
    // 当前 de-risk 读取 expert 0..active-1，runtime 版再补任意 expert id gather。
    std::memcpy([wbuf contents], src.w, static_cast<size_t>(active) * weight_per);
    std::memcpy([sbuf contents], src.s, static_cast<size_t>(active) * scale_per);
    std::memcpy([bbuf contents], src.b, static_cast<size_t>(active) * scale_per);
}

static std::vector<float> random_x(size_t n) {
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

static uint16_t fp32_to_bf16(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return static_cast<uint16_t>(u >> 16);
}

static void fill_synthetic(id<MTLBuffer> wbuf, id<MTLBuffer> sbuf, id<MTLBuffer> bbuf, uint32_t seed) {
    std::mt19937 rng(seed);
    auto* w = reinterpret_cast<uint32_t*>([wbuf contents]);
    size_t wn = [wbuf length] / sizeof(uint32_t);
    for (size_t i = 0; i < wn; ++i) w[i] = rng();

    auto* s = reinterpret_cast<uint16_t*>([sbuf contents]);
    auto* b = reinterpret_cast<uint16_t*>([bbuf contents]);
    size_t sn = [sbuf length] / sizeof(uint16_t);
    uint16_t scale = fp32_to_bf16(0.001f);
    uint16_t bias = fp32_to_bf16(-0.032f);
    for (size_t i = 0; i < sn; ++i) {
        s[i] = scale;
        b[i] = bias;
    }
}

static NSString* kernel_source() {
    return @R"(
    #include <metal_stdlib>
    using namespace metal;

    struct FusedParams {
      int experts;
      int hidden;
      int inter;
      int group_size;
      int bits;
    };

    inline float bf16_to_float(ushort v) {
      uint u = uint(v) << 16;
      return as_type<float>(u);
    }

    inline float qvalue(
      const device uint* w,
      const device ushort* s,
      const device ushort* b,
      uint expert,
      uint row,
      uint col,
      uint out_dim,
      uint in_dim,
      uint group_size,
      uint bits
    ) {
      uint words_per_row = (in_dim * bits) / 32;
      uint groups_per_row = in_dim / group_size;
      uint bit_offset = col * bits;
      uint word_idx = bit_offset / 32;
      uint shift = bit_offset % 32;
      uint base = (expert * out_dim + row) * words_per_row;
      uint word = w[base + word_idx];
      uint q = word >> shift;
      if (shift + bits > 32) {
        q |= w[base + word_idx + 1] << (32 - shift);
      }
      q &= (1u << bits) - 1u;
      uint group = col / group_size;
      uint sb = (expert * out_dim + row) * groups_per_row + group;
      return float(q) * bf16_to_float(s[sb]) + bf16_to_float(b[sb]);
    }

    kernel void fused_moe_one_expert(
      const device float* x [[buffer(0)]],
      const device uint* gate_w [[buffer(1)]],
      const device ushort* gate_s [[buffer(2)]],
      const device ushort* gate_b [[buffer(3)]],
      const device uint* up_w [[buffer(4)]],
      const device ushort* up_s [[buffer(5)]],
      const device ushort* up_b [[buffer(6)]],
      const device uint* down_w [[buffer(7)]],
      const device ushort* down_s [[buffer(8)]],
      const device ushort* down_b [[buffer(9)]],
      device float* y [[buffer(10)]],
      constant FusedParams& p [[buffer(11)]],
      uint tid [[thread_position_in_threadgroup]],
      uint gid [[thread_position_in_grid]]
    ) {
      constexpr uint block_size = 512;
      constexpr uint lanes_per_row = 16;
      constexpr uint rows_per_step = block_size / lanes_per_row;
      uint expert = gid / block_size;
      if (expert >= uint(p.experts)) return;
      uint lane = tid;
      threadgroup float act[1024];
      threadgroup float gate_part[block_size];
      threadgroup float up_part[block_size];

      // 多个 lane 协作计算同一个 row，减少每个 dot 的串行长度。
      uint local_row = tid / lanes_per_row;
      uint row_lane = tid % lanes_per_row;

      // 先计算 gate/up，并在 threadgroup 内得到 SwiGLU 激活。
      for (uint row_base = 0; row_base < uint(p.inter); row_base += rows_per_step) {
        uint j = row_base + local_row;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        if (j < uint(p.inter)) {
          for (uint col = row_lane; col < uint(p.hidden); col += lanes_per_row) {
            float xv = x[expert * p.hidden + col];
            gate_acc += qvalue(gate_w, gate_s, gate_b, expert, j, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
            up_acc += qvalue(up_w, up_s, up_b, expert, j, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
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
        if (row_lane == 0 && j < uint(p.inter)) {
          float gate_v = gate_part[tid];
          float up_v = up_part[tid];
          float sig = 1.0f / (1.0f + exp(-gate_v));
          act[j] = gate_v * sig * up_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // 再计算 down projection，输出最终 hidden。
      for (uint row_base = 0; row_base < uint(p.hidden); row_base += rows_per_step) {
        uint row = row_base + local_row;
        float acc = 0.0f;
        if (row < uint(p.hidden)) {
          for (uint col = row_lane; col < uint(p.inter); col += lanes_per_row) {
            acc += qvalue(down_w, down_s, down_b, expert, row, col, p.hidden, p.inter, p.group_size, p.bits) * act[col];
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
        if (row_lane == 0 && row < uint(p.hidden)) {
          y[expert * p.hidden + row] = gate_part[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    )";
}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string dir = arg_str(argc, argv, "--dir", "/tmp/qwen_compute_buffers");
        int layer = arg_int(argc, argv, "--layer", 43);
        int active = arg_int(argc, argv, "--active", 16);
        int hidden = arg_int(argc, argv, "--hidden", 2048);
        int inter = arg_int(argc, argv, "--inter", 512);
        int group = arg_int(argc, argv, "--group", 128);
        int bits = arg_int(argc, argv, "--bits", 6);
        int repeat = arg_int(argc, argv, "--repeat", 20);
        int synthetic = arg_int(argc, argv, "--synthetic", 0);

        MappedProj gate;
        MappedProj up;
        MappedProj down;
        if (!synthetic) {
            gate = map_proj(dir, layer, "gate_proj");
            up = map_proj(dir, layer, "up_proj");
            down = map_proj(dir, layer, "down_proj");
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernel_source() options:nil error:&error];
        if (!library) {
            std::cerr << [[error localizedDescription] UTF8String] << "\n";
            return 5;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"fused_moe_one_expert"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            std::cerr << [[error localizedDescription] UTF8String] << "\n";
            return 6;
        }

        int gu_words = (hidden * bits) / 32;
        int gu_groups = hidden / group;
        int down_words = (inter * bits) / 32;
        int down_groups = inter / group;
        size_t gu_w_bytes = static_cast<size_t>(active) * inter * gu_words * sizeof(uint32_t);
        size_t gu_s_bytes = static_cast<size_t>(active) * inter * gu_groups * sizeof(uint16_t);
        size_t down_w_bytes = static_cast<size_t>(active) * hidden * down_words * sizeof(uint32_t);
        size_t down_s_bytes = static_cast<size_t>(active) * hidden * down_groups * sizeof(uint16_t);

        std::vector<float> xh = random_x(static_cast<size_t>(active) * hidden);
        id<MTLBuffer> xbuf = [device newBufferWithBytes:xh.data()
                                                 length:xh.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> gw = [device newBufferWithLength:gu_w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> gs = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> gb = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> uw = [device newBufferWithLength:gu_w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> us = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ub = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> dw = [device newBufferWithLength:down_w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ds = [device newBufferWithLength:down_s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> db = [device newBufferWithLength:down_s_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> ybuf = [device newBufferWithLength:static_cast<size_t>(active) * hidden * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        FusedParams params{active, hidden, inter, group, bits};
        id<MTLBuffer> pbuf = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        if (synthetic) {
            fill_synthetic(gw, gs, gb, 1);
            fill_synthetic(uw, us, ub, 2);
            fill_synthetic(dw, ds, db, 3);
        } else {
            copy_projection(gate, active, inter, hidden, group, bits, gw, gs, gb);
            copy_projection(up, active, inter, hidden, group, bits, uw, us, ub);
            copy_projection(down, active, hidden, inter, group, bits, dw, ds, db);
        }

        auto run_once = [&]() {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:xbuf offset:0 atIndex:0];
            [enc setBuffer:gw offset:0 atIndex:1];
            [enc setBuffer:gs offset:0 atIndex:2];
            [enc setBuffer:gb offset:0 atIndex:3];
            [enc setBuffer:uw offset:0 atIndex:4];
            [enc setBuffer:us offset:0 atIndex:5];
            [enc setBuffer:ub offset:0 atIndex:6];
            [enc setBuffer:dw offset:0 atIndex:7];
            [enc setBuffer:ds offset:0 atIndex:8];
            [enc setBuffer:db offset:0 atIndex:9];
            [enc setBuffer:ybuf offset:0 atIndex:10];
            [enc setBuffer:pbuf offset:0 atIndex:11];
            MTLSize tg = MTLSizeMake(512, 1, 1);
            MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(active * 512), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };

        run_once();
        float* y = reinterpret_cast<float*>([ybuf contents]);
        double checksum = 0.0;
        for (int i = 0; i < active * hidden; i += std::max(1, hidden / 16)) checksum += y[i];

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; ++r) run_once();
        auto t1 = std::chrono::steady_clock::now();
        double kernel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;

        auto s0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; ++r) {
            if (synthetic) {
                fill_synthetic(gw, gs, gb, 1);
                fill_synthetic(uw, us, ub, 2);
                fill_synthetic(dw, ds, db, 3);
            } else {
                copy_projection(gate, active, inter, hidden, group, bits, gw, gs, gb);
                copy_projection(up, active, inter, hidden, group, bits, uw, us, ub);
                copy_projection(down, active, hidden, inter, group, bits, dw, ds, db);
            }
        }
        auto s1 = std::chrono::steady_clock::now();
        double stage_ms = std::chrono::duration<double, std::milli>(s1 - s0).count() / repeat;

        std::cout << "fused_moe layer=" << layer
                  << " active=" << active
                  << " synthetic=" << synthetic
                  << " bits=" << bits
                  << " hidden=" << hidden
                  << " inter=" << inter
                  << " stage_ms=" << stage_ms
                  << " kernel_ms=" << kernel_ms
                  << " total_ms=" << (stage_ms + kernel_ms)
                  << " checksum=" << checksum
                  << "\n";

        if (!synthetic) {
            unmap_proj(gate);
            unmap_proj(up);
            unmap_proj(down);
        }
    }
    return 0;
}
