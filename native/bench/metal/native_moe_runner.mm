#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
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

struct ProjectionShape {
    int out_dim;
    int in_dim;
    int words;
    int groups;
    size_t weight_per;
    size_t scale_per;
};

struct StageBuffers {
    id<MTLBuffer> gw;
    id<MTLBuffer> gs;
    id<MTLBuffer> gb;
    id<MTLBuffer> uw;
    id<MTLBuffer> us;
    id<MTLBuffer> ub;
    id<MTLBuffer> dw;
    id<MTLBuffer> ds;
    id<MTLBuffer> db;
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

static bool has_arg(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) return true;
    }
    return false;
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

static ProjectionShape proj_shape(int out_dim, int in_dim, int group, int bits) {
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    return ProjectionShape{
        out_dim,
        in_dim,
        words,
        groups,
        static_cast<size_t>(out_dim) * words * sizeof(uint32_t),
        static_cast<size_t>(out_dim) * groups * sizeof(uint16_t),
    };
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

static std::vector<int> parse_ints(const std::string& text) {
    std::vector<int> out;
    int sign = 1;
    int value = 0;
    bool in_num = false;
    for (char ch : text) {
        if (ch == '-' && !in_num) {
            sign = -1;
        } else if (std::isdigit(static_cast<unsigned char>(ch))) {
            value = value * 10 + (ch - '0');
            in_num = true;
        } else if (in_num) {
            out.push_back(sign * value);
            sign = 1;
            value = 0;
            in_num = false;
        } else {
            sign = 1;
        }
    }
    if (in_num) out.push_back(sign * value);
    return out;
}

static std::vector<std::vector<int>> build_steps(
    int argc,
    char** argv,
    int steps,
    int active,
    int all_experts) {
    std::string trace = arg_str(argc, argv, "--trace", "");
    if (!trace.empty()) {
        std::ifstream f(trace);
        if (!f) {
            std::cerr << "failed to open trace: " << trace << "\n";
            std::exit(7);
        }
        std::vector<std::vector<int>> rows;
        std::string line;
        while (std::getline(f, line)) {
            auto ids = parse_ints(line);
            if (!ids.empty()) {
                ids.resize(active, ids.back());
                rows.push_back(std::vector<int>(ids.begin(), ids.begin() + active));
            }
        }
        return rows;
    }
    std::string experts = arg_str(argc, argv, "--experts", "");
    if (!experts.empty()) {
        auto ids = parse_ints(experts);
        if (ids.empty()) {
            std::cerr << "--experts did not contain any ids\n";
            std::exit(8);
        }
        ids.resize(active, ids.back());
        std::vector<int> row(ids.begin(), ids.begin() + active);
        return std::vector<std::vector<int>>(steps, row);
    }
    std::vector<std::vector<int>> rows;
    for (int step = 0; step < steps; ++step) {
        std::vector<int> row;
        int start = (step * active) % std::max(1, all_experts - active + 1);
        for (int i = 0; i < active; ++i) row.push_back((start + i) % all_experts);
        rows.push_back(row);
    }
    return rows;
}

static void copy_one_projection(
    const MappedProj& src,
    const ProjectionShape& shape,
    const std::vector<int>& ids,
    id<MTLBuffer> wbuf,
    id<MTLBuffer> sbuf,
    id<MTLBuffer> bbuf) {
    auto* wd = static_cast<char*>([wbuf contents]);
    auto* sd = static_cast<char*>([sbuf contents]);
    auto* bd = static_cast<char*>([bbuf contents]);
    auto* ws = static_cast<const char*>(src.w);
    auto* ss = static_cast<const char*>(src.s);
    auto* bs = static_cast<const char*>(src.b);
    for (size_t local = 0; local < ids.size(); ++local) {
        size_t expert = static_cast<size_t>(ids[local]);
        std::memcpy(wd + local * shape.weight_per, ws + expert * shape.weight_per, shape.weight_per);
        std::memcpy(sd + local * shape.scale_per, ss + expert * shape.scale_per, shape.scale_per);
        std::memcpy(bd + local * shape.scale_per, bs + expert * shape.scale_per, shape.scale_per);
    }
}

static void fill_synthetic_projection(
    id<MTLBuffer> wbuf,
    id<MTLBuffer> sbuf,
    id<MTLBuffer> bbuf,
    uint32_t seed) {
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

    kernel void fused_moe_staged(
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
      uint local_row = tid / lanes_per_row;
      uint row_lane = tid % lanes_per_row;
      threadgroup float act[1024];
      threadgroup float gate_part[block_size];
      threadgroup float up_part[block_size];

      // gate/up 两个投影共用 x，先在 threadgroup 中生成 SwiGLU 激活。
      for (uint row_base = 0; row_base < uint(p.inter); row_base += rows_per_step) {
        uint row = row_base + local_row;
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        if (row < uint(p.inter)) {
          for (uint col = row_lane; col < uint(p.hidden); col += lanes_per_row) {
            float xv = x[expert * p.hidden + col];
            gate_acc += qvalue(gate_w, gate_s, gate_b, expert, row, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
            up_acc += qvalue(up_w, up_s, up_b, expert, row, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
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
        if (row_lane == 0 && row < uint(p.inter)) {
          float gate_v = gate_part[tid];
          float up_v = up_part[tid];
          act[row] = gate_v * (1.0f / (1.0f + exp(-gate_v))) * up_v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // down 投影输出每个 staged expert 的 hidden。
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
          if (row_lane < stride) gate_part[tid] += gate_part[tid + stride];
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

static StageBuffers make_stage_buffers(
    id<MTLDevice> device,
    int active,
    const ProjectionShape& gu,
    const ProjectionShape& down) {
    StageBuffers b;
    size_t gu_w_bytes = static_cast<size_t>(active) * gu.weight_per;
    size_t gu_s_bytes = static_cast<size_t>(active) * gu.scale_per;
    size_t down_w_bytes = static_cast<size_t>(active) * down.weight_per;
    size_t down_s_bytes = static_cast<size_t>(active) * down.scale_per;
    b.gw = [device newBufferWithLength:gu_w_bytes options:MTLResourceStorageModeShared];
    b.gs = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
    b.gb = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
    b.uw = [device newBufferWithLength:gu_w_bytes options:MTLResourceStorageModeShared];
    b.us = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
    b.ub = [device newBufferWithLength:gu_s_bytes options:MTLResourceStorageModeShared];
    b.dw = [device newBufferWithLength:down_w_bytes options:MTLResourceStorageModeShared];
    b.ds = [device newBufferWithLength:down_s_bytes options:MTLResourceStorageModeShared];
    b.db = [device newBufferWithLength:down_s_bytes options:MTLResourceStorageModeShared];
    return b;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string dir = arg_str(argc, argv, "--dir", "/tmp/qwen_compute_buffers");
        int layer = arg_int(argc, argv, "--layer", 43);
        int active = arg_int(argc, argv, "--active", 16);
        int steps = arg_int(argc, argv, "--steps", 16);
        int all_experts = arg_int(argc, argv, "--all-experts", 512);
        int hidden = arg_int(argc, argv, "--hidden", 2048);
        int inter = arg_int(argc, argv, "--inter", 512);
        int group = arg_int(argc, argv, "--group", 128);
        int bits = arg_int(argc, argv, "--bits", 6);
        int repeat = arg_int(argc, argv, "--repeat", 10);
        bool synthetic = arg_int(argc, argv, "--synthetic", 0) != 0;
        (void)has_arg; // 保留给后续命令行布尔参数扩展。

        auto step_ids = build_steps(argc, argv, steps, active, all_experts);
        steps = static_cast<int>(step_ids.size());
        if (steps <= 0 || active <= 0) {
            std::cerr << "steps and active must be positive\n";
            return 9;
        }

        ProjectionShape gu = proj_shape(inter, hidden, group, bits);
        ProjectionShape down = proj_shape(hidden, inter, group, bits);
        MappedProj gate;
        MappedProj up;
        MappedProj down_m;
        if (!synthetic) {
            gate = map_proj(dir, layer, "gate_proj");
            up = map_proj(dir, layer, "up_proj");
            down_m = map_proj(dir, layer, "down_proj");
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernel_source() options:nil error:&error];
        if (!library) {
            std::cerr << [[error localizedDescription] UTF8String] << "\n";
            return 5;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"fused_moe_staged"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            std::cerr << [[error localizedDescription] UTF8String] << "\n";
            return 6;
        }

        std::vector<float> xh = random_x(static_cast<size_t>(active) * hidden);
        id<MTLBuffer> xbuf = [device newBufferWithBytes:xh.data()
                                                 length:xh.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> ybuf = [device newBufferWithLength:static_cast<size_t>(active) * hidden * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        FusedParams params{active, hidden, inter, group, bits};
        id<MTLBuffer> pbuf = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

        StageBuffers slots[2] = {
            make_stage_buffers(device, active, gu, down),
            make_stage_buffers(device, active, gu, down),
        };

        auto stage = [&](int step, int slot_idx) {
            StageBuffers& b = slots[slot_idx];
            if (synthetic) {
                uint32_t base = static_cast<uint32_t>(step * 17 + slot_idx * 101);
                fill_synthetic_projection(b.gw, b.gs, b.gb, base + 1);
                fill_synthetic_projection(b.uw, b.us, b.ub, base + 2);
                fill_synthetic_projection(b.dw, b.ds, b.db, base + 3);
            } else {
                const auto& ids = step_ids[step % steps];
                copy_one_projection(gate, gu, ids, b.gw, b.gs, b.gb);
                copy_one_projection(up, gu, ids, b.uw, b.us, b.ub);
                copy_one_projection(down_m, down, ids, b.dw, b.ds, b.db);
            }
        };

        auto compute = [&](int slot_idx) {
            StageBuffers& b = slots[slot_idx];
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:xbuf offset:0 atIndex:0];
            [enc setBuffer:b.gw offset:0 atIndex:1];
            [enc setBuffer:b.gs offset:0 atIndex:2];
            [enc setBuffer:b.gb offset:0 atIndex:3];
            [enc setBuffer:b.uw offset:0 atIndex:4];
            [enc setBuffer:b.us offset:0 atIndex:5];
            [enc setBuffer:b.ub offset:0 atIndex:6];
            [enc setBuffer:b.dw offset:0 atIndex:7];
            [enc setBuffer:b.ds offset:0 atIndex:8];
            [enc setBuffer:b.db offset:0 atIndex:9];
            [enc setBuffer:ybuf offset:0 atIndex:10];
            [enc setBuffer:pbuf offset:0 atIndex:11];
            MTLSize tg = MTLSizeMake(512, 1, 1);
            MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(active * 512), 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };

        // 预热 Metal pipeline 和 page cache。
        stage(0, 0);
        compute(0);

        double stage_ms = 0.0;
        double kernel_ms = 0.0;
        auto sync_t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; ++r) {
            for (int step = 0; step < steps; ++step) {
                auto s0 = std::chrono::steady_clock::now();
                stage(step, 0);
                auto s1 = std::chrono::steady_clock::now();
                compute(0);
                auto s2 = std::chrono::steady_clock::now();
                stage_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
                kernel_ms += std::chrono::duration<double, std::milli>(s2 - s1).count();
            }
        }
        auto sync_t1 = std::chrono::steady_clock::now();
        int n = std::max(1, repeat * steps);
        stage_ms /= n;
        kernel_ms /= n;
        double sync_ms = std::chrono::duration<double, std::milli>(sync_t1 - sync_t0).count() / n;

        auto async_t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < repeat; ++r) {
            std::mutex mu;
            std::condition_variable cv;
            bool has_job = false;
            bool done_job = false;
            bool stop = false;
            int job_step = 0;
            int job_slot = 0;
            std::thread worker([&]() {
                while (true) {
                    int step = 0;
                    int slot = 0;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&]() { return has_job || stop; });
                        if (stop) return;
                        step = job_step;
                        slot = job_slot;
                        has_job = false;
                    }
                    stage(step, slot);
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        done_job = true;
                    }
                    cv.notify_all();
                }
            });
            auto submit = [&](int step, int slot) {
                std::lock_guard<std::mutex> lk(mu);
                job_step = step;
                job_slot = slot;
                done_job = false;
                has_job = true;
                cv.notify_all();
            };
            auto wait_ready = [&]() {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return done_job; });
            };
            submit(0, 0);
            wait_ready();
            for (int step = 0; step < steps; ++step) {
                int cur = step % 2;
                int next = 1 - cur;
                if (step + 1 < steps) submit(step + 1, next);
                compute(cur);
                if (step + 1 < steps) wait_ready();
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
            }
            cv.notify_all();
            worker.join();
        }
        auto async_t1 = std::chrono::steady_clock::now();
        double async_ms = std::chrono::duration<double, std::milli>(async_t1 - async_t0).count() / n;

        compute(0);
        float* y = reinterpret_cast<float*>([ybuf contents]);
        double checksum = 0.0;
        for (int i = 0; i < active * hidden; i += std::max(1, hidden / 16)) checksum += y[i];
        bool checksum_ok = std::isfinite(checksum);
        double total_ms = async_ms;
        double overlap_gain = sync_ms / std::max(async_ms, 1e-9);

        std::cout << "{"
                  << "\"layer\":" << layer
                  << ",\"active\":" << active
                  << ",\"steps\":" << steps
                  << ",\"bits\":" << bits
                  << ",\"hidden\":" << hidden
                  << ",\"inter\":" << inter
                  << ",\"synthetic\":" << (synthetic ? "true" : "false")
                  << ",\"async\":true"
                  << ",\"stage_ms\":" << stage_ms
                  << ",\"kernel_ms\":" << kernel_ms
                  << ",\"sync_ms\":" << sync_ms
                  << ",\"async_ms\":" << async_ms
                  << ",\"total_ms\":" << total_ms
                  << ",\"overlap_gain\":" << overlap_gain
                  << ",\"checksum\":" << checksum
                  << ",\"checksum_ok\":" << (checksum_ok ? "true" : "false")
                  << "}\n";

        if (!synthetic) {
            unmap_proj(gate);
            unmap_proj(up);
            unmap_proj(down_m);
        }
    }
    return 0;
}
