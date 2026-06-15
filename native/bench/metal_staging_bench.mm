#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

struct Params {
    int experts;
    int out_dim;
    int in_dim;
    int group_size;
    int bits;
    int rows_per_group;
};

static int arg_int(int argc, char** argv, const std::string& name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return std::stoi(argv[i + 1]);
    return fallback;
}

static std::string arg_str(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) if (argv[i] == name) return argv[i + 1];
    return fallback;
}

static bool has_flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) if (argv[i] == name) return true;
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
        perror("open");
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

static std::vector<float> random_x(size_t n) {
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    std::vector<float> out(n);
    for (auto& v : out) v = dist(rng);
    return out;
}

static void copy_stage(
    void* wm,
    void* sm,
    void* bm,
    int start_expert,
    int active,
    int out_dim,
    int in_dim,
    int group,
    int bits,
    id<MTLBuffer> wbuf,
    id<MTLBuffer> sbuf,
    id<MTLBuffer> bbuf
) {
    int words = (in_dim * bits) / 32;
    int groups = in_dim / group;
    size_t weight_per = static_cast<size_t>(out_dim) * words * 4;
    size_t scale_per = static_cast<size_t>(out_dim) * groups * 2;
    std::memcpy([wbuf contents], static_cast<char*>(wm) + static_cast<size_t>(start_expert) * weight_per,
                static_cast<size_t>(active) * weight_per);
    std::memcpy([sbuf contents], static_cast<char*>(sm) + static_cast<size_t>(start_expert) * scale_per,
                static_cast<size_t>(active) * scale_per);
    std::memcpy([bbuf contents], static_cast<char*>(bm) + static_cast<size_t>(start_expert) * scale_per,
                static_cast<size_t>(active) * scale_per);
}

static NSString* kernel_source() {
    return @R"(
    #include <metal_stdlib>
    using namespace metal;

    struct Params {
      int experts;
      int out_dim;
      int in_dim;
      int group_size;
      int bits;
      int rows_per_group;
    };

    inline float bf16_to_float(ushort v) {
      uint u = uint(v) << 16;
      return as_type<float>(u);
    }

    kernel void qlinear_stage(
      const device float* x [[buffer(0)]],
      const device uint* w [[buffer(1)]],
      const device ushort* scales [[buffer(2)]],
      const device ushort* biases [[buffer(3)]],
      device float* y [[buffer(4)]],
      constant Params& p [[buffer(5)]],
      uint tid [[thread_position_in_threadgroup]],
      uint gid [[thread_position_in_grid]]
    ) {
      constexpr int block_size = 256;
      int lanes_per_row = block_size / p.rows_per_group;
      uint group_id = gid / block_size;
      uint local_row = tid / lanes_per_row;
      uint lane = tid % lanes_per_row;
      uint global_row = group_id * p.rows_per_group + local_row;
      if (global_row >= uint(p.experts * p.out_dim)) return;
      uint expert = global_row / p.out_dim;
      uint out_row = global_row % p.out_dim;
      uint mask = (1u << p.bits) - 1u;
      int words_per_row = (p.in_dim * p.bits) / 32;
      int groups_per_row = p.in_dim / p.group_size;
      threadgroup float partial[block_size];
      float acc = 0.0f;
      for (int col = int(lane); col < p.in_dim; col += lanes_per_row) {
        int bit_offset = col * p.bits;
        int word_idx = bit_offset / 32;
        int shift = bit_offset % 32;
        uint base = (expert * p.out_dim + out_row) * words_per_row;
        uint word = w[base + word_idx];
        uint q = word >> shift;
        if (shift + p.bits > 32) {
          uint next_word = w[base + word_idx + 1];
          q |= (next_word << (32 - shift));
        }
        q &= mask;
        int g = col / p.group_size;
        uint sb = (expert * p.out_dim + out_row) * groups_per_row + g;
        float weight = float(q) * bf16_to_float(scales[sb]) + bf16_to_float(biases[sb]);
        acc += weight * x[expert * p.in_dim + col];
      }
      partial[tid] = acc;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
        if (lane < stride) partial[tid] += partial[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      if (lane == 0) y[global_row] = partial[tid];
    }
    )";
}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string weight_path = arg_str(argc, argv, "--weight", "");
        std::string scales_path = arg_str(argc, argv, "--scales", "");
        std::string biases_path = arg_str(argc, argv, "--biases", "");
        int all_experts = arg_int(argc, argv, "--all-experts", 512);
        int active = arg_int(argc, argv, "--active", 16);
        int in_dim = arg_int(argc, argv, "--in", 2048);
        int out_dim = arg_int(argc, argv, "--out", 512);
        int group = arg_int(argc, argv, "--group", 128);
        int bits = arg_int(argc, argv, "--bits", 6);
        int tile = arg_int(argc, argv, "--tile", 4);
        int repeat = arg_int(argc, argv, "--repeat", 30);
        bool async_mode = has_flag(argc, argv, "--async");

        int words = (in_dim * bits) / 32;
        int groups = in_dim / group;
        size_t weight_bytes = static_cast<size_t>(out_dim) * words * 4;
        size_t scale_bytes = static_cast<size_t>(out_dim) * groups * 2;

        int wfd, sfd, bfd;
        size_t wn, sn, bn;
        void* wm = map_file(weight_path, wn, wfd);
        void* sm = map_file(scales_path, sn, sfd);
        void* bm = map_file(biases_path, bn, bfd);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernel_source() options:nil error:&error];
        if (!library) {
            std::cerr << [[error localizedDescription] UTF8String] << "\n";
            return 5;
        }
        id<MTLFunction> fn = [library newFunctionWithName:@"qlinear_stage"];
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pso) {
            std::cerr << [[error localizedDescription] UTF8String] << "\n";
            return 6;
        }

        std::vector<float> xh = random_x(static_cast<size_t>(active) * in_dim);
        id<MTLBuffer> xbuf = [device newBufferWithBytes:xh.data()
                                                 length:xh.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> wbufs[2] = {
            [device newBufferWithLength:static_cast<size_t>(active) * weight_bytes options:MTLResourceStorageModeShared],
            [device newBufferWithLength:static_cast<size_t>(active) * weight_bytes options:MTLResourceStorageModeShared],
        };
        id<MTLBuffer> sbufs[2] = {
            [device newBufferWithLength:static_cast<size_t>(active) * scale_bytes options:MTLResourceStorageModeShared],
            [device newBufferWithLength:static_cast<size_t>(active) * scale_bytes options:MTLResourceStorageModeShared],
        };
        id<MTLBuffer> bbufs[2] = {
            [device newBufferWithLength:static_cast<size_t>(active) * scale_bytes options:MTLResourceStorageModeShared],
            [device newBufferWithLength:static_cast<size_t>(active) * scale_bytes options:MTLResourceStorageModeShared],
        };
        id<MTLBuffer> ybuf = [device newBufferWithLength:static_cast<size_t>(active) * out_dim * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
        Params params{active, out_dim, in_dim, group, bits, tile};
        id<MTLBuffer> pbuf = [device newBufferWithBytes:&params
                                                 length:sizeof(Params)
                                                options:MTLResourceStorageModeShared];

        auto start_for_step = [&](int step) {
            int limit = std::max(1, all_experts - active);
            return (step * active) % limit;
        };
        auto stage = [&](int step, int slot) {
            int start = start_for_step(step);
            std::memcpy([wbufs[slot] contents], static_cast<char*>(wm) + static_cast<size_t>(start) * weight_bytes,
                        static_cast<size_t>(active) * weight_bytes);
            std::memcpy([sbufs[slot] contents], static_cast<char*>(sm) + static_cast<size_t>(start) * scale_bytes,
                        static_cast<size_t>(active) * scale_bytes);
            std::memcpy([bbufs[slot] contents], static_cast<char*>(bm) + static_cast<size_t>(start) * scale_bytes,
                        static_cast<size_t>(active) * scale_bytes);
        };
        auto compute = [&](int slot) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:xbuf offset:0 atIndex:0];
            [enc setBuffer:wbufs[slot] offset:0 atIndex:1];
            [enc setBuffer:sbufs[slot] offset:0 atIndex:2];
            [enc setBuffer:bbufs[slot] offset:0 atIndex:3];
            [enc setBuffer:ybuf offset:0 atIndex:4];
            [enc setBuffer:pbuf offset:0 atIndex:5];
            NSUInteger grid = static_cast<NSUInteger>(((active * out_dim + tile - 1) / tile) * 256);
            [enc dispatchThreads:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        };
        auto t0 = std::chrono::steady_clock::now();
        if (!async_mode) {
            for (int r = 0; r < repeat; ++r) {
                stage(r, 0);
                compute(0);
            }
        } else {
            std::mutex mu;
            std::condition_variable cv;
            bool has_job = false, done_job = false, stop = false;
            int job_step = 0, job_slot = 0;
            std::thread worker([&]() {
                while (true) {
                    int step, slot;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&]() { return has_job || stop; });
                        if (stop) return;
                        step = job_step; slot = job_slot; has_job = false;
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
                job_step = step; job_slot = slot; done_job = false; has_job = true;
                cv.notify_all();
            };
            auto wait_ready = [&]() {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&]() { return done_job; });
            };
            submit(0, 0);
            wait_ready();
            for (int r = 0; r < repeat; ++r) {
                int cur = r % 2;
                int next = 1 - cur;
                if (r + 1 < repeat) submit(r + 1, next);
                compute(cur);
                if (r + 1 < repeat) wait_ready();
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                stop = true;
            }
            cv.notify_all();
            worker.join();
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;

        munmap(wm, wn);
        munmap(sm, sn);
        munmap(bm, bn);
        close(wfd);
        close(sfd);
        close(bfd);
        std::cout << "{"
                  << "\"active\":" << active
                  << ",\"in\":" << in_dim
                  << ",\"out\":" << out_dim
                  << ",\"bits\":" << bits
                  << ",\"tile\":" << tile
                  << ",\"async\":" << (async_mode ? "true" : "false")
                  << ",\"metal_ms\":" << ms
                  << "}\n";
    }
}
