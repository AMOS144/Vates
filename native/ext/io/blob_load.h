// blob 直读：把一组专家 blob 字节 pread 进新建的 MLX uint8[n,stride] 数组（惰性图节点）。
#pragma once
#include "../common.h"

mx::array blob_load(
    const std::string& path, const mx::array& expert_ids, int stride,
    mx::StreamOrDevice s);

// 将页对齐 blob 注册为 file-backed MLX uint8[experts,stride] array。
// Metal 统一内存直接引用 mmap 页面；实际专家由后续 GPU gather 决定，因此无需
// 先把 route ids 同步回 host，也不需要逐层 command-buffer completion handler。
mx::array blob_mmap(
    const std::string& path, int stride, int num_experts);

// GPU 根据尚未同步到 host 的真实 route ids，从 file-backed blob 收集紧凑专家行。
// 输出为 uint8[num_route_positions,stride]，可直接切成量化张量并交给 MLX QMM。
mx::array blob_mmap_gather(
    const std::string& path, const mx::array& expert_ids,
    int stride, int num_experts, mx::StreamOrDevice s);

// GPU-side page prefetch: each predicted expert touches one word per VM page.
// It returns a scalar dependency and never installs a command-buffer callback.
mx::array blob_mmap_prefetch(
    const std::string& path, const mx::array& expert_ids,
    int stride, int num_experts, mx::StreamOrDevice s);

// 诊断/回调端：在 host 上触碰同一个 file-backed Metal mapping 的每个 VM 页，
// 提前建立 GPU 后续 QMM 所需的 PTE。返回触碰页数。
long blob_mmap_prefault_host(
    const std::string& path, const mx::array& expert_ids,
    int stride, int num_experts);
