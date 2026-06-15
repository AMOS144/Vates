// native-fused-prefetch：blob 字节直读 + GPU 完成回调里读 id/派发 pread/写 staging。
// 实现见 native_prefetch.cpp；跨线程全局态(g_pf_*/g_stg_*)私有于该 TU。
#pragma once
#include "native_common.h"

// 把一组专家 blob 字节 pread 进新建的 MLX uint8[n,stride] 数组（惰性图节点）。
mx::array blob_load(
    const std::string& path, const mx::array& expert_ids, int stride,
    mx::StreamOrDevice s);

// 挂 GPU 完成回调：算完 inds 后在回调里读 id（可选预热字节），返回 dummy。
mx::array prefetch_on_complete(
    const mx::array& expert_ids, const std::string& path, int stride,
    bool do_read, mx::StreamOrDevice s);
std::vector<int> prefetch_on_complete_last_ids();
int prefetch_on_complete_fires();

// 回调把专家字节 pread 进调用方预分配的 MLX buffer（零拷贝物化候选）。
mx::array prefetch_into(
    const mx::array& dst, const mx::array& expert_ids, const std::string& path,
    int stride, mx::StreamOrDevice s);

// miss→hit：回调 pread 进 per-layer staging buffer，并原子记录 (gen,[expert,row])。
mx::array prefetch_into_staging(
    const mx::array& staging, const mx::array& expert_ids, int layer, long gen,
    const std::string& path, int stride, mx::StreamOrDevice s);
// 取走某层就绪记录：[gen, e0,r0,e1,r1,...]；空表示无就绪。
std::vector<long> prefetch_staging_take(int layer);
