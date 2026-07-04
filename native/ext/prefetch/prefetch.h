// GPU 完成回调预取：算完 inds 后在回调里读 id、pread 预热 page cache 或写 per-layer staging。
#pragma once
#include "../common.h"

// 挂 GPU 完成回调：算完 inds 后在回调里读 id 预热 page cache，返回 dummy。
// （无 staging 的轻量预取模式：仅把预测专家字节读进 page cache，不落池。）
mx::array prefetch_on_complete(
    const mx::array& expert_ids, const std::string& path, int stride,
    bool do_read, mx::StreamOrDevice s);

// miss→hit（方案B）：expert_ids 为预测宽集合(降序)，回调按 resident 过滤、取前 cap 个缺口
// pread 进 per-layer staging buffer(cap=buffer 行数)，并原子记录 (gen,[expert,row])。
mx::array prefetch_into_staging(
    const mx::array& staging, const mx::array& expert_ids, int layer, long gen,
    const std::string& path, int stride, const std::vector<int>& resident, int cap,
    bool parallel, mx::StreamOrDevice s);
// 取走某层就绪记录：[gen, e0,r0,e1,r1,...]；空表示无就绪。
std::vector<long> prefetch_staging_take(int layer);

// handler 触发时刻探针(诊断用)：enable 开关并清零；now 取同时钟当前秒；get 取 (gen,layer,t) 日志。
void staging_hprof_enable(bool on);
double staging_hprof_now();
std::vector<double> staging_hprof_get();   // 扁平 [gen,layer,t, ...]
