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
    bool parallel, int source_layer, int64_t forward_id, int priority,
    mx::StreamOrDevice s);
// expert IDs are already host-visible: start the staging read immediately,
// without inserting another Metal completed-handler boundary.
void prefetch_staging_ready(
    const mx::array& staging, const std::vector<int>& expert_ids, int layer,
    long gen, const std::string& path, int stride,
    const std::vector<int>& resident, int cap, bool parallel,
    int source_layer, int64_t forward_id, int priority);
// 取走某层就绪记录：[gen, e0,r0,e1,r1,...]；空表示无就绪。
std::vector<long> prefetch_staging_take(int layer, long generation = -1);
std::vector<long> prefetch_staging_take_for_demand(
    int layer, long generation);
void prefetch_staging_mark_consumed(int layer, long generation);
bool prefetch_staging_consumed(int layer, long generation);
bool prefetch_staging_finished(int layer, long generation);
void prefetch_staging_forget(int layer, long generation);
void prefetch_staging_wait_experts(
    int64_t forward_id, int layer, const mx::array& expert_ids);
void prefetch_staging_note_prejoin(
    int64_t forward_id, int layer, const mx::array& expert_ids);
void prefetch_staging_finish_demand(int64_t forward_id, int layer);
std::vector<long> prefetch_staging_wait_stats();
void prefetch_staging_wait_stats_reset();
void prefetch_staging_drain();

// handler 触发时刻探针(诊断用)：enable 开关并清零；now 取同时钟当前秒；get 取 (gen,layer,t) 日志。
void staging_hprof_enable(bool on);
double staging_hprof_now();
std::vector<double> staging_hprof_get();   // 扁平 [gen,layer,t, ...]
