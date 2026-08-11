// Phase 2 方案B：真实区槽状态 C++ 全接管（1 次同步版）+ demand_dual。
// 复刻 Python ResidentExpertPool 的 free/LFU 语义，成为 dual-source decode 真实区唯一权威。
#pragma once
#include "../common.h"

void real_init(int layer, int cap);                       // 初始化某层真实区(cap 槽全空闲)，幂等
void real_note_predictions(int layer, const uint32_t* expert_ids, size_t n);
std::vector<int> real_pin(int layer, const std::vector<int>& experts, int cap);
std::vector<int> real_pinned_contents(int layer);
std::vector<int> real_region_contents(int layer);         // [expert, slot, ...]
std::vector<int> real_verified_contents(int layer);       // excludes prediction-share rows
int real_region_count(int layer);
bool real_should_predict(int layer, int min_resident, int cooldown);
mx::array real_slot_table(int layer, int cap);
mx::array real_lease_table(int layer, int cap);
std::vector<std::pair<int, int>> real_prefetch_reserve(
    const uint32_t* expert_ids, size_t count, int layer, int cap,
    int speculative_limit, const std::vector<int>& resident);
void real_prefetch_publish(int layer, int expert, int row, int cap);
void real_prefetch_abort(int layer, int expert, int row, int cap);
void real_prefetch_wait_pending(
    int layer, const uint32_t* expert_ids, size_t count);
void real_prefetch_wait_all(int layer);
std::vector<int32_t> real_lookup_values(
    int layer, const uint32_t* expert_ids, size_t count);
void real_release_before_layer(int layer);
void real_reset();
void demand_deadline_snapshot(
    const mx::array& inds, int layer, int side_gen, bool use_side);
// demand 全接管：inds 惰性(内部 eval 一次)；side_gen 侧区代；pool_list 为 _segs 顺序池数组。
mx::array demand_dual(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen, const std::string& path,
    int stride, int cap, bool lfu, int decay_interval, int64_t forward_id,
    int sequence_length, bool use_side, bool record_deadline,
    mx::StreamOrDevice s);
mx::array demand_dual_async(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen,
    const std::string& path, int stride, int cap, bool lfu,
    int decay_interval, int64_t forward_id, int sequence_length,
    bool use_side, bool wait_for_pending, bool wait_for_refinement,
    bool evaluator_submit,
    mx::StreamOrDevice s);
mx::array demand_gpu_remap_only(
    const mx::array& inds, int layer, int side_gen, int cap,
    bool use_side, mx::StreamOrDevice s);
std::pair<mx::array, mx::array> demand_dual_split_async(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen,
    const std::string& path, int stride, int cap, bool lfu,
    int decay_interval, int64_t forward_id, int sequence_length,
    bool use_side, bool wait_for_pending, bool wait_for_refinement,
    bool evaluator_submit,
    mx::StreamOrDevice s);
std::vector<long> demand_async_stats();
void demand_async_stats_reset();
void demand_async_check();
mx::array demand_staged_multi(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, const std::string& path,
    int stride, int cap, bool lfu, int decay_interval, int spec_limit,
    const std::vector<mx::array>& staging_list,
    const std::vector<std::vector<int>>& staging_maps,
    int64_t forward_id, int sequence_length,
    mx::StreamOrDevice s);
std::vector<long> demand_last_stats();                    // [hitpos, misspos, loads, fallback012]
int late_promote_staged(
    const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int cap, int spec_limit,
    const mx::array& staging, const std::vector<int>& staging_map);
int demand_promote_staged(
    const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int cap, int spec_limit,
    const mx::array& staging, const std::vector<int>& staging_map,
    const mx::array& actual_ids);
// 目标层开始消费时的逐层唯一专家字节状态。每行：
// [layer, calls, actual_unique, real_resident, side_prefetch_complete, demand_fallback]。
// side_prefetch_complete 只在侧区已经 publish（整条 expert blob memcpy 完成）后计数。
std::vector<long> demand_deadline_stats();
void demand_deadline_stats_reset();
void demand_prejoin_note(
    int layer, const uint32_t* expert_ids, size_t n,
    const std::unordered_set<int>& staging_complete);
std::vector<long> demand_prejoin_stats();
void demand_prejoin_stats_reset();
std::vector<double> demand_timings();                     // [inds_eval, pool_eval, state, build] us
void demand_timing_enable(bool on);
// 测试壳：纯状态推进(不 pread/不侧区)，返回 local 槽位；供 LFU 驱逐等价对拍。
std::vector<int> real_debug_place(int layer, const std::vector<int>& experts_flat, int cap,
                                  bool lfu, int decay_interval);
