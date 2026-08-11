// 段散写持久侧区缓存（zero-copy dual-source 默认路径）：keep/evict/read-delta，
// 命中缺口时 pread blob 行并把各段 memcpy 进对应 per-key 池数组的 (base_row+cache_row) 行。
#pragma once
#include "../common.h"
#include <unordered_map>

mx::array prefetch_pool_sideregion(
    const std::vector<mx::array>& pool_list, const std::vector<int>& seg_nbytes,
    const mx::array& expert_ids, int layer, const std::string& path, int stride,
    const std::vector<int>& resident, int spec_slots, int base_row, int gen,
    int source_layer, int64_t forward_id, int priority,
    mx::StreamOrDevice s);
// The ids and destination pools are already materialized by a source-layer
// demand command buffer. Reserve final unified rows and start the same
// background read/publish path without creating a second MLX primitive or
// Metal completion handler.
void prefetch_unified_ready(
    const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes,
    const uint32_t* expert_ids, size_t count,
    int target_layer, const std::string& path, int stride,
    const std::vector<int>& resident, int speculative_limit, int real_cap,
    int source_layer, int64_t forward_id);
std::vector<int> sideregion_contents(int layer, int gen);   // [expert, phys_row, ...]
std::pair<mx::array, mx::array> sideregion_kv(int layer, int gen);  // (keys uint32, vals int32) device 数组
mx::array sideregion_slot_table(int layer, int gen);
// GPU-visible per-physical-row leases.  The remap kernel acquires a side-row
// lease before publishing its local index, then revalidates expert->row so a
// concurrent prefetch eviction can never turn a hit into a stale-row read.
mx::array sideregion_lease_table(int layer, int gen);
void sideregion_reset();
// 预取取证统计：[输入ID、过滤后唯一候选、侧区已有命中、真实预留读取、淘汰、pread成功、pread失败]。
std::vector<long> sideregion_prefetch_stats();
void sideregion_prefetch_stats_reset();

// 逐 forward/目标层的生产预取审计。提交侧记录 rerank 唯一候选与真实 I/O
// 时间线；demand 侧用同一 forward_id 配对真实路由，因而即使 callback 晚于
// demand 也不会错配到下一轮。返回固定 26 列，列定义由 runtime 汇总器校验。
std::vector<int64_t> prefetch_audit_stats();
void prefetch_audit_stats_reset();
bool prefetch_audit_enabled();
void prefetch_audit_note_submit(
    int64_t forward_id, int source_layer, int target_layer, int physical_gen);
void prefetch_audit_note_callback(
    int64_t forward_id, int target_layer, const uint32_t* ids, size_t n,
    const std::vector<int>& resident);
void prefetch_audit_note_pread(
    int64_t forward_id, int target_layer, size_t requested);
void prefetch_audit_note_publish(
    int64_t forward_id, int target_layer, size_t completed);
void prefetch_audit_note_demand(
    int64_t forward_id, int layer, int sequence_length,
    const uint32_t* actual_ids, size_t n,
    const std::unordered_map<int, int>& real,
    const std::unordered_map<int, int>& side);
// 排空侧区在途 fill：阻塞直到所有已提交的侧区预取字节写完。消费方在前向开头调用，
// 保证被消费的侧区行字节已完全写好（闭合异步写-读竞态）。
void sideregion_drain();
// 等待指定逻辑 forward/目标层的全部 early + refinement submission 完整发布。
void sideregion_wait_target(int64_t forward_id, int target_layer);
// 只等待 refinement callback 完成 reserve/evict，不等待任何 SSD 字节。
// demand 在取得 side snapshot 前调用，保证随后不会再有同层 callback 改写行账本。
void sideregion_wait_refinement(int64_t forward_id, int target_layer);
// 只等精确 route 中仍在 pending 的专家；选中集合里的假阳性可继续后台完成。
void sideregion_wait_experts(
    int64_t forward_id, int layer, int gen, const mx::array& expert_ids);
void sideregion_wait_expert_values(
    int64_t forward_id, int layer, int gen,
    const uint32_t* expert_ids, size_t count);
// Normal single-stage prefetch: wait only reservations that already exist;
// unlike progressive wait, there is no refinement-registration barrier.
void sideregion_wait_pending_values(
    int layer, int gen, const uint32_t* expert_ids, size_t count);
// Feed actual target-route usage back into side LFU. Prediction frequency
// alone cannot distinguish stable true hits from recurring false positives.
void sideregion_note_demand_values(
    int layer, int gen, const uint32_t* expert_ids, size_t count);

// Resolve only the experts in the current route against the published direct
// rows.  Unlike sideregion_snapshot this is O(route width), does not copy the
// complete ownership table, and is used after waiting for predicted-pending
// reads so those rows can bypass the CPU fallback allocator.
std::vector<int32_t> sideregion_lookup_values(
    int layer, int gen, const uint32_t* expert_ids, size_t count);

// Protect direct-pool rows while a GPU MoE command buffer may still read
// them. Demand acquires leases before releasing its event; a tiny dependency
// primitive releases them only when the consuming command buffer completes.
void sideregion_acquire_row_leases(
    int layer, const int32_t* rows, size_t count, int real_cap);
void sideregion_release_before_layer(int layer);

// 内部接口（不绑 Python）：取某层某代侧区 e2r 快照，供 demand_dual 在自己的核心状态机里
// 叠加侧区命中。等价于在 g_side 锁下拷贝 e2r（与原 demand_dual 内联循环一字不差）。
std::unordered_map<int, int> sideregion_snapshot(int layer, int gen);
