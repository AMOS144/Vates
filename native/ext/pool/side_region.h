// 段散写持久侧区缓存（zero-copy dual-source 默认路径）：keep/evict/read-delta，
// 命中缺口时 pread blob 行并把各段 memcpy 进对应 per-key 池数组的 (base_row+cache_row) 行。
#pragma once
#include "../common.h"
#include <unordered_map>

mx::array prefetch_pool_sideregion(
    const std::vector<mx::array>& pool_list, const std::vector<int>& seg_nbytes,
    const mx::array& expert_ids, int layer, const std::string& path, int stride,
    const std::vector<int>& resident, int spec_slots, int base_row, int gen,
    mx::StreamOrDevice s);
std::vector<int> sideregion_contents(int layer, int gen);   // [expert, phys_row, ...]
std::pair<mx::array, mx::array> sideregion_kv(int layer, int gen);  // (keys uint32, vals int32) device 数组
void sideregion_reset();
// 排空侧区在途 fill：阻塞直到所有已提交的侧区预取字节写完。消费方在前向开头调用，
// 保证被消费的侧区行字节已完全写好（闭合异步写-读竞态）。
void sideregion_drain();

// 内部接口（不绑 Python）：取某层某代侧区 e2r 快照，供 demand_dual 在自己的核心状态机里
// 叠加侧区命中。等价于在 g_side 锁下拷贝 e2r（与原 demand_dual 内联循环一字不差）。
std::unordered_map<int, int> sideregion_snapshot(int layer, int gen);
