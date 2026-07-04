// Phase 2 方案B：真实区槽状态 C++ 全接管（1 次同步版）+ demand_dual。
// 复刻 Python ResidentExpertPool 的 free/LFU 语义，成为 dual-source decode 真实区唯一权威。
#pragma once
#include "../common.h"

void real_init(int layer, int cap);                       // 初始化某层真实区(cap 槽全空闲)，幂等
std::vector<int> real_region_contents(int layer);         // [expert, slot, ...]
int real_region_count(int layer);
void real_reset();
// demand 全接管：inds 惰性(内部 eval 一次)；side_gen 侧区代；pool_list 为 _segs 顺序池数组。
mx::array demand_dual(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen, const std::string& path,
    int stride, int cap, bool lfu, int decay_interval, mx::StreamOrDevice s);
std::vector<long> demand_last_stats();                    // [hitpos, misspos, loads, fallback01]
std::vector<double> demand_timings();                     // [inds_eval, pool_eval, state, build] us
void demand_timing_enable(bool on);
// 测试壳：纯状态推进(不 pread/不侧区)，返回 local 槽位；供 LFU 驱逐等价对拍。
std::vector<int> real_debug_place(int layer, const std::vector<int>& experts_flat, int cap,
                                  bool lfu, int decay_interval);
