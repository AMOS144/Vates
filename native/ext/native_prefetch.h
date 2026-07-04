// native-fused-prefetch：blob 字节直读 + GPU 完成回调里读 id/派发 pread/写 staging/侧区。
// 实现见 native_prefetch.cpp；跨线程全局态(g_stg_*/g_side_*/g_real_*/g_owned_*)私有于该 TU。
#pragma once
#include <tuple>
#include <utility>
#include "native_common.h"

// 把一组专家 blob 字节 pread 进新建的 MLX uint8[n,stride] 数组（惰性图节点）。
mx::array blob_load(
    const std::string& path, const mx::array& expert_ids, int stride,
    mx::StreamOrDevice s);

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

// 段散写持久侧区缓存（zero-copy dual-source 专用）：keep/evict/read-delta，
// 命中缺口时 pread blob 行并把各段 memcpy 进对应 per-key 池数组的 (base_row+cache_row) 行。
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
// Route 3 Phase 1 底座：C++ 拥有的池 buffer 包成 mx.array（no-op deleter，进程内持有、永不迁移）。
// 供侧区/demand 后台 pread 安全直写；替代原 mx.zeros 建池 + 消费侧 MLX scatter。
mx::array pool_owned_zeros(const std::vector<int>& shape, const std::string& dtype);
// demand 真实区落池：把已加载专家段直接 memcpy 进 owned 池行（无 MLX scatter，保 buffer 稳定）。
void pool_write_rows(const std::vector<mx::array>& pool_list,
                     const std::vector<mx::array>& srcs_flat, const std::vector<int>& slots);
void pool_write_stacked(const std::vector<mx::array>& pool_list,
                        const std::vector<mx::array>& stacked_list, const std::vector<int>& slots);
// 诊断：mx.array 底层 buffer 原始指针（uintptr），用于对拍侧区写入 buffer 与 consume 读到 buffer 是否同一块。
uintptr_t array_data_ptr(const mx::array& a);

// ---- Phase 2 方案B：真实区槽状态 C++ 全接管（1 次同步版）----
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

// ---- 自由后台读线程（de-risk）：脱离 GPU 完成回调、零 GIL，pread 进调用方 MLX buffer ----
void bg_reader_start(int workers, int low_cap = 0);
long bg_reader_submit(const mx::array& dst, const std::vector<int>& experts,
                      const std::vector<int>& rows, const std::string& path,
                      int stride, long ticket, int prio = 0);
bool bg_reader_ready(long ticket);
void bg_reader_wait(long ticket);
void bg_reader_stop();
long bg_pread_into_pool(
    const std::vector<mx::array>& dst,
    const std::vector<long>& seg_off,
    const std::vector<long>& seg_nb,
    long slot, long expert,
    const std::string& path, long stride, long ticket, int prio = 0, bool nocache = true);
