// native-fused-prefetch：blob 字节直读 + GPU 完成回调里读 id/派发 pread/写 staging。
// 实现见 native_prefetch.cpp；跨线程全局态(g_pf_*/g_stg_*)私有于该 TU。
#pragma once
#include <tuple>
#include <utility>
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

// Task1 spike：图内 primitive 输出别名输入 buffer，验证下游 gather 依赖边机制。
mx::array materialize_spike(const mx::array& src, uint32_t fillval, mx::StreamOrDevice s);

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
    const std::string& path, long stride, long ticket, int prio = 0);
