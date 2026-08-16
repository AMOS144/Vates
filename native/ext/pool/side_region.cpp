// [4] 段散写持久侧区缓存（zero-copy dual-source 默认路径）。
// 与持久 staging 缓存同构，但目标是“多个结构化 per-key 池数组”：每个池数组形如
// (cap+spec, ...)，行 [base_row, base_row+spec) 为侧区。命中缺口时 pread blob 整行，
// 再把该行内按固定顺序拼接的各段 memcpy 进对应 per-key 数组的同一物理行。
#include "side_region.h"
#include "demand.h"
#include "../io/blob_io.h"
#include "../io/bg_reader.h"

#include <chrono>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <sys/uio.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

struct SideLayer {
  std::map<int, int> e2r;          // expert -> 物理侧区行 [base_row, base_row+spec)
  // 已经从 free/e2r 账本预留、但整条 expert blob 尚未完成并发布的行。
  // progressive rerank 会对同一 (layer, gen) 连续提交 early core 与 late fill；
  // 若只在 publish 后才记录 expert，第二个 callback 会重复预留同一 expert，
  // 最终泄漏物理行甚至让两个后台任务覆盖同一逻辑映射。
  std::map<int, int> pending_e2r;
  std::vector<int> free_rows;
  std::map<int, uint32_t> freq;    // expert -> 预测频次(LFU 分数;仅 SIDEREGION_LFU 用)
  bool inited = false;
  int base = 0;                    // 侧区起始物理行 base_row
  int spec = 0;                    // 侧区行数 spec_slots
  std::shared_ptr<mx::array> slot_table;
  std::shared_ptr<mx::array> lease_table;
};
static std::mutex g_side_mutex;
static std::map<std::pair<int, int>, SideLayer> g_side;   // 键 (layer, gen)：双缓冲两代独立
static std::map<std::pair<int, int>, uint32_t> g_side_row_leases;
static std::array<std::atomic<int64_t>, 64> g_target_consumed_forward{};

void sideregion_mark_target_consumed(int64_t forward_id, int layer) {
  if (forward_id < 0 || layer < 0 || layer >= 64) return;
  auto& value = g_target_consumed_forward[static_cast<size_t>(layer)];
  const int64_t encoded = forward_id + 1;
  int64_t observed = value.load(std::memory_order_relaxed);
  while (observed < encoded && !value.compare_exchange_weak(
      observed, encoded, std::memory_order_release,
      std::memory_order_relaxed)) {}
}

static bool target_already_consumed(int64_t forward_id, int layer) {
  const char* enabled = std::getenv("PREFETCH_CANCEL_STALE_QUEUED");
  return enabled && enabled[0] == '1' &&
      forward_id >= 0 && layer >= 0 && layer < 64 &&
      g_target_consumed_forward[static_cast<size_t>(layer)].load(
          std::memory_order_acquire) >= forward_id + 1;
}

static bool side_row_leased_locked(int layer, int row) {
  auto found = g_side_row_leases.find({layer, row});
  if (found != g_side_row_leases.end() && found->second > 0) return true;
  if (row < 0 || row >= 1024) return false;
  for (const auto& item : g_side) {
    if (item.first.first != layer || !item.second.lease_table) continue;
    const uint32_t* leases = item.second.lease_table->data<uint32_t>();
    if (__atomic_load_n(leases + row, __ATOMIC_ACQUIRE) != 0) return true;
  }
  return false;
}

void sideregion_acquire_row_leases(
    int layer, const int32_t* rows, size_t count, int real_cap) {
  const char* enabled = std::getenv("SIDEREGION_ROW_LEASES");
  if (!enabled || enabled[0] != '1') return;
  std::lock_guard<std::mutex> lk(g_side_mutex);
  for (size_t index = 0; index < count; ++index) {
    int row = static_cast<int>(rows[index]);
    if (row >= real_cap) g_side_row_leases[{layer, row}] += 1;
  }
}

void sideregion_release_before_layer(int layer) {
  const char* enabled = std::getenv("SIDEREGION_ROW_LEASES");
  if (!enabled || enabled[0] != '1') return;
  std::lock_guard<std::mutex> lk(g_side_mutex);
  if (layer <= 0) {
    // A new decoder traversal cannot begin before the previous traversal's
    // final output has completed, so every leftover tail-layer lease is dead.
    g_side_row_leases.clear();
    for (auto& item : g_side) {
      if (!item.second.lease_table) continue;
      uint32_t* leases = item.second.lease_table->data<uint32_t>();
      for (int row = 0; row < 1024; ++row)
        __atomic_store_n(leases + row, 0u, __ATOMIC_RELEASE);
    }
    return;
  }
  const int consumed_layer = layer - 1;
  for (auto it = g_side_row_leases.begin(); it != g_side_row_leases.end();) {
    if (it->first.first == consumed_layer) it = g_side_row_leases.erase(it);
    else ++it;
  }
  for (auto& item : g_side) {
    if (item.first.first != consumed_layer || !item.second.lease_table) continue;
    uint32_t* leases = item.second.lease_table->data<uint32_t>();
    for (int row = 0; row < 1024; ++row)
      __atomic_store_n(leases + row, 0u, __ATOMIC_RELEASE);
  }
}

static void ensure_side_table_locked(SideLayer& side) {
  if (!side.slot_table) {
    std::vector<int32_t> values(512, -1);
    side.slot_table = std::make_shared<mx::array>(
        values.data(), mx::Shape{512}, mx::int32);
  }
  if (!side.lease_table) {
    std::vector<uint32_t> leases(1024, 0);
    side.lease_table = std::make_shared<mx::array>(
        leases.data(), mx::Shape{1024}, mx::uint32);
  }
  int32_t* table = side.slot_table->data<int32_t>();
  for (const auto& item : side.e2r)
    if (item.first >= 0 && item.first < 512)
      __atomic_store_n(
          table + item.first, item.second, __ATOMIC_RELEASE);
}

static void side_table_set_locked(SideLayer& side, int expert, int row) {
  ensure_side_table_locked(side);
  if (expert >= 0 && expert < 512)
    __atomic_store_n(
      side.slot_table->data<int32_t>() + expert, row, __ATOMIC_RELEASE);
}

// Atomically close the GPU-remap/victim-selection race.  Merely checking the
// lease before clearing expert->row has a TOCTOU window: a remap can read the
// old row just after the check, acquire its lease, and then consume bytes that
// prefetch is already overwriting.  Invalidate first, then recheck.  A remap
// that observed the old value either publishes its lease before this check or
// fails its own table revalidation; in the former case restore ownership and
// leave the row untouched.  A transient GPU miss is safe (normal demand falls
// back), while a stale-row hit is a numerical correctness violation.
static bool side_try_unmap_unleased_locked(
    SideLayer& side, int layer, int expert, int row) {
  if (side_row_leased_locked(layer, row)) return false;
  side_table_set_locked(side, expert, -1);
  if (side_row_leased_locked(layer, row)) {
    side_table_set_locked(side, expert, row);
    return false;
  }
  return true;
}
// 预取诊断累计值：全部使用原子计数，默认路径只增加极小开销。
static std::atomic<long> g_stat_input_ids{0};
static std::atomic<long> g_stat_candidates{0};
static std::atomic<long> g_stat_side_hits{0};
static std::atomic<long> g_stat_reserved_reads{0};
static std::atomic<long> g_stat_evictions{0};
static std::atomic<long> g_stat_pread_ok{0};
static std::atomic<long> g_stat_pread_fail{0};
static std::atomic<bool> g_stats_enabled{false};
static std::atomic<long> g_stat_layer_reads[64]{};
static std::mutex g_stat_unique_mutex;
static std::unordered_set<uint64_t> g_stat_unique_reads;
static inline void side_stat_add(std::atomic<long>& counter, long value) {
  if (g_stats_enabled.load(std::memory_order_relaxed))
    counter.fetch_add(value, std::memory_order_relaxed);
}
static inline void side_stat_note_reads(
    int layer, const std::vector<std::pair<int, int>>& reads) {
  if (!g_stats_enabled.load(std::memory_order_relaxed) || reads.empty()) return;
  if (layer >= 0 && layer < 64)
    g_stat_layer_reads[layer].fetch_add(
        static_cast<long>(reads.size()), std::memory_order_relaxed);
  std::lock_guard<std::mutex> lk(g_stat_unique_mutex);
  for (const auto& item : reads)
    g_stat_unique_reads.insert(
        (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32)
        | static_cast<uint32_t>(item.first));
}

// 生产预取审计只在显式 reset 后开启。key=(逻辑 forward_id,target layer)，
// 不能用物理 side gen：持久 LFU 单缓冲下 gen 恒为 0，会把所有 forward 混在一起。
struct PrefetchAuditRecord {
  int64_t forward_id = -1;
  int source_layer = -1;
  int target_layer = -1;
  int physical_gen = -1;
  int sequence_length = -1;
  int64_t submit_eval_us = -1;
  int64_t callback_us = -1;
  int64_t pread_start_us = -1;
  int64_t publish_end_us = -1;
  int64_t demand_us = -1;
  std::unordered_set<int> candidates;
  std::unordered_set<int> source_resident;
  std::unordered_set<int> actual;
  std::unordered_set<int> deadline_complete;
  long deadline_real = 0;
  long deadline_side = 0;
  long fallback = 0;
  long pread_requested = 0;
  long pread_completed = 0;
  long submission_count = 0;
  long demand_count = 0;
  bool source_resident_recorded = false;
};
static std::mutex g_prefetch_audit_mutex;
static std::map<std::pair<int64_t, int>, PrefetchAuditRecord> g_prefetch_audit;
static std::atomic<bool> g_prefetch_audit_enabled{false};

static int64_t audit_now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool prefetch_audit_enabled() {
  return g_prefetch_audit_enabled.load(std::memory_order_relaxed);
}

static PrefetchAuditRecord& audit_record_locked(int64_t forward_id, int target_layer) {
  PrefetchAuditRecord& row = g_prefetch_audit[{forward_id, target_layer}];
  row.forward_id = forward_id;
  row.target_layer = target_layer;
  return row;
}

void prefetch_audit_note_submit(
    int64_t forward_id, int source_layer, int target_layer, int physical_gen) {
  if (!prefetch_audit_enabled() || forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  PrefetchAuditRecord& row = audit_record_locked(forward_id, target_layer);
  // 同一 target 可有 early core + late refinement。审计窗口必须保留最早的
  // 原 main source，而不是被晚期 T-1 refinement 覆盖。
  if (row.source_layer < 0 || source_layer < row.source_layer)
    row.source_layer = source_layer;
  row.physical_gen = physical_gen;
  int64_t now = audit_now_us();
  if (row.submit_eval_us < 0 || now < row.submit_eval_us) row.submit_eval_us = now;
  row.submission_count += 1;
}

void prefetch_audit_note_callback(
    int64_t forward_id, int target_layer, const uint32_t* ids, size_t n,
    const std::vector<int>& resident) {
  if (!prefetch_audit_enabled() || forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  PrefetchAuditRecord& row = audit_record_locked(forward_id, target_layer);
  int64_t now = audit_now_us();
  if (row.callback_us < 0 || now < row.callback_us) row.callback_us = now;
  for (size_t i = 0; i < n; ++i) row.candidates.insert(static_cast<int>(ids[i]));
  // 第一份提交时 resident 才是原 source 边界的权威快照。后续 refinement
  // 只能扩充候选，不能改写 coverage 的 resident 基线。
  if (!row.source_resident_recorded) {
    row.source_resident.insert(resident.begin(), resident.end());
    row.source_resident_recorded = true;
  }
}

void prefetch_audit_note_pread(
    int64_t forward_id, int target_layer, size_t requested) {
  if (!prefetch_audit_enabled() || forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  PrefetchAuditRecord& row = audit_record_locked(forward_id, target_layer);
  int64_t now = audit_now_us();
  if (row.pread_start_us < 0 || now < row.pread_start_us) row.pread_start_us = now;
  row.pread_requested += static_cast<long>(requested);
}

void prefetch_audit_note_publish(
    int64_t forward_id, int target_layer, size_t completed) {
  if (!prefetch_audit_enabled() || forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  PrefetchAuditRecord& row = audit_record_locked(forward_id, target_layer);
  int64_t now = audit_now_us();
  if (row.publish_end_us < 0 || now > row.publish_end_us) row.publish_end_us = now;
  row.pread_completed += static_cast<long>(completed);
}

void prefetch_audit_note_demand(
    int64_t forward_id, int layer, int sequence_length,
    const uint32_t* actual_ids, size_t n,
    const std::unordered_map<int, int>& real,
    const std::unordered_map<int, int>& side) {
  if (!prefetch_audit_enabled() || forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  PrefetchAuditRecord& row = audit_record_locked(forward_id, layer);
  row.sequence_length = sequence_length;
  row.demand_us = audit_now_us();
  row.demand_count += 1;
  row.actual.clear();
  row.deadline_complete.clear();
  row.deadline_real = row.deadline_side = row.fallback = 0;
  for (size_t i = 0; i < n; ++i) row.actual.insert(static_cast<int>(actual_ids[i]));
  for (int expert : row.actual) {
    if (side.count(expert)) {
      ++row.deadline_side;
      row.deadline_complete.insert(expert);
    } else if (real.count(expert)) {
      ++row.deadline_real;
      row.deadline_complete.insert(expert);
    }
    else ++row.fallback;
  }
}

std::vector<int64_t> prefetch_audit_stats() {
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  std::vector<int64_t> out;
  out.reserve(g_prefetch_audit.size() * 26);
  for (const auto& item : g_prefetch_audit) {
    const PrefetchAuditRecord& row = item.second;
    long candidate_hits = 0, source_resident_hits = 0, system_hits = 0;
    long candidate_complete_hits = 0, system_complete_hits = 0;
    for (int expert : row.actual) {
      bool candidate = row.candidates.count(expert);
      bool resident = row.source_resident.count(expert);
      if (candidate) ++candidate_hits;
      if (resident) ++source_resident_hits;
      if (candidate || resident) ++system_hits;
      if (candidate && row.deadline_complete.count(expert)) ++candidate_complete_hits;
      if ((candidate || resident) && row.deadline_complete.count(expert))
        ++system_complete_hits;
    }
    out.insert(out.end(), {
        row.forward_id, row.source_layer, row.target_layer, row.physical_gen,
        row.sequence_length,
        row.submit_eval_us, row.callback_us, row.pread_start_us, row.publish_end_us,
        row.demand_us, static_cast<int64_t>(row.candidates.size()),
        static_cast<int64_t>(row.source_resident.size()),
        static_cast<int64_t>(row.actual.size()), candidate_hits, source_resident_hits,
        system_hits, candidate_complete_hits, system_complete_hits,
        row.deadline_real, row.deadline_side, row.fallback,
        row.pread_requested, row.pread_completed, row.submission_count, row.demand_count,
        row.callback_us >= 0 && row.demand_us >= 0 && row.callback_us <= row.demand_us ? 1 : 0,
    });
  }
  return out;
}

void prefetch_audit_stats_reset() {
  std::lock_guard<std::mutex> lk(g_prefetch_audit_mutex);
  g_prefetch_audit.clear();
  g_prefetch_audit_enabled.store(true, std::memory_order_relaxed);
}

// 侧区 fill 在途计数：eval_gpu 提交预取时 +1，后台 read_publish 写完字节后 -1。
// 消费方在前向开头 sideregion_drain() 排空上一前向的 fill，保证被消费的侧区行字节
// 已完全写好（含 GPU 完成回调滞后的情形）→ 消灭「GPU 消费 kernel 读到半写侧区行」竞态。
static std::atomic<long> g_side_inflight{0};
static std::mutex g_side_drain_mutex;
static std::condition_variable g_side_drain_cv;
// 细粒度 deadline barrier：只等同一逻辑 forward 的同一目标层两次提交，
// 不把无关层的 early 预取也串到关键路径。
static std::map<std::pair<int64_t, int>, long> g_side_target_inflight;
static std::map<std::pair<int64_t, int>, bool> g_side_refinement_ready;
static inline void side_inflight_start(int64_t forward_id, int target_layer) {
  std::lock_guard<std::mutex> lk(g_side_drain_mutex);
  g_side_inflight.fetch_add(1);
  if (forward_id >= 0) g_side_target_inflight[{forward_id, target_layer}] += 1;
}
static inline void side_inflight_done(int64_t forward_id, int target_layer) {
  {
    std::lock_guard<std::mutex> lk(g_side_drain_mutex);
    g_side_inflight.fetch_sub(1);
    if (forward_id >= 0) {
      auto key = std::make_pair(forward_id, target_layer);
      auto it = g_side_target_inflight.find(key);
      if (it != g_side_target_inflight.end()) {
        if (--it->second == 0) g_side_target_inflight.erase(it);
      }
    }
  }
  g_side_drain_cv.notify_all();
}
void sideregion_drain() {
  std::unique_lock<std::mutex> lk(g_side_drain_mutex);
  g_side_drain_cv.wait(lk, [] { return g_side_inflight.load() == 0; });
}
void sideregion_wait_target(int64_t forward_id, int target_layer) {
  std::unique_lock<std::mutex> lk(g_side_drain_mutex);
  auto key = std::make_pair(forward_id, target_layer);
  g_side_drain_cv.wait(lk, [&] { return !g_side_target_inflight.count(key); });
}
void sideregion_wait_refinement(int64_t forward_id, int target_layer) {
  std::unique_lock<std::mutex> lk(g_side_drain_mutex);
  auto key = std::make_pair(forward_id, target_layer);
  g_side_drain_cv.wait(lk, [&] { return g_side_refinement_ready.count(key); });
  g_side_refinement_ready.erase(key);
}
void sideregion_wait_expert_values(
    int64_t forward_id, int layer, int gen,
    const uint32_t* values, size_t count) {
  std::unordered_set<int> wanted;
  for (size_t index = 0; index < count; ++index)
    wanted.insert(static_cast<int>(values[index]));
  std::unique_lock<std::mutex> lk(g_side_drain_mutex);
  auto key = std::make_pair(forward_id, layer);
  g_side_drain_cv.wait(lk, [&] {
    if (!g_side_refinement_ready.count(key)) return false;
    std::lock_guard<std::mutex> side_lk(g_side_mutex);
    auto side = g_side.find({layer, gen});
    if (side == g_side.end()) return true;
    for (int expert : wanted)
      if (side->second.pending_e2r.count(expert)) return false;
    return true;
  });
  g_side_refinement_ready.erase(key);
}
void sideregion_wait_experts(
    int64_t forward_id, int layer, int gen, const mx::array& expert_ids) {
  mx::array ids = expert_ids;
  ids.eval();
  sideregion_wait_expert_values(
      forward_id, layer, gen, ids.data<uint32_t>(), ids.size());
}
void sideregion_wait_pending_values(
    int layer, int gen, const uint32_t* values, size_t count) {
  std::unordered_set<int> wanted;
  for (size_t index = 0; index < count; ++index)
    wanted.insert(static_cast<int>(values[index]));
  std::unique_lock<std::mutex> lk(g_side_drain_mutex);
  g_side_drain_cv.wait(lk, [&] {
    std::lock_guard<std::mutex> side_lk(g_side_mutex);
    auto side = g_side.find({layer, gen});
    if (side == g_side.end()) return true;
    for (int expert : wanted)
      if (side->second.pending_e2r.count(expert)) return false;
    return true;
  });
}
void sideregion_note_demand_values(
    int layer, int gen, const uint32_t* values, size_t count) {
  // Progressive submits both an early core and a late refinement. One true
  // route occurrence must therefore outweigh several speculative sightings,
  // otherwise recurring false positives become artificially hot.
  static const uint32_t kDemandBoost = []() {
    const char* value = std::getenv("SIDEREGION_DEMAND_BOOST");
    if (!value) return uint32_t{8};
    long parsed = std::strtol(value, nullptr, 10);
    return static_cast<uint32_t>(std::max<long>(0, parsed));
  }();
  if (kDemandBoost == 0 || !values || count == 0) return;
  std::lock_guard<std::mutex> lk(g_side_mutex);
  auto found = g_side.find({layer, gen});
  if (found == g_side.end()) return;
  SideLayer& side = found->second;
  std::unordered_set<int> seen;
  for (size_t index = 0; index < count; ++index) {
    int expert = static_cast<int>(values[index]);
    if (!seen.insert(expert).second) continue;
    if (!side.e2r.count(expert)) continue;
    uint32_t& freq = side.freq[expert];
    freq = std::numeric_limits<uint32_t>::max() - freq < kDemandBoost
        ? std::numeric_limits<uint32_t>::max()
        : freq + kDemandBoost;
  }
}
static inline void side_refinement_ready(int64_t forward_id, int target_layer) {
  if (forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_side_drain_mutex);
  g_side_refinement_ready[{forward_id, target_layer}] = true;
  g_side_drain_cv.notify_all();
}
static inline void side_progress_notify() {
  // 与 wait_experts 共用 drain mutex，避免“predicate 刚检查完、尚未进入 wait”
  // 时 publish 的通知丢失。
  std::lock_guard<std::mutex> lk(g_side_drain_mutex);
  g_side_drain_cv.notify_all();
}

// 临时诊断：只追踪指定 (layer,row) 的所有账本/字节事件（SIDE_TRACE_LAYER/ROW）。
static inline bool side_trace_hit(int layer, int row) {
  const char* le = std::getenv("SIDE_TRACE_LAYER");
  const char* re = std::getenv("SIDE_TRACE_ROW");
  if (!le || !re) return false;
  return layer == atoi(le) && row == atoi(re);
}

// 临时诊断：跨线程全局事件序号 + 线程短 id，用于把 reserve(Metal 回调线程)/read_publish(bg 线程)/
// sideregion_kv(主线程) 的事件按发生顺序排出来（root-cause 取证，SIDE_TRACE_* 命中时才用）。
static std::atomic<uint64_t> g_side_ev{0};
static inline unsigned side_tid() {
  return static_cast<unsigned>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff);
}

struct SideReadyState {
  NS::SharedPtr<MTL::SharedEvent> event;
  uint64_t value = 0;
};
static std::mutex g_side_ready_event_mutex;
static NS::SharedPtr<MTL::SharedEvent> g_side_ready_event;
static MTL::Device* g_side_ready_event_device = nullptr;
static std::atomic<uint64_t> g_side_ready_event_value{1};

static void side_ready_signal(const std::shared_ptr<SideReadyState>& state) {
  if (state && state->event) state->event->setSignaledValue(state->value);
}

static void side_wait_pending_empty(int layer, int gen) {
  std::unique_lock<std::mutex> lock(g_side_drain_mutex);
  g_side_drain_cv.wait(lock, [&] {
    std::lock_guard<std::mutex> side_lock(g_side_mutex);
    auto found = g_side.find({layer, gen});
    return found == g_side.end() || found->second.pending_e2r.empty();
  });
}

static bool refinement_wait_all_enabled() {
  const char* value = std::getenv("PREFETCH_REFINEMENT_WAIT_ALL");
  return value && value[0] == '1';
}

class PrefetchPoolSideRegionPrimitive : public mx::Primitive {
 public:
  PrefetchPoolSideRegionPrimitive(mx::Stream s, std::vector<int> seg_nbytes, int layer, int gen,
                                  std::string path, size_t stride, std::vector<int> resident,
                                  int spec_slots, int base_row, int source_layer,
                                  int64_t forward_id, int priority,
                                  std::shared_ptr<SideReadyState> ready_state)
      : Primitive(s), seg_(std::move(seg_nbytes)), layer_(layer), gen_(gen), path_(std::move(path)),
        stride_(stride), resident_(std::move(resident)), spec_(spec_slots), base_(base_row),
        source_layer_(source_layer), forward_id_(forward_id), priority_(priority),
        ready_state_(std::move(ready_state)) {}
  const char* name() const override { return "PrefetchPoolSideRegionPrimitive"; }

  void eval_cpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    run(in);
  }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    std::vector<uint8_t*> ptrs;
    ptrs.reserve(in.size() - 1);
    for (size_t i = 1; i < in.size(); ++i) {
      mx::array a = in[i];                 // 非 const 拷贝才能取可写指针
      ptrs.push_back(a.data<uint8_t>());
    }
    mx::array ids = in[0];
    const uint32_t* idp = ids.data<uint32_t>();
    size_t n = ids.size();
    std::vector<int> seg = seg_;
    int layer = layer_;
    int gen = gen_;
    std::string path = path_;
    size_t stride = stride_;
    std::vector<int> resident = resident_;
    int spec = spec_, base = base_;
    const bool unified = base < 0;
    const int real_cap = unified ? -base : 0;
    int source_layer = source_layer_;
    int64_t forward_id = forward_id_;
    int priority = priority_;
    auto& enc = mx::metal::get_command_encoder(stream());
    if (ready_state_) {
      auto& device = mx::metal::device(stream().device);
      std::lock_guard<std::mutex> lock(g_side_ready_event_mutex);
      if (!g_side_ready_event || g_side_ready_event_device != device.mtl_device()) {
        g_side_ready_event = NS::TransferPtr(device.mtl_device()->newSharedEvent());
        g_side_ready_event_device = device.mtl_device();
        g_side_ready_event_value.store(1, std::memory_order_relaxed);
      }
      ready_state_->event = g_side_ready_event;
      ready_state_->value =
          g_side_ready_event_value.fetch_add(1, std::memory_order_relaxed);
    }
    auto ready_state = ready_state_;
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    prefetch_audit_note_submit(forward_id, source_layer, layer, gen);
    // 提交即计在途：在 eval_gpu（预取提交）时 +1，直到后台字节写完才 -1。这样消费方前向开头
    // sideregion_drain() 能等到「即使 GPU 完成回调尚未触发」的 fill，闭合跨前向的写-读竞态。
    side_inflight_start(forward_id, layer);
    // in 按值捕获 → 保活 expert_ids 与所有池数组 buffer；idp/ptrs 在回调里指针有效。
    cb->addCompletedHandler(
        [in, ptrs, seg, idp, n, layer, gen, path, stride, resident, spec, base,
         forward_id, priority, ready_state, unified,
         real_cap](MTL::CommandBuffer*) {
          // 阶段1（回调线程、持锁极短）：读惰性 id（此刻已算完）、预留侧区行。
          // 必须在回调里——id 只有 command buffer 完成后才有效。
          prefetch_audit_note_callback(forward_id, layer, idp, n, resident);
          // Progressive refinement resubmits the complete legal union, which
          // contains the immutable early core.  With core<=15 and final
          // width<=26/32, early+tail jointly fit the admission share, so
          // waiting for *all* early reads only serializes route-critical I/O
          // behind false positives and destroys the T-1 window.  Reservation
          // is concurrency-safe: an already-pending expert is deduplicated and
          // each new tail expert receives a distinct row. Keep the old full
          // drain only as a diagnostic compatibility switch.
          if (priority >= 2 && refinement_wait_all_enabled()) {
            if (unified) real_prefetch_wait_all(layer);
            else side_wait_pending_empty(layer, gen);
          }
          auto to_read = unified
              ? real_prefetch_reserve(
                    idp, n, layer, real_cap, spec, resident, priority)
              : reserve(
                    idp, n, layer, gen, resident, spec, base, priority);
          // Unified direct-slot prefetch bypasses reserve(), where the legacy
          // side-cache counters are normally updated. Count its actual SSD
          // reservations here so per-layer profiling reflects the production
          // path instead of reporting an empty map beside nonzero pool loads.
          if (unified) side_stat_note_reads(layer, to_read);
          if (priority > 0) side_refinement_ready(forward_id, layer);
          if (to_read.empty()) {
            prefetch_audit_note_pread(forward_id, layer, 0);
            prefetch_audit_note_publish(forward_id, layer, 0);
            side_inflight_done(forward_id, layer);  // 无缺口可读：立即消账，避免 drain 空等
            side_ready_signal(ready_state);
            return;
          }
          // 诊断门控 SIDEREGION_SYNC=1：回调内同步 pread+memcpy+publish（不派 bg），
          // 消除「异步 bg fill 与下一前向 gather 竞态」这一变量，用于 systematic-debugging 取证。
          static const bool kSync = []() {
            const char* e = std::getenv("SIDEREGION_SYNC");
            return e && e[0] == '1';
          }();
          if (kSync) {
            read_publish(
                ptrs, seg, to_read, path, stride, layer, gen, forward_id,
                priority > 0, unified, real_cap);
            side_inflight_done(forward_id, layer);
            side_ready_signal(ready_state);
            return;
          }
          static const bool kFirstRowHighPriority = []() {
            const char* value = std::getenv(
                "PREFETCH_FIRST_ROW_HIGH_PRIORITY");
            return value && value[0] == '1';
          }();
          if (priority == 0 && kFirstRowHighPriority && !to_read.empty()) {
            // The first physical row has much higher precision than the tail
            // (about 87% on held-out Qwen traces).  Give only that row demand-
            // class queueing latency while preserving early/F_NOCACHE I/O
            // semantics; the lower-confidence tail remains low priority.
            const size_t task_count = to_read.size() > 1 ? 2 : 1;
            auto remaining = std::make_shared<std::atomic<size_t>>(task_count);
            auto finish = [forward_id, layer, ready_state, remaining]() {
              if (remaining->fetch_sub(1) == 1) {
                side_inflight_done(forward_id, layer);
                side_ready_signal(ready_state);
              }
            };
            const auto first = to_read.front();
            bg_submit_task(
                [in, ptrs, seg, first, path, stride, layer, gen,
                 forward_id, ready_state, unified, real_cap, finish]() {
                  read_publish(
                      ptrs, seg, {first}, path, stride, layer, gen,
                      forward_id, /*cached=*/false, unified, real_cap);
                  finish();
                },
                /*priority=*/1);
            if (to_read.size() > 1) {
              std::vector<std::pair<int, int>> tail(
                  to_read.begin() + 1, to_read.end());
              bg_submit_task(
                  [in, ptrs, seg, tail, path, stride, layer, gen,
                   forward_id, ready_state, unified, real_cap, finish]() {
                    read_publish(
                        ptrs, seg, tail, path, stride, layer, gen,
                        forward_id, /*cached=*/false, unified, real_cap);
                    finish();
                  },
                  /*priority=*/0);
            }
            return;
          }
          static const bool kSplitEarlyExperts = []() {
            const char* value = std::getenv("PREFETCH_SPLIT_EARLY_EXPERTS");
            return value && value[0] == '1';
          }();
          if ((priority >= 2 || kSplitEarlyExperts) && to_read.size() > 1) {
            // Route-critical exact tail: one batch task would serialize up to
            // ~18 random expert records on a single worker. Split by expert so
            // the reserved high-priority workers can issue independent preadv
            // calls and publish each completed row immediately. The logical
            // submission remains one inflight item/event and is released only
            // after the final expert finishes.
            auto remaining = std::make_shared<std::atomic<size_t>>(
                to_read.size());
            for (auto placement : to_read) {
              bg_submit_task(
                  [in, ptrs, seg, placement, path, stride, layer, gen,
                   forward_id, priority, ready_state, unified, real_cap,
                   remaining]() {
                    read_publish(
                        ptrs, seg, {placement}, path, stride, layer, gen,
                        forward_id, priority > 0, unified, real_cap);
                    if (remaining->fetch_sub(1) == 1) {
                      side_inflight_done(forward_id, layer);
                      side_ready_signal(ready_state);
                    }
                  },
                  priority);
            }
            return;
          }
          // 阶段2+3 派给自由后台线程：pread + 逐行发布脱离 Metal 回调线程，
          // 与主 stream 后续层计算、多层预取互相并发。
          bg_submit_task([in, ptrs, seg, to_read, path, stride, layer, gen,
                          forward_id, priority, ready_state, unified,
                          real_cap]() {
            read_publish(
                ptrs, seg, to_read, path, stride, layer, gen, forward_id,
                priority > 0, unified, real_cap);
            side_inflight_done(forward_id, layer);
            side_ready_signal(ready_state);
          }, priority);
        });
    if (ready_state_) {
      // The ready wait is a dependent MLX primitive. Force an evaluator-owned
      // submission boundary so it cannot be encoded into the producer command
      // buffer whose completion callback is responsible for signaling ready.
      auto& device = mx::metal::device(stream().device);
      auto source = R"METAL(
        #include <metal_stdlib>
        using namespace metal;
        kernel void side_ready_submit(device uchar* guard [[buffer(0)]],
                                      uint tid [[thread_position_in_grid]]) {
          if (tid == 0) guard[0] = guard[0];
        }
      )METAL";
      auto library = device.get_library(
          "side_ready_submit", [source] { return std::string(source); });
      auto submit = device.get_kernel("side_ready_submit", library);
      const int max_ops = std::get<0>(device.get_max_ops_mb_per_buffer());
      for (int index = 0; index <= max_ops && !enc.needs_commit(); ++index) {
        enc.set_compute_pipeline_state(submit);
        enc.set_output_array(out[0], 0);
        enc.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
      }
    }
  }

 private:
  void run(const std::vector<mx::array>& in) {
    std::vector<uint8_t*> ptrs;
    ptrs.reserve(in.size() - 1);
    for (size_t i = 1; i < in.size(); ++i) {
      mx::array a = in[i];
      a.eval();                            // 与 PrefetchStagingCachedPrimitive::eval_cpu 一致：先物化输入
      ptrs.push_back(a.data<uint8_t>());
    }
    mx::array ids = in[0];
    ids.eval();
    // CPU 路径同步执行（测试/无 GPU 时）：预留 + 读发布一气呵成。
    prefetch_audit_note_submit(forward_id_, source_layer_, layer_, gen_);
    prefetch_audit_note_callback(
        forward_id_, layer_, ids.data<uint32_t>(), ids.size(), resident_);
    const bool unified = base_ < 0;
    const int real_cap = unified ? -base_ : 0;
    if (priority_ >= 2 && refinement_wait_all_enabled()) {
      if (unified) real_prefetch_wait_all(layer_);
      else side_wait_pending_empty(layer_, gen_);
    }
    auto to_read = unified
        ? real_prefetch_reserve(
              ids.data<uint32_t>(), ids.size(), layer_, real_cap, spec_, resident_,
              priority_)
        : reserve(
              ids.data<uint32_t>(), ids.size(), layer_, gen_, resident_, spec_,
              base_, priority_);
    read_publish(
        ptrs, seg_, to_read, path_, stride_, layer_, gen_, forward_id_,
        priority_ > 0, unified, real_cap);
  }

  // 阶段1：过滤常驻/去重 → 淘汰 ∉P 的旧行 → 为缺口预留物理行（出 free，暂不入 e2r，
  // 避免消费者在字节写好前看到 e2r 命中而 gather 到脏字节）。返回 (expert, 预留行)。
  static std::vector<std::pair<int, int>> reserve(
      const uint32_t* idp, size_t n, int layer, int gen, const std::vector<int>& resident,
      int spec, int base, int priority) {
    side_stat_add(g_stat_input_ids, static_cast<long>(n));
    const char* lfu_env = std::getenv("SIDEREGION_LFU");   // 每次读,便于测试切换
    // 默认开:持久 LFU 单缓冲=生产路径(cli 默认)。仅显式 SIDEREGION_LFU=0 回退旧 legacy 双缓冲。
    bool lfu = !lfu_env || lfu_env[0] != '0';
    std::unordered_set<int> res(resident.begin(), resident.end());
    std::vector<int> P;
    std::unordered_set<int> Pset, seen;
    for (size_t i = 0; i < n; ++i) {
      int e = static_cast<int>(idp[i]);
      if (res.count(e) || !seen.insert(e).second) continue;
      P.push_back(e);
      Pset.insert(e);
    }
    side_stat_add(g_stat_candidates, static_cast<long>(P.size()));
    std::vector<std::pair<int, int>> to_read;
    // Early rerank and exact T-1 refinement have different precision.  Keep
    // speculative early I/O narrow, but let a route-critical refinement fill
    // every still-missing real route in one callback.  A non-positive
    // refinement budget means unlimited, matching the historical convention.
    const char* read_budget_env = std::getenv(
        priority >= 2
            ? "PREFETCH_REFINEMENT_READ_BUDGET"
            : "PREFETCH_PHYSICAL_READ_BUDGET");
    const int read_budget = (
        read_budget_env && read_budget_env[0]
        ? std::max(0, std::atoi(read_budget_env)) : 0);
    std::lock_guard<std::mutex> lk(g_side_mutex);
    SideLayer& c = g_side[{layer, gen}];
    ensure_side_table_locked(c);
    if (!c.inited) {
      for (int r = 0; r < spec; ++r) c.free_rows.push_back(base + r);
      c.base = base;
      c.spec = spec;
      c.inited = true;
    }
    // A fallback may have loaded an expert into the real region while an old
    // complete copy still occupies a persistent side row.  The source-time
    // resident snapshot is stable until this target layer is consumed, so
    // keeping both copies only shrinks the effective 32+spec working set.
    // Reclaim complete duplicates before selecting victims; pending rows keep
    // their reservation until their writer finishes and are handled on a
    // later source occurrence.
    static const bool kReclaimRealDuplicates = []() {
      const char* value = std::getenv("SIDEREGION_RECLAIM_REAL_DUPLICATES");
      return !value || value[0] != '0';
    }();
    if (kReclaimRealDuplicates) {
      for (int expert : res) {
        auto duplicate = c.e2r.find(expert);
        if (duplicate == c.e2r.end()) continue;
        if (!side_try_unmap_unleased_locked(
                c, layer, expert, duplicate->second)) continue;
        c.free_rows.push_back(duplicate->second);
        c.e2r.erase(duplicate);
        c.freq.erase(expert);
      }
    }
    if (!lfu) {
      // 旧行为:∉P 全弃(一次性预取批)。
      for (auto it = c.e2r.begin(); it != c.e2r.end();) {
        if (!Pset.count(it->first)
            && side_try_unmap_unleased_locked(
                c, layer, it->first, it->second)) {
          c.free_rows.push_back(it->second);
          it = c.e2r.erase(it);
        } else {
          ++it;
        }
      }
      for (int e : P) {
        if (read_budget > 0 &&
            static_cast<int>(to_read.size()) >= read_budget) break;
        if (c.e2r.count(e)) {
          side_stat_add(g_stat_side_hits, 1);
          continue;
        }
        if (c.pending_e2r.count(e)) continue;
        if (c.free_rows.empty()) continue;
        int row = c.free_rows.back();
        to_read.emplace_back(e, row);
        c.pending_e2r[e] = row;
        c.free_rows.pop_back();
      }
      side_stat_add(g_stat_reserved_reads, static_cast<long>(to_read.size()));
      side_stat_note_reads(layer, to_read);
      return to_read;
    }
    // LFU 持久:∉P 不清;再预测命中已驻专家 freq+1(越常预测越热)。
    for (int e : P) {
      if (c.e2r.count(e)) {
        c.freq[e] += 1;
        side_stat_add(g_stat_side_hits, 1);
      } else if (c.pending_e2r.count(e)) {
        // 在途行尚不能算 deadline-complete side hit，但必须视作已预留，
        // 避免 refinement callback 为同一 expert 再占一行。
        continue;
      }
    }
    for (int e : P) {
      if (read_budget > 0 &&
          static_cast<int>(to_read.size()) >= read_budget) break;
      if (c.e2r.count(e) || c.pending_e2r.count(e))
        continue;                                 // 已驻或在途,跳过(不重读)
      int row;
      if (!c.free_rows.empty()) {
        row = c.free_rows.back();
        c.free_rows.pop_back();
        if (side_trace_hit(layer, row))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d RESERVE_FROM_FREE row=%d expert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, e);
      } else {
        // free 空:LFU 淘汰 e2r 中 freq 最小且 ∉P 者(tie-break:最小 expert id)。
        int victim = -1;
        uint32_t best = 0;
        for (auto& kv : c.e2r) {
          if (Pset.count(kv.first)) continue;     // 不淘本步要用的
          if (side_row_leased_locked(layer, kv.second)) continue;
          uint32_t f = c.freq.count(kv.first) ? c.freq[kv.first] : 0;
          if (victim < 0 || f < best || (f == best && kv.first < victim)) {
            victim = kv.first;
            best = f;
          }
        }
        if (victim < 0) continue;                 // 全是 P 热,无可淘 → 本步不读
        row = c.e2r[victim];
        if (!side_try_unmap_unleased_locked(
                c, layer, victim, row)) continue;
        c.e2r.erase(victim);
        c.freq.erase(victim);
        side_stat_add(g_stat_evictions, 1);
        if (side_trace_hit(layer, row))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d EVICT_REUSE row=%d victim=%d newExpert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, victim, e);
      }
      to_read.emplace_back(e, row);
      c.pending_e2r[e] = row;
    }
    side_stat_add(g_stat_reserved_reads, static_cast<long>(to_read.size()));
    side_stat_note_reads(layer, to_read);
    // ===== 临时诊断（SIDE_AUDIT=1）：reserve 结束时审计侧区行账本自洽性 =====
    if (std::getenv("SIDE_AUDIT")) side_audit(c, layer, gen, "reserve", to_read);
    return to_read;
  }

  // 审计：检测「一行被两专家占用」「free 与 e2r 行重叠」「to_read 行仍被 e2r 占用」。
  static void side_audit(SideLayer& c, int layer, int gen, const char* where,
                         const std::vector<std::pair<int, int>>& to_read) {
    std::map<int, int> row_owner;                 // row -> expert
    for (auto& p : c.e2r) {
      auto it = row_owner.find(p.second);
      if (it != row_owner.end())
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d DOUBLE_OWN row=%d experts=%d,%d\n",
                where, layer, gen, p.second, it->second, p.first);
      row_owner[p.second] = p.first;
    }
    for (auto& p : c.pending_e2r) {
      auto it = row_owner.find(p.second);
      if (it != row_owner.end())
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d PENDING_DOUBLE_OWN row=%d experts=%d,%d\n",
                where, layer, gen, p.second, it->second, p.first);
      row_owner[p.second] = p.first;
    }
    std::unordered_set<int> free_set(c.free_rows.begin(), c.free_rows.end());
    for (auto& p : c.e2r)
      if (free_set.count(p.second))
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d FREE_E2R_OVERLAP row=%d owned_by=%d\n",
                where, layer, gen, p.second, p.first);
    for (auto& p : c.pending_e2r)
      if (free_set.count(p.second))
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d FREE_PENDING_OVERLAP row=%d pending_by=%d\n",
                where, layer, gen, p.second, p.first);
    if (c.free_rows.size() != free_set.size())
      fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d FREE_DUP free_n=%zu uniq=%zu\n",
              where, layer, gen, c.free_rows.size(), free_set.size());
    for (auto& pr : to_read) {
      auto pending = c.pending_e2r.find(pr.first);
      bool owns_pending = (
          pending != c.pending_e2r.end() && pending->second == pr.second);
      if (row_owner.count(pr.second) && !owns_pending)
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d TOREAD_LIVE_ROW row=%d assigned_to=%d still_owned_by=%d\n",
                where, layer, gen, pr.second, pr.first, row_owner[pr.second]);
    }
  }

 public:
  // 阶段2+3：pread blob 整行 + 各段 memcpy 直写进对应 per-key 池数组的物理侧区行（不持锁），
  // 写完后持锁发布 e2r。ptrs[i] 为第 i 个池 key 数组的 buffer 指针（C++ 拥有、地址恒定不迁移，
  // 由 Route 3 底座保证——见 pool_owned_zeros），故后台异步直写安全、消费侧读同一 buffer。
  // 在后台线程跑：与计算并发，且不阻塞消费侧的 sideregion_kv。
  static void read_publish(const std::vector<uint8_t*>& ptrs, const std::vector<int>& seg,
                           const std::vector<std::pair<int, int>>& to_read,
                           const std::string& path, size_t stride, int layer, int gen,
                           int64_t forward_id, bool cached, bool unified,
                           int real_cap) {
    // Reservation happens in the Metal completion callback, before this low
    // priority task enters its worker queue. If target demand has arrived in
    // the meantime, release the untouched rows so demand can issue the same
    // experts immediately on its high-priority reader pool.
    if (target_already_consumed(forward_id, layer)) {
      if (unified) {
        for (const auto& item : to_read)
          real_prefetch_abort(layer, item.first, item.second, real_cap);
      } else {
        std::lock_guard<std::mutex> lock(g_side_mutex);
        SideLayer& side = g_side[{layer, gen}];
        for (const auto& item : to_read) {
          auto pending = side.pending_e2r.find(item.first);
          if (pending == side.pending_e2r.end() ||
              pending->second != item.second) continue;
          side.pending_e2r.erase(pending);
          side.free_rows.push_back(item.second);
        }
      }
      side_progress_notify();
      prefetch_audit_note_pread(forward_id, layer, 0);
      prefetch_audit_note_publish(forward_id, layer, 0);
      return;
    }
    prefetch_audit_note_pread(forward_id, layer, to_read.size());
    const bool partial_projection = unified && !cached && seg.size() == 9 && [] {
      const char* value = std::getenv("PREFETCH_PARTIAL_PROJECTIONS");
      const char* demand_async = std::getenv("DEMAND_ASYNC");
      // run_mtp_spec deliberately switches async demand off during prompt
      // ingestion. Keep full-row publication there; synchronous demand has no
      // two-stage consumer and must never wait on a prefix-only reservation.
      return value && value[0] == '1' &&
          demand_async && demand_async[0] == '1';
    }();
    const bool partial_down_first = partial_projection && [] {
      const char* value = std::getenv("PREFETCH_PARTIAL_ORDER");
      return value && std::strcmp(value, "down_first") == 0;
    }();
    const int partial_full_rows = partial_projection ? [] {
      const char* value = std::getenv("PREFETCH_PARTIAL_FULL_ROWS");
      return value && value[0] ? std::max(0, std::atoi(value)) : 0;
    }() : 0;
    // 段在 blob 记录内的偏移（段顺序 = seg 顺序 = ptrs/池 key 顺序）。
    std::vector<size_t> seg_off(seg.size(), 0);
    size_t acc = 0;
    for (size_t i = 0; i < seg.size(); ++i) { seg_off[i] = acc; acc += static_cast<size_t>(seg[i]); }
    // Late route-critical rows are small (~1.7 MiB each) and may be demanded
    // immediately. Normal cached pread avoids the large non-cached random-I/O
    // latency and lets a subsequent exact fallback reuse the page. Early
    // broad speculation stays F_NOCACHE by default.  On machines with enough
    // spare unified memory an opt-in page-cache experiment can retain the
    // repeatedly-read hot expert records without enlarging the MLX pool.
    static const int kEarlyPageCacheRows = []() {
      const char* rows = std::getenv("PREFETCH_EARLY_PAGE_CACHE_ROWS");
      if (rows && rows[0]) return std::max(0, std::atoi(rows));
      const char* value = std::getenv("PREFETCH_EARLY_PAGE_CACHE");
      return value && value[0] == '1' ? INT_MAX : 0;
    }();
    const int cached_rows = cached
        ? static_cast<int>(to_read.size())
        : std::min(kEarlyPageCacheRows, static_cast<int>(to_read.size()));
    int cached_fd = cached_rows > 0 ? ::open(path.c_str(), O_RDONLY) : -1;
    int nocache_fd = cached_rows < static_cast<int>(to_read.size())
        ? open_blob_nocache(path.c_str()) : -1;
    if ((cached_rows > 0 && cached_fd < 0) ||
        (cached_rows < static_cast<int>(to_read.size()) && nocache_fd < 0)) {
      if (cached_fd >= 0) ::close(cached_fd);
      if (nocache_fd >= 0) ::close(nocache_fd);
      side_stat_add(g_stat_pread_fail, static_cast<long>(to_read.size()));
      if (unified) {
        for (auto& pr : to_read)
          real_prefetch_abort(layer, pr.first, pr.second, real_cap);
      } else {
        std::lock_guard<std::mutex> lk(g_side_mutex);
        SideLayer& c = g_side[{layer, gen}];
        for (auto& pr : to_read) {
          auto pending = c.pending_e2r.find(pr.first);
          if (pending != c.pending_e2r.end() && pending->second == pr.second)
            c.pending_e2r.erase(pending);
          c.free_rows.push_back(pr.second);
        }
      }
      side_progress_notify();
      prefetch_audit_note_publish(forward_id, layer, 0);
      return;
    }
    static const bool kPreadv = []() {
      const char* value = std::getenv("SIDEREGION_PREADV");
      return !value || value[0] != '0';
    }();
    // Compatibility path keeps one reusable per-worker record buffer.  The
    // preadv path below writes each blob segment directly into its final pool
    // row and therefore needs neither this buffer nor a second memcpy pass.
    static thread_local std::vector<uint8_t> rec;
    if (!kPreadv && rec.size() < stride) rec.resize(stride);
    size_t completed = 0;
    for (size_t read_index = 0; read_index < to_read.size(); ++read_index) {
      // A low-priority batch is score ordered, but one worker processes its
      // rows serially.  Once target demand has entered, continuing through the
      // low-ranked tail only delays the high-priority exact reads.  Release
      // untouched reservations so the demand waiter can immediately allocate
      // and fetch the true route itself.  The existing stale-cancel switch
      // gates this behavior and remains off by default.
      if (read_index > 0 && target_already_consumed(forward_id, layer)) {
        if (unified) {
          for (size_t pending = read_index; pending < to_read.size(); ++pending)
            real_prefetch_abort(
                layer, to_read[pending].first, to_read[pending].second,
                real_cap);
        } else {
          std::lock_guard<std::mutex> lock(g_side_mutex);
          SideLayer& side = g_side[{layer, gen}];
          for (size_t pending_index = read_index;
               pending_index < to_read.size(); ++pending_index) {
            const auto& item = to_read[pending_index];
            auto pending = side.pending_e2r.find(item.first);
            if (pending == side.pending_e2r.end() ||
                pending->second != item.second) continue;
            side.pending_e2r.erase(pending);
            side.free_rows.push_back(item.second);
          }
          side_progress_notify();
        }
        break;
      }
      auto& pr = to_read[read_index];
      const bool row_partial = partial_projection &&
          static_cast<int>(read_index) >= partial_full_rows;
      const size_t first_segment = row_partial && partial_down_first ? 6 : 0;
      const size_t io_segments = row_partial
          ? (partial_down_first ? seg.size() - first_segment : 6)
          : seg.size();
      int fd = static_cast<int>(read_index) < cached_rows
          ? cached_fd : nocache_fd;
      int e = pr.first, row = pr.second;
      ssize_t bytes_read = -1;
      if (kPreadv) {
        std::vector<iovec> vectors(io_segments);
        size_t expected = 0;
        for (size_t local = 0; local < io_segments; ++local) {
          const size_t i = first_segment + local;
          vectors[local].iov_base = ptrs[i] + static_cast<size_t>(row) *
              static_cast<size_t>(seg[i]);
          vectors[local].iov_len = static_cast<size_t>(seg[i]);
          expected += static_cast<size_t>(seg[i]);
        }
        bytes_read = ::preadv(
            fd, vectors.data(), static_cast<int>(vectors.size()),
            static_cast<off_t>(
                static_cast<size_t>(e) * stride + seg_off[first_segment]));
      } else {
        bytes_read = ::pread(
            fd, rec.data(), stride,
            static_cast<off_t>(static_cast<size_t>(e) * stride));
      }
      const size_t expected_bytes = (kPreadv && row_partial)
          ? std::accumulate(
                seg.begin() + first_segment,
                seg.begin() + first_segment + io_segments, size_t{0})
          : stride;
      if (bytes_read != static_cast<ssize_t>(expected_bytes)) {
        side_stat_add(g_stat_pread_fail, 1);
        if (unified) {
          real_prefetch_abort(layer, e, row, real_cap);
        } else {                                              // 读失败：行还回 free
          std::lock_guard<std::mutex> lk(g_side_mutex);
          SideLayer& c = g_side[{layer, gen}];
          auto pending = c.pending_e2r.find(e);
          if (pending != c.pending_e2r.end() && pending->second == row)
            c.pending_e2r.erase(pending);
          c.free_rows.push_back(row);
        }
        side_progress_notify();
        continue;
      }
      side_stat_add(g_stat_pread_ok, 1);
      if (!kPreadv) {
        // Compatibility path: scatter the contiguous record into final rows.
        for (size_t i = 0; i < seg.size(); ++i)
          std::memcpy(ptrs[i] + static_cast<size_t>(row) * static_cast<size_t>(seg[i]),
                      rec.data() + seg_off[i], static_cast<size_t>(seg[i]));
      }
      if (side_trace_hit(layer, row))
        fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d WRITEPOOL row=%d expert=%d\n",
                (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, e);
      // 每个专家的整条 blob 与所有 segment 都写完后立即发布。旧实现等整批
      // 候选全部读完才统一发布，使排序最前面的专家虽已完整到达却仍在 target
      // deadline 不可见。逐专家发布保留相同的“半写行永不暴露”不变量，同时
      // 让 rerank 顺序真正成为 SSD deadline 的优先级顺序。
      if (unified) {
        if (row_partial)
          real_prefetch_publish_partial(layer, e, row, real_cap);
        else
          real_prefetch_publish(layer, e, row, real_cap);
      } else {
        std::lock_guard<std::mutex> lk(g_side_mutex);
        SideLayer& c = g_side[{layer, gen}];
        auto pending = c.pending_e2r.find(e);
        if (pending == c.pending_e2r.end() || pending->second != row) {
          // reset/异常账本变化后绝不能发布一个失去 reservation 的行。
          c.free_rows.push_back(row);
          continue;
        }
        c.pending_e2r.erase(pending);
        c.e2r[e] = row;
        side_table_set_locked(c, e, row);
        if (!c.freq.count(e)) c.freq[e] = 1;
        if (side_trace_hit(layer, row))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d PUBLISH row=%d expert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, e);
        if (std::getenv("SIDE_AUDIT")) side_audit(c, layer, gen, "publish-one", {});
      }
      side_progress_notify();
      ++completed;
      static const bool kPreemptForDemand = []() {
        const char* value = std::getenv("PREFETCH_PREEMPT_FOR_DEMAND");
        return value && value[0] == '1';
      }();
      if (!cached && kPreemptForDemand)
        bg_reader_wait_high_idle();
    }
    if (cached_fd >= 0) ::close(cached_fd);
    if (nocache_fd >= 0) ::close(nocache_fd);
    if (!unified && std::getenv("SIDE_AUDIT")) {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      side_audit(c, layer, gen, "publish-batch-end", {});
    }
    prefetch_audit_note_publish(forward_id, layer, completed);
  }

  std::vector<int> seg_;
  int layer_;
  int gen_;
  std::string path_;
  size_t stride_;
  std::vector<int> resident_;
  int spec_;
  int base_;
  int source_layer_;
  int64_t forward_id_;
  int priority_;
  std::shared_ptr<SideReadyState> ready_state_;
};

void prefetch_unified_ready(
    const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes,
    const uint32_t* expert_ids, size_t count,
    int target_layer, const std::string& path, int stride,
    const std::vector<int>& resident, int speculative_limit, int real_cap,
    int source_layer, int64_t forward_id) {
  prefetch_audit_note_submit(
      forward_id, source_layer, target_layer, /*physical_gen=*/0);
  prefetch_audit_note_callback(
      forward_id, target_layer, expert_ids, count, resident);
  side_inflight_start(forward_id, target_layer);
  auto reads = real_prefetch_reserve(
      expert_ids, count, target_layer, real_cap,
      speculative_limit, resident, /*priority=*/0);
  side_stat_note_reads(target_layer, reads);
  if (reads.empty()) {
    prefetch_audit_note_pread(forward_id, target_layer, 0);
    prefetch_audit_note_publish(forward_id, target_layer, 0);
    side_inflight_done(forward_id, target_layer);
    return;
  }
  std::vector<uint8_t*> pointers;
  pointers.reserve(pool_list.size());
  for (auto pool : pool_list) pointers.push_back(pool.data<uint8_t>());
  const char* parallel_rows_env = std::getenv("PREFETCH_PARALLEL_ROWS");
  const bool parallel_rows = parallel_rows_env && parallel_rows_env[0] == '1';
  if (parallel_rows && reads.size() > 1) {
    // A batch is otherwise one low-priority job, so one worker reads all
    // predicted experts serially even when PREFETCH_LOW_WORKERS=2.  Split the
    // tiny physical prefix into independent jobs: the reader's low-priority
    // concurrency cap still protects high-priority demand I/O, while the
    // first two ranked experts can reach the deadline in parallel.
    auto remaining = std::make_shared<std::atomic<size_t>>(reads.size());
    for (const auto& read : reads) {
      std::vector<std::pair<int, int>> one{read};
      bg_submit_task(
          [pool_list, pointers, seg_nbytes, one, path, stride, target_layer,
           forward_id, real_cap, remaining]() {
            PrefetchPoolSideRegionPrimitive::read_publish(
                pointers, seg_nbytes, one, path,
                static_cast<size_t>(stride), target_layer, /*gen=*/0,
                forward_id, /*cached=*/false, /*unified=*/true, real_cap);
            if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1)
              side_inflight_done(forward_id, target_layer);
          },
          /*priority=*/0);
    }
    return;
  }
  // Keep every C++-owned MLX buffer alive until the background worker has
  // finished writing and publishing all reserved rows.
  bg_submit_task(
      [pool_list, pointers, seg_nbytes, reads, path, stride, target_layer,
       forward_id, real_cap]() {
        PrefetchPoolSideRegionPrimitive::read_publish(
            pointers, seg_nbytes, reads, path,
            static_cast<size_t>(stride), target_layer, /*gen=*/0,
            forward_id, /*cached=*/false, /*unified=*/true, real_cap);
        side_inflight_done(forward_id, target_layer);
      },
      /*priority=*/0);
}

class SideReadyWaitPrimitive : public mx::Primitive {
 public:
  SideReadyWaitPrimitive(mx::Stream stream, std::shared_ptr<SideReadyState> state)
      : Primitive(stream), state_(std::move(state)) {}
  const char* name() const override { return "SideReadyWaitPrimitive"; }
  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    std::memcpy(
        outputs[0].data<uint8_t>(), inputs[0].data<uint8_t>(),
        outputs[0].nbytes());
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.end_encoding();
    encoder.get_command_buffer()->encodeWait(state_->event.get(), state_->value);
    auto source = R"METAL(
      #include <metal_stdlib>
      using namespace metal;
      kernel void side_ready_copy(
          device const uchar* src [[buffer(0)]],
          device uchar* dst [[buffer(1)]],
          uint tid [[thread_position_in_grid]]) { dst[tid] = src[tid]; }
    )METAL";
    auto library = device.get_library(
        "side_ready_wait", [source] { return std::string(source); });
    auto copy = device.get_kernel("side_ready_copy", library);
    encoder.set_compute_pipeline_state(copy);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_output_array(outputs[0], 1);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  }
 private:
  std::shared_ptr<SideReadyState> state_;
};

mx::array prefetch_pool_sideregion(
    const std::vector<mx::array>& pool_list, const std::vector<int>& seg_nbytes,
    const mx::array& expert_ids, int layer, const std::string& path, int stride,
    const std::vector<int>& resident, int spec_slots, int base_row, int gen,
    int source_layer, int64_t forward_id, int priority,
    mx::StreamOrDevice s = {}) {
  // 防御：段数必须与池数组个数一一对应（每段映射唯一一个 per-key 数组）。
  if (seg_nbytes.size() != pool_list.size()) {
    throw std::invalid_argument(
        "prefetch_pool_sideregion: seg_nbytes.size()=" + std::to_string(seg_nbytes.size()) +
        " != pool_list.size()=" + std::to_string(pool_list.size()));
  }
  // 防御：各段字节之和必须等于 stride（blob 记录恰好是各段的拼接）。
  size_t seg_sum = 0;
  for (int b : seg_nbytes) seg_sum += static_cast<size_t>(b);
  if (seg_sum != static_cast<size_t>(stride)) {
    throw std::invalid_argument(
        "prefetch_pool_sideregion: sum(seg_nbytes)=" + std::to_string(seg_sum) +
        " != stride=" + std::to_string(stride));
  }
  std::vector<mx::array> inputs;
  inputs.push_back(expert_ids);
  for (auto& a : pool_list) inputs.push_back(a);
  auto ready_state = priority >= 2 ? std::make_shared<SideReadyState>() : nullptr;
  mx::array raw(
      mx::Shape{1}, mx::uint8,
      std::make_shared<PrefetchPoolSideRegionPrimitive>(
          mx::to_stream(s), seg_nbytes, layer, gen, path, static_cast<size_t>(stride), resident,
          spec_slots, base_row, source_layer, forward_id, priority, ready_state),
      inputs);
  if (!ready_state) return raw;
  mx::Stream stream = mx::to_stream(s);
  return mx::array(
      raw.shape(), mx::uint8,
      std::make_shared<SideReadyWaitPrimitive>(stream, ready_state), {raw});
}

std::vector<int> sideregion_contents(int layer, int gen) {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  std::vector<int> out;
  auto it = g_side.find({layer, gen});
  if (it != g_side.end())
    for (auto& p : it->second.e2r) { out.push_back(p.first); out.push_back(p.second); }
  return out;
}

// 侧区 e2r → 两个 device mx.array (keys uint32, vals int32),直接在 C++ 从 map 建连续 buffer。
// 消掉 Python 侧 dict 构建 + list(...)→mx.array 的每层 host 胶水。
std::pair<mx::array, mx::array> sideregion_kv(int layer, int gen) {
  std::vector<uint32_t> keys;
  std::vector<int32_t> vals;
  {
    std::lock_guard<std::mutex> lk(g_side_mutex);
    auto it = g_side.find({layer, gen});
    if (it != g_side.end()) {
      keys.reserve(it->second.e2r.size());
      vals.reserve(it->second.e2r.size());
      for (auto& p : it->second.e2r) {
        keys.push_back(static_cast<uint32_t>(p.first));
        vals.push_back(static_cast<int32_t>(p.second));
        if (side_trace_hit(layer, p.second))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d CONSUME_KV row=%d expert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, p.second, p.first);
      }
    }
  }
  int n = static_cast<int>(keys.size());
  return {mx::array(keys.data(), mx::Shape{n}, mx::uint32),
          mx::array(vals.data(), mx::Shape{n}, mx::int32)};
}

mx::array sideregion_slot_table(int layer, int gen) {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  SideLayer& side = g_side[{layer, gen}];
  ensure_side_table_locked(side);
  return *side.slot_table;
}

mx::array sideregion_lease_table(int layer, int gen) {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  SideLayer& side = g_side[{layer, gen}];
  ensure_side_table_locked(side);
  return *side.lease_table;
}

void sideregion_reset() {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  for (auto& item : g_side) {
    SideLayer& side = item.second;
    if (side.slot_table)
      std::fill(
          side.slot_table->data<int32_t>(),
          side.slot_table->data<int32_t>() + 512, -1);
    if (side.lease_table)
      std::fill(
          side.lease_table->data<uint32_t>(),
          side.lease_table->data<uint32_t>() + 1024, 0u);
  }
  g_side.clear();
  g_side_row_leases.clear();
}

std::vector<long> sideregion_prefetch_stats() {
  return {
      g_stat_input_ids.load(std::memory_order_relaxed),
      g_stat_candidates.load(std::memory_order_relaxed),
      g_stat_side_hits.load(std::memory_order_relaxed),
      g_stat_reserved_reads.load(std::memory_order_relaxed),
      g_stat_evictions.load(std::memory_order_relaxed),
      g_stat_pread_ok.load(std::memory_order_relaxed),
      g_stat_pread_fail.load(std::memory_order_relaxed),
      [&]() {
        std::lock_guard<std::mutex> lk(g_stat_unique_mutex);
        return static_cast<long>(g_stat_unique_reads.size());
      }(),
  };
}

std::vector<long> sideregion_prefetch_reads_by_layer() {
  std::vector<long> out(64, 0);
  for (int layer = 0; layer < 64; ++layer)
    out[layer] = g_stat_layer_reads[layer].load(std::memory_order_relaxed);
  return out;
}

void sideregion_prefetch_stats_reset() {
  g_stat_input_ids.store(0, std::memory_order_relaxed);
  g_stat_candidates.store(0, std::memory_order_relaxed);
  g_stat_side_hits.store(0, std::memory_order_relaxed);
  g_stat_reserved_reads.store(0, std::memory_order_relaxed);
  g_stat_evictions.store(0, std::memory_order_relaxed);
  g_stat_pread_ok.store(0, std::memory_order_relaxed);
  g_stat_pread_fail.store(0, std::memory_order_relaxed);
  for (auto& value : g_stat_layer_reads)
    value.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(g_stat_unique_mutex);
    g_stat_unique_reads.clear();
  }
  g_stats_enabled.store(true, std::memory_order_relaxed);
}

std::unordered_map<int, int> sideregion_snapshot(int layer, int gen) {
  std::unordered_map<int, int> side;
  std::lock_guard<std::mutex> lk(g_side_mutex);
  auto it = g_side.find({layer, gen});
  if (it != g_side.end()) for (auto& p : it->second.e2r) side[p.first] = p.second;
  return side;
}

std::vector<int32_t> sideregion_lookup_values(
    int layer, int gen, const uint32_t* expert_ids, size_t count) {
  std::vector<int32_t> rows(count, -1);
  std::lock_guard<std::mutex> lk(g_side_mutex);
  auto found = g_side.find({layer, gen});
  if (found == g_side.end()) return rows;
  const auto& e2r = found->second.e2r;
  const char* lease_env = std::getenv("SIDEREGION_ROW_LEASES");
  const bool acquire_leases = lease_env && lease_env[0] == '1';
  for (size_t index = 0; index < count; ++index) {
    auto row = e2r.find(static_cast<int>(expert_ids[index]));
    if (row != e2r.end()) {
      rows[index] = row->second;
      if (acquire_leases)
        g_side_row_leases[{layer, row->second}] += 1;
    }
  }
  return rows;
}
