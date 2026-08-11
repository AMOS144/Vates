// [6] Phase 2 方案B：真实区槽状态 C++ 全接管（1 次同步版）。
// 复刻 Python ResidentExpertPool 的 _slot_of/_free/_freq + _choose_victim/_alloc_slot 语义：
// - free 优先(front pop，free 初始 [0,cap))；free 空 → LFU 驱逐，受害者槽直接复用(不回 free)。
// - _choose_victim：candidates=插入序中 ∉current 者；victim=min(freq, 候选下标)，驱逐不删 freq。
// - dual 语义：真实区命中不移动插入序；仅新放入 miss 追加序尾。
#include "demand.h"
#include "side_region.h"
#include "../io/bg_reader.h"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>

struct RealLayer {
  std::vector<int> order;                    // 插入序(LRU tie-break)，与 e2r 同步维护
  std::unordered_map<int, int> e2r;          // expert -> slot [0,cap)
  std::vector<int> free_rows;                // 空闲槽(front pop，仿 free.pop(0))
  std::unordered_map<int, uint32_t> freq;    // LFU 频次(驱逐不删，与 Python 一致)
  std::unordered_map<int, uint32_t> pred_freq; // 预测频次
  std::unordered_set<int> speculative;       // 尚未被真实路由命中的预测驻留
  std::unordered_set<int> pinned;            // 启动期安全集：真实区内不可驱逐
  std::unordered_map<int, int> pending_e2r;  // 已预留但字节尚未完整就位
  int cap = 0;
  long access = 0;                           // 累计访问(decay 用)
  bool inited = false;
  std::shared_ptr<mx::array> slot_table;
  std::shared_ptr<mx::array> lease_table;
};
static std::mutex g_real_mutex;
static std::map<int, RealLayer> g_real;
static std::map<int, int> g_predict_cooldown;
static std::map<int, int> g_predict_budget;
static std::condition_variable g_real_progress;
static std::mutex g_pred_pending_mutex;
static std::map<int, std::unordered_map<int, uint32_t>> g_pred_pending;

// demand 统计：累计 + 本次(供 Python 更新 rp.hits/misses/gpu_fastpath/gpu_fallback)。
static std::mutex g_dstat_mutex;
static long g_d_last[6] = {0, 0, 0, 0, 0, 0};
static std::atomic<long> g_demand_ticket{1000000000};   // demand 并行 pread 用的独立 ticket 段
static std::atomic<long> g_async_calls{0};
static std::atomic<long> g_async_fast{0};
static std::atomic<long> g_async_fallback{0};
static std::atomic<long> g_async_loads{0};
static std::atomic<long> g_async_hit_positions{0};
static std::atomic<long> g_async_positions{0};
static std::atomic<long> g_async_true_fallback{0};
static std::atomic<long> g_async_pending_rescued{0};
static std::atomic<long> g_async_pending_wait_us{0};
static std::atomic<long> g_async_fallback_wait_us{0};
static std::mutex g_async_error_mutex;
static std::string g_async_error;
static std::mutex g_async_active_mutex;
static std::condition_variable g_async_active_cv;
static long g_async_active = 0;

struct DemandDeadlineLayer {
  long calls = 0;
  long actual_unique = 0;
  long real_resident = 0;
  long side_prefetch_complete = 0;
  long demand_fallback = 0;
};
static std::mutex g_deadline_mutex;
static std::map<int, DemandDeadlineLayer> g_deadline;
static std::atomic<bool> g_deadline_enabled{false};
static std::mutex g_prejoin_mutex;
static std::map<int, DemandDeadlineLayer> g_prejoin;
static std::atomic<bool> g_prejoin_enabled{false};

// 必须在 demand 分配 miss 槽之前取快照：此刻就是目标 MoE 的字节 deadline。
// side_region 的 e2r 只在完整 pread + 全段 memcpy 后 publish，因此 side 命中
// 等价于该专家全部字节已在 deadline 前到达，而不是仅“提交过”。调用方持
// g_real_mutex，保证 real resident 快照与随后 demand_core 使用的状态一致。
static void note_deadline_locked(
    int layer, RealLayer& c, const uint32_t* ip, size_t n,
    const std::unordered_map<int, int>& side) {
  if (!g_deadline_enabled.load(std::memory_order_relaxed)) return;
  std::unordered_set<int> seen;
  long real = 0, prefetched = 0, fallback = 0;
  for (size_t i = 0; i < n; ++i) {
    int e = static_cast<int>(ip[i]);
    if (!seen.insert(e).second) continue;
    // 与 demand_core 一致：侧区优先于真实区。两边同时存在也只计一次覆盖。
    if (side.count(e)) ++prefetched;
    else if (c.e2r.count(e)) ++real;
    else ++fallback;
  }
  std::lock_guard<std::mutex> lk(g_deadline_mutex);
  DemandDeadlineLayer& row = g_deadline[layer];
  row.calls += 1;
  row.actual_unique += static_cast<long>(seen.size());
  row.real_resident += real;
  row.side_prefetch_complete += prefetched;
  row.demand_fallback += fallback;
}

static void note_deadline_from_gpu_local(
    int layer, const uint32_t* ids, const int32_t* local,
    size_t count, int real_cap) {
  if (!g_deadline_enabled.load(std::memory_order_relaxed)) return;
  std::unordered_set<int> seen;
  long real = 0, side = 0, fallback = 0;
  {
    std::lock_guard<std::mutex> real_lock(g_real_mutex);
    auto found = g_real.find(layer);
    for (size_t index = 0; index < count; ++index) {
      int expert = static_cast<int>(ids[index]);
      if (!seen.insert(expert).second) continue;
      if (local[index] < 0) ++fallback;
      else if (local[index] >= real_cap) ++side;
      else if (found != g_real.end() &&
               found->second.speculative.count(expert)) ++side;
      else ++real;
    }
  }
  std::lock_guard<std::mutex> lock(g_deadline_mutex);
  DemandDeadlineLayer& row = g_deadline[layer];
  ++row.calls;
  row.actual_unique += static_cast<long>(seen.size());
  row.real_resident += real;
  row.side_prefetch_complete += side;
  row.demand_fallback += fallback;
}

void demand_deadline_snapshot(
    const mx::array& inds, int layer, int side_gen, bool use_side) {
  if (!g_deadline_enabled.load(std::memory_order_relaxed)) return;
  mx::array ids = mx::contiguous(inds);
  ids.eval();
  auto side = use_side
      ? sideregion_snapshot(layer, side_gen)
      : std::unordered_map<int, int>{};
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& real = g_real[layer];
  note_deadline_locked(
      layer, real, ids.data<uint32_t>(), ids.size(), side);
}

void demand_prejoin_note(
    int layer, const uint32_t* ip, size_t n,
    const std::unordered_set<int>& staging_complete) {
  if (!g_prejoin_enabled.load(std::memory_order_relaxed)) return;
  std::unordered_set<int> seen;
  long real = 0, prefetched = 0, fallback = 0;
  {
    std::lock_guard<std::mutex> real_lk(g_real_mutex);
    auto found = g_real.find(layer);
    for (size_t i = 0; i < n; ++i) {
      int expert = static_cast<int>(ip[i]);
      if (!seen.insert(expert).second) continue;
      if (staging_complete.count(expert)) ++prefetched;
      else if (found != g_real.end() && found->second.e2r.count(expert)) ++real;
      else ++fallback;
    }
  }
  std::lock_guard<std::mutex> lk(g_prejoin_mutex);
  auto& row = g_prejoin[layer];
  ++row.calls;
  row.actual_unique += static_cast<long>(seen.size());
  row.real_resident += real;
  row.side_prefetch_complete += prefetched;
  row.demand_fallback += fallback;
}

std::vector<long> demand_prejoin_stats() {
  std::lock_guard<std::mutex> lk(g_prejoin_mutex);
  std::vector<long> out;
  out.reserve(g_prejoin.size() * 6);
  for (const auto& item : g_prejoin) {
    const auto& row = item.second;
    out.insert(out.end(), {item.first, row.calls, row.actual_unique,
                           row.real_resident, row.side_prefetch_complete,
                           row.demand_fallback});
  }
  return out;
}

void demand_prejoin_stats_reset() {
  std::lock_guard<std::mutex> lk(g_prejoin_mutex);
  g_prejoin.clear();
  g_prejoin_enabled.store(true, std::memory_order_relaxed);
}

// 诊断计时(DEMAND_TIMING=1)：累计各段主线程微秒，供定位结构性开销。默认关。
static bool g_dt_on = false;
static double g_dt[6] = {0, 0, 0, 0, 0, 0};  // [inds_eval, pool_eval, side_snap, real_lock, core, build]
static inline double dt_now_us() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}
std::vector<double> demand_timings() { return {g_dt[0], g_dt[1], g_dt[2], g_dt[3], g_dt[4], g_dt[5]}; }
void demand_timing_enable(bool on) {
  g_dt_on = on;
  for (int i = 0; i < 6; ++i) g_dt[i] = 0;
}

static void real_ensure_locked(RealLayer& c, int cap) {
  if (c.inited) {
    if (c.cap != cap)
      throw std::invalid_argument("real region cap changed after initialization");
    return;
  }
  c.cap = cap;
  std::vector<int32_t> values(512, -1);
  c.slot_table = std::make_shared<mx::array>(
      values.data(), mx::Shape{512}, mx::int32);
  std::vector<uint32_t> leases(1024, 0);
  c.lease_table = std::make_shared<mx::array>(
      leases.data(), mx::Shape{1024}, mx::uint32);
  c.free_rows.clear();
  for (int r = 0; r < cap; ++r) c.free_rows.push_back(r);   // free 初始 [0,cap)
  c.inited = true;
}

static void real_table_set_locked(RealLayer& layer, int expert, int slot) {
  if (expert >= 0 && expert < 512)
    __atomic_store_n(
        layer.slot_table->data<int32_t>() + expert, slot, __ATOMIC_RELEASE);
}

static bool real_row_leased_locked(const RealLayer& layer, int row) {
  return row >= 0 && row < 1024 && layer.lease_table &&
      __atomic_load_n(
          layer.lease_table->data<uint32_t>() + row, __ATOMIC_ACQUIRE) != 0;
}

static int alloc_slot_locked(
    RealLayer& c, int e, const std::unordered_set<int>& current);

void real_init(int layer, int cap) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  real_ensure_locked(g_real[layer], cap);
}

void real_note_predictions(int layer, const uint32_t* expert_ids, size_t n) {
  std::unordered_set<int> seen;
  std::lock_guard<std::mutex> lk(g_pred_pending_mutex);
  auto& pending = g_pred_pending[layer];
  for (size_t i = 0; i < n; ++i) {
    int e = static_cast<int>(expert_ids[i]);
    if (seen.insert(e).second) pending[e] += 1;
  }
}

static void merge_prediction_freq_locked(int layer, RealLayer& c) {
  std::lock_guard<std::mutex> lk(g_pred_pending_mutex);
  auto it = g_pred_pending.find(layer);
  if (it == g_pred_pending.end()) return;
  for (const auto& item : it->second) c.pred_freq[item.first] += item.second;
  g_pred_pending.erase(it);
}

std::vector<int> real_pin(int layer, const std::vector<int>& experts, int cap) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& c = g_real[layer];
  real_ensure_locked(c, cap);
  std::vector<int> unique;
  std::unordered_set<int> requested;
  for (int e : experts) {
    if (e < 0) throw std::invalid_argument("real_pin expert must be non-negative");
    if (requested.insert(e).second) unique.push_back(e);
  }
  std::unordered_set<int> combined_pins = c.pinned;
  combined_pins.insert(requested.begin(), requested.end());
  if (combined_pins.size() > static_cast<size_t>(cap))
    throw std::invalid_argument("real_pin count exceeds real region cap");
  // 先保护本批全部目标，避免批内后放入的 pin 驱逐刚放入的 pin。
  c.pinned.insert(requested.begin(), requested.end());
  std::vector<int> slots;
  slots.reserve(unique.size());
  for (int e : unique) {
    auto it = c.e2r.find(e);
    if (it != c.e2r.end()) {
      slots.push_back(it->second);
      continue;
    }
    int slot = alloc_slot_locked(c, e, requested);
    if (slot < 0)
      throw std::runtime_error("real_pin has no evictable slot");
    slots.push_back(slot);
  }
  return slots;
}

std::vector<int> real_pinned_contents(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  std::vector<int> out;
  auto it = g_real.find(layer);
  if (it == g_real.end()) return out;
  out.assign(it->second.pinned.begin(), it->second.pinned.end());
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<int> real_region_contents(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  std::vector<int> out;
  auto it = g_real.find(layer);
  if (it != g_real.end())
    for (auto& p : it->second.e2r) { out.push_back(p.first); out.push_back(p.second); }
  return out;
}

std::vector<int> real_verified_contents(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  std::vector<int> out;
  auto it = g_real.find(layer);
  if (it != g_real.end()) {
    for (const auto& item : it->second.e2r) {
      if (it->second.speculative.count(item.first)) continue;
      out.push_back(item.first);
      out.push_back(item.second);
    }
  }
  return out;
}

int real_region_count(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  auto it = g_real.find(layer);
  return it == g_real.end() ? 0 : static_cast<int>(it->second.e2r.size());
}

bool real_should_predict(int layer, int min_resident, int cooldown) {
  std::lock_guard<std::mutex> lock(g_real_mutex);
  g_predict_budget[layer] = std::max(1, cooldown);
  auto found = g_real.find(layer);
  // An empty layer needs the first prediction.  After that, occupancy alone
  // must not keep the predictor permanently enabled: genuinely cold layers
  // can have a stable working set below a capacity-derived fill floor.  True
  // demand loads and useful speculative promotions explicitly rearm the
  // bounded cooldown below.
  (void)min_resident;
  if (found == g_real.end() || found->second.e2r.empty()) return true;
  int& remaining = g_predict_cooldown[layer];
  if (remaining <= 0) return false;
  --remaining;
  return true;
}

static void mark_predict_cooldown_locked(int layer) {
  int& remaining = g_predict_cooldown[layer];
  remaining = std::max(remaining, std::max(1, g_predict_budget[layer]));
}

mx::array real_slot_table(int layer, int cap) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& real = g_real[layer];
  real_ensure_locked(real, cap);
  return *real.slot_table;
}

mx::array real_lease_table(int layer, int cap) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& real = g_real[layer];
  real_ensure_locked(real, cap);
  return *real.lease_table;
}

void real_release_before_layer(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  if (layer <= 0) {
    for (auto& item : g_real) {
      RealLayer& real = item.second;
      if (!real.lease_table) continue;
      for (int row = 0; row < real.cap; ++row)
        __atomic_store_n(
            real.lease_table->data<uint32_t>() + row, 0u, __ATOMIC_RELEASE);
    }
    return;
  }
  auto found = g_real.find(layer - 1);
  if (found == g_real.end() || !found->second.lease_table) return;
  RealLayer& real = found->second;
  for (int row = 0; row < real.cap; ++row)
    __atomic_store_n(
        real.lease_table->data<uint32_t>() + row, 0u, __ATOMIC_RELEASE);
}

void real_reset() {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  g_real.clear();
  g_predict_cooldown.clear();
  g_predict_budget.clear();
  std::lock_guard<std::mutex> pred_lk(g_pred_pending_mutex);
  g_pred_pending.clear();
}

// 复刻 _choose_victim：遍历插入序(order)选 ∉current 且 freq 最小者，并列取最早(候选下标最小)。
// 返回 expert id；-1 表示无可驱逐。调用方须持 g_real_mutex。
static int choose_victim_locked(RealLayer& c, const std::unordered_set<int>& current) {
  int victim = -1;
  uint32_t best = 0;
  // Demand owns the verified share and must not churn the bounded prediction
  // share on every true miss.  Prefer a cold non-speculative row first; this
  // reproduces the old 32 real + 32 prediction isolation while retaining one
  // allocation and one expert->slot table.  A speculative row is the last
  // resort only when every verified row is current/pinned/leased.
  for (int e : c.order) {
    if (!c.e2r.count(e) || c.pinned.count(e) || current.count(e) ||
        real_row_leased_locked(c, c.e2r[e]) ||
        c.speculative.count(e)) continue;
    uint32_t f = c.freq.count(e) ? c.freq[e] : 0;
    if (victim < 0 || f < best) { victim = e; best = f; }
  }
  if (victim >= 0) return victim;
  for (int e : c.order) {
    if (!c.e2r.count(e) || c.pinned.count(e) || current.count(e) ||
        real_row_leased_locked(c, c.e2r[e]) ||
        !c.speculative.count(e)) continue;
    uint32_t f = c.pred_freq.count(e) ? c.pred_freq[e] : 0;
    if (victim < 0 || f < best) { victim = e; best = f; }   // 并列不更新 → 保留更早者
  }
  return victim;
}

// 复刻 _alloc_slot（e 为 miss，不会已在 e2r）：free 优先，否则 LFU 驱逐复用受害者槽。
// 返回 slot；-1 表示无可驱逐(超容量，不该在 dual 发生)。调用方须持 g_real_mutex。
static int alloc_slot_locked(RealLayer& c, int e, const std::unordered_set<int>& current) {
  auto it = c.e2r.find(e);
  if (it != c.e2r.end()) return it->second;
  int slot;
  if (!c.free_rows.empty()) {
    slot = c.free_rows.front();
    c.free_rows.erase(c.free_rows.begin());       // free.pop(0)
  } else {
    int victim = choose_victim_locked(c, current);
    if (victim < 0) return -1;
    slot = c.e2r[victim];
    real_table_set_locked(c, victim, -1);
    c.e2r.erase(victim);
    c.speculative.erase(victim);
    c.pred_freq.erase(victim);
    for (auto oit = c.order.begin(); oit != c.order.end(); ++oit)
      if (*oit == victim) { c.order.erase(oit); break; }
  }
  c.e2r[e] = slot;
  real_table_set_locked(c, e, slot);
  c.order.push_back(e);
  return slot;
}

// LFU 频次 bump（canonical：本次全部唯一专家各 +1）+ decay。调用方须持 g_real_mutex。
static void note_access_locked(int layer, RealLayer& c,
                               const std::vector<int>& uniq_access,
                               bool lfu, int decay_interval) {
  // A correctly predicted row remains in the bounded speculative share after
  // use, exactly as it did in the old side cache.  Promoting its *label* to a
  // verified row on every hit drains the speculative share and forces one new
  // SSD read per layer/step even though the bytes are still useful.  Unified
  // ownership does not require that churn: one table owns the row either way.
  static const uint32_t kPredictionDemandBoost = []() {
    const char* value = std::getenv("SIDEREGION_DEMAND_BOOST");
    if (!value) return uint32_t{8};
    long parsed = std::strtol(value, nullptr, 10);
    return static_cast<uint32_t>(std::max<long>(0, parsed));
  }();
  static const bool kPromotePredictionHitsWhileUnderfilled = []() {
    const char* value = std::getenv("UNIFIED_PROMOTE_FREE_HITS");
    return !value || value[0] != '0';
  }();
  static const double kAdaptiveFill = []() {
    const char* value = std::getenv("PREFETCH_ADAPTIVE_FILL");
    if (!value) return 0.85;
    return std::max(0.0, std::min(1.0, std::strtod(value, nullptr)));
  }();
  // While the unified allocation still has unused physical rows, a
  // prediction that is consumed by the real route has proved itself useful.
  // Move only its ownership label into the verified/history share so the
  // bounded speculative quota can populate another free row on a later
  // occurrence.  This lets a decoupled pool (for example 88 physical rows,
  // 32 speculative admissions) warm past 32 rows and eventually reach the
  // adaptive predictor's fill threshold.
  //
  // Stop as soon as the same fill threshold used by real_should_predict is
  // reached.  Filling all remaining rows would keep reopening speculative
  // admissions after the adaptive predictor is already meant to be quiet,
  // causing needless SSD churn and allocating cold capacity.
  const int promote_until = std::max(
      1, static_cast<int>(static_cast<double>(c.cap) * kAdaptiveFill));
  if (kPromotePredictionHitsWhileUnderfilled &&
      static_cast<int>(c.e2r.size()) < promote_until) {
    bool promoted = false;
    for (int e : uniq_access) {
      auto speculative = c.speculative.find(e);
      if (speculative == c.speculative.end()) continue;
      c.speculative.erase(speculative);
      c.pred_freq.erase(e);
      promoted = true;
    }
    if (promoted) mark_predict_cooldown_locked(layer);
  }
  for (int e : uniq_access) {
    if (!c.speculative.count(e) || kPredictionDemandBoost == 0) continue;
    uint32_t& frequency = c.pred_freq[e];
    frequency = std::numeric_limits<uint32_t>::max() - frequency <
            kPredictionDemandBoost
        ? std::numeric_limits<uint32_t>::max()
        : frequency + kPredictionDemandBoost;
  }
  if (!lfu) return;
  for (int e : uniq_access) c.freq[e] += 1;
  c.access += static_cast<long>(uniq_access.size());
  if (decay_interval > 0 && c.access >= decay_interval) {
    for (auto it = c.freq.begin(); it != c.freq.end();) {
      it->second /= 2;
      if (it->second == 0) it = c.freq.erase(it);
      else ++it;
    }
    c.access = 0;
  }
}

static int alloc_speculative_locked(
    RealLayer& c, int e, int spec_limit,
    const std::unordered_set<int>& protect) {
  auto existing = c.e2r.find(e);
  if (existing != c.e2r.end()) return existing->second;
  if (spec_limit <= 0) return -1;

  int slot = -1;
  if (static_cast<int>(c.speculative.size()) < spec_limit &&
      !c.free_rows.empty()) {
    slot = c.free_rows.front();
    c.free_rows.erase(c.free_rows.begin());
  } else {
    int victim = -1;
    uint32_t best = 0;
    for (int candidate : c.order) {
      if (!c.e2r.count(candidate) || !c.speculative.count(candidate) ||
          real_row_leased_locked(c, c.e2r[candidate]) ||
          c.pinned.count(candidate) || protect.count(candidate)) continue;
      uint32_t f = c.pred_freq.count(candidate) ? c.pred_freq[candidate] : 0;
      if (victim < 0 || f < best) { victim = candidate; best = f; }
    }
    if (victim < 0 || best > c.pred_freq[e]) return -1;
    slot = c.e2r[victim];
    real_table_set_locked(c, victim, -1);
    c.e2r.erase(victim);
    c.speculative.erase(victim);
    c.pred_freq.erase(victim);
    for (auto it = c.order.begin(); it != c.order.end(); ++it) {
      if (*it == victim) { c.order.erase(it); break; }
    }
  }
  c.e2r[e] = slot;
  real_table_set_locked(c, e, slot);
  c.order.push_back(e);
  c.speculative.insert(e);
  return slot;
}

std::vector<std::pair<int, int>> real_prefetch_reserve(
    const uint32_t* expert_ids, size_t count, int layer, int cap,
    int speculative_limit, const std::vector<int>& resident) {
  std::vector<int> order;
  // ``resident`` is an audit snapshot, not an immutable ownership set.  Once
  // the unified pool is full, protecting every source-time row leaves no slot
  // for a newly selected expert and makes a 64-row pool unable to behave like
  // the intended 32 verified + 32 speculative cache.  Protect this
  // occurrence's selected set (plus pinned/leased rows in victim selection),
  // while LFU is allowed to replace cold verified history up to spec_limit.
  (void)resident;
  std::unordered_set<int> protect;
  std::unordered_set<int> seen;
  for (size_t index = 0; index < count; ++index) {
    int expert = static_cast<int>(expert_ids[index]);
    if (seen.insert(expert).second)
      order.push_back(expert);
  }
  protect.insert(order.begin(), order.end());

  std::vector<std::pair<int, int>> reads;
  std::lock_guard<std::mutex> lock(g_real_mutex);
  RealLayer& real = g_real[layer];
  real_ensure_locked(real, cap);
  merge_prediction_freq_locked(layer, real);
  for (int expert : order) real.pred_freq[expert] += 1;

  for (int expert : order) {
    if (real.e2r.count(expert) || real.pending_e2r.count(expert)) continue;
    int row = -1;
    const int speculative_inflight = static_cast<int>(
        real.speculative.size() + real.pending_e2r.size());
    if (speculative_inflight < speculative_limit && !real.free_rows.empty()) {
      row = real.free_rows.front();
      real.free_rows.erase(real.free_rows.begin());
    } else {
      int victim = -1;
      uint32_t best = 0;
      for (int candidate : real.order) {
        if (!real.e2r.count(candidate) ||
            !real.speculative.count(candidate) ||
            real.pinned.count(candidate) || protect.count(candidate) ||
            real_row_leased_locked(real, real.e2r[candidate])) continue;
        uint32_t frequency = real.pred_freq.count(candidate)
            ? real.pred_freq[candidate] : 0;
        if (victim < 0 || frequency < best) {
          victim = candidate;
          best = frequency;
        }
      }
      if (victim < 0 && speculative_inflight < speculative_limit) {
        for (int candidate : real.order) {
          if (!real.e2r.count(candidate) || real.speculative.count(candidate) ||
              real.pinned.count(candidate) || protect.count(candidate) ||
              real_row_leased_locked(real, real.e2r[candidate])) continue;
          uint32_t frequency = real.freq.count(candidate)
              ? real.freq[candidate] : 0;
          if (victim < 0 || frequency < best) {
            victim = candidate;
            best = frequency;
          }
        }
      }
      if (victim < 0) continue;
      // The current occurrence's selected set is authoritative.  LFU chooses
      // which stale prediction leaves; it must not veto admission entirely,
      // otherwise logical rerank recall never becomes physical byte coverage.
      row = real.e2r[victim];
      real_table_set_locked(real, victim, -1);
      real.e2r.erase(victim);
      real.speculative.erase(victim);
      real.pred_freq.erase(victim);
      for (auto it = real.order.begin(); it != real.order.end(); ++it) {
        if (*it == victim) { real.order.erase(it); break; }
      }
    }
    real.pending_e2r[expert] = row;
    reads.emplace_back(expert, row);
  }
  return reads;
}

void real_prefetch_publish(int layer, int expert, int row, int cap) {
  {
    std::lock_guard<std::mutex> lock(g_real_mutex);
    RealLayer& real = g_real[layer];
    real_ensure_locked(real, cap);
    auto pending = real.pending_e2r.find(expert);
    if (pending == real.pending_e2r.end() || pending->second != row) return;
    real.pending_e2r.erase(pending);
    real.e2r[expert] = row;
    real.order.push_back(expert);
    real.speculative.insert(expert);
    real_table_set_locked(real, expert, row);
  }
  g_real_progress.notify_all();
}

void real_prefetch_abort(int layer, int expert, int row, int cap) {
  {
    std::lock_guard<std::mutex> lock(g_real_mutex);
    RealLayer& real = g_real[layer];
    real_ensure_locked(real, cap);
    auto pending = real.pending_e2r.find(expert);
    if (pending == real.pending_e2r.end() || pending->second != row) return;
    real.pending_e2r.erase(pending);
    real.free_rows.push_back(row);
  }
  g_real_progress.notify_all();
}

void real_prefetch_wait_pending(
    int layer, const uint32_t* expert_ids, size_t count) {
  std::unordered_set<int> route;
  for (size_t index = 0; index < count; ++index)
    route.insert(static_cast<int>(expert_ids[index]));
  std::unique_lock<std::mutex> lock(g_real_mutex);
  g_real_progress.wait(lock, [&] {
    auto found = g_real.find(layer);
    if (found == g_real.end()) return true;
    for (int expert : route)
      if (found->second.pending_e2r.count(expert)) return false;
    return true;
  });
}

void real_prefetch_wait_all(int layer) {
  std::unique_lock<std::mutex> lock(g_real_mutex);
  g_real_progress.wait(lock, [&] {
    auto found = g_real.find(layer);
    return found == g_real.end() || found->second.pending_e2r.empty();
  });
}

std::vector<int32_t> real_lookup_values(
    int layer, const uint32_t* expert_ids, size_t count) {
  std::vector<int32_t> rows(count, -1);
  std::lock_guard<std::mutex> lock(g_real_mutex);
  auto found = g_real.find(layer);
  if (found == g_real.end()) return rows;
  for (size_t index = 0; index < count; ++index) {
    auto row = found->second.e2r.find(static_cast<int>(expert_ids[index]));
    if (row != found->second.e2r.end()) rows[index] = row->second;
  }
  return rows;
}

// 核心状态机（纯 CPU、无 I/O）：给定 host inds(ip[n]) + 侧区快照(side)，算 local、分配 miss 槽，
// 把需要落盘的新放入 (expert, slot) 追加到 placements（字节由调用方在锁外并行 pread 落池）。
// 返回 local(int32 vector)。stats: [hitpos, misspos, loads(=placements 数)]。
static std::vector<int32_t> demand_core_locked(
    int layer, RealLayer& c, const uint32_t* ip, size_t n,
    const std::unordered_map<int, int>& side,
    bool lfu, int decay_interval, std::vector<std::pair<int, int>>& placements, long stats[3],
    bool* overcap = nullptr) {
  std::vector<int32_t> local(n, -1);
  std::vector<int> uniq_miss, access_order;
  std::unordered_set<int> miss_seen, access_seen, real_needed;
  int hitpos = 0;
  // pass1：算命中(侧区覆盖真实区)、收集 miss(首见序)、收集唯一访问(freq)。
  for (size_t i = 0; i < n; ++i) {
    int e = static_cast<int>(ip[i]);
    if (access_seen.insert(e).second) access_order.push_back(e);
    auto sit = side.find(e);
    if (sit != side.end()) { local[i] = sit->second; ++hitpos; continue; }
    real_needed.insert(e);
    auto rit = c.e2r.find(e);
    if (rit != c.e2r.end()) { local[i] = rit->second; ++hitpos; continue; }
    if (miss_seen.insert(e).second) uniq_miss.push_back(e);
  }
  note_access_locked(layer, c, access_order, lfu, decay_interval);
  // 如果本次所有非 side 专家与不可驱逐的无关 pin 无法同时放进真实区，绝不能沿用
  // 旧的“多余 miss 映射到 slot0”兜底：那会让多个专家读取同一槽并产生静默数值错误。
  // 返回 fallback=2，让 Python 用临时 stacked experts 完整执行本次 MoE；这里不改变
  // e2r/order/free，后续正常 demand 仍从原 resident 状态继续。
  size_t unavailable_pins = 0;
  for (int e : c.pinned)
    if (!real_needed.count(e)) ++unavailable_pins;
  if (real_needed.size() + unavailable_pins > static_cast<size_t>(c.cap)) {
    if (overcap != nullptr) *overcap = true;
    stats[0] = hitpos;
    stats[1] = static_cast<long>(n) - hitpos;
    stats[2] = 0;
    return std::vector<int32_t>(n, 0);
  }
  // pass2：miss 分配槽（不落盘）。current = 本前向全部唯一路由专家(命中+miss)：绝不驱逐本前向要读的
  // 任何非 side 专家的槽（否则真实区命中专家的槽被 miss 复写 → 脏字节）。side 已经
  // 有完整副本的专家不需要保护其重复 real 槽，否则会无谓减少可驱逐槽。
  std::unordered_map<int, int> new_slot;
  for (int e : uniq_miss) {
    int slot = alloc_slot_locked(c, e, real_needed);
    if (slot < 0)
      throw std::runtime_error("demand_dual preflight passed but no evictable real slot");
    new_slot[e] = slot;
    placements.emplace_back(e, slot);
  }
  // pass3：回填 miss 位置。
  int misspos = 0;
  for (size_t i = 0; i < n; ++i) {
    if (local[i] < 0) {
      auto it = new_slot.find(static_cast<int>(ip[i]));
      local[i] = (it != new_slot.end()) ? it->second : 0;
      ++misspos;
    }
  }
  stats[0] = hitpos; stats[1] = misspos; stats[2] = static_cast<long>(placements.size());
  return local;
}

static std::vector<long> submit_demand_reads(
    const std::vector<mx::array>& pools,
    const std::vector<long>& seg_off,
    const std::vector<long>& seg_nb,
    const std::vector<std::pair<int, int>>& placements,
    const std::string& path, long stride) {
  std::vector<long> tickets;
  tickets.reserve(placements.size() * pools.size());
  for (const auto& placement : placements) {
    // A typical routed layer misses only one expert.  One job per expert then
    // leaves seven of eight readers idle while that worker performs all six
    // projection/quantization-segment reads serially.  Each segment targets a
    // disjoint pool array and file range, so they can safely run in parallel.
    for (size_t segment = 0; segment < pools.size(); ++segment) {
      const long ticket = g_demand_ticket.fetch_add(1);
      bg_pread_into_pool(
          {pools[segment]}, {seg_off[segment]}, {seg_nb[segment]},
          placement.second, placement.first, path, stride, ticket,
          /*prio=*/1, /*nocache=*/false);
      tickets.push_back(ticket);
    }
  }
  return tickets;
}

// demand 全接管：inds 惰性(内部 eval 一次=1 次同步)；side_gen 指定侧区代；pool_list 为 _segs 顺序
// 的 per-key 池数组(已 eval、指针稳定)。返回 local(int32, inds.shape)。
mx::array demand_dual(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen, const std::string& path,
    int stride, int cap, bool lfu, int decay_interval, int64_t forward_id,
    int sequence_length, bool use_side, bool record_deadline,
    mx::StreamOrDevice s = {}) {
  if (seg_nbytes.size() != pool_list.size())
    throw std::invalid_argument("demand_dual: seg_nbytes.size() != pool_list.size()");
  double t0 = g_dt_on ? dt_now_us() : 0;
  // 关键:inds 常是 argpartition(...)[..., -k:] 切片 → 非连续 strided 视图(每行按父数组 stride E
  // 偏移)。若直接按连续读 data(),仅首行(token0)正确,后续 token 读到错位内存 → 装错专家、seq≥2 全错
  // (投机 verify 全体受害)。contiguous() 强制物化为连续,读取才逐位正确。
  mx::array ids = mx::contiguous(inds);
  ids.eval();                                   // 唯一同步
  size_t n = ids.size();
  const uint32_t* ip = ids.data<uint32_t>();
  double t1 = g_dt_on ? dt_now_us() : 0;
  std::vector<uint8_t*> ptrs;
  ptrs.reserve(pool_list.size());
  for (auto& a0 : pool_list) { mx::array a = a0; a.eval(); ptrs.push_back(a.data<uint8_t>()); }
  double t2 = g_dt_on ? dt_now_us() : 0;
  std::unordered_map<int, int> side = use_side
      ? sideregion_snapshot(layer, side_gen)
      : std::unordered_map<int, int>{};
  double ta = g_dt_on ? dt_now_us() : 0;
  long stats[3];
  std::vector<int32_t> local;
  std::vector<std::pair<int, int>> placements;   // (expert, slot)：锁外并行落盘
  bool overcap = false;
  double tb;
  {
    std::lock_guard<std::mutex> lk(g_real_mutex);
    tb = g_dt_on ? dt_now_us() : 0;
    RealLayer& c = g_real[layer];
    real_ensure_locked(c, cap);
    // 与提交侧用 (forward_id,target layer) 精确配对。放在任何 miss 槽分配之前，
    // 保证这里记录的是目标 MoE 开始消费时的真实 deadline 状态。
    prefetch_audit_note_demand(
        forward_id, layer, sequence_length, ip, n, c.e2r, side);
    if (record_deadline) note_deadline_locked(layer, c, ip, n, side);
    local = demand_core_locked(
        layer, c, ip, n, side, lfu, decay_interval, placements, stats, &overcap);
  }
  // 锁外：把 miss 专家的字节 pread 落真实区槽。复刻基线并发模型——多 miss 派给 BgReader worker
  // 并行 pread（高优队列、直写池段行，无主线程 tmp/memcpy），本层等待其完成（并行 → 远快于串行）。
  static const bool kSkipIO = []() {
    const char* e = std::getenv("DEMAND_SKIP_IO");
    return e && e[0] == '1';
  }();
  if (!placements.empty() && !kSkipIO) {
    std::vector<long> seg_off(seg_nbytes.size()), seg_nb(seg_nbytes.size());
    long acc = 0;
    for (size_t k = 0; k < seg_nbytes.size(); ++k) {
      seg_off[k] = acc; seg_nb[k] = seg_nbytes[k]; acc += seg_nbytes[k];
    }
    auto tickets = submit_demand_reads(
        pool_list, seg_off, seg_nb, placements, path,
        static_cast<long>(stride));
    for (long tk : tickets) bg_reader_wait(tk);   // 并行 pread 完成 → 池槽字节就绪
  }
  if (!placements.empty()) {
    std::lock_guard<std::mutex> lock(g_real_mutex);
    mark_predict_cooldown_locked(layer);
  }
  double t3 = g_dt_on ? dt_now_us() : 0;
  {
    std::lock_guard<std::mutex> lk(g_dstat_mutex);
    long side_hits = 0;
    if (use_side) {
      for (size_t i = 0; i < n; ++i)
        if (side.count(static_cast<int>(ip[i]))) ++side_hits;
    }
    g_d_last[0] = stats[0]; g_d_last[1] = stats[1]; g_d_last[2] = stats[2];
    // 0=全命中，1=正常 real demand load，2=真实区装不下、调用方必须临时 fetch。
    g_d_last[3] = overcap ? 2 : ((stats[1] == 0) ? 0 : 1);
    g_d_last[4] = side_hits;
    g_d_last[5] = stats[0] - side_hits;
  }
  mx::array out = mx::array(local.data(), ids.shape(), mx::int32);
  if (g_dt_on) {
    double t4 = dt_now_us();
    g_dt[0] += t1 - t0; g_dt[1] += t2 - t1; g_dt[2] += ta - t2;
    g_dt[3] += tb - ta; g_dt[4] += t3 - tb; g_dt[5] += t4 - t3;
  }
  return out;
}

namespace {

struct AsyncDemandState {
  NS::SharedPtr<MTL::SharedEvent> event;
  uint64_t demand_value = 0;
};

static std::mutex g_async_event_mutex;
static NS::SharedPtr<MTL::SharedEvent> g_async_event;
static MTL::Device* g_async_event_device = nullptr;
static std::atomic<uint64_t> g_async_event_value{1};
static std::string async_demand_metal_source() {
  return R"METAL(
    #include <metal_stdlib>
    using namespace metal;
    kernel void demand_async_noop(
        device int* guard [[buffer(0)]],
        uint tid [[thread_position_in_grid]]) {
      if (tid == 0) guard[0] = guard[0];
    }
    kernel void demand_async_copy(
        device const int* src [[buffer(0)]],
        device int* dst [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
      dst[tid] = src[tid];
    }
    kernel void demand_gpu_remap(
        device const uint* ids [[buffer(0)]],
        device atomic_int* real_table [[buffer(1)]],
        device atomic_int* side_table [[buffer(2)]],
        device atomic_uint* row_leases [[buffer(3)]],
        device int* local [[buffer(4)]],
        constant int4& params [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
      int side = params.x != 0
          ? atomic_load_explicit(
                side_table + ids[tid], memory_order_relaxed)
          : -1;
      int real = atomic_load_explicit(
          real_table + ids[tid], memory_order_relaxed);
      int row = side >= 0 ? side : real;
      const bool lease_side = params.w != 0 && row >= params.y;
      const bool lease_real = params.z != 0 && row >= 0 && row < params.y;
      if ((lease_side || lease_real) && row < 1024) {
        atomic_fetch_add_explicit(
            row_leases + row, 1u, memory_order_relaxed);
        int confirmed = lease_side
            ? atomic_load_explicit(
                  side_table + ids[tid], memory_order_relaxed)
            : atomic_load_explicit(
                  real_table + ids[tid], memory_order_relaxed);
        if (confirmed != row) {
          atomic_fetch_sub_explicit(
              row_leases + row, 1u, memory_order_relaxed);
          row = -1;
        }
      }
      local[tid] = row;
    }
  )METAL";
}

static void force_async_commit(
    mx::metal::Device& device, mx::metal::CommandEncoder& encoder,
    mx::array& guard, bool evaluator_submit) {
  if (evaluator_submit) {
    // The caller evaluates entry_local with mx.async_eval(), which gives MLX
    // a normal evaluator-owned submission boundary.  Do not pad that buffer.
    return;
  }
  auto library = device.get_library(
      "streaming_async_demand", [] { return async_demand_metal_source(); });
  auto noop = device.get_kernel("demand_async_noop", library);
  const int max_ops = std::get<0>(device.get_max_ops_mb_per_buffer());
  // The command buffer already contains the upstream gate/remap graph.  Pad
  // only until MLX reports that this *current* buffer needs submission;
  // blindly dispatching max_ops+1 no-ops ignores the work already encoded and
  // can spill gratuitous kernels into a fresh buffer.
  for (int index = 0; index <= max_ops && !encoder.needs_commit(); ++index) {
    encoder.set_compute_pipeline_state(noop);
    encoder.set_output_array(guard, 0);
    encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  }
}

class AsyncDemandPrimitive : public mx::Primitive {
 public:
  AsyncDemandPrimitive(
      mx::Stream stream, std::shared_ptr<AsyncDemandState> state,
      std::vector<int> seg_nbytes, int layer, int side_gen,
      std::string path, int stride, int cap, bool lfu,
      int decay_interval, int64_t forward_id, int sequence_length,
      bool use_side, bool wait_for_pending, bool wait_for_refinement,
      bool evaluator_submit,
      std::vector<int> prefetch_seg_nbytes = {},
      int prefetch_layer = -1, std::string prefetch_path = {},
      int prefetch_stride = 0, int prefetch_cap = 0,
      int prefetch_spec_limit = 0,
      std::vector<int> prefetch_resident = {})
      : Primitive(stream), state_(std::move(state)),
        seg_nbytes_(std::move(seg_nbytes)), layer_(layer),
        side_gen_(side_gen), path_(std::move(path)), stride_(stride),
        cap_(cap), lfu_(lfu), decay_interval_(decay_interval),
        forward_id_(forward_id), sequence_length_(sequence_length),
        use_side_(use_side), wait_for_pending_(wait_for_pending),
        wait_for_refinement_(wait_for_refinement),
        evaluator_submit_(evaluator_submit),
        prefetch_seg_nbytes_(std::move(prefetch_seg_nbytes)),
        prefetch_layer_(prefetch_layer),
        prefetch_path_(std::move(prefetch_path)),
        prefetch_stride_(prefetch_stride), prefetch_cap_(prefetch_cap),
        prefetch_spec_limit_(prefetch_spec_limit),
        prefetch_resident_(std::move(prefetch_resident)) {}

  const char* name() const override { return "AsyncDemandPrimitive"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("AsyncDemandPrimitive requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    const bool fused_prefetch = prefetch_layer_ >= 0;
    const size_t demand_input_count = seg_nbytes_.size() + 4;
    const size_t expected_inputs = demand_input_count + (
        fused_prefetch ? 1 + prefetch_seg_nbytes_.size() : 0);
    if (inputs.size() < 5 || expected_inputs != inputs.size())
      throw std::invalid_argument("async demand input/segment mismatch");
    if (outputs.size() != 2)
      throw std::invalid_argument("async demand requires entry/final outputs");
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    outputs[1].set_data(mx::allocator::malloc(outputs[1].nbytes()));
    auto entry_local = outputs[0];
    auto final_local = outputs[1];
    int32_t* entry_ptr = entry_local.data<int32_t>();
    int32_t* final_ptr = final_local.data<int32_t>();
    const mx::array ids = inputs[0];
    const uint32_t* id_ptr = ids.data<uint32_t>();
    const size_t count = ids.size();
    std::vector<mx::array> pools(
        inputs.begin() + 1, inputs.begin() + 1 + seg_nbytes_.size());
    const mx::array real_table = inputs[1 + seg_nbytes_.size()];
    const mx::array side_table = inputs[2 + seg_nbytes_.size()];
    const mx::array row_leases = inputs[3 + seg_nbytes_.size()];
    std::vector<uint8_t*> pool_ptrs;
    pool_ptrs.reserve(pools.size());
    for (auto pool : pools) pool_ptrs.push_back(pool.data<uint8_t>());
    mx::array prefetch_ids = ids;
    const uint32_t* prefetch_id_ptr = nullptr;
    size_t prefetch_count = 0;
    std::vector<mx::array> prefetch_pools;
    if (fused_prefetch) {
      prefetch_ids = inputs[demand_input_count];
      prefetch_id_ptr = prefetch_ids.data<uint32_t>();
      prefetch_count = prefetch_ids.size();
      prefetch_pools.assign(
          inputs.begin() + demand_input_count + 1, inputs.end());
    }

    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    auto state = state_;
    {
      std::lock_guard<std::mutex> lock(g_async_event_mutex);
      if (!g_async_event || g_async_event_device != device.mtl_device()) {
        g_async_event = NS::TransferPtr(device.mtl_device()->newSharedEvent());
        g_async_event_device = device.mtl_device();
        g_async_event_value.store(1, std::memory_order_relaxed);
      }
      state->event = g_async_event;
      state->demand_value =
          g_async_event_value.fetch_add(1, std::memory_order_relaxed);
    }
    auto seg_nbytes = seg_nbytes_;
    const int layer = layer_, side_gen = side_gen_, stride = stride_, cap = cap_;
    const bool lfu = lfu_, use_side = use_side_;
    const bool wait_for_pending = wait_for_pending_;
    const bool wait_for_refinement = wait_for_refinement_;
    const int decay = decay_interval_, sequence_length = sequence_length_;
    const int64_t forward_id = forward_id_;
    const std::string path = path_;
    const auto prefetch_seg_nbytes = prefetch_seg_nbytes_;
    const int prefetch_layer = prefetch_layer_;
    const std::string prefetch_path = prefetch_path_;
    const int prefetch_stride = prefetch_stride_;
    const int prefetch_cap = prefetch_cap_;
    const int prefetch_spec_limit = prefetch_spec_limit_;
    const auto prefetch_resident = prefetch_resident_;
    // Persistent shared tables are updated when real/side ownership changes.
    // Remap therefore follows gate computation in the same command buffer;
    // there is no host table snapshot or pre-remap SharedEvent round trip.
    auto library = device.get_library(
        "streaming_async_demand", [] { return async_demand_metal_source(); });
    auto remap = device.get_kernel("demand_gpu_remap", library);
    encoder.set_compute_pipeline_state(remap);
    encoder.set_input_array(ids, 0);
    encoder.set_input_array(real_table, 1);
    encoder.set_input_array(side_table, 2);
    encoder.set_input_array(row_leases, 3);
    encoder.set_output_array(outputs[0], 4);
    const char* lease_env = std::getenv("SIDEREGION_ROW_LEASES");
    struct LeaseParams {
      int side_enabled; int real_cap; int real_enabled; int side_lease_enabled;
    };
    const LeaseParams lease_params = {
        use_side ? 1 : 0,
        cap,
        use_side ? 0 : 1,
        use_side && lease_env && lease_env[0] == '1' ? 1 : 0,
    };
    encoder.set_bytes(lease_params, 5);
    encoder.dispatch_threads(
        MTL::Size(count, 1, 1),
        MTL::Size(std::min<size_t>(count, 256), 1, 1));

    // Submit the route-producing buffer before encoding the separate
    // final-local wait primitive.
    force_async_commit(device, encoder, outputs[0], evaluator_submit_);
    MTL::CommandBuffer* command_buffer = encoder.get_command_buffer();
    {
      std::lock_guard<std::mutex> lock(g_async_active_mutex);
      ++g_async_active;
    }
    command_buffer->addCompletedHandler(
        [inputs, pools, pool_ptrs, prefetch_ids, prefetch_pools,
         prefetch_id_ptr, prefetch_count, entry_local, final_local, entry_ptr,
         final_ptr, id_ptr, count,
         state, seg_nbytes, layer, side_gen, path, stride, cap, lfu,
         decay, forward_id, sequence_length, use_side,
         wait_for_pending, wait_for_refinement, prefetch_seg_nbytes,
         prefetch_layer, prefetch_path, prefetch_stride, prefetch_cap,
         prefetch_spec_limit, prefetch_resident](MTL::CommandBuffer*) {
          try {
            if (use_side) sideregion_release_before_layer(layer);
            else real_release_before_layer(layer);
            long stats[3] = {0, 0, 0};
            bool overcap = false;
            std::vector<int32_t> local;
            std::vector<std::pair<int, int>> placements;
            bool any_miss = false;
            long entry_hit_positions = 0;
            for (size_t index = 0; index < count; ++index) {
              any_miss = any_miss || entry_ptr[index] < 0;
              entry_hit_positions += entry_ptr[index] >= 0;
            }
            // The GPU remap is the authoritative entry snapshot.  Do not copy
            // the complete side ownership map or take the real-pool lock on
            // every production all-hit layer merely for disabled diagnostics.
            // Those maps can contain dozens of entries and this completion
            // handler gates the target MoE command buffer.
            if (prefetch_audit_enabled()) {
              auto entry_side = use_side
                  ? sideregion_snapshot(layer, side_gen)
                  : std::unordered_map<int, int>{};
              std::lock_guard<std::mutex> lock(g_real_mutex);
              RealLayer& real = g_real[layer];
              real_ensure_locked(real, cap);
              prefetch_audit_note_demand(
                  forward_id, layer, sequence_length, id_ptr, count,
                  real.e2r, entry_side);
            }
            note_deadline_from_gpu_local(
                layer, id_ptr, entry_ptr, count, cap);
            if (any_miss && wait_for_pending) {
              auto pending_t0 = std::chrono::steady_clock::now();
              if (use_side) {
                if (wait_for_refinement)
                  sideregion_wait_expert_values(
                      forward_id, layer, side_gen, id_ptr, count);
                else
                  sideregion_wait_pending_values(
                      layer, side_gen, id_ptr, count);
              } else {
                if (wait_for_refinement)
                  sideregion_wait_refinement(forward_id, layer);
                real_prefetch_wait_pending(layer, id_ptr, count);
              }
              g_async_pending_wait_us.fetch_add(
                  static_cast<long>(std::chrono::duration_cast<
                      std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - pending_t0).count()),
                  std::memory_order_relaxed);
            }
            if (!any_miss) {
              stats[0] = static_cast<long>(count);
              std::memcpy(final_ptr, entry_ptr, count * sizeof(int32_t));
              if (use_side)
                sideregion_note_demand_values(
                    layer, side_gen, id_ptr, count);
              std::vector<int> access;
              access.reserve(count);
              std::unordered_set<int> seen;
              for (size_t index = 0; index < count; ++index) {
                int expert = static_cast<int>(id_ptr[index]);
                if (seen.insert(expert).second) access.push_back(expert);
              }
              std::lock_guard<std::mutex> lock(g_real_mutex);
              RealLayer& real = g_real[layer];
              note_access_locked(layer, real, access, lfu, decay);
            } else {
              // The entry GPU table can miss a prediction whose SSD read was
              // already reserved but had not published yet.  After the
              // targeted wait, resolve just this route against the current
              // direct rows.  Do not send those rescued hits through the
              // allocator/eviction/fallback state machine.
              auto side_local = use_side
                  ? sideregion_lookup_values(
                        layer, side_gen, id_ptr, count)
                  : real_lookup_values(layer, id_ptr, count);
              std::unordered_map<int, int> route_side;
              route_side.reserve(count);
              local.resize(count, -1);
              bool true_miss = false;
              for (size_t index = 0; index < count; ++index) {
                const int expert = static_cast<int>(id_ptr[index]);
                const int entry_row = static_cast<int>(entry_ptr[index]);
                const int side_row = static_cast<int>(side_local[index]);
                if (side_row >= 0) route_side[expert] = side_row;
                // Real rows are stable until this handler takes the real lock.
                // A side row must be revalidated because a prefetch callback
                // may have evicted/reassigned it after the entry GPU snapshot.
                const int resolved = side_row >= 0 ? side_row : entry_row;
                local[index] = resolved;
                true_miss = true_miss || resolved < 0;
              }
              if (use_side)
                sideregion_note_demand_values(
                    layer, side_gen, id_ptr, count);
              if (!true_miss) {
                stats[0] = static_cast<long>(count);
                std::vector<int> access;
                access.reserve(count);
                std::unordered_set<int> seen;
                for (size_t index = 0; index < count; ++index) {
                  int expert = static_cast<int>(id_ptr[index]);
                  if (seen.insert(expert).second) access.push_back(expert);
                }
                std::lock_guard<std::mutex> lock(g_real_mutex);
                RealLayer& real = g_real[layer];
                note_access_locked(layer, real, access, lfu, decay);
              } else {
                // Only experts still absent after the targeted pending wait
                // enter allocation and synchronous SSD fallback.  Include all
                // current route-side rows so the core never duplicates them in
                // the real region.
                std::lock_guard<std::mutex> lock(g_real_mutex);
                RealLayer& real = g_real[layer];
                real_ensure_locked(real, cap);
                local = demand_core_locked(
                    layer, real, id_ptr, count, route_side, lfu, decay,
                    placements, stats, &overcap);
              }
            }
            if (overcap)
              throw std::runtime_error(
                  "async demand route set exceeds real pool capacity");

            if (!placements.empty()) {
              auto fallback_t0 = std::chrono::steady_clock::now();
              std::vector<long> seg_off(seg_nbytes.size());
              std::vector<long> seg_nb(seg_nbytes.size());
              long offset = 0;
              for (size_t index = 0; index < seg_nbytes.size(); ++index) {
                seg_off[index] = offset;
                seg_nb[index] = seg_nbytes[index];
                offset += seg_nbytes[index];
              }
              auto tickets = submit_demand_reads(
                  pools, seg_off, seg_nb, placements, path,
                  static_cast<long>(stride));
              for (long ticket : tickets) bg_reader_wait(ticket);
              g_async_fallback_wait_us.fetch_add(
                  static_cast<long>(std::chrono::duration_cast<
                      std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - fallback_t0).count()),
                  std::memory_order_relaxed);
            }
            // Current-layer true misses are deadline-critical. Starting the
            // next layer's broad speculative batch before them can occupy a
            // reader worker with false positives and lengthen the current
            // event wait. Submit prediction only after current demand bytes
            // are complete, but still before releasing this layer's MoE event;
            // it therefore retains the full source-MoE compute window.
            if (prefetch_layer >= 0) {
              prefetch_unified_ready(
                  prefetch_pools, prefetch_seg_nbytes,
                  prefetch_id_ptr, prefetch_count,
                  prefetch_layer, prefetch_path, prefetch_stride,
                  prefetch_resident, prefetch_spec_limit, prefetch_cap,
                  layer, forward_id);
            }
            if (any_miss)
              std::memcpy(final_ptr, local.data(), count * sizeof(int32_t));
            if (use_side)
              sideregion_acquire_row_leases(
                  layer, final_ptr, count, cap);
            g_async_calls.fetch_add(1, std::memory_order_relaxed);
            g_async_loads.fetch_add(stats[2], std::memory_order_relaxed);
            // This is the GPU remap result at target entry.  Hits acquired by
            // waiting for pending side reads below must not inflate deadline
            // coverage or be presented as zero-wait hits.
            g_async_hit_positions.fetch_add(
                entry_hit_positions, std::memory_order_relaxed);
            g_async_positions.fetch_add(
                static_cast<long>(count), std::memory_order_relaxed);
            const long rescued = std::max<long>(
                0, stats[0] - entry_hit_positions);
            g_async_pending_rescued.fetch_add(
                rescued, std::memory_order_relaxed);
            if (stats[2] > 0)
              g_async_true_fallback.fetch_add(1, std::memory_order_relaxed);
            if (stats[2] > 0) {
              std::lock_guard<std::mutex> lock(g_real_mutex);
              mark_predict_cooldown_locked(layer);
            }
            if (!any_miss)
              g_async_fast.fetch_add(1, std::memory_order_relaxed);
            else
              g_async_fallback.fetch_add(1, std::memory_order_relaxed);
          } catch (const std::exception& error) {
            std::fill(final_ptr, final_ptr + count, 0);
            std::lock_guard<std::mutex> lock(g_async_error_mutex);
            if (g_async_error.empty()) g_async_error = error.what();
          }
          // CPU writes to shared pool/local memory happen-before all GPU work
          // after the command-buffer event wait below.
          state->event->setSignaledValue(state->demand_value);
          {
            std::lock_guard<std::mutex> lock(g_async_active_mutex);
            --g_async_active;
          }
          g_async_active_cv.notify_all();
        });
  }

 private:
  std::shared_ptr<AsyncDemandState> state_;
  std::vector<int> seg_nbytes_;
  int layer_, side_gen_;
  std::string path_;
  int stride_, cap_;
  bool lfu_;
  int decay_interval_;
  int64_t forward_id_;
  int sequence_length_;
  bool use_side_;
  bool wait_for_pending_;
  bool wait_for_refinement_;
  bool evaluator_submit_;
  std::vector<int> prefetch_seg_nbytes_;
  int prefetch_layer_;
  std::string prefetch_path_;
  int prefetch_stride_;
  int prefetch_cap_;
  int prefetch_spec_limit_;
  std::vector<int> prefetch_resident_;
};

class AsyncDemandWaitPrimitive : public mx::Primitive {
 public:
  AsyncDemandWaitPrimitive(
      mx::Stream stream, std::shared_ptr<AsyncDemandState> state)
      : Primitive(stream), state_(std::move(state)) {}
  const char* name() const override { return "AsyncDemandWaitPrimitive"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("AsyncDemandWaitPrimitive requires Metal");
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.end_encoding();
    encoder.get_command_buffer()->encodeWait(
        state_->event.get(), state_->demand_value);
    auto library = device.get_library(
        "streaming_async_demand", [] { return async_demand_metal_source(); });
    auto copy = device.get_kernel("demand_async_copy", library);
    encoder.set_compute_pipeline_state(copy);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_output_array(outputs[0], 1);
    encoder.dispatch_threads(
        MTL::Size(outputs[0].size(), 1, 1),
        MTL::Size(std::min<size_t>(outputs[0].size(), 256), 1, 1));
  }

 private:
  std::shared_ptr<AsyncDemandState> state_;
};

class GpuOnlyDemandPrimitive : public mx::Primitive {
 public:
  GpuOnlyDemandPrimitive(mx::Stream stream, int cap, bool use_side)
      : Primitive(stream), cap_(cap), use_side_(use_side) {}
  const char* name() const override { return "GpuOnlyDemandPrimitive"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("GpuOnlyDemandPrimitive requires Metal");
  }
  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    auto library = device.get_library(
        "streaming_async_demand", [] { return async_demand_metal_source(); });
    auto remap = device.get_kernel("demand_gpu_remap", library);
    encoder.set_compute_pipeline_state(remap);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_output_array(outputs[0], 4);
    struct LeaseParams {
      int side_enabled; int real_cap; int real_enabled; int side_lease_enabled;
    };
    const LeaseParams params = {
        use_side_ ? 1 : 0, cap_, use_side_ ? 0 : 1, 0,
    };
    encoder.set_bytes(params, 5);
    encoder.dispatch_threads(
        MTL::Size(outputs[0].size(), 1, 1),
        MTL::Size(std::min<size_t>(outputs[0].size(), 256), 1, 1));
  }
 private:
  int cap_;
  bool use_side_;
};

}  // namespace

mx::array demand_gpu_remap_only(
    const mx::array& inds, int layer, int side_gen, int cap,
    bool use_side, mx::StreamOrDevice s = {}) {
  mx::Stream stream = mx::to_stream(s);
  mx::array ids = mx::contiguous(inds, false, stream);
  mx::array real_table = real_slot_table(layer, cap);
  mx::array side_table = use_side
      ? sideregion_slot_table(layer, side_gen)
      : sideregion_slot_table(-1, -1);
  mx::array side_leases = use_side
      ? sideregion_lease_table(layer, side_gen)
      : real_lease_table(layer, cap);
  return mx::array(
      ids.shape(), mx::int32,
      std::make_shared<GpuOnlyDemandPrimitive>(stream, cap, use_side),
      {ids, real_table, side_table, side_leases});
}

std::pair<mx::array, mx::array> demand_dual_split_async(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen,
    const std::string& path, int stride, int cap, bool lfu,
    int decay_interval, int64_t forward_id, int sequence_length,
    bool use_side, bool wait_for_pending, bool wait_for_refinement,
    bool evaluator_submit,
    mx::StreamOrDevice s = {}) {
  if (seg_nbytes.size() != pool_list.size())
    throw std::invalid_argument("demand_dual_async segment mismatch");
  mx::Stream stream = mx::to_stream(s);
  mx::array ids = mx::contiguous(inds, false, stream);
  auto state = std::make_shared<AsyncDemandState>();
  mx::array real_table = real_slot_table(layer, cap);
  mx::array side_table = use_side
      ? sideregion_slot_table(layer, side_gen)
      : sideregion_slot_table(-1, -1);
  mx::array side_leases = use_side
      ? sideregion_lease_table(layer, side_gen)
      : real_lease_table(layer, cap);
  std::vector<mx::array> inputs;
  inputs.reserve(pool_list.size() + 4);
  inputs.push_back(ids);
  inputs.insert(inputs.end(), pool_list.begin(), pool_list.end());
  inputs.push_back(real_table);
  inputs.push_back(side_table);
  inputs.push_back(side_leases);
  auto primitive = std::make_shared<AsyncDemandPrimitive>(
      stream, state, seg_nbytes, layer, side_gen, path, stride, cap,
      lfu, decay_interval, forward_id, sequence_length, use_side,
      wait_for_pending, wait_for_refinement, evaluator_submit);
  auto raw = mx::array::make_arrays(
      {ids.shape(), ids.shape()}, {mx::int32, mx::int32},
      primitive, inputs);
  mx::array gated_final(
      raw[1].shape(), mx::int32,
      std::make_shared<AsyncDemandWaitPrimitive>(stream, state),
      {raw[1]});
  return {raw[0], gated_final};
}

mx::array demand_dual_async(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen,
    const std::string& path, int stride, int cap, bool lfu,
    int decay_interval, int64_t forward_id, int sequence_length,
    bool use_side, bool wait_for_pending, bool wait_for_refinement,
    bool evaluator_submit,
    mx::StreamOrDevice s = {}) {
  return demand_dual_split_async(
      inds, pool_list, seg_nbytes, layer, side_gen, path, stride, cap,
      lfu, decay_interval, forward_id, sequence_length, use_side,
      wait_for_pending, wait_for_refinement, evaluator_submit, s).second;
}

std::vector<long> demand_async_stats() {
  return {
      g_async_calls.load(std::memory_order_relaxed),
      g_async_fast.load(std::memory_order_relaxed),
      g_async_fallback.load(std::memory_order_relaxed),
      g_async_loads.load(std::memory_order_relaxed),
      g_async_hit_positions.load(std::memory_order_relaxed),
      g_async_positions.load(std::memory_order_relaxed),
      g_async_true_fallback.load(std::memory_order_relaxed),
      g_async_pending_rescued.load(std::memory_order_relaxed),
      g_async_pending_wait_us.load(std::memory_order_relaxed),
      g_async_fallback_wait_us.load(std::memory_order_relaxed),
  };
}

void demand_async_stats_reset() {
  {
    std::unique_lock<std::mutex> lock(g_async_active_mutex);
    g_async_active_cv.wait(lock, [] { return g_async_active == 0; });
  }
  g_async_calls.store(0, std::memory_order_relaxed);
  g_async_fast.store(0, std::memory_order_relaxed);
  g_async_fallback.store(0, std::memory_order_relaxed);
  g_async_loads.store(0, std::memory_order_relaxed);
  g_async_hit_positions.store(0, std::memory_order_relaxed);
  g_async_positions.store(0, std::memory_order_relaxed);
  g_async_true_fallback.store(0, std::memory_order_relaxed);
  g_async_pending_rescued.store(0, std::memory_order_relaxed);
  g_async_pending_wait_us.store(0, std::memory_order_relaxed);
  g_async_fallback_wait_us.store(0, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(g_async_error_mutex);
  g_async_error.clear();
}

void demand_async_check() {
  {
    std::unique_lock<std::mutex> lock(g_async_active_mutex);
    g_async_active_cv.wait(lock, [] { return g_async_active == 0; });
  }
  std::lock_guard<std::mutex> lock(g_async_error_mutex);
  if (!g_async_error.empty())
    throw std::runtime_error("asynchronous demand failed: " + g_async_error);
}

mx::array demand_staged_multi(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, const std::string& path,
    int stride, int cap, bool lfu, int decay_interval, int spec_limit,
    const std::vector<mx::array>& staging_list,
    const std::vector<std::vector<int>>& staging_maps,
    int64_t forward_id, int sequence_length,
    mx::StreamOrDevice s = {}) {
  const double t0 = g_dt_on ? dt_now_us() : 0;
  if (seg_nbytes.size() != pool_list.size())
    throw std::invalid_argument(
        "demand_staged_multi: seg_nbytes.size() != pool_list.size()");
  if (staging_list.size() != staging_maps.size())
    throw std::invalid_argument(
        "demand_staged_multi: staging_list/maps size mismatch");
  size_t seg_sum = 0;
  for (int nb : seg_nbytes) {
    if (nb < 0) throw std::invalid_argument("demand_staged_multi: negative segment");
    seg_sum += static_cast<size_t>(nb);
  }
  if (seg_sum != static_cast<size_t>(stride))
    throw std::invalid_argument("demand_staged_multi: segment sum != stride");

  mx::array ids = mx::contiguous(inds);
  ids.eval();
  const double t1 = g_dt_on ? dt_now_us() : 0;
  const uint32_t* ip = ids.data<uint32_t>();
  const size_t n = ids.size();
  std::vector<uint8_t*> ptrs;
  ptrs.reserve(pool_list.size());
  for (auto& value : pool_list) {
    mx::array array = value;
    array.eval();
    ptrs.push_back(array.data<uint8_t>());
  }
  std::vector<const uint8_t*> staging_ptrs;
  staging_ptrs.reserve(staging_list.size());
  for (auto& value : staging_list) {
    mx::array array = value;
    array.eval();
    staging_ptrs.push_back(array.data<uint8_t>());
  }
  const double t2 = g_dt_on ? dt_now_us() : 0;

  struct StagedRow { int bank; int row; };
  std::unordered_map<int, StagedRow> staged;
  std::vector<int> staged_order;
  std::unordered_set<int> staged_seen;
  for (size_t bank = 0; bank < staging_maps.size(); ++bank) {
    const auto& mapping = staging_maps[bank];
    if (mapping.size() % 2 != 0)
      throw std::invalid_argument("demand_staged_multi: map must be expert,row pairs");
    for (size_t i = 0; i < mapping.size(); i += 2) {
      int expert = mapping[i], row = mapping[i + 1];
      if (row < 0 || static_cast<size_t>(row + 1) * seg_sum
              > staging_list[bank].nbytes()) continue;
      if (staged_seen.insert(expert).second) {
        staged[expert] = {static_cast<int>(bank), row};
        staged_order.push_back(expert);
      }
    }
  }
  const double t3 = g_dt_on ? dt_now_us() : 0;

  std::vector<int32_t> local(n, 0);
  std::vector<int> access_order;
  std::unordered_set<int> access_set;
  for (size_t i = 0; i < n; ++i) {
    int expert = static_cast<int>(ip[i]);
    if (access_set.insert(expert).second) access_order.push_back(expert);
  }
  std::vector<std::tuple<int, int, int>> promotions; // bank,row,slot
  std::vector<std::pair<int, int>> demand_reads;
  std::unordered_map<int, int> assigned;
  std::unordered_set<int> routed_promotions, routed_reads;
  long real_hitpos = 0, staged_hitpos = 0, demand_pos = 0;
  bool overcap = false;
  {
    std::unique_lock<std::mutex> lk(g_real_mutex);
    RealLayer& c = g_real[layer];
    real_ensure_locked(c, cap);
    merge_prediction_freq_locked(layer, c);

    std::unordered_map<int, int> staged_audit;
    for (const auto& item : staged) staged_audit[item.first] = item.second.row;
    prefetch_audit_note_demand(
        forward_id, layer, sequence_length, ip, n, c.e2r, staged_audit);
    note_deadline_locked(layer, c, ip, n, staged_audit);

    size_t unavailable_pins = 0;
    for (int expert : c.pinned)
      if (!access_set.count(expert)) ++unavailable_pins;
    if (access_set.size() + unavailable_pins > static_cast<size_t>(cap)) {
      overcap = true;
    } else {
      note_access_locked(layer, c, access_order, lfu, decay_interval);
      for (int expert : access_order) {
        auto resident = c.e2r.find(expert);
        if (resident != c.e2r.end()) {
          assigned[expert] = resident->second;
          continue;
        }
        int slot = alloc_slot_locked(c, expert, access_set);
        if (slot < 0)
          throw std::runtime_error("demand_staged_multi has no real slot");
        assigned[expert] = slot;
        auto ready = staged.find(expert);
        if (ready != staged.end()) {
          promotions.emplace_back(ready->second.bank, ready->second.row, slot);
          routed_promotions.insert(expert);
        } else {
          demand_reads.emplace_back(expert, slot);
          routed_reads.insert(expert);
        }
      }

      std::unordered_set<int> prediction_protect = access_set;
      prediction_protect.insert(staged_seen.begin(), staged_seen.end());
      for (int expert : staged_order) {
        if (access_set.count(expert)) continue;
        bool resident = c.e2r.count(expert) != 0;
        int slot = alloc_speculative_locked(
            c, expert, spec_limit, prediction_protect);
        if (slot >= 0 && !resident) {
          const auto row = staged[expert];
          promotions.emplace_back(row.bank, row.row, slot);
        }
      }

      for (size_t i = 0; i < n; ++i) {
        int expert = static_cast<int>(ip[i]);
        local[i] = assigned[expert];
        if (routed_promotions.count(expert)) ++staged_hitpos;
        else if (routed_reads.count(expert)) ++demand_pos;
        else ++real_hitpos;
      }

      std::vector<size_t> offsets(seg_nbytes.size(), 0);
      for (size_t i = 1; i < seg_nbytes.size(); ++i)
        offsets[i] = offsets[i - 1] + static_cast<size_t>(seg_nbytes[i - 1]);
      for (const auto& promotion : promotions) {
        int bank = std::get<0>(promotion), row = std::get<1>(promotion);
        int slot = std::get<2>(promotion);
        const uint8_t* source = staging_ptrs[bank]
            + static_cast<size_t>(row) * static_cast<size_t>(stride);
        for (size_t segment = 0; segment < seg_nbytes.size(); ++segment) {
          std::memcpy(
              ptrs[segment] + static_cast<size_t>(slot) * seg_nbytes[segment],
              source + offsets[segment], static_cast<size_t>(seg_nbytes[segment]));
        }
      }
      if (!demand_reads.empty()) {
        std::vector<long> offsets_long(offsets.begin(), offsets.end());
        std::vector<long> sizes(seg_nbytes.begin(), seg_nbytes.end());
        std::vector<long> tickets;
        tickets.reserve(demand_reads.size());
        for (const auto& request : demand_reads) {
          long ticket = g_demand_ticket.fetch_add(1);
          bg_pread_into_pool(
              pool_list, offsets_long, sizes, request.second, request.first,
              path, static_cast<long>(stride), ticket, 1, false);
          tickets.push_back(ticket);
        }
        for (long ticket : tickets) bg_reader_wait(ticket);
      }
    }
  }
  const double t4 = g_dt_on ? dt_now_us() : 0;

  {
    std::lock_guard<std::mutex> lk(g_dstat_mutex);
    if (overcap) {
      g_d_last[0] = 0; g_d_last[1] = static_cast<long>(n);
      g_d_last[2] = 0; g_d_last[3] = 2;
      g_d_last[4] = 0; g_d_last[5] = 0;
    } else {
      g_d_last[0] = real_hitpos + staged_hitpos;
      g_d_last[1] = demand_pos;
      g_d_last[2] = static_cast<long>(demand_reads.size());
      g_d_last[3] = demand_reads.empty() ? 0 : 1;
      g_d_last[4] = staged_hitpos;
      g_d_last[5] = real_hitpos;
    }
  }
  mx::array out = mx::array(local.data(), ids.shape(), mx::int32);
  if (g_dt_on) {
    const double t5 = dt_now_us();
    // Reuse the existing six-bucket diagnostic ABI:
    // ids eval / pool+staging eval / staged-map build / state+promotion+I/O /
    // output construction / total residual (zero for this implementation).
    g_dt[0] += t1 - t0;
    g_dt[1] += t2 - t1;
    g_dt[2] += t3 - t2;
    g_dt[3] += t4 - t3;
    g_dt[4] += t5 - t4;
  }
  return out;
}

int late_promote_staged(
    const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int cap, int spec_limit,
    const mx::array& staging, const std::vector<int>& staging_map) {
  if (seg_nbytes.size() != pool_list.size())
    throw std::invalid_argument(
        "late_promote_staged: seg_nbytes.size() != pool_list.size()");
  if (staging_map.size() % 2 != 0)
    throw std::invalid_argument(
        "late_promote_staged: staging_map must contain expert,row pairs");

  size_t stride = 0;
  std::vector<size_t> seg_off(seg_nbytes.size(), 0);
  for (size_t i = 0; i < seg_nbytes.size(); ++i) {
    if (seg_nbytes[i] < 0)
      throw std::invalid_argument("late_promote_staged: negative segment");
    seg_off[i] = stride;
    stride += static_cast<size_t>(seg_nbytes[i]);
  }
  std::vector<uint8_t*> ptrs;
  ptrs.reserve(pool_list.size());
  for (auto& value : pool_list) {
    mx::array array = value;
    array.eval();
    ptrs.push_back(array.data<uint8_t>());
  }
  mx::array stg = staging;
  stg.eval();
  const uint8_t* source_base = stg.data<uint8_t>();

  std::vector<int> order;
  std::unordered_map<int, int> rows;
  std::unordered_set<int> protect;
  for (size_t i = 0; i < staging_map.size(); i += 2) {
    int expert = staging_map[i], row = staging_map[i + 1];
    if (row < 0 || static_cast<size_t>(row + 1) * stride > stg.nbytes())
      continue;
    rows[expert] = row;
    if (protect.insert(expert).second) order.push_back(expert);
  }

  int promoted = 0;
  // Slot ownership and byte publication are one critical section: demand can
  // never observe an e2r entry before every segment has been copied.
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& c = g_real[layer];
  real_ensure_locked(c, cap);
  merge_prediction_freq_locked(layer, c);
  for (int expert : order) {
    if (c.e2r.count(expert)) continue;
    int slot = alloc_speculative_locked(
        c, expert, spec_limit, protect);
    if (slot < 0) continue;
    const uint8_t* source = source_base
        + static_cast<size_t>(rows[expert]) * stride;
    for (size_t segment = 0; segment < seg_nbytes.size(); ++segment) {
      std::memcpy(
          ptrs[segment] + static_cast<size_t>(slot) * seg_nbytes[segment],
          source + seg_off[segment],
          static_cast<size_t>(seg_nbytes[segment]));
    }
    ++promoted;
  }
  return promoted;
}

int demand_promote_staged(
    const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int cap, int spec_limit,
    const mx::array& staging, const std::vector<int>& staging_map,
    const mx::array& actual_ids) {
  if (seg_nbytes.size() != pool_list.size())
    throw std::invalid_argument(
        "demand_promote_staged: seg_nbytes.size() != pool_list.size()");
  if (staging_map.size() % 2 != 0)
    throw std::invalid_argument(
        "demand_promote_staged: staging_map must contain expert,row pairs");

  mx::array actual = mx::contiguous(actual_ids);
  actual.eval();
  const uint32_t* actual_ptr = actual.data<uint32_t>();
  std::unordered_set<int> routed;
  for (size_t i = 0; i < actual.size(); ++i)
    routed.insert(static_cast<int>(actual_ptr[i]));

  size_t stride = 0;
  std::vector<size_t> seg_off(seg_nbytes.size(), 0);
  for (size_t i = 0; i < seg_nbytes.size(); ++i) {
    if (seg_nbytes[i] < 0)
      throw std::invalid_argument("demand_promote_staged: negative segment");
    seg_off[i] = stride;
    stride += static_cast<size_t>(seg_nbytes[i]);
  }
  std::vector<uint8_t*> ptrs;
  ptrs.reserve(pool_list.size());
  for (auto& value : pool_list) {
    mx::array array = value;
    array.eval();
    ptrs.push_back(array.data<uint8_t>());
  }
  mx::array stg = staging;
  stg.eval();
  const uint8_t* source_base = stg.data<uint8_t>();

  std::vector<int> order;
  std::unordered_map<int, int> rows;
  std::unordered_set<int> staged;
  for (size_t i = 0; i < staging_map.size(); i += 2) {
    int expert = staging_map[i], row = staging_map[i + 1];
    if (row < 0 || static_cast<size_t>(row + 1) * stride > stg.nbytes())
      continue;
    rows[expert] = row;
    if (staged.insert(expert).second) order.push_back(expert);
  }

  int promoted = 0;
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& c = g_real[layer];
  real_ensure_locked(c, cap);
  merge_prediction_freq_locked(layer, c);

  // First reserve every correctly predicted route as a normal demand row.
  // It may evict stale speculative rows and is never rejected merely because
  // the speculative quota has drained to zero.
  for (int expert : order) {
    if (!routed.count(expert)) continue;
    auto existing = c.e2r.find(expert);
    if (existing != c.e2r.end()) {
      c.speculative.erase(expert);
      continue;
    }
    int slot = alloc_slot_locked(c, expert, routed);
    if (slot < 0) continue;
    const uint8_t* source = source_base
        + static_cast<size_t>(rows[expert]) * stride;
    for (size_t segment = 0; segment < seg_nbytes.size(); ++segment) {
      std::memcpy(
          ptrs[segment] + static_cast<size_t>(slot) * seg_nbytes[segment],
          source + seg_off[segment],
          static_cast<size_t>(seg_nbytes[segment]));
    }
    ++promoted;
  }

  // Non-routed predictions use only the bounded speculative share.
  std::unordered_set<int> protect = routed;
  protect.insert(staged.begin(), staged.end());
  for (int expert : order) {
    if (routed.count(expert) || c.e2r.count(expert)) continue;
    int slot = alloc_speculative_locked(c, expert, spec_limit, protect);
    if (slot < 0) continue;
    const uint8_t* source = source_base
        + static_cast<size_t>(rows[expert]) * stride;
    for (size_t segment = 0; segment < seg_nbytes.size(); ++segment) {
      std::memcpy(
          ptrs[segment] + static_cast<size_t>(slot) * seg_nbytes[segment],
          source + seg_off[segment],
          static_cast<size_t>(seg_nbytes[segment]));
    }
    ++promoted;
  }
  return promoted;
}

// 取本次 demand 统计 [hitpos, misspos, loads, fallback01]（主线程串行，安全）。
std::vector<long> demand_last_stats() {
  std::lock_guard<std::mutex> lk(g_dstat_mutex);
  return {g_d_last[0], g_d_last[1], g_d_last[2],
          g_d_last[3], g_d_last[4], g_d_last[5]};
}

std::vector<long> demand_deadline_stats() {
  std::lock_guard<std::mutex> lk(g_deadline_mutex);
  std::vector<long> out;
  out.reserve(g_deadline.size() * 6);
  for (const auto& item : g_deadline) {
    const DemandDeadlineLayer& row = item.second;
    out.push_back(item.first);
    out.push_back(row.calls);
    out.push_back(row.actual_unique);
    out.push_back(row.real_resident);
    out.push_back(row.side_prefetch_complete);
    out.push_back(row.demand_fallback);
  }
  return out;
}

void demand_deadline_stats_reset() {
  std::lock_guard<std::mutex> lk(g_deadline_mutex);
  g_deadline.clear();
  g_deadline_enabled.store(true, std::memory_order_relaxed);
}

// 测试壳：纯状态推进(不 pread/不侧区)，把 experts_flat 当一次 demand 的 inds，返回 local 槽位。
// 供 LFU 驱逐语义逐步等价单测(与 Python 参考对拍)。
std::vector<int> real_debug_place(int layer, const std::vector<int>& experts_flat, int cap,
                                  bool lfu, int decay_interval) {
  std::vector<uint32_t> u(experts_flat.begin(), experts_flat.end());
  long stats[3];
  std::vector<std::pair<int, int>> placements;
  std::lock_guard<std::mutex> lk(g_real_mutex);
  RealLayer& c = g_real[layer];
  real_ensure_locked(c, cap);
  std::unordered_map<int, int> empty_side;
  auto local = demand_core_locked(layer, c, u.data(), u.size(), empty_side, lfu, decay_interval,
                                  placements, stats);
  return std::vector<int>(local.begin(), local.end());
}
