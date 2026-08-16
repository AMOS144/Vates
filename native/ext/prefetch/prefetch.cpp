// [2] 轻量预取(无 staging，仅预热 page cache) + [3] staging 版 miss→hit + STAGING_HPROF 探针。
#include "prefetch.h"
#include "../io/bg_reader.h"
#include "../pool/demand.h"
#include "../pool/side_region.h"

#include <chrono>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <unordered_set>
#include <fcntl.h>
#include <unistd.h>

// ====== [2] 轻量预取(无 staging 模式)：GPU 完成回调里读 inds、pread 预热 page cache ======
// 命门验证:回调在 command buffer 完成后触发,此时 inds 已算完,读到的是正确值。
class PrefetchOnCompletePrimitive : public mx::Primitive {
 public:
  PrefetchOnCompletePrimitive(mx::Stream s, std::string path, size_t stride, bool do_read)
      : Primitive(s), path_(std::move(path)), stride_(stride), do_read_(do_read) {}
  const char* name() const override { return "PrefetchOnCompletePrimitive"; }

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0];
    ids.eval();
    record_and_read(ids.data<uint32_t>(), ids.size(), path_, stride_, do_read_);
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0];                 // by-value：保活到回调结束
    const uint32_t* ptr = ids.data<uint32_t>();  // 指针此刻有效（buffer 已分配），值待 GPU 算完
    size_t n = ids.size();
    std::string path = path_;
    size_t stride = stride_;
    bool do_read = do_read_;
    auto& enc = mx::metal::get_command_encoder(stream());
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    cb->addCompletedHandler([ids, ptr, n, path, stride, do_read](MTL::CommandBuffer*) {
      // 此时 buffer 已完成 → ptr 指向已算好的 inds 值。ids 按值捕获保证 buffer 不被释放。
      record_and_read(ptr, n, path, stride, do_read);
    });
  }

 private:
  static void record_and_read(const uint32_t* p, size_t n, const std::string& path,
                              size_t stride, bool do_read) {
    int fd = (do_read && !path.empty()) ? ::open(path.c_str(), O_RDONLY) : -1;
    static thread_local std::vector<uint8_t> buf;
    if (do_read && buf.size() < stride) buf.resize(stride);
    for (size_t i = 0; i < n; ++i) {
      int e = static_cast<int>(p[i]);
      if (fd >= 0) ::pread(fd, buf.data(), stride, static_cast<off_t>(static_cast<size_t>(e) * stride));
    }
    if (fd >= 0) ::close(fd);
  }
  std::string path_;
  size_t stride_;
  bool do_read_;
};

// 返回一个 dummy 输出（挂进图里，eval 时触发上面的回调）。
mx::array prefetch_on_complete(
    const mx::array& expert_ids,
    const std::string& path,
    int stride,
    bool do_read = true,
    mx::StreamOrDevice s = {}) {
  return mx::array(
      mx::Shape{1},
      mx::uint8,
      std::make_shared<PrefetchOnCompletePrimitive>(
          mx::to_stream(s), path, static_cast<size_t>(stride), do_read),
      std::vector<mx::array>{expert_ids});
}

// ---- handler 触发时刻探针(STAGING_HPROF)：记录每次 staging 完成回调被 Metal 触发的时刻 ----
// 用于实测"完成回调到底扎堆在 eval 尾、还是逐层铺开"。仅诊断用，默认关。
static std::mutex g_hprof_mutex;
static std::vector<std::tuple<long, int, double>> g_hprof_log;  // (gen, layer, t_fire_seconds)
static bool g_hprof_on = false;

static inline double hprof_steady_now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ====== [3] staging 版 miss→hit：handler pread 进 per-layer staging + 记录 (expert→row) ======
static std::mutex g_stg_mutex;
// (layer,gen) -> [(expert,row)].  Progressive rerank can have an immutable
// early core and a later refinement for the same target layer; keying only by
// layer let the second callback overwrite the first completed bank.
static std::map<std::pair<int, long>, std::vector<std::pair<int, int>>> g_stg_ready;
// Multi-step progressive state is keyed by logical (forward,target), never by
// the small physical generation.  ``seen`` deduplicates early/refinement
// submissions, while ``pending`` contains only rows whose bytes are not yet
// complete.  Demand waits for the intersection of its real route and pending.
using StagingTarget = std::pair<int64_t, int>;
static std::map<StagingTarget, std::unordered_set<int>> g_stg_seen;
static std::map<StagingTarget, std::unordered_set<int>> g_stg_pending;
static std::map<StagingTarget, std::unordered_set<int>> g_stg_complete;
static std::map<StagingTarget, long> g_stg_target_inflight;
static std::set<StagingTarget> g_stg_refinement_ready;
static std::set<StagingTarget> g_stg_demand_finished;
static std::map<std::pair<int, long>, bool> g_stg_finished;
static std::set<std::pair<int, long>> g_stg_consumed;
static std::condition_variable g_stg_progress_cv;
struct StagingWaitStats {
  long calls = 0;
  long actual_unique = 0;
  long complete_at_entry = 0;
  long pending_at_entry = 0;
  long wait_us = 0;
};
static std::mutex g_stg_wait_stats_mutex;
static std::map<int, StagingWaitStats> g_stg_wait_stats;
static std::atomic<bool> g_stg_wait_stats_enabled{false};
static std::mutex g_stg_drain_mutex;
static std::condition_variable g_stg_drain_cv;
static long g_stg_inflight = 0;

static void staging_inflight_start(int64_t forward_id, int layer) {
  {
    std::lock_guard<std::mutex> lk(g_stg_drain_mutex);
    ++g_stg_inflight;
  }
  if (forward_id >= 0) {
    std::lock_guard<std::mutex> stg_lk(g_stg_mutex);
    g_stg_target_inflight[{forward_id, layer}] += 1;
  }
}

static void staging_inflight_done(int64_t forward_id, int layer) {
  if (forward_id >= 0) {
    std::lock_guard<std::mutex> stg_lk(g_stg_mutex);
    auto key = StagingTarget{forward_id, layer};
    auto it = g_stg_target_inflight.find(key);
    if (it != g_stg_target_inflight.end() && --it->second == 0) {
      g_stg_target_inflight.erase(it);
      if (g_stg_demand_finished.erase(key)) {
        g_stg_seen.erase(key);
        g_stg_pending.erase(key);
        g_stg_complete.erase(key);
        g_stg_refinement_ready.erase(key);
      }
    }
    g_stg_progress_cv.notify_all();
  }
  {
    std::lock_guard<std::mutex> lk(g_stg_drain_mutex);
    --g_stg_inflight;
  }
  g_stg_drain_cv.notify_all();
}

static std::pair<mx::array, std::unordered_set<int>> staging_prejoin_snapshot(
    int64_t forward_id, int layer, const mx::array& expert_ids) {
  mx::array ids = mx::contiguous(expert_ids);
  ids.eval();
  const uint32_t* values = ids.data<uint32_t>();
  const auto key = StagingTarget{forward_id, layer};
  std::unordered_set<int> complete_at_entry;
  {
    std::lock_guard<std::mutex> lk(g_stg_mutex);
    auto complete = g_stg_complete.find(key);
    if (complete != g_stg_complete.end()) complete_at_entry = complete->second;
  }
  demand_prejoin_note(layer, values, ids.size(), complete_at_entry);
  return {ids, complete_at_entry};
}

void prefetch_staging_note_prejoin(
    int64_t forward_id, int layer, const mx::array& expert_ids) {
  staging_prejoin_snapshot(forward_id, layer, expert_ids);
}

void prefetch_staging_wait_experts(
    int64_t forward_id, int layer, const mx::array& expert_ids) {
  auto snapshot = staging_prejoin_snapshot(forward_id, layer, expert_ids);
  mx::array ids = snapshot.first;
  const auto& complete_at_entry = snapshot.second;
  const uint32_t* values = ids.data<uint32_t>();
  std::unordered_set<int> wanted;
  for (size_t i = 0; i < ids.size(); ++i)
    wanted.insert(static_cast<int>(values[i]));
  const auto key = StagingTarget{forward_id, layer};
  long pending_at_entry = 0;
  {
    std::lock_guard<std::mutex> lk(g_stg_mutex);
    auto pending = g_stg_pending.find(key);
    if (pending != g_stg_pending.end()) {
      for (int expert : wanted)
        if (pending->second.count(expert)) ++pending_at_entry;
    }
  }
  auto wait_started = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lk(g_stg_mutex);
  g_stg_progress_cv.wait(lk, [&] {
    // The refinement callback is the registration barrier: once visible,
    // both early and tail candidate sets have entered seen/pending.
    if (!g_stg_refinement_ready.count(key)) return false;
    auto pending = g_stg_pending.find(key);
    if (pending == g_stg_pending.end()) return true;
    for (int expert : wanted)
      if (pending->second.count(expert)) return false;
    return true;
  });
  lk.unlock();
  long wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - wait_started).count();
  if (g_stg_wait_stats_enabled.load(std::memory_order_relaxed)) {
    long complete_hits = 0;
    for (int expert : wanted)
      if (complete_at_entry.count(expert)) ++complete_hits;
    std::lock_guard<std::mutex> stat_lk(g_stg_wait_stats_mutex);
    auto& row = g_stg_wait_stats[layer];
    ++row.calls;
    row.actual_unique += static_cast<long>(wanted.size());
    row.complete_at_entry += complete_hits;
    row.pending_at_entry += pending_at_entry;
    row.wait_us += wait_us;
  }
}

void prefetch_staging_finish_demand(int64_t forward_id, int layer) {
  if (forward_id < 0) return;
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  const auto key = StagingTarget{forward_id, layer};
  if (!g_stg_target_inflight.count(key)) {
    g_stg_seen.erase(key);
    g_stg_pending.erase(key);
    g_stg_complete.erase(key);
    g_stg_refinement_ready.erase(key);
  } else {
    g_stg_demand_finished.insert(key);
  }
}

std::vector<long> prefetch_staging_wait_stats() {
  std::lock_guard<std::mutex> lk(g_stg_wait_stats_mutex);
  std::vector<long> out;
  out.reserve(g_stg_wait_stats.size() * 6);
  for (const auto& item : g_stg_wait_stats) {
    const auto& row = item.second;
    out.insert(out.end(), {item.first, row.calls, row.actual_unique,
                           row.complete_at_entry, row.pending_at_entry,
                           row.wait_us});
  }
  return out;
}

void prefetch_staging_wait_stats_reset() {
  std::lock_guard<std::mutex> lk(g_stg_wait_stats_mutex);
  g_stg_wait_stats.clear();
  g_stg_wait_stats_enabled.store(true, std::memory_order_relaxed);
}

void prefetch_staging_drain() {
  std::unique_lock<std::mutex> lk(g_stg_drain_mutex);
  g_stg_drain_cv.wait(lk, [] { return g_stg_inflight == 0; });
}

class PrefetchStagingPrimitive : public mx::Primitive {
 public:
  // 方案B：expert_ids 是"预测宽集合"(top-N，按门控分降序)；resident 是目标层当前常驻专家。
  // handler 在回调里过滤掉常驻、按降序取前 cap 个缺口 pread 进 staging（cap=buffer 行数）。
  // 这样预测可以很宽(高 recall)，而 staging 内存只按 cap 预留(小，覆盖缺口分布即可)。
  PrefetchStagingPrimitive(mx::Stream s, int layer, long gen, std::string path, size_t stride,
                           std::vector<int> resident, int cap, bool parallel,
                           int source_layer, int64_t forward_id, int priority)
      : Primitive(s), layer_(layer), gen_(gen), path_(std::move(path)), stride_(stride),
        resident_(std::move(resident)), cap_(cap), parallel_(parallel),
        source_layer_(source_layer), forward_id_(forward_id), priority_(priority) {}
  const char* name() const override { return "PrefetchStagingPrimitive"; }

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0]; ids.eval();
    mx::array stg = inputs[1]; stg.eval();
    prefetch_audit_note_submit(forward_id_, source_layer_, layer_, gen_);
    staging_inflight_start(forward_id_, layer_);
    fill(ids.data<uint32_t>(), ids.size(), stg.data<uint8_t>(), layer_, gen_, path_, stride_,
         resident_, cap_, forward_id_, priority_);
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0];
    mx::array stg = inputs[1];
    const uint32_t* idp = ids.data<uint32_t>();
    uint8_t* sp = stg.data<uint8_t>();
    size_t n = ids.size();
    int layer = layer_; long gen = gen_; std::string path = path_; size_t stride = stride_;
    std::vector<int> resident = resident_; int cap = cap_; bool parallel = parallel_;
    int source_layer = source_layer_; int64_t forward_id = forward_id_;
    int priority = priority_;
    auto& enc = mx::metal::get_command_encoder(stream());
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    prefetch_audit_note_submit(forward_id, source_layer, layer, gen);
    staging_inflight_start(forward_id, layer);
    cb->addCompletedHandler(
        [ids, stg, idp, sp, n, layer, gen, path, stride, resident, cap,
         parallel, forward_id, priority](MTL::CommandBuffer*) {
          // 探针：记录本回调被 Metal 触发的时刻(= 该层 pread 能开始跑的时刻)。
          if (g_hprof_on) {
            double t = hprof_steady_now();
            std::lock_guard<std::mutex> lk(g_hprof_mutex);
            g_hprof_log.emplace_back(gen, layer, t);
          }
          // ids/stg 按值捕获保活 buffer；派后台线程时再拷一份保活到 fill 跑完。
          if (parallel) {
            bg_submit_task([ids, stg, idp, sp, n, layer, gen, path, stride,
                            resident, cap, forward_id, priority]() {
              fill(idp, n, sp, layer, gen, path, stride, resident, cap,
                   forward_id, priority);
            }, priority);
          } else {
            fill(idp, n, sp, layer, gen, path, stride, resident, cap,
                 forward_id, priority);
          }
        });
  }

  static void submit_ready(
      const mx::array& staging, std::vector<uint32_t> ids, int layer,
      long gen, std::string path, size_t stride, std::vector<int> resident,
      int cap, bool parallel, int source_layer, int64_t forward_id,
      int priority) {
    mx::array stg = mx::contiguous(staging);
    stg.eval();
    prefetch_audit_note_submit(forward_id, source_layer, layer, gen);
    staging_inflight_start(forward_id, layer);
    auto work = [stg, ids = std::move(ids), layer, gen,
                 path = std::move(path), stride,
                 resident = std::move(resident), cap, forward_id,
                 priority]() mutable {
      fill(ids.data(), ids.size(), stg.data<uint8_t>(), layer, gen, path,
           stride, resident, cap, forward_id, priority);
    };
    if (parallel)
      bg_submit_task(std::move(work), priority);
    else
      work();
  }

 private:
  static void fill(const uint32_t* idp, size_t n, uint8_t* stg,
                   int layer, long gen, const std::string& path, size_t stride,
                   const std::vector<int>& resident, int cap, int64_t forward_id,
                   int priority) {
    real_note_predictions(layer, idp, n);
    prefetch_audit_note_callback(forward_id, layer, idp, n, resident);
    prefetch_audit_note_pread(forward_id, layer, 0);
    std::unordered_set<int> res(resident.begin(), resident.end());
    // The source-time Python snapshot can be several layers old.  Recheck the
    // authoritative unified pool at callback time so staging does not reread
    // experts that demand or another prediction has already made resident.
    for (int expert : real_present_experts(layer)) res.insert(expert);
    std::unordered_set<int> done;
    std::vector<std::pair<int, int>> reserved;
    reserved.reserve(static_cast<size_t>(cap));
    {
      std::lock_guard<std::mutex> lk(g_stg_mutex);
      auto key = StagingTarget{forward_id, layer};
      auto& seen = g_stg_seen[key];
      auto& pending = g_stg_pending[key];
      int row = 0;
      for (size_t i = 0; i < n && row < cap; ++i) {
        int e = static_cast<int>(idp[i]);
        if (res.count(e) || !done.insert(e).second) continue;
        if (forward_id >= 0 && !seen.insert(e).second) continue;
        reserved.emplace_back(e, row++);
        if (forward_id >= 0) pending.insert(e);
      }
      if (priority > 0 && forward_id >= 0)
        g_stg_refinement_ready.insert(key);
      g_stg_progress_cv.notify_all();
    }
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      {
        std::lock_guard<std::mutex> lk(g_stg_mutex);
        auto key = StagingTarget{forward_id, layer};
        for (const auto& item : reserved) g_stg_pending[key].erase(item.first);
        g_stg_finished[{layer, gen}] = true;
        g_stg_progress_cv.notify_all();
      }
      prefetch_audit_note_publish(forward_id, layer, 0);
      staging_inflight_done(forward_id, layer);
      return;
    }
    size_t requested = 0;
    size_t completed = 0;
    for (const auto& item : reserved) {
      int e = item.first;
      int row = item.second;
      ++requested;
      if (::pread(fd, stg + static_cast<size_t>(row) * stride, stride,
                  static_cast<off_t>(static_cast<size_t>(e) * stride))
          == static_cast<ssize_t>(stride)) {
        ++completed;
        std::lock_guard<std::mutex> lk(g_stg_mutex);
        g_stg_ready[{layer, gen}].emplace_back(e, row);
        if (forward_id >= 0) {
          g_stg_pending[{forward_id, layer}].erase(e);
          g_stg_complete[{forward_id, layer}].insert(e);
        }
        g_stg_progress_cv.notify_all();
      } else {
        std::lock_guard<std::mutex> lk(g_stg_mutex);
        if (forward_id >= 0) g_stg_pending[{forward_id, layer}].erase(e);
        g_stg_progress_cv.notify_all();
      }
    }
    ::close(fd);
    prefetch_audit_note_pread(forward_id, layer, requested);
    prefetch_audit_note_publish(forward_id, layer, completed);
    {
      std::lock_guard<std::mutex> lk(g_stg_mutex);
      // Ensure an empty completed generation can still be consumed/released.
      if (!g_stg_ready.count({layer, gen})) g_stg_ready[{layer, gen}] = {};
      g_stg_finished[{layer, gen}] = true;
      g_stg_progress_cv.notify_all();
    }
    staging_inflight_done(forward_id, layer);
  }
  int layer_;
  long gen_;
  std::string path_;
  size_t stride_;
  std::vector<int> resident_;   // 目标层提交时刻的常驻专家快照（过滤用）
  int cap_;                     // staging buffer 行数上限
  bool parallel_;               // true: fill 派后台线程池并行；false: 回调线程同步(旧行为)
  int source_layer_;
  int64_t forward_id_;
  int priority_;
};

void prefetch_staging_ready(
    const mx::array& staging, const std::vector<int>& expert_ids, int layer,
    long gen, const std::string& path, int stride,
    const std::vector<int>& resident, int cap, bool parallel,
    int source_layer, int64_t forward_id, int priority) {
  std::vector<uint32_t> ids(expert_ids.begin(), expert_ids.end());
  PrefetchStagingPrimitive::submit_ready(
      staging, std::move(ids), layer, gen, path,
      static_cast<size_t>(stride), resident, cap, parallel, source_layer,
      forward_id, priority);
}

mx::array prefetch_into_staging(
    const mx::array& staging, const mx::array& expert_ids, int layer, long gen,
    const std::string& path, int stride, const std::vector<int>& resident, int cap,
    bool parallel, int source_layer, int64_t forward_id, int priority,
    mx::StreamOrDevice s = {}) {
  return mx::array(
      mx::Shape{1}, mx::uint8,
      std::make_shared<PrefetchStagingPrimitive>(
          mx::to_stream(s), layer, gen, path, static_cast<size_t>(stride), resident,
          cap, parallel, source_layer, forward_id, priority),
      std::vector<mx::array>{expert_ids, staging});
}

bool prefetch_staging_finished(int layer, long generation) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  auto it = g_stg_finished.find({layer, generation});
  return it != g_stg_finished.end() && it->second;
}

void prefetch_staging_forget(int layer, long generation) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  g_stg_ready.erase({layer, generation});
  g_stg_finished.erase({layer, generation});
  g_stg_consumed.erase({layer, generation});
}

// 取走某层就绪记录：[gen, e0,r0,e1,r1,...]（首元素是 generation）；空表示无就绪。
std::vector<long> prefetch_staging_take(int layer, long generation) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  std::vector<long> out;
  auto it = generation >= 0
      ? g_stg_ready.find({layer, generation})
      : g_stg_ready.lower_bound({layer, std::numeric_limits<long>::min()});
  if (it != g_stg_ready.end() && it->first.first != layer)
    it = g_stg_ready.end();
  if (it != g_stg_ready.end()) {
    out.push_back(it->first.second);
    for (auto& p : it->second) { out.push_back(p.first); out.push_back(p.second); }
    g_stg_ready.erase(it);
  }
  return out;
}

std::vector<long> prefetch_staging_take_for_demand(
    int layer, long generation) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  const auto key = std::make_pair(layer, generation);
  std::vector<long> out;
  auto it = g_stg_ready.find(key);
  if (it != g_stg_ready.end()) {
    out.push_back(generation);
    for (const auto& row : it->second) {
      out.push_back(row.first);
      out.push_back(row.second);
    }
    g_stg_ready.erase(it);
  }
  return out;
}

void prefetch_staging_mark_consumed(int layer, long generation) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  g_stg_consumed.insert({layer, generation});
}

bool prefetch_staging_consumed(int layer, long generation) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  return g_stg_consumed.count({layer, generation}) != 0;
}

// ---- handler 触发时刻探针接口 ----
void staging_hprof_enable(bool on) {
  std::lock_guard<std::mutex> lk(g_hprof_mutex);
  g_hprof_on = on;
  g_hprof_log.clear();             // 开启即清零，便于每次采集干净
}
double staging_hprof_now() { return hprof_steady_now(); }  // 与日志同一时钟，供 Python 标 eval 边界
// 扁平返回 [gen0,layer0,t0, gen1,layer1,t1, ...]（避开 nanobind tuple caster）。
std::vector<double> staging_hprof_get() {
  std::lock_guard<std::mutex> lk(g_hprof_mutex);
  std::vector<double> out;
  out.reserve(g_hprof_log.size() * 3);
  for (auto& r : g_hprof_log) {
    out.push_back(static_cast<double>(std::get<0>(r)));
    out.push_back(static_cast<double>(std::get<1>(r)));
    out.push_back(std::get<2>(r));
  }
  return out;
}
