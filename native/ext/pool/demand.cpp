// [6] Phase 2 方案B：真实区槽状态 C++ 全接管（1 次同步版）。
// 复刻 Python ResidentExpertPool 的 _slot_of/_free/_freq + _choose_victim/_alloc_slot 语义：
// - free 优先(front pop，free 初始 [0,cap))；free 空 → LFU 驱逐，受害者槽直接复用(不回 free)。
// - _choose_victim：candidates=插入序中 ∉current 者；victim=min(freq, 候选下标)，驱逐不删 freq。
// - dual 语义：真实区命中不移动插入序；仅新放入 miss 追加序尾。
#include "demand.h"
#include "side_region.h"
#include "../io/bg_reader.h"

#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>

struct RealLayer {
  std::vector<int> order;                    // 插入序(LRU tie-break)，与 e2r 同步维护
  std::unordered_map<int, int> e2r;          // expert -> slot [0,cap)
  std::vector<int> free_rows;                // 空闲槽(front pop，仿 free.pop(0))
  std::unordered_map<int, uint32_t> freq;    // LFU 频次(驱逐不删，与 Python 一致)
  int cap = 0;
  long access = 0;                           // 累计访问(decay 用)
  bool inited = false;
};
static std::mutex g_real_mutex;
static std::map<int, RealLayer> g_real;

// demand 统计：累计 + 本次(供 Python 更新 rp.hits/misses/gpu_fastpath/gpu_fallback)。
static std::mutex g_dstat_mutex;
static long g_d_last[4] = {0, 0, 0, 0};      // 本次 [hitpos, misspos, loads, fallback01]
static std::atomic<long> g_demand_ticket{1000000000};   // demand 并行 pread 用的独立 ticket 段

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
  if (c.inited) return;
  c.cap = cap;
  c.free_rows.clear();
  for (int r = 0; r < cap; ++r) c.free_rows.push_back(r);   // free 初始 [0,cap)
  c.inited = true;
}

void real_init(int layer, int cap) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  real_ensure_locked(g_real[layer], cap);
}

std::vector<int> real_region_contents(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  std::vector<int> out;
  auto it = g_real.find(layer);
  if (it != g_real.end())
    for (auto& p : it->second.e2r) { out.push_back(p.first); out.push_back(p.second); }
  return out;
}

int real_region_count(int layer) {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  auto it = g_real.find(layer);
  return it == g_real.end() ? 0 : static_cast<int>(it->second.e2r.size());
}

void real_reset() {
  std::lock_guard<std::mutex> lk(g_real_mutex);
  g_real.clear();
}

// 复刻 _choose_victim：遍历插入序(order)选 ∉current 且 freq 最小者，并列取最早(候选下标最小)。
// 返回 expert id；-1 表示无可驱逐。调用方须持 g_real_mutex。
static int choose_victim_locked(RealLayer& c, const std::unordered_set<int>& current) {
  int victim = -1;
  uint32_t best = 0;
  for (int e : c.order) {
    if (!c.e2r.count(e) || current.count(e)) continue;
    uint32_t f = c.freq.count(e) ? c.freq[e] : 0;
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
    c.e2r.erase(victim);
    for (auto oit = c.order.begin(); oit != c.order.end(); ++oit)
      if (*oit == victim) { c.order.erase(oit); break; }
  }
  c.e2r[e] = slot;
  c.order.push_back(e);
  return slot;
}

// LFU 频次 bump（canonical：本次全部唯一专家各 +1）+ decay。调用方须持 g_real_mutex。
static void note_access_locked(RealLayer& c, const std::vector<int>& uniq_access,
                               bool lfu, int decay_interval) {
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

// 核心状态机（纯 CPU、无 I/O）：给定 host inds(ip[n]) + 侧区快照(side)，算 local、分配 miss 槽，
// 把需要落盘的新放入 (expert, slot) 追加到 placements（字节由调用方在锁外并行 pread 落池）。
// 返回 local(int32 vector)。stats: [hitpos, misspos, loads(=placements 数)]。
static std::vector<int32_t> demand_core_locked(
    RealLayer& c, const uint32_t* ip, size_t n, const std::unordered_map<int, int>& side,
    bool lfu, int decay_interval, std::vector<std::pair<int, int>>& placements, long stats[3]) {
  std::vector<int32_t> local(n, -1);
  std::vector<int> uniq_miss, access_order;
  std::unordered_set<int> miss_seen, access_seen;
  int hitpos = 0;
  // pass1：算命中(侧区覆盖真实区)、收集 miss(首见序)、收集唯一访问(freq)。
  for (size_t i = 0; i < n; ++i) {
    int e = static_cast<int>(ip[i]);
    if (access_seen.insert(e).second) access_order.push_back(e);
    auto sit = side.find(e);
    if (sit != side.end()) { local[i] = sit->second; ++hitpos; continue; }
    auto rit = c.e2r.find(e);
    if (rit != c.e2r.end()) { local[i] = rit->second; ++hitpos; continue; }
    if (miss_seen.insert(e).second) uniq_miss.push_back(e);
  }
  note_access_locked(c, access_order, lfu, decay_interval);
  // pass2：miss 分配槽（不落盘）。current = 本前向全部唯一路由专家(命中+miss)：绝不驱逐本前向要读的
  // 任何专家的槽（否则真实区命中专家的槽被 miss 复写 → 脏字节。比 Python 仅护 miss 更严格、更正确）。
  std::unordered_map<int, int> new_slot;
  for (int e : uniq_miss) {
    int slot = alloc_slot_locked(c, e, access_seen);
    if (slot < 0) { new_slot[e] = 0; continue; }               // 超容量兜底
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

// demand 全接管：inds 惰性(内部 eval 一次=1 次同步)；side_gen 指定侧区代；pool_list 为 _segs 顺序
// 的 per-key 池数组(已 eval、指针稳定)。返回 local(int32, inds.shape)。
mx::array demand_dual(
    const mx::array& inds, const std::vector<mx::array>& pool_list,
    const std::vector<int>& seg_nbytes, int layer, int side_gen, const std::string& path,
    int stride, int cap, bool lfu, int decay_interval, mx::StreamOrDevice s = {}) {
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
  std::unordered_map<int, int> side = sideregion_snapshot(layer, side_gen);   // 侧区快照(该代)
  double ta = g_dt_on ? dt_now_us() : 0;
  long stats[3];
  std::vector<int32_t> local;
  std::vector<std::pair<int, int>> placements;   // (expert, slot)：锁外并行落盘
  double tb;
  {
    std::lock_guard<std::mutex> lk(g_real_mutex);
    tb = g_dt_on ? dt_now_us() : 0;
    RealLayer& c = g_real[layer];
    real_ensure_locked(c, cap);
    local = demand_core_locked(c, ip, n, side, lfu, decay_interval, placements, stats);
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
    std::vector<long> tickets;
    tickets.reserve(placements.size());
    for (auto& pr : placements) {
      long tk = g_demand_ticket.fetch_add(1);
      bg_pread_into_pool(pool_list, seg_off, seg_nb, pr.second, pr.first, path,
                         static_cast<long>(stride), tk, /*prio=*/1,
                         /*nocache=*/false);          // demand route 读走 cache：段偏移非页对齐
      tickets.push_back(tk);
    }
    for (long tk : tickets) bg_reader_wait(tk);   // 并行 pread 完成 → 池槽字节就绪
  }
  double t3 = g_dt_on ? dt_now_us() : 0;
  {
    std::lock_guard<std::mutex> lk(g_dstat_mutex);
    g_d_last[0] = stats[0]; g_d_last[1] = stats[1]; g_d_last[2] = stats[2];
    g_d_last[3] = (stats[1] == 0) ? 0 : 1;
  }
  mx::array out = mx::array(local.data(), ids.shape(), mx::int32);
  if (g_dt_on) {
    double t4 = dt_now_us();
    g_dt[0] += t1 - t0; g_dt[1] += t2 - t1; g_dt[2] += ta - t2;
    g_dt[3] += tb - ta; g_dt[4] += t3 - tb; g_dt[5] += t4 - t3;
  }
  return out;
}

// 取本次 demand 统计 [hitpos, misspos, loads, fallback01]（主线程串行，安全）。
std::vector<long> demand_last_stats() {
  std::lock_guard<std::mutex> lk(g_dstat_mutex);
  return {g_d_last[0], g_d_last[1], g_d_last[2], g_d_last[3]};
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
  auto local = demand_core_locked(c, u.data(), u.size(), empty_side, lfu, decay_interval,
                                  placements, stats);
  return std::vector<int>(local.begin(), local.end());
}
