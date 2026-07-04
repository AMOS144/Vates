// [4] 段散写持久侧区缓存（zero-copy dual-source 默认路径）。
// 与持久 staging 缓存同构，但目标是“多个结构化 per-key 池数组”：每个池数组形如
// (cap+spec, ...)，行 [base_row, base_row+spec) 为侧区。命中缺口时 pread blob 整行，
// 再把该行内按固定顺序拼接的各段 memcpy 进对应 per-key 数组的同一物理行。
#include "side_region.h"
#include "../io/blob_io.h"
#include "../io/bg_reader.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <unordered_set>

struct SideLayer {
  std::map<int, int> e2r;          // expert -> 物理侧区行 [base_row, base_row+spec)
  std::vector<int> free_rows;
  std::map<int, uint32_t> freq;    // expert -> 预测频次(LFU 分数;仅 SIDEREGION_LFU 用)
  bool inited = false;
  int base = 0;                    // 侧区起始物理行 base_row
  int spec = 0;                    // 侧区行数 spec_slots
};
static std::mutex g_side_mutex;
static std::map<std::pair<int, int>, SideLayer> g_side;   // 键 (layer, gen)：双缓冲两代独立

// 侧区 fill 在途计数：eval_gpu 提交预取时 +1，后台 read_publish 写完字节后 -1。
// 消费方在前向开头 sideregion_drain() 排空上一前向的 fill，保证被消费的侧区行字节
// 已完全写好（含 GPU 完成回调滞后的情形）→ 消灭「GPU 消费 kernel 读到半写侧区行」竞态。
static std::atomic<long> g_side_inflight{0};
static std::mutex g_side_drain_mutex;
static std::condition_variable g_side_drain_cv;
static inline void side_inflight_done() {
  {
    std::lock_guard<std::mutex> lk(g_side_drain_mutex);
    g_side_inflight.fetch_sub(1);
  }
  g_side_drain_cv.notify_all();
}
void sideregion_drain() {
  std::unique_lock<std::mutex> lk(g_side_drain_mutex);
  g_side_drain_cv.wait(lk, [] { return g_side_inflight.load() == 0; });
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

class PrefetchPoolSideRegionPrimitive : public mx::Primitive {
 public:
  PrefetchPoolSideRegionPrimitive(mx::Stream s, std::vector<int> seg_nbytes, int layer, int gen,
                                  std::string path, size_t stride, std::vector<int> resident,
                                  int spec_slots, int base_row)
      : Primitive(s), seg_(std::move(seg_nbytes)), layer_(layer), gen_(gen), path_(std::move(path)),
        stride_(stride), resident_(std::move(resident)), spec_(spec_slots), base_(base_row) {}
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
    auto& enc = mx::metal::get_command_encoder(stream());
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    // 提交即计在途：在 eval_gpu（预取提交）时 +1，直到后台字节写完才 -1。这样消费方前向开头
    // sideregion_drain() 能等到「即使 GPU 完成回调尚未触发」的 fill，闭合跨前向的写-读竞态。
    g_side_inflight.fetch_add(1);
    // in 按值捕获 → 保活 expert_ids 与所有池数组 buffer；idp/ptrs 在回调里指针有效。
    cb->addCompletedHandler(
        [in, ptrs, seg, idp, n, layer, gen, path, stride, resident, spec, base](MTL::CommandBuffer*) {
          // 阶段1（回调线程、持锁极短）：读惰性 id（此刻已算完）、预留侧区行。
          // 必须在回调里——id 只有 command buffer 完成后才有效。
          auto to_read = reserve(idp, n, layer, gen, resident, spec, base);
          if (to_read.empty()) {
            side_inflight_done();               // 无缺口可读：立即消账，避免 drain 空等
            return;
          }
          // 诊断门控 SIDEREGION_SYNC=1：回调内同步 pread+memcpy+publish（不派 bg），
          // 消除「异步 bg fill 与下一前向 gather 竞态」这一变量，用于 systematic-debugging 取证。
          static const bool kSync = []() {
            const char* e = std::getenv("SIDEREGION_SYNC");
            return e && e[0] == '1';
          }();
          if (kSync) {
            read_publish(ptrs, seg, to_read, path, stride, layer, gen);
            side_inflight_done();
            return;
          }
          // 阶段2+3 派给自由后台线程：~40MB pread + memcpy + 发布 e2r 脱离 Metal 回调线程，
          // 与主 stream 后续层计算、多层预取互相并发 → 真正恢复 I/O/计算重叠。
          // 闭包再次按值捕获 in：保活池/ids buffer 到后台读完（ptrs 指向其内存）。
          bg_submit_task([in, ptrs, seg, to_read, path, stride, layer, gen]() {
            read_publish(ptrs, seg, to_read, path, stride, layer, gen);
            side_inflight_done();               // 字节写完才消账 → drain 保证被消费行已就绪
          });
        });
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
    auto to_read = reserve(ids.data<uint32_t>(), ids.size(), layer_, gen_, resident_, spec_, base_);
    read_publish(ptrs, seg_, to_read, path_, stride_, layer_, gen_);
  }

  // 阶段1：过滤常驻/去重 → 淘汰 ∉P 的旧行 → 为缺口预留物理行（出 free，暂不入 e2r，
  // 避免消费者在字节写好前看到 e2r 命中而 gather 到脏字节）。返回 (expert, 预留行)。
  static std::vector<std::pair<int, int>> reserve(
      const uint32_t* idp, size_t n, int layer, int gen, const std::vector<int>& resident,
      int spec, int base) {
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
    std::vector<std::pair<int, int>> to_read;
    std::lock_guard<std::mutex> lk(g_side_mutex);
    SideLayer& c = g_side[{layer, gen}];
    if (!c.inited) {
      for (int r = 0; r < spec; ++r) c.free_rows.push_back(base + r);
      c.base = base;
      c.spec = spec;
      c.inited = true;
    }
    if (!lfu) {
      // 旧行为:∉P 全弃(一次性预取批)。
      for (auto it = c.e2r.begin(); it != c.e2r.end();) {
        if (!Pset.count(it->first)) {
          c.free_rows.push_back(it->second);
          it = c.e2r.erase(it);
        } else {
          ++it;
        }
      }
      for (int e : P) {
        if (c.e2r.count(e) || c.free_rows.empty()) continue;
        to_read.emplace_back(e, c.free_rows.back());
        c.free_rows.pop_back();
      }
      return to_read;
    }
    // LFU 持久:∉P 不清;再预测命中已驻专家 freq+1(越常预测越热)。
    for (int e : P) {
      if (c.e2r.count(e)) c.freq[e] += 1;
    }
    for (int e : P) {
      if (c.e2r.count(e)) continue;               // 已驻,跳过(不重读)
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
          uint32_t f = c.freq.count(kv.first) ? c.freq[kv.first] : 0;
          if (victim < 0 || f < best || (f == best && kv.first < victim)) {
            victim = kv.first;
            best = f;
          }
        }
        if (victim < 0) continue;                 // 全是 P 热,无可淘 → 本步不读
        row = c.e2r[victim];
        c.e2r.erase(victim);
        c.freq.erase(victim);
        if (side_trace_hit(layer, row))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d EVICT_REUSE row=%d victim=%d newExpert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, victim, e);
      }
      to_read.emplace_back(e, row);
    }
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
    std::unordered_set<int> free_set(c.free_rows.begin(), c.free_rows.end());
    for (auto& p : c.e2r)
      if (free_set.count(p.second))
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d FREE_E2R_OVERLAP row=%d owned_by=%d\n",
                where, layer, gen, p.second, p.first);
    if (c.free_rows.size() != free_set.size())
      fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d FREE_DUP free_n=%zu uniq=%zu\n",
              where, layer, gen, c.free_rows.size(), free_set.size());
    for (auto& pr : to_read)
      if (row_owner.count(pr.second))
        fprintf(stderr, "[SIDE_AUDIT] %s L%d gen%d TOREAD_LIVE_ROW row=%d assigned_to=%d still_owned_by=%d\n",
                where, layer, gen, pr.second, pr.first, row_owner[pr.second]);
  }

  // 阶段2+3：pread blob 整行 + 各段 memcpy 直写进对应 per-key 池数组的物理侧区行（不持锁），
  // 写完后持锁发布 e2r。ptrs[i] 为第 i 个池 key 数组的 buffer 指针（C++ 拥有、地址恒定不迁移，
  // 由 Route 3 底座保证——见 pool_owned_zeros），故后台异步直写安全、消费侧读同一 buffer。
  // 在后台线程跑：与计算并发，且不阻塞消费侧的 sideregion_kv。
  static void read_publish(const std::vector<uint8_t*>& ptrs, const std::vector<int>& seg,
                           const std::vector<std::pair<int, int>>& to_read,
                           const std::string& path, size_t stride, int layer, int gen) {
    // 段在 blob 记录内的偏移（段顺序 = seg 顺序 = ptrs/池 key 顺序）。
    std::vector<size_t> seg_off(seg.size(), 0);
    size_t acc = 0;
    for (size_t i = 0; i < seg.size(); ++i) { seg_off[i] = acc; acc += static_cast<size_t>(seg[i]); }
    int fd = open_blob_nocache(path.c_str());
    if (fd < 0) {                                             // 失败则把预留行还回 free
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : to_read) c.free_rows.push_back(pr.second);
      return;
    }
    std::vector<uint8_t> rec(stride);                          // 整条 blob 记录临时缓冲
    std::vector<std::pair<int, int>> done;
    for (auto& pr : to_read) {
      int e = pr.first, row = pr.second;
      if (::pread(fd, rec.data(), stride, static_cast<off_t>(static_cast<size_t>(e) * stride)) !=
          static_cast<ssize_t>(stride)) {
        std::lock_guard<std::mutex> lk(g_side_mutex);         // 读失败：行还回 free
        g_side[{layer, gen}].free_rows.push_back(row);
        continue;
      }
      // 各段 memcpy 进对应 per-key 池数组的物理行 row（(n_slots,*shape) 布局 → 行偏移 = row*seg[i]）。
      for (size_t i = 0; i < seg.size(); ++i)
        std::memcpy(ptrs[i] + static_cast<size_t>(row) * static_cast<size_t>(seg[i]),
                    rec.data() + seg_off[i], static_cast<size_t>(seg[i]));
      if (side_trace_hit(layer, row))
        fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d WRITEPOOL row=%d expert=%d\n",
                (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, row, e);
      done.emplace_back(e, row);
    }
    ::close(fd);
    {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : done) {                                  // 字节就绪后才发布 e2r
        c.e2r[pr.first] = pr.second;
        if (!c.freq.count(pr.first)) c.freq[pr.first] = 1;     // 新专家初始 freq
        if (side_trace_hit(layer, pr.second))
          fprintf(stderr, "[SIDE_TRACE ev=%llu tid=%u] L%d gen%d PUBLISH row=%d expert=%d\n",
                  (unsigned long long)g_side_ev.fetch_add(1), side_tid(), layer, gen, pr.second,
                  pr.first);
      }
      if (std::getenv("SIDE_AUDIT")) side_audit(c, layer, gen, "publish", {});
    }
  }
  std::vector<int> seg_;
  int layer_;
  int gen_;
  std::string path_;
  size_t stride_;
  std::vector<int> resident_;
  int spec_;
  int base_;
};

mx::array prefetch_pool_sideregion(
    const std::vector<mx::array>& pool_list, const std::vector<int>& seg_nbytes,
    const mx::array& expert_ids, int layer, const std::string& path, int stride,
    const std::vector<int>& resident, int spec_slots, int base_row, int gen,
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
  return mx::array(
      mx::Shape{1}, mx::uint8,
      std::make_shared<PrefetchPoolSideRegionPrimitive>(
          mx::to_stream(s), seg_nbytes, layer, gen, path, static_cast<size_t>(stride), resident,
          spec_slots, base_row),
      inputs);
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

void sideregion_reset() {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  g_side.clear();
}

std::unordered_map<int, int> sideregion_snapshot(int layer, int gen) {
  std::unordered_map<int, int> side;
  std::lock_guard<std::mutex> lk(g_side_mutex);
  auto it = g_side.find({layer, gen});
  if (it != g_side.end()) for (auto& p : it->second.e2r) side[p.first] = p.second;
  return side;
}
