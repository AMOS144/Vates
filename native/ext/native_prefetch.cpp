#include "native_prefetch.h"

#include <unordered_set>
#include <thread>
#include <queue>
#include <condition_variable>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <tuple>
#include <map>
#include <utility>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

#ifndef F_NOCACHE
#define F_NOCACHE 48          // macOS：提示内核读过的页不留 page cache
#endif

// 打开 blob 并设 F_NOCACHE：与 demand 侧 blob_loader 对齐，避免预取读把 page cache 灌满 →
// 在内存受限机上累积压力触发"双稳态"慢挡翻转（实测 zerocopy 每轮翻慢挡的根因）。
static inline int open_blob_nocache(const char* path) {
  int fd = ::open(path, O_RDONLY);
  if (fd >= 0) ::fcntl(fd, F_NOCACHE, 1);
  return fd;
}

// 自由后台读线程池的通用任务入口（定义在文件末尾的匿名命名空间里）。
// 侧区预取的 GPU 完成回调用它把 pread 派到后台线程，脱离 Metal 回调线程 → 真正与计算重叠。
namespace { void bg_submit_task(std::function<void()> fn); }

// 把一组专家的 blob 字节直接 pread 进 MLX 自有 buffer（无 kernel、无额外拷贝）。
// load 作为惰性图节点：在批量 eval 中执行，避免 Python 侧 per-expert mx.eval 同步。
class BlobLoadPrimitive : public mx::Primitive {
 public:
  BlobLoadPrimitive(mx::Stream stream, std::string path, size_t stride, std::vector<int> experts)
      : Primitive(stream), path_(std::move(path)), stride_(stride), experts_(std::move(experts)) {}
  const char* name() const override { return "BlobLoadPrimitive"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>& outputs) override {
    load(outputs[0]);
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>& outputs) override {
    load(outputs[0]);
  }

 private:
  void load(mx::array& out) {
    out.set_data(mx::allocator::malloc(out.nbytes()));
    uint8_t* dst = out.data<uint8_t>();
    int fd = ::open(path_.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("blob open failed: " + path_);
    for (size_t i = 0; i < experts_.size(); ++i) {
      size_t off = static_cast<size_t>(experts_[i]) * stride_;
      ssize_t n = ::pread(fd, dst + i * stride_, stride_, static_cast<off_t>(off));
      if (n != static_cast<ssize_t>(stride_)) {
        ::close(fd);
        throw std::runtime_error("blob pread short read");
      }
    }
    ::close(fd);
  }
  std::string path_;
  size_t stride_;
  std::vector<int> experts_;
};

mx::array blob_load(
    const std::string& path,
    const mx::array& expert_ids,
    int stride,
    mx::StreamOrDevice s = {}) {
  auto ids = expert_ids;
  ids.eval();
  std::vector<int> ev(ids.size());
  const uint32_t* p = ids.data<uint32_t>();
  for (size_t i = 0; i < ids.size(); ++i) ev[i] = static_cast<int>(p[i]);
  int n = static_cast<int>(ev.size());
  return mx::array(
      mx::Shape{n, stride},
      mx::uint8,
      std::make_shared<BlobLoadPrimitive>(mx::to_stream(s), path, static_cast<size_t>(stride), std::move(ev)),
      std::vector<mx::array>{});
}

// ---- native-fused-prefetch (de-risk): GPU 完成回调里读 id + 派发 pread ----
// 命门验证:回调在 command buffer 完成后触发,此时 inds 已算完,读到的是正确值。
static std::mutex g_pf_mutex;
static std::vector<int> g_pf_last_ids;
static std::atomic<int> g_pf_fires{0};

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
    std::vector<int> seen;
    seen.reserve(n);
    int fd = (do_read && !path.empty()) ? ::open(path.c_str(), O_RDONLY) : -1;
    static thread_local std::vector<uint8_t> buf;
    if (do_read && buf.size() < stride) buf.resize(stride);
    for (size_t i = 0; i < n; ++i) {
      int e = static_cast<int>(p[i]);
      seen.push_back(e);
      if (fd >= 0) ::pread(fd, buf.data(), stride, static_cast<off_t>(static_cast<size_t>(e) * stride));
    }
    if (fd >= 0) ::close(fd);
    std::lock_guard<std::mutex> lk(g_pf_mutex);
    g_pf_last_ids = std::move(seen);
    g_pf_fires.fetch_add(1);
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

std::vector<int> prefetch_on_complete_last_ids() {
  std::lock_guard<std::mutex> lk(g_pf_mutex);
  return g_pf_last_ids;
}

int prefetch_on_complete_fires() { return g_pf_fires.load(); }

// de-risk: 完成回调把专家字节 pread 进调用方预分配的 MLX buffer(零拷贝物化候选)。
// 验证:主线程后续读这块 buffer 能拿到正确字节、quantized_matmul 不崩。
class PrefetchIntoPrimitive : public mx::Primitive {
 public:
  PrefetchIntoPrimitive(mx::Stream s, std::string path, size_t stride)
      : Primitive(s), path_(std::move(path)), stride_(stride) {}
  const char* name() const override { return "PrefetchIntoPrimitive"; }

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0];
    ids.eval();
    mx::array dst = inputs[1];
    dst.eval();
    pread_into(ids.data<uint32_t>(), ids.size(), dst.data<uint8_t>(), path_, stride_);
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0];
    mx::array dst = inputs[1];
    const uint32_t* idp = ids.data<uint32_t>();
    uint8_t* dstp = dst.data<uint8_t>();   // 预分配 buffer 的指针(统一内存,CPU 可写)
    size_t n = ids.size();
    std::string path = path_;
    size_t stride = stride_;
    auto& enc = mx::metal::get_command_encoder(stream());
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    cb->addCompletedHandler([ids, dst, idp, dstp, n, path, stride](MTL::CommandBuffer*) {
      pread_into(idp, n, dstp, path, stride);
    });
  }

 private:
  static void pread_into(const uint32_t* idp, size_t n, uint8_t* dst,
                         const std::string& path, size_t stride) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    for (size_t i = 0; i < n; ++i) {
      size_t off = static_cast<size_t>(idp[i]) * stride;
      ::pread(fd, dst + i * stride, stride, static_cast<off_t>(off));
    }
    ::close(fd);
    {
      std::lock_guard<std::mutex> lk(g_pf_mutex);
      g_pf_fires.fetch_add(1);
    }
  }
  std::string path_;
  size_t stride_;
};

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

// ---- staging 版 miss→hit：handler pread 进 per-layer staging + 记录 (expert→row) ----
static std::mutex g_stg_mutex;
// layer -> (gen, [(expert, row)])；handler 原子写 gen+映射，主线程按 gen 匹配 buffer 后 take。
static std::map<int, std::pair<long, std::vector<std::pair<int, int>>>> g_stg_ready;

class PrefetchStagingPrimitive : public mx::Primitive {
 public:
  // 方案B：expert_ids 是"预测宽集合"(top-N，按门控分降序)；resident 是目标层当前常驻专家。
  // handler 在回调里过滤掉常驻、按降序取前 cap 个缺口 pread 进 staging（cap=buffer 行数）。
  // 这样预测可以很宽(高 recall)，而 staging 内存只按 cap 预留(小，覆盖缺口分布即可)。
  PrefetchStagingPrimitive(mx::Stream s, int layer, long gen, std::string path, size_t stride,
                           std::vector<int> resident, int cap, bool parallel)
      : Primitive(s), layer_(layer), gen_(gen), path_(std::move(path)), stride_(stride),
        resident_(std::move(resident)), cap_(cap), parallel_(parallel) {}
  const char* name() const override { return "PrefetchStagingPrimitive"; }

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0]; ids.eval();
    mx::array stg = inputs[1]; stg.eval();
    fill(ids.data<uint32_t>(), ids.size(), stg.data<uint8_t>(), layer_, gen_, path_, stride_,
         resident_, cap_);
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
    auto& enc = mx::metal::get_command_encoder(stream());
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    cb->addCompletedHandler(
        [ids, stg, idp, sp, n, layer, gen, path, stride, resident, cap, parallel](MTL::CommandBuffer*) {
          // 探针：记录本回调被 Metal 触发的时刻(= 该层 pread 能开始跑的时刻)。
          if (g_hprof_on) {
            double t = hprof_steady_now();
            std::lock_guard<std::mutex> lk(g_hprof_mutex);
            g_hprof_log.emplace_back(gen, layer, t);
          }
          // ids/stg 按值捕获保活 buffer；派后台线程时再拷一份保活到 fill 跑完。
          if (parallel) {
            bg_submit_task([ids, stg, idp, sp, n, layer, gen, path, stride, resident, cap]() {
              fill(idp, n, sp, layer, gen, path, stride, resident, cap);
            });
          } else {
            fill(idp, n, sp, layer, gen, path, stride, resident, cap);
          }
        });
  }

 private:
  static void fill(const uint32_t* idp, size_t n, uint8_t* stg,
                   int layer, long gen, const std::string& path, size_t stride,
                   const std::vector<int>& resident, int cap) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    std::unordered_set<int> res(resident.begin(), resident.end());
    std::unordered_set<int> done;                 // 去重：同一缺口只 pread 一次
    std::vector<std::pair<int, int>> ready;
    ready.reserve(static_cast<size_t>(cap));
    int row = 0;                                  // 写入的 staging 行（≤ cap）
    for (size_t i = 0; i < n && row < cap; ++i) {
      int e = static_cast<int>(idp[i]);
      if (res.count(e) || !done.insert(e).second) continue;  // 已常驻/已取过 → 跳过
      if (::pread(fd, stg + static_cast<size_t>(row) * stride, stride,
                  static_cast<off_t>(static_cast<size_t>(e) * stride))
          == static_cast<ssize_t>(stride)) {
        ready.emplace_back(e, row);
        ++row;
      }
    }
    ::close(fd);
    std::lock_guard<std::mutex> lk(g_stg_mutex);
    g_stg_ready[layer] = {gen, std::move(ready)};   // 原子：gen 与映射一起写
    g_pf_fires.fetch_add(1);
  }
  int layer_;
  long gen_;
  std::string path_;
  size_t stride_;
  std::vector<int> resident_;   // 目标层提交时刻的常驻专家快照（过滤用）
  int cap_;                     // staging buffer 行数上限
  bool parallel_;               // true: fill 派后台线程池并行；false: 回调线程同步(旧行为)
};

mx::array prefetch_into_staging(
    const mx::array& staging, const mx::array& expert_ids, int layer, long gen,
    const std::string& path, int stride, const std::vector<int>& resident, int cap,
    bool parallel, mx::StreamOrDevice s = {}) {
  return mx::array(
      mx::Shape{1}, mx::uint8,
      std::make_shared<PrefetchStagingPrimitive>(
          mx::to_stream(s), layer, gen, path, static_cast<size_t>(stride), resident, cap, parallel),
      std::vector<mx::array>{expert_ids, staging});
}

// 取走某层就绪记录：[gen, e0,r0,e1,r1,...]（首元素是 generation）；空表示无就绪。
std::vector<long> prefetch_staging_take(int layer) {
  std::lock_guard<std::mutex> lk(g_stg_mutex);
  std::vector<long> out;
  auto it = g_stg_ready.find(layer);
  if (it != g_stg_ready.end()) {
    out.push_back(it->second.first);                 // gen
    for (auto& p : it->second.second) { out.push_back(p.first); out.push_back(p.second); }
    g_stg_ready.erase(it);
  }
  return out;
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

// ---- 段散写持久侧区缓存 ----
// 与持久 staging 缓存同构，但目标是“多个结构化 per-key 池数组”：每个池数组形如
// (cap+spec, ...)，行 [base_row, base_row+spec) 为侧区。命中缺口时 pread blob 整行，
// 再把该行内按固定顺序拼接的各段 memcpy 进对应 per-key 数组的同一物理行。
struct SideLayer {
  std::map<int, int> e2r;          // expert -> 物理侧区行 [base_row, base_row+spec)
  std::vector<int> free_rows;
  bool inited = false;
};
static std::mutex g_side_mutex;
static std::map<std::pair<int, int>, SideLayer> g_side;   // 键 (layer, gen)：双缓冲两代独立

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
    // in 按值捕获 → 保活 expert_ids 与所有池数组 buffer；idp/ptrs 在回调里指针有效。
    cb->addCompletedHandler(
        [in, ptrs, seg, idp, n, layer, gen, path, stride, resident, spec, base](MTL::CommandBuffer*) {
          // 阶段1（回调线程、持锁极短）：读惰性 id（此刻已算完）、预留侧区行。
          // 必须在回调里——id 只有 command buffer 完成后才有效。
          auto to_read = reserve(idp, n, layer, gen, resident, spec, base);
          if (to_read.empty()) return;
          // 诊断门控 SIDEREGION_SYNC=1：回调内同步 pread+memcpy+publish（不派 bg），
          // 消除「异步 bg fill 与下一前向 gather 竞态」这一变量，用于 systematic-debugging 取证。
          static const bool kSync = []() {
            const char* e = std::getenv("SIDEREGION_SYNC");
            return e && e[0] == '1';
          }();
          if (kSync) {
            read_publish(ptrs, seg, to_read, path, stride, layer, gen);
            return;
          }
          // 阶段2+3 派给自由后台线程：~40MB pread + memcpy + 发布 e2r 脱离 Metal 回调线程，
          // 与主 stream 后续层计算、多层预取互相并发 → 真正恢复 I/O/计算重叠。
          // 闭包再次按值捕获 in：保活池/ids buffer 到后台读完（ptrs 指向其内存）。
          bg_submit_task([in, ptrs, seg, to_read, path, stride, layer, gen]() {
            read_publish(ptrs, seg, to_read, path, stride, layer, gen);
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
      c.inited = true;
    }
    for (auto it = c.e2r.begin(); it != c.e2r.end();) {       // 淘汰 ∉P 的行
      if (!Pset.count(it->first)) {
        c.free_rows.push_back(it->second);
        it = c.e2r.erase(it);
      } else {
        ++it;
      }
    }
    for (int e : P) {                                         // 只为缺口预留行
      if (c.e2r.count(e) || c.free_rows.empty()) continue;
      to_read.emplace_back(e, c.free_rows.back());
      c.free_rows.pop_back();
    }
    return to_read;
  }

  // 阶段2+3：pread + memcpy 进已预留的独占行（不持锁），写完后持锁发布 e2r。
  // 在后台线程跑：与计算并发，且不阻塞消费侧的 sideregion_contents。
  static void read_publish(const std::vector<uint8_t*>& ptrs, const std::vector<int>& seg,
                           const std::vector<std::pair<int, int>>& to_read,
                           const std::string& path, size_t stride, int layer, int gen) {
    int fd = open_blob_nocache(path.c_str());
    if (fd < 0) {                                             // 失败则把预留行还回 free
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : to_read) c.free_rows.push_back(pr.second);
      return;
    }
    std::vector<uint8_t> tmp(stride);
    std::vector<std::pair<int, int>> done;
    for (auto& pr : to_read) {
      int e = pr.first, row = pr.second;
      if (::pread(fd, tmp.data(), stride, static_cast<off_t>(static_cast<size_t>(e) * stride)) !=
          static_cast<ssize_t>(stride)) {
        std::lock_guard<std::mutex> lk(g_side_mutex);         // 读失败：行还回 free
        g_side[{layer, gen}].free_rows.push_back(row);
        continue;
      }
      size_t off = 0;
      for (size_t k = 0; k < seg.size(); ++k) {
        std::memcpy(ptrs[k] + static_cast<size_t>(row) * seg[k], tmp.data() + off, seg[k]);
        off += static_cast<size_t>(seg[k]);
      }
      done.emplace_back(e, row);
    }
    ::close(fd);
    {
      std::lock_guard<std::mutex> lk(g_side_mutex);
      SideLayer& c = g_side[{layer, gen}];
      for (auto& pr : done) c.e2r[pr.first] = pr.second;       // 字节就绪后才发布 e2r
    }
    g_pf_fires.fetch_add(1);
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

void sideregion_reset() {
  std::lock_guard<std::mutex> lk(g_side_mutex);
  g_side.clear();
}

// ====== Task1 spike：验证「输出别名输入 buffer + eval_gpu 回调里 memcpy 填值」机制 ======
// 单输入 src(uint32 [N])；eval 把 src 的 buffer 原地写成常量 fill、并把输入别名为输出。
// 下游 take(out) 必须确定性读到 fill —— 证明「图内 fill primitive 输出 → gather 依赖边」成立。
class MaterializeSpikePrimitive : public mx::Primitive {
 public:
  MaterializeSpikePrimitive(mx::Stream s, uint32_t fillval) : Primitive(s), fill_(fillval) {}
  const char* name() const override { return "MaterializeSpikePrimitive"; }
  void eval_cpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    out[0].set_data(mx::allocator::malloc(out[0].nbytes()));
    uint32_t* p = out[0].data<uint32_t>();
    for (size_t i = 0; i < out[0].size(); ++i) p[i] = fill_;
  }
  void eval_gpu(const std::vector<mx::array>& in, std::vector<mx::array>& out) override {
    // 别名零拷贝：out 共享 in[0] buffer，再原地只写「偶数行」（模拟只填侧区行、保留其它行）。
    out[0].copy_shared_buffer(in[0]);
    uint32_t* p = out[0].data<uint32_t>();
    for (size_t i = 0; i < out[0].size(); i += 2) p[i] = fill_;
  }
 private:
  uint32_t fill_;
};

mx::array materialize_spike(const mx::array& src, uint32_t fillval, mx::StreamOrDevice s = {}) {
  return mx::array(src.shape(), src.dtype(),
                   std::make_shared<MaterializeSpikePrimitive>(mx::to_stream(s), fillval),
                   std::vector<mx::array>{src});
}

// dst: 预分配 uint8 [n, stride] MLX 数组(调用方已 eval)；回调把 n 个专家 pread 进它。
// 返回 dummy(折进图触发)。
mx::array prefetch_into(
    const mx::array& dst,
    const mx::array& expert_ids,
    const std::string& path,
    int stride,
    mx::StreamOrDevice s = {}) {
  return mx::array(
      mx::Shape{1},
      mx::uint8,
      std::make_shared<PrefetchIntoPrimitive>(mx::to_stream(s), path, static_cast<size_t>(stride)),
      std::vector<mx::array>{expert_ids, dst});
}

// ====== 自由后台读线程（de-risk）======
// 线程全程只碰 dst 原始指针 / 路径 / 整数，绝不接触 Python 对象 → 无需 GIL。
namespace {
struct ReadOp { uint8_t* dst; size_t nbytes; off_t file_off; };
struct BgJob {
  std::vector<ReadOp> ops;
  std::string path;
  long ticket;
  std::vector<mx::array> keep;     // 持 buffer 引用，保证 dst 指针在读期间存活
  std::function<void()> task;      // 若设置，worker 直接执行它（侧区异步读用），不走 ops/ticket
  int prio = 0;                    // >0=高优(route 读)，=0=低优(投机兜底)
};

// 双队列 + 低优并发上限：高优(route)读永远优先取、且独占大多数 worker；
// 低优(投机)读最多占 low_cap_ 个 worker → 给 route 读留出 worker 和 SSD 带宽，
// 实现"高分抢带宽先就绪、低分延后不阻塞关键路径"的真 IO 优先级。
class BgReader {
 public:
  ~BgReader() { stop(); }            // 进程退出兜底：避免 joinable 线程析构触发 std::terminate
  void start(int workers, int low_cap) {
    { std::lock_guard<std::mutex> lk(m_); low_cap_ = low_cap; }
    ensure_started(workers);
  }
  // 懒启动：侧区预取的 GPU 回调可能在 Python 没调用 bg_reader_start 时就要派任务，
  // 故 submit/submit_task 自动确保线程已起（默认 4 worker），调用方无需显式 start。
  void ensure_started(int workers) {
    std::lock_guard<std::mutex> lk(m_);
    if (running_) return;
    running_ = true;
    for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { loop(); });
  }
  void submit(BgJob job) {
    ensure_started(4);
    { std::lock_guard<std::mutex> lk(m_);
      if (job.prio > 0) high_q_.push(std::move(job)); else low_q_.push(std::move(job)); }
    cv_.notify_all();
  }
  void submit_task(std::function<void()> fn) {
    ensure_started(4);
    BgJob job;
    job.task = std::move(fn);
    { std::lock_guard<std::mutex> lk(m_); low_q_.push(std::move(job)); }   // 通用任务走低优
    cv_.notify_all();
  }
  bool ready(long ticket) {
    std::lock_guard<std::mutex> lk(dm_);
    return done_.count(ticket) > 0;
  }
  void wait(long ticket) {
    std::unique_lock<std::mutex> lk(dm_);
    dcv_.wait(lk, [&] { return done_.count(ticket) > 0; });
  }
  void stop() {
    { std::lock_guard<std::mutex> lk(m_); running_ = false; }
    cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
    threads_.clear();
    { std::lock_guard<std::mutex> lk(dm_); done_.clear(); }
    { std::lock_guard<std::mutex> lk(m_); active_low_ = 0; }
  }
 private:
  int low_budget() const { return low_cap_ <= 0 ? 1 << 30 : low_cap_; }  // <=0 视为不限流
  bool can_take_low() const { return !low_q_.empty() && active_low_ < low_budget(); }
  void loop() {
    std::unordered_map<std::string, int> fds;       // 每线程各自缓存 fd
    while (true) {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this] { return !high_q_.empty() || can_take_low() || !running_; });
      if (!running_ && high_q_.empty() && low_q_.empty()) break;
      if (!running_ && high_q_.empty() && !can_take_low()) break;  // 退出时低优可超额排空
      bool is_low = false;
      BgJob job;
      if (!high_q_.empty()) {                          // 高优(route)读永远先取
        job = std::move(high_q_.front()); high_q_.pop();
      } else if (can_take_low()) {                     // 低优限流：仅在额度内取
        job = std::move(low_q_.front()); low_q_.pop(); ++active_low_; is_low = true;
      } else {
        continue;                                      // 假醒（低优已满额）→ 回去等
      }
      lk.unlock();
      if (job.task) { job.task(); }                    // 通用任务（侧区异步读）：直接执行，无 ticket
      else {
        int fd;
        auto it = fds.find(job.path);
        if (it == fds.end()) { fd = open_blob_nocache(job.path.c_str()); fds[job.path] = fd; }
        else fd = it->second;
        if (fd >= 0)
          for (auto& op : job.ops) ::pread(fd, op.dst, op.nbytes, op.file_off);
        { std::lock_guard<std::mutex> lk2(dm_); done_.insert(job.ticket); }
        dcv_.notify_all();
      }
      if (is_low) {                                    // 释放低优额度 → 唤醒别的 worker 再取
        std::lock_guard<std::mutex> lk2(m_); --active_low_; cv_.notify_all();
      }
    }
    for (auto& kv : fds) if (kv.second >= 0) ::close(kv.second);
  }
  std::mutex m_, dm_;
  std::condition_variable cv_, dcv_;
  std::queue<BgJob> high_q_, low_q_;
  int active_low_ = 0;             // 当前正在执行的低优读数
  int low_cap_ = 0;               // 低优并发上限（<=0 不限流，保持旧行为）
  std::unordered_set<long> done_;
  std::vector<std::thread> threads_;
  bool running_ = false;
};
BgReader g_bg;
void bg_submit_task(std::function<void()> fn) { g_bg.submit_task(std::move(fn)); }
}  // namespace

void bg_reader_start(int workers, int low_cap) { g_bg.start(workers, low_cap); }

long bg_reader_submit(const mx::array& dst, const std::vector<int>& experts,
                      const std::vector<int>& rows, const std::string& path,
                      int stride, long ticket, int prio) {
  mx::array d = dst;
  d.eval();
  uint8_t* base = d.data<uint8_t>();
  size_t st = static_cast<size_t>(stride);
  BgJob job;
  job.path = path;
  job.ticket = ticket;
  job.prio = prio;
  job.keep.push_back(d);
  for (size_t i = 0; i < experts.size(); ++i)
    job.ops.push_back(ReadOp{base + static_cast<size_t>(rows[i]) * st, st,
                             static_cast<off_t>(static_cast<size_t>(experts[i]) * st)});
  g_bg.submit(std::move(job));
  return ticket;
}

// 专家各段直写进多个池段张量的 slot 行（消费侧零 MLX 算子）。
long bg_pread_into_pool(
    const std::vector<mx::array>& dst,
    const std::vector<long>& seg_off,
    const std::vector<long>& seg_nb,
    long slot, long expert,
    const std::string& path, long stride, long ticket, int prio) {
  BgJob job;
  job.path = path;
  job.ticket = ticket;
  job.prio = prio;
  for (size_t i = 0; i < dst.size(); ++i) {
    mx::array d = dst[i];
    d.eval();
    uint8_t* base = d.data<uint8_t>();
    job.keep.push_back(d);
    job.ops.push_back(ReadOp{
        base + static_cast<size_t>(slot) * static_cast<size_t>(seg_nb[i]),
        static_cast<size_t>(seg_nb[i]),
        static_cast<off_t>(static_cast<size_t>(expert) * static_cast<size_t>(stride)
                           + static_cast<size_t>(seg_off[i]))});
  }
  g_bg.submit(std::move(job));
  return ticket;
}

bool bg_reader_ready(long ticket) { return g_bg.ready(ticket); }
void bg_reader_wait(long ticket) { g_bg.wait(ticket); }
void bg_reader_stop() { g_bg.stop(); }
