// [2] 轻量预取(无 staging，仅预热 page cache) + [3] staging 版 miss→hit + STAGING_HPROF 探针。
#include "prefetch.h"
#include "../io/bg_reader.h"

#include <chrono>
#include <functional>
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
