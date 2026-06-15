#include "native_prefetch.h"

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

// ---- staging 版 miss→hit：handler pread 进 per-layer staging + 记录 (expert→row) ----
static std::mutex g_stg_mutex;
// layer -> (gen, [(expert, row)])；handler 原子写 gen+映射，主线程按 gen 匹配 buffer 后 take。
static std::map<int, std::pair<long, std::vector<std::pair<int, int>>>> g_stg_ready;

class PrefetchStagingPrimitive : public mx::Primitive {
 public:
  PrefetchStagingPrimitive(mx::Stream s, int layer, long gen, std::string path, size_t stride)
      : Primitive(s), layer_(layer), gen_(gen), path_(std::move(path)), stride_(stride) {}
  const char* name() const override { return "PrefetchStagingPrimitive"; }

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0]; ids.eval();
    mx::array stg = inputs[1]; stg.eval();
    fill(ids.data<uint32_t>(), ids.size(), stg.data<uint8_t>(), layer_, gen_, path_, stride_);
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    mx::array ids = inputs[0];
    mx::array stg = inputs[1];
    const uint32_t* idp = ids.data<uint32_t>();
    uint8_t* sp = stg.data<uint8_t>();
    size_t n = ids.size();
    int layer = layer_; long gen = gen_; std::string path = path_; size_t stride = stride_;
    auto& enc = mx::metal::get_command_encoder(stream());
    MTL::CommandBuffer* cb = enc.get_command_buffer();
    cb->addCompletedHandler([ids, stg, idp, sp, n, layer, gen, path, stride](MTL::CommandBuffer*) {
      fill(idp, n, sp, layer, gen, path, stride);
    });
  }

 private:
  static void fill(const uint32_t* idp, size_t n, uint8_t* stg,
                   int layer, long gen, const std::string& path, size_t stride) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    std::vector<std::pair<int, int>> ready;
    ready.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      int e = static_cast<int>(idp[i]);
      if (::pread(fd, stg + i * stride, stride, static_cast<off_t>(static_cast<size_t>(e) * stride))
          == static_cast<ssize_t>(stride)) {
        ready.emplace_back(e, static_cast<int>(i));
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
};

mx::array prefetch_into_staging(
    const mx::array& staging, const mx::array& expert_ids, int layer, long gen,
    const std::string& path, int stride, mx::StreamOrDevice s = {}) {
  return mx::array(
      mx::Shape{1}, mx::uint8,
      std::make_shared<PrefetchStagingPrimitive>(
          mx::to_stream(s), layer, gen, path, static_cast<size_t>(stride)),
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
