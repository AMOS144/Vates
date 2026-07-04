// [1] blob 直读：pread 专家字节进 MLX buffer（惰性图节点）。
// 把一组专家的 blob 字节直接 pread 进 MLX 自有 buffer（无 kernel、无额外拷贝）。
// load 作为惰性图节点：在批量 eval 中执行，避免 Python 侧 per-expert mx.eval 同步。
#include "blob_load.h"

#include <fcntl.h>
#include <unistd.h>

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
