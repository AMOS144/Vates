// [1] blob 直读：pread 专家字节进 MLX buffer（惰性图节点）。
// 把一组专家的 blob 字节直接 pread 进 MLX 自有 buffer（无 kernel、无额外拷贝）。
// load 作为惰性图节点：在批量 eval 中执行，避免 Python 侧 per-expert mx.eval 同步。
#include "blob_load.h"

#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
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

mx::array blob_mmap(
    const std::string& path, int stride, int num_experts) {
  if (stride <= 0 || num_experts <= 0) {
    throw std::invalid_argument("blob_mmap requires positive dimensions");
  }
  const size_t expected =
      static_cast<size_t>(stride) * static_cast<size_t>(num_experts);
  const long page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0 || expected % static_cast<size_t>(page_size) != 0) {
    throw std::invalid_argument("blob_mmap byte length must be page aligned");
  }
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("blob_mmap open failed: " + path);
  struct stat info {};
  if (::fstat(fd, &info) != 0 || static_cast<size_t>(info.st_size) != expected) {
    ::close(fd);
    throw std::runtime_error("blob_mmap file size does not match shape");
  }
  // Metal's bytes-no-copy registration requires writable virtual memory even
  // though kernels only read it. MAP_PRIVATE preserves the underlying file.
  void* mapping = ::mmap(
      nullptr, expected, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  if (mapping == MAP_FAILED) {
    ::close(fd);
    throw std::runtime_error("blob_mmap mmap failed: " + path);
  }
  try {
    return mx::array(
        mapping,
        mx::Shape{num_experts, stride},
        mx::uint8,
        [expected, fd](void* pointer) {
          ::munmap(pointer, expected);
          ::close(fd);
        });
  } catch (...) {
    ::munmap(mapping, expected);
    ::close(fd);
    throw;
  }
}

namespace {

struct MappedBlob {
  std::string path;
  size_t length;
  void* mapping;
  int fd;
  std::mutex mutex;
  MTL::Device* device{nullptr};
  NS::SharedPtr<MTL::Buffer> buffer;

  MappedBlob(std::string path_, size_t expected)
      : path(std::move(path_)), length(expected),
        mapping(MAP_FAILED), fd(-1) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("blob mmap gather open failed: " + path);
    struct stat info {};
    if (::fstat(fd, &info) != 0 || static_cast<size_t>(info.st_size) != length) {
      ::close(fd);
      fd = -1;
      throw std::runtime_error("blob mmap gather file size mismatch");
    }
    mapping = ::mmap(
        nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
      ::close(fd);
      fd = -1;
      throw std::runtime_error("blob mmap gather mmap failed: " + path);
    }
  }

  ~MappedBlob() {
    buffer.reset();
    if (mapping != MAP_FAILED) ::munmap(mapping, length);
    if (fd >= 0) ::close(fd);
  }

  MTL::Buffer* metal_buffer(MTL::Device* requested) {
    std::lock_guard<std::mutex> lock(mutex);
    if (buffer && device != requested) {
      throw std::runtime_error("blob mmap gather cannot span Metal devices");
    }
    if (!buffer) {
      device = requested;
      buffer = NS::TransferPtr(requested->newBuffer(
          mapping, length, MTL::ResourceStorageModeShared,
          ^(void*, NS::UInteger) {}));
      if (!buffer) {
        throw std::runtime_error("Metal rejected file-backed blob buffer");
      }
    }
    return buffer.get();
  }
};

static std::mutex g_mapped_blob_mutex;
static std::unordered_map<std::string, std::shared_ptr<MappedBlob>>
    g_mapped_blobs;

static std::shared_ptr<MappedBlob> mapped_blob(
    const std::string& path, size_t expected) {
  std::lock_guard<std::mutex> lock(g_mapped_blob_mutex);
  auto found = g_mapped_blobs.find(path);
  if (found != g_mapped_blobs.end()) {
    if (found->second->length != expected) {
      throw std::runtime_error("blob mmap gather shape changed for cached path");
    }
    return found->second;
  }
  auto value = std::make_shared<MappedBlob>(path, expected);
  g_mapped_blobs.emplace(path, value);
  return value;
}

static std::string blob_gather_metal_source() {
  return R"METAL(
    #include <metal_stdlib>
    using namespace metal;
    struct BlobGatherParams { uint vectors_per_expert; };
    kernel void blob_mmap_gather_u4(
        device const uint4* blob [[buffer(0)]],
        device const uint* experts [[buffer(1)]],
        device uint4* output [[buffer(2)]],
        constant BlobGatherParams& params [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
      const uint route_position = tid / params.vectors_per_expert;
      const uint local_vector = tid % params.vectors_per_expert;
      output[tid] = blob[
          experts[route_position] * params.vectors_per_expert + local_vector];
    }
    struct BlobPrefetchParams { uint words_per_expert; uint words_per_page; };
    kernel void blob_mmap_prefetch_pages(
        device const uint* blob [[buffer(0)]],
        device const uint* experts [[buffer(1)]],
        device uint* checksums [[buffer(2)]],
        constant BlobPrefetchParams& params [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
      uint checksum = 0;
      const uint base = experts[tid] * params.words_per_expert;
      for (uint word = 0; word < params.words_per_expert;
           word += params.words_per_page) {
        checksum ^= blob[base + word];
      }
      checksums[tid] = checksum;
    }
  )METAL";
}

class BlobMmapGatherPrimitive : public mx::Primitive {
 public:
  BlobMmapGatherPrimitive(
      mx::Stream stream, std::shared_ptr<MappedBlob> blob, int stride)
      : Primitive(stream), blob_(std::move(blob)), stride_(stride) {}

  const char* name() const override { return "BlobMmapGatherPrimitive"; }

  void eval_cpu(
      const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("blob_mmap_gather requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    auto& output = outputs[0];
    output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    auto library = device.get_library(
        "streaming_blob_mmap_gather", [] { return blob_gather_metal_source(); });
    auto kernel = device.get_kernel("blob_mmap_gather_u4", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_buffer(blob_->metal_buffer(device.mtl_device()), 0);
    encoder.set_input_array(inputs[0], 1);
    encoder.set_output_array(output, 2);
    struct BlobGatherParams { uint32_t vectors_per_expert; };
    const BlobGatherParams params{static_cast<uint32_t>(stride_ / 16)};
    encoder.set_bytes(params, 3);
    const size_t vectors = output.nbytes() / 16;
    encoder.dispatch_threads(
        MTL::Size(vectors, 1, 1),
        MTL::Size(std::min<size_t>(vectors, 256), 1, 1));
  }

 private:
  std::shared_ptr<MappedBlob> blob_;
  int stride_;
};

class BlobMmapPrefetchPrimitive : public mx::Primitive {
 public:
  BlobMmapPrefetchPrimitive(
      mx::Stream stream, std::shared_ptr<MappedBlob> blob, int stride)
      : Primitive(stream), blob_(std::move(blob)), stride_(stride) {}

  const char* name() const override { return "BlobMmapPrefetchPrimitive"; }

  void eval_cpu(
      const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("blob_mmap_prefetch requires Metal");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    auto& output = outputs[0];
    output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& device = mx::metal::device(stream().device);
    auto& encoder = mx::metal::get_command_encoder(stream());
    auto library = device.get_library(
        "streaming_blob_mmap_gather", [] { return blob_gather_metal_source(); });
    auto kernel = device.get_kernel("blob_mmap_prefetch_pages", library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_buffer(blob_->metal_buffer(device.mtl_device()), 0);
    encoder.set_input_array(inputs[0], 1);
    encoder.set_output_array(output, 2);
    struct BlobPrefetchParams {
      uint32_t words_per_expert;
      uint32_t words_per_page;
    };
    const BlobPrefetchParams params{
        static_cast<uint32_t>(stride_ / 4),
        static_cast<uint32_t>(::sysconf(_SC_PAGESIZE) / 4),
    };
    encoder.set_bytes(params, 3);
    encoder.dispatch_threads(
        MTL::Size(output.size(), 1, 1),
        MTL::Size(std::min<size_t>(output.size(), 256), 1, 1));
  }

 private:
  std::shared_ptr<MappedBlob> blob_;
  int stride_;
};

}  // namespace

mx::array blob_mmap_gather(
    const std::string& path, const mx::array& expert_ids,
    int stride, int num_experts, mx::StreamOrDevice s = {}) {
  if (stride <= 0 || stride % 16 != 0 || num_experts <= 0) {
    throw std::invalid_argument(
        "blob_mmap_gather requires positive 16-byte-aligned dimensions");
  }
  const size_t expected =
      static_cast<size_t>(stride) * static_cast<size_t>(num_experts);
  auto stream = mx::to_stream(s);
  auto ids = mx::contiguous(expert_ids, false, stream);
  if (ids.dtype() != mx::uint32) {
    throw std::invalid_argument("blob_mmap_gather expert_ids must be uint32");
  }
  return mx::array(
      mx::Shape{static_cast<int>(ids.size()), stride},
      mx::uint8,
      std::make_shared<BlobMmapGatherPrimitive>(
          stream, mapped_blob(path, expected), stride),
      std::vector<mx::array>{ids});
}

mx::array blob_mmap_prefetch(
    const std::string& path, const mx::array& expert_ids,
    int stride, int num_experts, mx::StreamOrDevice s = {}) {
  if (stride <= 0 || stride % 4 != 0 || num_experts <= 0) {
    throw std::invalid_argument("blob_mmap_prefetch dimensions are invalid");
  }
  const size_t expected =
      static_cast<size_t>(stride) * static_cast<size_t>(num_experts);
  auto stream = mx::to_stream(s);
  auto ids = mx::contiguous(expert_ids, false, stream);
  if (ids.dtype() != mx::uint32) {
    throw std::invalid_argument("blob_mmap_prefetch expert_ids must be uint32");
  }
  auto touched = mx::array(
      mx::Shape{static_cast<int>(ids.size())},
      mx::uint32,
      std::make_shared<BlobMmapPrefetchPrimitive>(
          stream, mapped_blob(path, expected), stride),
      std::vector<mx::array>{ids});
  return mx::sum(touched, stream);
}

long blob_mmap_prefault_host(
    const std::string& path, const mx::array& expert_ids,
    int stride, int num_experts) {
  if (stride <= 0 || num_experts <= 0) {
    throw std::invalid_argument("blob_mmap_prefault_host dimensions are invalid");
  }
  const size_t expected =
      static_cast<size_t>(stride) * static_cast<size_t>(num_experts);
  auto blob = mapped_blob(path, expected);
  auto ids = mx::contiguous(expert_ids);
  ids.eval();
  if (ids.dtype() != mx::uint32) {
    throw std::invalid_argument("blob_mmap_prefault_host ids must be uint32");
  }
  const auto* values = ids.data<uint32_t>();
  const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
  volatile uint8_t checksum = 0;
  long pages = 0;
  for (size_t index = 0; index < ids.size(); ++index) {
    const uint32_t expert = values[index];
    if (expert >= static_cast<uint32_t>(num_experts)) {
      throw std::out_of_range("blob_mmap_prefault_host expert out of range");
    }
    auto* start = static_cast<uint8_t*>(blob->mapping)
        + static_cast<size_t>(expert) * static_cast<size_t>(stride);
    ::madvise(start, static_cast<size_t>(stride), MADV_WILLNEED);
    for (size_t offset = 0; offset < static_cast<size_t>(stride); offset += page) {
      checksum ^= start[offset];
      ++pages;
    }
  }
  // Keep the volatile page loads observable without leaking model bytes.
  (void)checksum;
  return pages;
}
