// [5] Route 3 Phase 1 底座：C++ 拥有的池 buffer。
// 用 mx::allocator::malloc 分配 buffer、no-op deleter 建叶子 mx.array（C++ 经 g_owned_bufs
// 持有 Buffer 句柄，进程内不释放）。因 C++ 独占持有、MLX 只读，该 buffer 永不被 MLX
// donation/迁移（spike 已证），故侧区/demand 的后台异步 pread 可安全直写、消费侧读同一块。
#include "owned_pool.h"

static std::mutex g_owned_mutex;
static std::vector<mx::allocator::Buffer> g_owned_bufs;

static mx::Dtype dtype_from_str(const std::string& s) {
  if (s == "uint32") return mx::uint32;
  if (s == "uint16") return mx::uint16;
  if (s == "uint8") return mx::uint8;
  if (s == "int32") return mx::int32;
  if (s == "int16") return mx::int16;
  if (s == "bfloat16") return mx::bfloat16;
  if (s == "float16") return mx::float16;
  if (s == "float32") return mx::float32;
  throw std::runtime_error("pool_owned_zeros: unsupported dtype " + s);
}

mx::array pool_owned_zeros(const std::vector<int>& shape, const std::string& dtype) {
  mx::Dtype dt = dtype_from_str(dtype);
  mx::Shape shp(shape.begin(), shape.end());
  size_t n = 1;
  for (int d : shape) n *= static_cast<size_t>(d);
  size_t nbytes = n * static_cast<size_t>(mx::size_of(dt));
  auto buf = mx::allocator::malloc(nbytes);
  std::memset(buf.raw_ptr(), 0, nbytes);
  {
    std::lock_guard<std::mutex> lk(g_owned_mutex);
    g_owned_bufs.push_back(buf);   // C++ 持有，保证进程内不被释放
  }
  return mx::array(buf, shp, dt, [](mx::allocator::Buffer) {});
}

// demand 真实区落池：把一批已加载专家段 memcpy 进 owned 池行（无 MLX scatter → pool buffer 永不重绑）。
// pool_list[i]：第 i 个 key 的池数组；srcs_flat 长 K*m，key-major：[k0行0,k0行1,...,k1行0,...]；
// slots[j]：第 j 个专家的目标物理行。CPU 端直写，调用点须保证此刻池未被 GPU 并发读该行。
void pool_write_rows(const std::vector<mx::array>& pool_list,
                     const std::vector<mx::array>& srcs_flat,
                     const std::vector<int>& slots) {
  int K = static_cast<int>(pool_list.size());
  int m = static_cast<int>(slots.size());
  if (m == 0) return;
  if (static_cast<int>(srcs_flat.size()) != K * m)
    throw std::runtime_error("pool_write_rows: srcs_flat 长度须 == K*m");
  for (int i = 0; i < K; ++i) {
    mx::array p = pool_list[i];
    p.eval();
    uint8_t* base = p.data<uint8_t>();
    for (int j = 0; j < m; ++j) {
      mx::array s = srcs_flat[static_cast<size_t>(i) * m + j];
      s.eval();
      size_t nb = s.nbytes();
      std::memcpy(base + static_cast<size_t>(slots[j]) * nb, s.data<uint8_t>(), nb);
    }
  }
}

// 同上，但源是每 key 预堆叠的 (m,*shape) 整块 stacked_list[i]，按行 memcpy 到 slots[j]。
void pool_write_stacked(const std::vector<mx::array>& pool_list,
                        const std::vector<mx::array>& stacked_list,
                        const std::vector<int>& slots) {
  int K = static_cast<int>(pool_list.size());
  int m = static_cast<int>(slots.size());
  if (m == 0) return;
  for (int i = 0; i < K; ++i) {
    mx::array p = pool_list[i];
    p.eval();
    mx::array st = stacked_list[i];
    st.eval();
    uint8_t* base = p.data<uint8_t>();
    const uint8_t* src = st.data<uint8_t>();
    size_t rownb = st.nbytes() / static_cast<size_t>(m);   // stacked 为 (m,*shape) → 每行字节
    for (int j = 0; j < m; ++j)
      std::memcpy(base + static_cast<size_t>(slots[j]) * rownb,
                  src + static_cast<size_t>(j) * rownb, rownb);
  }
}

// 临时诊断：返回某 mx.array 底层 buffer 的原始数据指针（uintptr）。用于对拍
// 「C++ 侧区 memcpy 写入的 buffer 指针」与「Python consume/verify 读到的 pool buffer 指针」
// 是否同一块——若不同即证明 MLX 在两者之间重分配了池 buffer，raw 写入被落单。
uintptr_t array_data_ptr(const mx::array& a) {
  mx::array b = a;
  b.eval();
  return reinterpret_cast<uintptr_t>(b.data<uint8_t>());
}
