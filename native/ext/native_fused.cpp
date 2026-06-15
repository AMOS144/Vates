#include "native_fused.h"

struct FusedParams {
  int experts;
  int hidden;
  int inter;
  int group_size;
  int bits;
  int k;
};

struct MappedProj {
  void* w{MAP_FAILED};
  void* s{MAP_FAILED};
  void* b{MAP_FAILED};
  size_t wn{0}, sn{0}, bn{0};
  int wfd{-1}, sfd{-1}, bfd{-1};
};

struct ProjectionShape {
  int out_dim;
  int in_dim;
  size_t weight_per;
  size_t scale_per;
};

static size_t file_size(int fd) {
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    throw std::runtime_error("fstat failed");
  }
  return static_cast<size_t>(st.st_size);
}

static void* map_file(const std::string& path, size_t& nbytes, int& fd) {
  fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("failed to open " + path);
  }
  nbytes = file_size(fd);
  void* p = mmap(nullptr, nbytes, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) {
    throw std::runtime_error("failed to mmap " + path);
  }
  return p;
}

static MappedProj map_proj(const std::string& dir, int layer, const std::string& proj) {
  MappedProj m;
  std::string layer_name = "layer" + std::string(layer < 10 ? "0" : "") + std::to_string(layer);
  std::string base = dir + "/" + layer_name + "." + proj;
  m.w = map_file(base + ".weight.bin", m.wn, m.wfd);
  m.s = map_file(base + ".scales.bin", m.sn, m.sfd);
  m.b = map_file(base + ".biases.bin", m.bn, m.bfd);
  return m;
}

static void unmap_proj(MappedProj& m) {
  if (m.w != MAP_FAILED) munmap(m.w, m.wn);
  if (m.s != MAP_FAILED) munmap(m.s, m.sn);
  if (m.b != MAP_FAILED) munmap(m.b, m.bn);
  if (m.wfd >= 0) close(m.wfd);
  if (m.sfd >= 0) close(m.sfd);
  if (m.bfd >= 0) close(m.bfd);
}

static ProjectionShape proj_shape(int out_dim, int in_dim, int group, int bits) {
  int words = (in_dim * bits) / 32;
  int groups = in_dim / group;
  return ProjectionShape{
      out_dim,
      in_dim,
      static_cast<size_t>(out_dim) * words * sizeof(uint32_t),
      static_cast<size_t>(out_dim) * groups * sizeof(uint16_t),
  };
}

static uint16_t fp32_to_bf16(float x) {
  uint32_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return static_cast<uint16_t>(u >> 16);
}

static void fill_synthetic(
    MTL::Buffer* wbuf,
    MTL::Buffer* sbuf,
    MTL::Buffer* bbuf,
    uint32_t seed) {
  std::mt19937 rng(seed);
  auto* w = reinterpret_cast<uint32_t*>(wbuf->contents());
  size_t wn = wbuf->length() / sizeof(uint32_t);
  for (size_t i = 0; i < wn; ++i) w[i] = rng();
  auto* s = reinterpret_cast<uint16_t*>(sbuf->contents());
  auto* b = reinterpret_cast<uint16_t*>(bbuf->contents());
  size_t sn = sbuf->length() / sizeof(uint16_t);
  uint16_t scale = fp32_to_bf16(0.001f);
  uint16_t bias = fp32_to_bf16(-0.032f);
  for (size_t i = 0; i < sn; ++i) {
    s[i] = scale;
    b[i] = bias;
  }
}

static void copy_projection(
    const MappedProj& src,
    const ProjectionShape& shape,
    const std::vector<int>& ids,
    MTL::Buffer* wbuf,
    MTL::Buffer* sbuf,
    MTL::Buffer* bbuf) {
  auto* wd = static_cast<char*>(wbuf->contents());
  auto* sd = static_cast<char*>(sbuf->contents());
  auto* bd = static_cast<char*>(bbuf->contents());
  auto* ws = static_cast<const char*>(src.w);
  auto* ss = static_cast<const char*>(src.s);
  auto* bs = static_cast<const char*>(src.b);
  for (size_t local = 0; local < ids.size(); ++local) {
    size_t expert = static_cast<size_t>(ids[local]);
    std::memcpy(wd + local * shape.weight_per, ws + expert * shape.weight_per, shape.weight_per);
    std::memcpy(sd + local * shape.scale_per, ss + expert * shape.scale_per, shape.scale_per);
    std::memcpy(bd + local * shape.scale_per, bs + expert * shape.scale_per, shape.scale_per);
  }
}

static std::string metal_source() {
  return R"(
    #include <metal_stdlib>
    using namespace metal;
    struct FusedParams { int experts; int hidden; int inter; int group_size; int bits; int k; };
    inline float bf16_to_float(ushort v) { uint u = uint(v) << 16; return as_type<float>(u); }
    inline float qvalue(const device uint* w, const device ushort* s, const device ushort* b,
                        uint expert, uint row, uint col, uint out_dim, uint in_dim,
                        uint group_size, uint bits) {
      uint words_per_row = (in_dim * bits) / 32;
      uint groups_per_row = in_dim / group_size;
      uint bit_offset = col * bits;
      uint word_idx = bit_offset / 32;
      uint shift = bit_offset % 32;
      uint base = (expert * out_dim + row) * words_per_row;
      uint q = w[base + word_idx] >> shift;
      if (shift + bits > 32) q |= w[base + word_idx + 1] << (32 - shift);
      q &= (1u << bits) - 1u;
      uint sb = (expert * out_dim + row) * groups_per_row + col / group_size;
      return float(q) * bf16_to_float(s[sb]) + bf16_to_float(b[sb]);
    }
    kernel void synthetic_moe(
      const device float* x [[buffer(0)]],
      device float* y [[buffer(1)]],
      const device float* scores [[buffer(2)]],
      constant FusedParams& p [[buffer(3)]],
      uint gid [[thread_position_in_grid]]) {
      uint token = gid / p.hidden;
      uint col = gid % p.hidden;
      float acc = 0.0f;
      for (int j = 0; j < p.k; ++j) {
        acc += scores[token * p.k + j];
      }
      y[gid] = x[token * p.hidden + col] * acc;
    }
    kernel void fused_moe_pairs(
      const device float* x [[buffer(0)]],
      const device uint* gate_w [[buffer(1)]], const device ushort* gate_s [[buffer(2)]], const device ushort* gate_b [[buffer(3)]],
      const device uint* up_w [[buffer(4)]], const device ushort* up_s [[buffer(5)]], const device ushort* up_b [[buffer(6)]],
      const device uint* down_w [[buffer(7)]], const device ushort* down_s [[buffer(8)]], const device ushort* down_b [[buffer(9)]],
      device float* y [[buffer(10)]], constant FusedParams& p [[buffer(11)]],
      uint tid [[thread_position_in_threadgroup]], uint gid [[thread_position_in_grid]]) {
      constexpr uint block_size = 512;
      constexpr uint lanes_per_row = 16;
      constexpr uint rows_per_step = block_size / lanes_per_row;
      uint expert = gid / block_size;
      if (expert >= uint(p.experts)) return;
      uint token = expert / p.k;
      uint local_row = tid / lanes_per_row;
      uint row_lane = tid % lanes_per_row;
      threadgroup float act[1024];
      threadgroup float gate_part[block_size];
      threadgroup float up_part[block_size];
      for (uint row_base = 0; row_base < uint(p.inter); row_base += rows_per_step) {
        uint row = row_base + local_row;
        float gate_acc = 0.0f, up_acc = 0.0f;
        if (row < uint(p.inter)) {
          for (uint col = row_lane; col < uint(p.hidden); col += lanes_per_row) {
            float xv = x[token * p.hidden + col];
            gate_acc += qvalue(gate_w, gate_s, gate_b, expert, row, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
            up_acc += qvalue(up_w, up_s, up_b, expert, row, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
          }
        }
        gate_part[tid] = gate_acc;
        up_part[tid] = up_acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
          if (row_lane < stride) { gate_part[tid] += gate_part[tid + stride]; up_part[tid] += up_part[tid + stride]; }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row_lane == 0 && row < uint(p.inter)) {
          float g = gate_part[tid];
          act[row] = g * (1.0f / (1.0f + exp(-g))) * up_part[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint row_base = 0; row_base < uint(p.hidden); row_base += rows_per_step) {
        uint row = row_base + local_row;
        float acc = 0.0f;
        if (row < uint(p.hidden)) {
          for (uint col = row_lane; col < uint(p.inter); col += lanes_per_row) {
            acc += qvalue(down_w, down_s, down_b, expert, row, col, p.hidden, p.inter, p.group_size, p.bits) * act[col];
          }
        }
        gate_part[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
          if (row_lane < stride) gate_part[tid] += gate_part[tid + stride];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row_lane == 0 && row < uint(p.hidden)) {
          y[expert * p.hidden + row] = gate_part[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    kernel void fused_moe_slots_pairs(
      const device float* x [[buffer(0)]],
      const device uint* gate_w [[buffer(1)]], const device ushort* gate_s [[buffer(2)]], const device ushort* gate_b [[buffer(3)]],
      const device uint* up_w [[buffer(4)]], const device ushort* up_s [[buffer(5)]], const device ushort* up_b [[buffer(6)]],
      const device uint* down_w [[buffer(7)]], const device ushort* down_s [[buffer(8)]], const device ushort* down_b [[buffer(9)]],
      device float* y [[buffer(10)]], constant FusedParams& p [[buffer(11)]],
      const device uint* local_slots [[buffer(12)]],
      uint tid [[thread_position_in_threadgroup]], uint gid [[thread_position_in_grid]]) {
      constexpr uint block_size = 512;
      constexpr uint lanes_per_row = 16;
      constexpr uint rows_per_step = block_size / lanes_per_row;
      uint active_idx = gid / block_size;
      if (active_idx >= uint(p.experts)) return;
      uint slot = local_slots[active_idx];
      uint token = active_idx / p.k;
      uint local_row = tid / lanes_per_row;
      uint row_lane = tid % lanes_per_row;
      threadgroup float act[1024];
      threadgroup float gate_part[block_size];
      threadgroup float up_part[block_size];
      for (uint row_base = 0; row_base < uint(p.inter); row_base += rows_per_step) {
        uint row = row_base + local_row;
        float gate_acc = 0.0f, up_acc = 0.0f;
        if (row < uint(p.inter)) {
          for (uint col = row_lane; col < uint(p.hidden); col += lanes_per_row) {
            float xv = x[token * p.hidden + col];
            gate_acc += qvalue(gate_w, gate_s, gate_b, slot, row, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
            up_acc += qvalue(up_w, up_s, up_b, slot, row, col, p.inter, p.hidden, p.group_size, p.bits) * xv;
          }
        }
        gate_part[tid] = gate_acc;
        up_part[tid] = up_acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
          if (row_lane < stride) { gate_part[tid] += gate_part[tid + stride]; up_part[tid] += up_part[tid + stride]; }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row_lane == 0 && row < uint(p.inter)) {
          float g = gate_part[tid];
          act[row] = g * (1.0f / (1.0f + exp(-g))) * up_part[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint row_base = 0; row_base < uint(p.hidden); row_base += rows_per_step) {
        uint row = row_base + local_row;
        float acc = 0.0f;
        if (row < uint(p.hidden)) {
          for (uint col = row_lane; col < uint(p.inter); col += lanes_per_row) {
            acc += qvalue(down_w, down_s, down_b, slot, row, col, p.hidden, p.inter, p.group_size, p.bits) * act[col];
          }
        }
        gate_part[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes_per_row / 2; stride > 0; stride >>= 1) {
          if (row_lane < stride) gate_part[tid] += gate_part[tid + stride];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (row_lane == 0 && row < uint(p.hidden)) {
          y[active_idx * p.hidden + row] = gate_part[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }
    kernel void reduce_pairs(
      const device float* pair_y [[buffer(0)]],
      const device float* scores [[buffer(1)]],
      device float* out [[buffer(2)]],
      constant FusedParams& p [[buffer(3)]],
      uint gid [[thread_position_in_grid]]) {
      uint token = gid / p.hidden;
      uint col = gid % p.hidden;
      float acc = 0.0f;
      for (int j = 0; j < p.k; ++j) {
        uint pair = token * p.k + uint(j);
        acc += pair_y[pair * p.hidden + col] * scores[pair];
      }
      out[gid] = acc;
    }
  )";
}

class FusedMoePrimitive : public mx::Primitive {
 public:
  FusedMoePrimitive(
      mx::Stream stream,
      std::string compute_dir,
      int layer,
      int hidden,
      int inter,
      int group,
      int bits,
      int num_experts,
      bool synthetic,
      std::vector<int> expert_ids)
      : Primitive(stream),
        compute_dir_(std::move(compute_dir)),
        layer_(layer),
        hidden_(hidden),
        inter_(inter),
        group_(group),
        bits_(bits),
        num_experts_(num_experts),
        synthetic_(synthetic),
        expert_ids_(std::move(expert_ids)) {
    if (!synthetic_) {
      gate_ = map_proj(compute_dir_, layer_, "gate_proj");
      up_ = map_proj(compute_dir_, layer_, "up_proj");
      down_ = map_proj(compute_dir_, layer_, "down_proj");
    }
  }

  ~FusedMoePrimitive() override {
    if (!synthetic_) {
      unmap_proj(gate_);
      unmap_proj(up_);
      unmap_proj(down_);
    }
  }

  const char* name() const override { return "FusedMoePrimitive"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("FusedMoePrimitive only supports GPU evaluation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& x = inputs[0];
    auto& scores = inputs[1];
    auto& out = outputs[0];
    int active = static_cast<int>(expert_ids_.size());
    int tokens = static_cast<int>(x.size() / hidden_);
    int k = active / std::max(1, tokens);
    if (active <= 0 || tokens <= 0 || active != static_cast<int>(scores.size())) {
      throw std::runtime_error("FusedMoePrimitive shape mismatch");
    }
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& d = mx::metal::device(stream().device);
    auto& enc = mx::metal::get_command_encoder(stream());
    auto lib = d.get_library("native_moe_mlx_ext", []() { return metal_source(); });
    if (synthetic_) {
      auto synthetic_kernel = d.get_kernel("synthetic_moe", lib);
      enc.set_compute_pipeline_state(synthetic_kernel);
      enc.set_input_array(x, 0);
      enc.set_output_array(out, 1);
      enc.set_input_array(scores, 2);
      FusedParams params{active, hidden_, inter_, group_, bits_, k};
      enc.set_bytes(params, 3);
      enc.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(std::min<size_t>(out.size(), 256), 1, 1));
      return;
    }
    auto fused = d.get_kernel("fused_moe_pairs", lib);
    auto reduce = d.get_kernel("reduce_pairs", lib);

    ProjectionShape gu = proj_shape(inter_, hidden_, group_, bits_);
    ProjectionShape down = proj_shape(hidden_, inter_, group_, bits_);
    size_t gu_w_bytes = static_cast<size_t>(active) * gu.weight_per;
    size_t gu_s_bytes = static_cast<size_t>(active) * gu.scale_per;
    size_t down_w_bytes = static_cast<size_t>(active) * down.weight_per;
    size_t down_s_bytes = static_cast<size_t>(active) * down.scale_per;
    buffers_.clear();
    auto make_buffer = [&](size_t nbytes) -> MTL::Buffer* {
      buffers_.push_back(NS::TransferPtr(d.mtl_device()->newBuffer(nbytes, MTL::ResourceStorageModeShared)));
      return buffers_.back().get();
    };
    MTL::Buffer* gw = make_buffer(gu_w_bytes);
    MTL::Buffer* gs = make_buffer(gu_s_bytes);
    MTL::Buffer* gb = make_buffer(gu_s_bytes);
    MTL::Buffer* uw = make_buffer(gu_w_bytes);
    MTL::Buffer* us = make_buffer(gu_s_bytes);
    MTL::Buffer* ub = make_buffer(gu_s_bytes);
    MTL::Buffer* dw = make_buffer(down_w_bytes);
    MTL::Buffer* ds = make_buffer(down_s_bytes);
    MTL::Buffer* db = make_buffer(down_s_bytes);
    if (synthetic_) {
      fill_synthetic(gw, gs, gb, 1);
      fill_synthetic(uw, us, ub, 2);
      fill_synthetic(dw, ds, db, 3);
    } else {
      copy_projection(gate_, gu, expert_ids_, gw, gs, gb);
      copy_projection(up_, gu, expert_ids_, uw, us, ub);
      copy_projection(down_, down, expert_ids_, dw, ds, db);
    }

    auto pair_out = mx::zeros(mx::Shape{active, hidden_}, mx::float32, stream());
    pair_out.set_data(mx::allocator::malloc(pair_out.nbytes()));

    enc.set_compute_pipeline_state(fused);
    enc.set_input_array(x, 0);
    enc.set_buffer(gw, 1);
    enc.set_buffer(gs, 2);
    enc.set_buffer(gb, 3);
    enc.set_buffer(uw, 4);
    enc.set_buffer(us, 5);
    enc.set_buffer(ub, 6);
    enc.set_buffer(dw, 7);
    enc.set_buffer(ds, 8);
    enc.set_buffer(db, 9);
    enc.set_output_array(out, 10);
    FusedParams params{active, hidden_, inter_, group_, bits_, k};
    enc.set_bytes(params, 11);
    enc.set_input_array(scores, 12);
    enc.add_temporary(x);
    enc.add_temporary(scores);
    enc.dispatch_threads(MTL::Size(static_cast<size_t>(active) * 512, 1, 1), MTL::Size(512, 1, 1));
  }

 private:
  std::string compute_dir_;
  int layer_;
  int hidden_;
  int inter_;
  int group_;
  int bits_;
  int num_experts_;
  bool synthetic_;
  std::vector<int> expert_ids_;
  MappedProj gate_;
  MappedProj up_;
  MappedProj down_;
  std::vector<NS::SharedPtr<MTL::Buffer>> buffers_;
};

class StagedFusedMoePrimitive : public mx::Primitive {
 public:
  StagedFusedMoePrimitive(mx::Stream stream, int hidden, int inter, int group, int bits)
      : Primitive(stream), hidden_(hidden), inter_(inter), group_(group), bits_(bits) {}

  const char* name() const override { return "StagedFusedMoePrimitive"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("StagedFusedMoePrimitive only supports GPU evaluation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& x = inputs[0];
    auto& scores = inputs[1];
    auto& out = outputs[0];
    int tokens = static_cast<int>(x.size() / hidden_);
    int active = static_cast<int>(scores.size());
    int k = active / std::max(1, tokens);
    if (active <= 0 || tokens <= 0 || active != static_cast<int>(scores.size())) {
      throw std::runtime_error("StagedFusedMoePrimitive shape mismatch");
    }
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& d = mx::metal::device(stream().device);
    auto& enc = mx::metal::get_command_encoder(stream());
    auto lib = d.get_library("native_moe_mlx_ext", []() { return metal_source(); });
    auto fused = d.get_kernel("fused_moe_pairs", lib);
    auto reduce = d.get_kernel("reduce_pairs", lib);
    auto pair_out = mx::zeros(mx::Shape{active, hidden_}, mx::float32, stream());
    pair_out.set_data(mx::allocator::malloc(pair_out.nbytes()));
    enc.set_compute_pipeline_state(fused);
    enc.set_input_array(x, 0);
    enc.set_input_array(inputs[2], 1);
    enc.set_input_array(inputs[3], 2);
    enc.set_input_array(inputs[4], 3);
    enc.set_input_array(inputs[5], 4);
    enc.set_input_array(inputs[6], 5);
    enc.set_input_array(inputs[7], 6);
    enc.set_input_array(inputs[8], 7);
    enc.set_input_array(inputs[9], 8);
    enc.set_input_array(inputs[10], 9);
    enc.set_output_array(pair_out, 10);
    FusedParams params{active, hidden_, inter_, group_, bits_, k};
    enc.set_bytes(params, 11);
    enc.dispatch_threads(MTL::Size(static_cast<size_t>(active) * 512, 1, 1), MTL::Size(512, 1, 1));
    enc.set_compute_pipeline_state(reduce);
    enc.set_input_array(pair_out, 0);
    enc.set_input_array(scores, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.add_temporary(pair_out);
    enc.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(std::min<size_t>(out.size(), 256), 1, 1));
  }

 private:
  int hidden_;
  int inter_;
  int group_;
  int bits_;
};

class SlotFusedMoePrimitive : public mx::Primitive {
 public:
  SlotFusedMoePrimitive(mx::Stream stream, int hidden, int inter, int group, int bits)
      : Primitive(stream), hidden_(hidden), inter_(inter), group_(group), bits_(bits) {}

  const char* name() const override { return "SlotFusedMoePrimitive"; }

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("SlotFusedMoePrimitive only supports GPU evaluation");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& x = inputs[0];
    auto& local_slots = inputs[1];
    auto& scores = inputs[2];
    auto& out = outputs[0];
    int tokens = static_cast<int>(x.size() / hidden_);
    int active = static_cast<int>(scores.size());
    int k = active / std::max(1, tokens);
    if (active <= 0 || tokens <= 0 || active != static_cast<int>(local_slots.size())) {
      throw std::runtime_error("SlotFusedMoePrimitive shape mismatch");
    }
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& d = mx::metal::device(stream().device);
    auto& enc = mx::metal::get_command_encoder(stream());
    auto lib = d.get_library("native_moe_mlx_ext", []() { return metal_source(); });
    auto fused = d.get_kernel("fused_moe_slots_pairs", lib);
    auto reduce = d.get_kernel("reduce_pairs", lib);
    auto pair_out = mx::zeros(mx::Shape{active, hidden_}, mx::float32, stream());
    pair_out.set_data(mx::allocator::malloc(pair_out.nbytes()));
    enc.set_compute_pipeline_state(fused);
    enc.set_input_array(x, 0);
    enc.set_input_array(inputs[3], 1);
    enc.set_input_array(inputs[4], 2);
    enc.set_input_array(inputs[5], 3);
    enc.set_input_array(inputs[6], 4);
    enc.set_input_array(inputs[7], 5);
    enc.set_input_array(inputs[8], 6);
    enc.set_input_array(inputs[9], 7);
    enc.set_input_array(inputs[10], 8);
    enc.set_input_array(inputs[11], 9);
    enc.set_output_array(pair_out, 10);
    FusedParams params{active, hidden_, inter_, group_, bits_, k};
    enc.set_bytes(params, 11);
    enc.set_input_array(local_slots, 12);
    enc.dispatch_threads(MTL::Size(static_cast<size_t>(active) * 512, 1, 1), MTL::Size(512, 1, 1));
    enc.set_compute_pipeline_state(reduce);
    enc.set_input_array(pair_out, 0);
    enc.set_input_array(scores, 1);
    enc.set_output_array(out, 2);
    enc.set_bytes(params, 3);
    enc.add_temporary(pair_out);
    enc.dispatch_threads(MTL::Size(out.size(), 1, 1), MTL::Size(std::min<size_t>(out.size(), 256), 1, 1));
  }

 private:
  int hidden_;
  int inter_;
  int group_;
  int bits_;
};
mx::array fused_moe(
    const mx::array& x,
    const mx::array& expert_ids,
    const mx::array& scores,
    const std::string& compute_dir,
    int layer,
    int hidden,
    int inter,
    int group,
    int bits,
    int num_experts,
    bool synthetic,
    mx::StreamOrDevice s = {}) {
  if (!synthetic) {
    throw std::runtime_error(
        "real mmap staging in MLX Primitive needs MLX-managed staging buffers; use synthetic for bridge tests");
  }
  auto ids = expert_ids;
  ids.eval();
  std::vector<int> expert_vec(ids.size());
  const uint32_t* idp = ids.data<uint32_t>();
  for (size_t i = 0; i < ids.size(); ++i) expert_vec[i] = static_cast<int>(idp[i]);
  auto out_shape = x.shape();
  out_shape.back() = hidden;
  return mx::array(
      out_shape,
      mx::float32,
      std::make_shared<FusedMoePrimitive>(
          mx::to_stream(s),
          compute_dir,
          layer,
          hidden,
          inter,
          group,
          bits,
          num_experts,
          synthetic,
          std::move(expert_vec)),
      std::vector<mx::array>{x, scores});
}

mx::array fused_moe_staged(
    const mx::array& x,
    const mx::array& scores,
    const mx::array& gate_w,
    const mx::array& gate_s,
    const mx::array& gate_b,
    const mx::array& up_w,
    const mx::array& up_s,
    const mx::array& up_b,
    const mx::array& down_w,
    const mx::array& down_s,
    const mx::array& down_b,
    int hidden,
    int inter,
    int group,
    int bits,
    mx::StreamOrDevice s = {}) {
  auto out_shape = x.shape();
  out_shape.back() = hidden;
  return mx::array(
      out_shape,
      mx::float32,
      std::make_shared<StagedFusedMoePrimitive>(mx::to_stream(s), hidden, inter, group, bits),
      std::vector<mx::array>{
          x, scores,
          gate_w, gate_s, gate_b,
          up_w, up_s, up_b,
          down_w, down_s, down_b});
}

mx::array fused_moe_slots(
    const mx::array& x,
    const mx::array& local_slots,
    const mx::array& scores,
    const mx::array& gate_w,
    const mx::array& gate_s,
    const mx::array& gate_b,
    const mx::array& up_w,
    const mx::array& up_s,
    const mx::array& up_b,
    const mx::array& down_w,
    const mx::array& down_s,
    const mx::array& down_b,
    int hidden,
    int inter,
    int group,
    int bits,
    mx::StreamOrDevice s = {}) {
  auto out_shape = x.shape();
  out_shape.back() = hidden;
  return mx::array(
      out_shape,
      mx::float32,
      std::make_shared<SlotFusedMoePrimitive>(mx::to_stream(s), hidden, inter, group, bits),
      std::vector<mx::array>{
          x, local_slots, scores,
          gate_w, gate_s, gate_b,
          up_w, up_s, up_b,
          down_w, down_s, down_b});
}
