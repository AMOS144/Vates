#include "quant_attention.h"

#include <dlfcn.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace {
using namespace mlx::core;
using namespace mlx::steel;

std::string binary_dir() {
  static std::string path = [] {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) {
      throw std::runtime_error("cannot resolve native_moe_ext directory");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return path;
}

bool contiguous_last(const array& a) {
  return a.strides(-1) == 1;
}

class K4V3AttentionPrimitive : public Primitive {
 public:
  K4V3AttentionPrimitive(Stream stream, float scale, int bq, int bk)
      : Primitive(stream), scale_(scale), bq_(bq), bk_(bk) {}

  void eval_cpu(
      const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("k4v3_fused_causal_attention is GPU-only");
  }

  void eval_gpu(
      const std::vector<array>& in, std::vector<array>& out) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    const auto& q = in[0];
    auto& o = out[0];
    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    constexpr int D = 256;
    const int HK = in[1].shape(1);
    const int kL = in[1].shape(2);
    const int wm = bq_ == 16 ? 2 : 4;
    const bool align_q = qL % bq_ == 0;
    const bool align_k = kL % bk_ == 0;
    const bool no = false;
    const bool yes = true;
    metal::MTLFCList constants = {
        {&align_q, MTL::DataType::DataTypeBool, 200},
        {&align_k, MTL::DataType::DataTypeBool, 201},
        {&no, MTL::DataType::DataTypeBool, 300},
        {&yes, MTL::DataType::DataTypeBool, 301},
        {&no, MTL::DataType::DataTypeBool, 302},
        {&no, MTL::DataType::DataTypeBool, 303},
        {&no, MTL::DataType::DataTypeBool, 304},
        {&no, MTL::DataType::DataTypeBool, 305},
        {&no, MTL::DataType::DataTypeBool, 306}};

    std::string name;
    concatenate(name, "vates_k4v3_fa256_", type_to_name(q), "_bq", bq_,
                "_bk", bk_, "_bd256_wm", wm, "_wn1_mask", type_to_name(q));
    std::string hash;
    concatenate(hash, name, "_aq", align_q ? 't' : 'n', "_ak",
                align_k ? 't' : 'n');
    auto lib = device.get_library("vates_quant_attention", binary_dir());
    auto kernel = device.get_kernel(name, lib, hash, constants);
    encoder.set_compute_pipeline_state(kernel);

    o.set_data(allocator::malloc(o.nbytes()));
    const int nq = (qL + bq_ - 1) / bq_;
    const int nk = (kL + bk_ - 1) / bk_;
    AttnParams params{
        B, H, D, qL, kL, H / HK, scale_, nq, nk,
        qL / bq_, kL / bk_, qL % bq_, kL % bk_, kL - qL,
        {q.strides(0), q.strides(1), q.strides(2)},
        {in[1].strides(0), in[1].strides(1), in[1].strides(2)},
        {0, 0, 0},
        {o.strides(0), o.strides(1), o.strides(2)}};
    for (int i = 0; i < 7; ++i) encoder.set_input_array(in[i], i);
    encoder.set_output_array(o, 7);
    encoder.set_bytes(params, 8);
    encoder.dispatch_threadgroups(
        MTL::Size(nq, H, B), MTL::Size(32, wm, 1));
  }

  DEFINE_NAME(VatesK4V3FusedCausalAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const K4V3AttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && bq_ == rhs.bq_ && bk_ == rhs.bk_;
  }

 private:
  float scale_;
  int bq_;
  int bk_;
};

class DenseAttentionPrimitive : public Primitive {
 public:
  DenseAttentionPrimitive(Stream stream, float scale)
      : Primitive(stream), scale_(scale) {}
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("dense_fused_causal_attention is GPU-only");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    auto& encoder = metal::get_command_encoder(stream);
    const auto& q = in[0];
    const auto& k = in[1];
    const auto& v = in[2];
    auto& o = out[0];
    const int B = q.shape(0), H = q.shape(1), qL = q.shape(2);
    const int HK = k.shape(1), kL = k.shape(2);
    constexpr int bq = 32, bk = 8, D = 256, wm = 4;
    const bool aq = qL % bq == 0, ak = kL % bk == 0;
    const bool no = false, yes = true;
    metal::MTLFCList constants = {
        {&aq, MTL::DataType::DataTypeBool, 200},
        {&ak, MTL::DataType::DataTypeBool, 201},
        {&no, MTL::DataType::DataTypeBool, 300},
        {&yes, MTL::DataType::DataTypeBool, 301},
        {&no, MTL::DataType::DataTypeBool, 302},
        {&no, MTL::DataType::DataTypeBool, 303},
        {&no, MTL::DataType::DataTypeBool, 304},
        {&no, MTL::DataType::DataTypeBool, 305},
        {&no, MTL::DataType::DataTypeBool, 306}};
    std::string name;
    concatenate(name, "vates_dense_fa256_", type_to_name(q),
                "_bq32_bk8_bd256_wm4_wn1_mask", type_to_name(q));
    std::string hash;
    concatenate(hash, name, "_aq", aq ? 't' : 'n', "_ak", ak ? 't' : 'n');
    auto lib = device.get_library("vates_quant_attention", binary_dir());
    auto kernel = device.get_kernel(name, lib, hash, constants);
    encoder.set_compute_pipeline_state(kernel);
    o.set_data(allocator::malloc(o.nbytes()));
    const int nq = (qL + bq - 1) / bq, nk = (kL + bk - 1) / bk;
    AttnParams params{
        B, H, D, qL, kL, H / HK, scale_, nq, nk,
        qL / bq, kL / bk, qL % bq, kL % bk, kL - qL,
        {q.strides(0), q.strides(1), q.strides(2)},
        {k.strides(0), k.strides(1), k.strides(2)},
        {v.strides(0), v.strides(1), v.strides(2)},
        {o.strides(0), o.strides(1), o.strides(2)}};
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_output_array(o, 3);
    encoder.set_bytes(params, 4);
    encoder.dispatch_threadgroups(MTL::Size(nq, H, B), MTL::Size(32, wm, 1));
  }
  DEFINE_NAME(VatesDenseFusedCausalAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    return scale_ == static_cast<const DenseAttentionPrimitive&>(other).scale_;
  }
 private:
  float scale_;
};
}  // namespace

mx::array k4v3_fused_causal_attention(
    const mx::array& q, const mx::array& kw, const mx::array& ks,
    const mx::array& kb, const mx::array& vw, const mx::array& vs,
    const mx::array& vb, float scale, int q_block, int k_block,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (stream.device == mx::Device::cpu || q.ndim() != 4 ||
      (q.dtype() != mx::float16 && q.dtype() != mx::bfloat16) ||
      q.shape(3) != 256 || (q_block != 16 && q_block != 32) ||
      (k_block != 8 && k_block != 16)) {
    throw std::invalid_argument("unsupported K4/V3 fused attention Q shape");
  }
  const int B = q.shape(0), HK = kw.shape(1), L = kw.shape(2);
  if (kw.shape() != mx::Shape{B, HK, L, 32} ||
      vw.shape() != mx::Shape{B, HK, L, 24} ||
      ks.shape() != mx::Shape{B, HK, L, 4} || kb.shape() != ks.shape() ||
      vs.shape() != mx::Shape{B, HK, L, 4} || vb.shape() != vs.shape() ||
      kw.dtype() != mx::uint32 || vw.dtype() != mx::uint32 ||
      ks.dtype() != q.dtype() || kb.dtype() != q.dtype() ||
      vs.dtype() != q.dtype() || vb.dtype() != q.dtype() ||
      q.shape(1) % HK != 0 || q.shape(2) > L ||
      !contiguous_last(q) || !kw.flags().row_contiguous ||
      !ks.flags().row_contiguous || !kb.flags().row_contiguous ||
      !vw.flags().row_contiguous || !vs.flags().row_contiguous ||
      !vb.flags().row_contiguous) {
    throw std::invalid_argument("incompatible contiguous K4/V3 cache tensors");
  }
  mx::Shape shape{q.shape(0), q.shape(1), q.shape(2), 256};
  std::vector<mx::array> inputs{q, kw, ks, kb, vw, vs, vb};
  return mx::array(
      std::move(shape), q.dtype(),
      std::make_shared<K4V3AttentionPrimitive>(
          stream, scale, q_block, k_block),
      std::move(inputs));
}

mx::array dense_fused_causal_attention(
    const mx::array& q, const mx::array& k, const mx::array& v,
    float scale, mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  if (stream.device == mx::Device::cpu || q.ndim() != 4 || k.ndim() != 4 ||
      v.ndim() != 4 || q.dtype() != k.dtype() || q.dtype() != v.dtype() ||
      (q.dtype() != mx::float16 && q.dtype() != mx::bfloat16) ||
      q.shape(0) != k.shape(0) || k.shape(0) != v.shape(0) ||
      q.shape(1) % k.shape(1) || k.shape(1) != v.shape(1) ||
      k.shape(2) != v.shape(2) || q.shape(2) > k.shape(2) ||
      q.shape(3) != 256 || k.shape(3) != 256 || v.shape(3) != 256 ||
      !contiguous_last(q) || !contiguous_last(k) || !contiguous_last(v)) {
    throw std::invalid_argument("unsupported dense head256 attention tensors");
  }
  mx::Shape shape{q.shape(0), q.shape(1), q.shape(2), 256};
  std::vector<mx::array> inputs{q, k, v};
  return mx::array(
      std::move(shape), q.dtype(),
      std::make_shared<DenseAttentionPrimitive>(stream, scale),
      std::move(inputs));
}
