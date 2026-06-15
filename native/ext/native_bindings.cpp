#include "native_fused.h"
#include "native_prefetch.h"
using namespace nb::literals;

NB_MODULE(native_moe_ext, m) {
  nb::module_::import_("mlx.core");
  m.doc() = "Native MLX MoE extension.";
  m.def(
      "fused_moe",
      &fused_moe,
      "x"_a,
      "expert_ids"_a,
      "scores"_a,
      "compute_dir"_a,
      "layer"_a,
      "hidden"_a,
      "inter"_a,
      "group"_a,
      "bits"_a,
      "num_experts"_a,
      "synthetic"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "fused_moe_staged",
      &fused_moe_staged,
      "x"_a,
      "scores"_a,
      "gate_w"_a,
      "gate_s"_a,
      "gate_b"_a,
      "up_w"_a,
      "up_s"_a,
      "up_b"_a,
      "down_w"_a,
      "down_s"_a,
      "down_b"_a,
      "hidden"_a,
      "inter"_a,
      "group"_a,
      "bits"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "blob_load",
      &blob_load,
      "path"_a,
      "expert_ids"_a,
      "stride"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "prefetch_on_complete",
      &prefetch_on_complete,
      "expert_ids"_a,
      "path"_a,
      "stride"_a,
      "do_read"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def("prefetch_on_complete_last_ids", &prefetch_on_complete_last_ids);
  m.def("prefetch_on_complete_fires", &prefetch_on_complete_fires);
  m.def(
      "prefetch_into",
      &prefetch_into,
      "dst"_a,
      "expert_ids"_a,
      "path"_a,
      "stride"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "prefetch_into_staging",
      &prefetch_into_staging,
      "staging"_a,
      "expert_ids"_a,
      "layer"_a,
      "gen"_a,
      "path"_a,
      "stride"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def("prefetch_staging_take", &prefetch_staging_take, "layer"_a);
  m.def(
      "fused_moe_slots",
      &fused_moe_slots,
      "x"_a,
      "local_slots"_a,
      "scores"_a,
      "gate_w"_a,
      "gate_s"_a,
      "gate_b"_a,
      "up_w"_a,
      "up_s"_a,
      "up_b"_a,
      "down_w"_a,
      "down_s"_a,
      "down_b"_a,
      "hidden"_a,
      "inter"_a,
      "group"_a,
      "bits"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
}
