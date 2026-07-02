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
      "resident"_a,
      "cap"_a,
      "parallel"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def("prefetch_staging_take", &prefetch_staging_take, "layer"_a);
  m.def("staging_hprof_enable", &staging_hprof_enable, "on"_a);
  m.def("staging_hprof_now", &staging_hprof_now);
  m.def("staging_hprof_get", &staging_hprof_get);
  m.def("prefetch_pool_sideregion", &prefetch_pool_sideregion,
        "pool_list"_a, "seg_nbytes"_a, "expert_ids"_a, "layer"_a, "path"_a, "stride"_a,
        "resident"_a, "spec_slots"_a, "base_row"_a, "gen"_a = 0, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("sideregion_contents", &sideregion_contents, "layer"_a, "gen"_a = 0);
  m.def("sideregion_kv", &sideregion_kv, "layer"_a, "gen"_a = 0);
  m.def("sideregion_reset", &sideregion_reset);
  m.def("materialize_spike", &materialize_spike, "src"_a, "fillval"_a, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("demand_probe", &demand_probe, "inds"_a, "offset"_a, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("demand_probe_handler", &demand_probe_handler, "inds"_a, "offset"_a, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("real_init", &real_init, "layer"_a, "cap"_a);
  m.def("real_region_contents", &real_region_contents, "layer"_a);
  m.def("real_region_count", &real_region_count, "layer"_a);
  m.def("real_reset", &real_reset);
  m.def("demand_dual", &demand_dual, "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "side_gen"_a, "path"_a, "stride"_a, "cap"_a, "lfu"_a, "decay_interval"_a, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("demand_last_stats", &demand_last_stats);
  m.def("demand_timings", &demand_timings);
  m.def("demand_timing_enable", &demand_timing_enable, "on"_a);
  m.def("real_debug_place", &real_debug_place, "layer"_a, "experts_flat"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a);
  m.def("real_freq", &real_freq, "layer"_a, "e"_a);
  m.def("bg_reader_start", &bg_reader_start, "workers"_a = 1, "low_cap"_a = 0);
  m.def("bg_reader_submit", &bg_reader_submit,
        "dst"_a, "experts"_a, "rows"_a, "path"_a, "stride"_a, "ticket"_a, "prio"_a = 0);
  m.def("bg_reader_ready", &bg_reader_ready, "ticket"_a);
  m.def("bg_reader_wait", &bg_reader_wait, "ticket"_a,
        nb::call_guard<nb::gil_scoped_release>());   // 阻塞等时释放 GIL
  m.def("bg_reader_stop", &bg_reader_stop);
  m.def("bg_pread_into_pool", &bg_pread_into_pool,
        "dst"_a, "seg_off"_a, "seg_nb"_a, "slot"_a, "expert"_a,
        "path"_a, "stride"_a, "ticket"_a, "prio"_a = 0, "nocache"_a = true);
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
