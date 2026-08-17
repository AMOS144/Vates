#include "compute/fused_moe.h"
#include "io/blob_load.h"
#include "io/bg_reader.h"
#include "prefetch/prefetch.h"
#include "pool/owned_pool.h"
#include "pool/side_region.h"
#include "pool/demand.h"
using namespace nb::literals;

NB_MODULE(native_moe_ext, m) {
  nb::module_::import_("mlx.core");
  m.doc() = "Native MLX MoE extension.";

  // ---- 融合 MoE 计算核（native_fused.cpp）----
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
  // ---- [1] blob 直读 ----
  m.def(
      "blob_load",
      &blob_load,
      "path"_a,
      "expert_ids"_a,
      "stride"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "blob_mmap",
      &blob_mmap,
      "path"_a,
      "stride"_a,
      "num_experts"_a);
  m.def(
      "blob_mmap_gather",
      &blob_mmap_gather,
      "path"_a,
      "expert_ids"_a,
      "stride"_a,
      "num_experts"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "blob_mmap_prefetch",
      &blob_mmap_prefetch,
      "path"_a,
      "expert_ids"_a,
      "stride"_a,
      "num_experts"_a,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def(
      "blob_mmap_prefault_host",
      &blob_mmap_prefault_host,
      "path"_a,
      "expert_ids"_a,
      "stride"_a,
      "num_experts"_a,
      nb::call_guard<nb::gil_scoped_release>());
  // ---- [2] 轻量预取(无 staging，仅预热 page cache) ----
  m.def(
      "prefetch_on_complete",
      &prefetch_on_complete,
      "expert_ids"_a,
      "path"_a,
      "stride"_a,
      "do_read"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none());
  // ---- [3] staging miss→hit + 完成回调时刻探针（诊断）----
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
      "source_layer"_a = -1,
      "forward_id"_a = -1,
      "priority"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none());
  m.def("prefetch_staging_ready_ids", &prefetch_staging_ready,
        "staging"_a, "expert_ids"_a, "layer"_a, "gen"_a, "path"_a,
        "stride"_a, "resident"_a, "cap"_a, "parallel"_a,
        "source_layer"_a = -1, "forward_id"_a = -1, "priority"_a = 0);
  m.def("prefetch_staging_take", &prefetch_staging_take,
        "layer"_a, "generation"_a = -1);
  m.def("prefetch_staging_consumed", &prefetch_staging_consumed,
        "layer"_a, "generation"_a);
  m.def("prefetch_staging_drain", &prefetch_staging_drain,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("prefetch_staging_finished", &prefetch_staging_finished,
        "layer"_a, "generation"_a);
  m.def("prefetch_staging_forget", &prefetch_staging_forget,
        "layer"_a, "generation"_a);
  m.def("prefetch_staging_wait_experts", &prefetch_staging_wait_experts,
        "forward_id"_a, "layer"_a, "expert_ids"_a,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("prefetch_staging_note_prejoin", &prefetch_staging_note_prejoin,
        "forward_id"_a, "layer"_a, "expert_ids"_a,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("prefetch_staging_finish_demand", &prefetch_staging_finish_demand,
        "forward_id"_a, "layer"_a);
  m.def("prefetch_staging_wait_stats", &prefetch_staging_wait_stats);
  m.def("prefetch_staging_wait_stats_reset", &prefetch_staging_wait_stats_reset);
  m.def("staging_hprof_enable", &staging_hprof_enable, "on"_a);
  m.def("staging_hprof_now", &staging_hprof_now);
  m.def("staging_hprof_get", &staging_hprof_get);
  // ---- [4] 段散写侧区缓存（zero-copy dual-source 默认路径）----
  m.def("prefetch_pool_sideregion", &prefetch_pool_sideregion,
        "pool_list"_a, "seg_nbytes"_a, "expert_ids"_a, "layer"_a, "path"_a, "stride"_a,
        "resident"_a, "spec_slots"_a, "base_row"_a, "gen"_a = 0,
        "source_layer"_a = -1, "forward_id"_a = -1, "priority"_a = 0,
        nb::kw_only(),
        "stream"_a = nb::none());
  m.def("prefetch_unified_ready_ids",
        [](const std::vector<mx::array>& pool_list,
           const std::vector<int>& seg_nbytes,
           const std::vector<int>& expert_ids, int layer,
           const std::string& path, int stride,
           const std::vector<int>& resident, int speculative_limit,
           int real_cap, int source_layer, int64_t forward_id) {
          std::vector<uint32_t> ids(expert_ids.begin(), expert_ids.end());
          prefetch_unified_ready(
              pool_list, seg_nbytes, ids.data(), ids.size(), layer, path,
              stride, resident, speculative_limit, real_cap, source_layer,
              forward_id);
        },
        "pool_list"_a, "seg_nbytes"_a, "expert_ids"_a, "layer"_a,
        "path"_a, "stride"_a, "resident"_a, "speculative_limit"_a,
        "real_cap"_a, "source_layer"_a, "forward_id"_a);
  m.def("sideregion_contents", &sideregion_contents, "layer"_a, "gen"_a = 0);
  m.def("sideregion_kv", &sideregion_kv, "layer"_a, "gen"_a = 0);
  m.def("sideregion_reset", &sideregion_reset);
  m.def("sideregion_prefetch_stats", &sideregion_prefetch_stats);
  m.def("sideregion_prefetch_reads_by_layer", &sideregion_prefetch_reads_by_layer);
  m.def("sideregion_prefetch_stats_reset", &sideregion_prefetch_stats_reset);
  m.def("prefetch_audit_stats", &prefetch_audit_stats);
  m.def("prefetch_audit_stats_reset", &prefetch_audit_stats_reset);
  m.def("sideregion_drain", &sideregion_drain,
        nb::call_guard<nb::gil_scoped_release>());   // 等待时释放 GIL
  m.def("sideregion_wait_target", &sideregion_wait_target,
        "forward_id"_a, "target_layer"_a,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("sideregion_wait_refinement", &sideregion_wait_refinement,
        "forward_id"_a, "target_layer"_a,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("sideregion_wait_experts", &sideregion_wait_experts,
        "forward_id"_a, "layer"_a, "gen"_a, "expert_ids"_a,
        nb::call_guard<nb::gil_scoped_release>());
  // ---- [5] Route 3 owned 池底座（C++ 拥有 buffer + 直写，删 MLX scatter）----
  m.def("pool_owned_zeros", &pool_owned_zeros, "shape"_a, "dtype"_a);
  m.def("pool_write_rows", &pool_write_rows, "pool_list"_a, "srcs_flat"_a, "slots"_a);
  m.def("pool_write_stacked", &pool_write_stacked, "pool_list"_a, "stacked_list"_a, "slots"_a);
  m.def("array_data_ptr", &array_data_ptr, "a"_a);
  // ---- [6] 方案B 真实区槽状态 C++ 接管 + demand_dual ----
  m.def("real_init", &real_init, "layer"_a, "cap"_a);
  m.def("real_pin", &real_pin, "layer"_a, "experts"_a, "cap"_a);
  m.def("real_pinned_contents", &real_pinned_contents, "layer"_a);
  m.def("real_region_contents", &real_region_contents, "layer"_a);
  m.def("real_verified_contents", &real_verified_contents, "layer"_a);
  m.def("real_region_count", &real_region_count, "layer"_a);
  m.def("real_should_predict", &real_should_predict,
        "layer"_a, "min_resident"_a, "cooldown"_a);
  m.def("real_reset", &real_reset);
  m.def("demand_deadline_snapshot", &demand_deadline_snapshot,
        "inds"_a, "layer"_a, "side_gen"_a, "use_side"_a = true);
  m.def("demand_dual", &demand_dual, "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "side_gen"_a, "path"_a, "stride"_a, "cap"_a, "lfu"_a, "decay_interval"_a,
        "forward_id"_a = -1, "sequence_length"_a = -1,
        "use_side"_a = true, "record_deadline"_a = true, nb::kw_only(),
        "stream"_a = nb::none());
  m.def("demand_dual_async", &demand_dual_async,
        "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "side_gen"_a, "path"_a, "stride"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a, "forward_id"_a = -1,
        "sequence_length"_a = -1, "use_side"_a = true,
        "wait_for_pending"_a = false,
        "wait_for_refinement"_a = false,
        "evaluator_submit"_a = false,
        nb::kw_only(), "stream"_a = nb::none());
  m.def("demand_dual_async_prefetch", &demand_dual_async_prefetch,
        "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "side_gen"_a, "path"_a, "stride"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a, "forward_id"_a, "sequence_length"_a,
        "use_side"_a, "wait_for_pending"_a, "wait_for_refinement"_a,
        "evaluator_submit"_a, "prefetch_ids"_a,
        "prefetch_pool_list"_a, "prefetch_seg_nbytes"_a,
        "prefetch_layer"_a, "prefetch_path"_a, "prefetch_stride"_a,
        "prefetch_cap"_a, "prefetch_spec_limit"_a,
        "prefetch_resident"_a, nb::kw_only(), "stream"_a = nb::none());
  m.def("demand_gpu_remap_only", &demand_gpu_remap_only,
        "inds"_a, "layer"_a, "side_gen"_a, "cap"_a,
        "use_side"_a = true, nb::kw_only(), "stream"_a = nb::none());
  m.def("demand_dual_split_async", &demand_dual_split_async,
        "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "side_gen"_a, "path"_a, "stride"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a, "forward_id"_a = -1,
        "sequence_length"_a = -1, "use_side"_a = true,
        "wait_for_pending"_a = false,
        "wait_for_refinement"_a = false,
        "evaluator_submit"_a = false,
        nb::kw_only(), "stream"_a = nb::none());
  m.def("demand_staged_split_async", &demand_staged_split_async,
        "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "path"_a, "stride"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a, "forward_id"_a,
        "sequence_length"_a, "evaluator_submit"_a, "spec_limit"_a,
        "staging_buffers"_a, "staging_generations"_a,
        nb::kw_only(), "stream"_a = nb::none());
  m.def("demand_dual_projection_split_async",
        &demand_dual_projection_split_async,
        "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "side_gen"_a, "path"_a, "stride"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a, "forward_id"_a = -1,
        "sequence_length"_a = -1, "use_side"_a = false,
        "wait_for_pending"_a = false,
        "wait_for_refinement"_a = false,
        "evaluator_submit"_a = false,
        nb::kw_only(), "stream"_a = nb::none());
  m.def("demand_async_stats", &demand_async_stats);
  m.def("demand_async_layer_stats", &demand_async_layer_stats);
  m.def("unified_prefetch_reads_by_layer", &unified_prefetch_reads_by_layer);
  m.def("demand_async_miss_histogram", &demand_async_miss_histogram);
  m.def("demand_async_seq_miss_histogram",
        &demand_async_seq_miss_histogram);
  m.def("demand_async_stats_reset", &demand_async_stats_reset);
  m.def("demand_async_check", &demand_async_check);
  m.def("demand_last_stats", &demand_last_stats);
  m.def("demand_staged_multi", &demand_staged_multi,
        "inds"_a, "pool_list"_a, "seg_nbytes"_a, "layer"_a,
        "path"_a, "stride"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a, "spec_limit"_a, "staging_list"_a,
        "staging_maps"_a, "forward_id"_a = -1,
        "sequence_length"_a = -1, nb::kw_only(), "stream"_a = nb::none());
  m.def("late_promote_staged", &late_promote_staged,
        "pool_list"_a, "seg_nbytes"_a, "layer"_a, "cap"_a,
        "spec_limit"_a, "staging"_a, "staging_map"_a);
  m.def("demand_promote_staged", &demand_promote_staged,
        "pool_list"_a, "seg_nbytes"_a, "layer"_a, "cap"_a,
        "spec_limit"_a, "staging"_a, "staging_map"_a, "actual_ids"_a);
  m.def("demand_deadline_stats", &demand_deadline_stats);
  m.def("demand_deadline_stats_reset", &demand_deadline_stats_reset);
  m.def("demand_prejoin_stats", &demand_prejoin_stats);
  m.def("demand_prejoin_stats_reset", &demand_prejoin_stats_reset);
  m.def("demand_timings", &demand_timings);
  m.def("demand_timing_enable", &demand_timing_enable, "on"_a);
  m.def("real_debug_place", &real_debug_place, "layer"_a, "experts_flat"_a, "cap"_a, "lfu"_a,
        "decay_interval"_a);
  // ---- [7] 自由后台读线程 ----
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
  // ---- 融合 MoE 计算核（slots 版，native_fused.cpp）----
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
