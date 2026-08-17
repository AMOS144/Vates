"""Reproducible Qwen3-Next K=3 streaming profile for interactive use."""
import os
import sys
from pathlib import Path


FINAL_DEFAULTS = {
    "STREAM_BLOB_LOADER": "1",
    # Expert-major deliberately streams the 48-layer expert corpus once per
    # prompt.  Do not disguise that I/O with tens of GB of macOS page cache or
    # let the cache inflate machine-wide memory outside MLX accounting.
    "STREAM_BLOB_NOCACHE": "1",
    # A 32K superblock is still layer-major Expert-major (not the legacy tiny
    # token chunk).  At 256K it bounds activations while the compact K4/V3 KV
    # cache carries exact context across eight blocks.  Re-reading experts is
    # negligible beside exact long-context attention and avoids a >10 GB peak.
    "PREFILL_CHUNK": "32768",
    "EXPERT_MAJOR_GROUP_EXPERTS": "128",
    "EXPERT_MAJOR_MAX_ASSIGNMENTS": "32768",
    "EXPERT_MAJOR_TRANSIENT_BANK": "1",
    "EXPERT_MAJOR_LAYER_BARRIER": "1",
    "EXPERT_MAJOR_CLEAR_CACHE": "1",
    "EXPERT_MAJOR_ATTENTION_TILE": "2048",
    "EXPERT_MAJOR_GDN_TILE": "512",
    # Register-resident blocked recurrence removes the stock kernel's
    # redundant q/k reads.  32K: 416.35 -> 464.44 tok/s at the same 5.14 GB
    # peak; a 1024-token prefill + 32 greedy tokens matched exactly.
    "EXPERT_MAJOR_FUSED_GDN": "1",
    # These completed experimental kernels remain opt-in because real-model
    # A/B did not meet the production speed/accuracy gate.
    "EXPERT_MAJOR_FUSED_ATTENTION": "0",
    "EXPERT_MAJOR_DENSE_STEEL_ATTENTION": "0",
    "EXPERT_MAJOR_METAL_GEMM": "0",
    "EXPERT_MAJOR_DOUBLE_BUFFER": "0",
    # 16M score elements permits a 512-query tile at 32K and remains bounded
    # at longer contexts.  On the 80B model this improved 32K prefill from
    # 328.64 to 363.67 tok/s while peak memory moved only 4.80 -> 4.97 GB.
    "EXPERT_MAJOR_ATTENTION_SCORE_BUDGET": "16777216",
    "KV_QUANT": "1",
    "KV_K_BITS": "4",
    "KV_V_BITS": "3",
    "KV_GROUP_SIZE": "64",
    "KV_ROTATE": "1",
    "EXPERT_SLOTS": "152",
    "EXPERT_POOL_PROFILE": (
        "benchmarks/results/qwen_k3_prefetch_wait_rebalanced_same10g.json"
    ),
    "LAYER0_SLOTS": "216",
    "ZEROCOPY_DUAL_SOURCE": "1",
    "POOL_SPEC_SLOTS": "0",
    "POOL_ADMISSION_SLOTS": "32",
    "PREFETCH_DIRECT_SLOTS": "1",
    "PREFETCH_ISOLATED_SIDE": "1",
    "SIDEREGION_ROW_LEASES": "1",
    "DEMAND_ASYNC": "0",
    # Prompt ingestion has much wider route unions and stays synchronous. The
    # CLI enables async demand only at the exact prefill/decode boundary.
    "SPEC_SPLIT_DEMAND_AFTER_PREFILL": "1",
    "DEMAND_ASYNC_PY_SUBMIT": "0",
    "DEMAND_WORKERS": "16",
    "PREFETCH_LOW_WORKERS": "2",
    # Batch true demand reads into one preadv call; this halves the remaining
    # fallback wait in the fresh-process 40 tok/s profile.
    "DEMAND_PREADV": "1",
    "DEMAND_PREADV_GROUPS": "1",
    "MTP_BITS": "4",
    "MTP_GROUP_SIZE": "64",
    "MTP_STREAM_EXPERTS": "1",
    # 128 slots measured 39.81 tok/s; 256 is the smallest point that stays
    # above 40 tok/s (40.89) while keeping active memory near 10 GiB.
    "MTP_EXPERT_SLOTS": "256",
    "MTP_EXPERT_DIR": "models/qn_mtp_experts_4bit_g64",
    "MTP_ADAPTIVE_DEPTH": "1",
    "MTP_CONF_TAU": "0.3",
    "MTP_DEPTH_MAX": "3",
    "NATIVE_FUSED_PREFETCH": "1",
    "NATIVE_NO_SUBMIT": "0",
    # The logical rerank remains 15/26-wide for the recall audit, while only
    # its highest-ranked missing rows are issued to SSD at each target window.
    # Three rows is the measured throughput knee for the 10.6 GiB profile.
    "PREFETCH_PHYSICAL_READ_BUDGET": "3",
    "PREFETCH_PHYSICAL_READ_BUDGET_PROFILE": "2:1",
    "PREFETCH_PREDICT_GATE_BITS": "4",
    "PREFETCH_ASYNC_PREDICT": "1",
    # Fixed layer selection is faster than rebuilding/skipping predictors via
    # the adaptive state machine (38.24 vs 37.98 tok/s before MTP tuning).
    "PREFETCH_ADAPTIVE": "0",
    "PREFETCH_ADAPTIVE_FILL": "0.85",
    "PREFETCH_ADAPTIVE_COOLDOWN": "32",
    "PREFETCH_PROGRESSIVE": "0",
    "CROSS_LAYER_AHEAD_PROFILE": "5-6:2",
    "POOL_LAYER_CAP_OVERRIDES": "2:216,5:152",
    "PREFETCH_RERANK": "noisy_or",
    "PREFETCH_RERANK_ROUTER_PATHS": "",
    "PREFETCH_RERANK_CANDIDATE_WIDTH": "64",
    "PREFETCH_RERANK_MAX_WIDTH": "26",
    "PREFETCH_RERANK_BACKFILL_EXTRA": "4",
    "PREFETCH_RERANK_WIDTH_POLICY": "predicted_route_union",
    "PREFETCH_RERANK_RANKING_POLICY": "topk_union_fast",
    # Keep enough of the raw top-64 candidate union to retain at least 95%
    # of its route coverage.  A margin of 4 under-selected on L15/L22/L23;
    # margin 0 passed the 1.5x width and 95% relative-recall audit on all three.
    "PREFETCH_RERANK_UNION_MARGIN": "0",
    # These eight layers were already cache-stable under the 10 GiB profile.
    # Omitting their shadow gates saves more compute than their prefetches hide.
    "PREFETCH_TARGET_LAYERS": (
        "1-7,10-16,18-21,23,25-26,28-36,38-41,43-47"
    ),
    "CROSS_LAYER_AHEAD_LO": "1",
    "CROSS_LAYER_AHEAD_HI": "2",
    "CROSS_LAYER_CUTOFF": "6",
    "PREFETCH_RERANK_RESIDUAL_SCALE_OVERRIDES": "3-47:0.5",
    "DEMAND_SPARSE_MISS_BUDGET": "17",
    "DEMAND_SPARSE_PARTITION": "1",
    "K": "3",
}


def configure() -> None:
    for name, value in FINAL_DEFAULTS.items():
        os.environ.setdefault(name, value)
    # Worktrees do not duplicate the large model directory.  When the caller
    # supplies an absolute target MODEL, resolve the default MTP expert shard
    # beside it instead of relative to the worktree's current directory.
    mtp_dir = Path(os.environ["MTP_EXPERT_DIR"])
    model = os.environ.get("MODEL")
    if not mtp_dir.is_absolute() and model and not mtp_dir.exists():
        sibling = Path(model).expanduser().resolve().parent / mtp_dir.name
        if sibling.exists():
            os.environ["MTP_EXPERT_DIR"] = str(sibling)


def _chat_argv(argv: list[str]) -> list[str] | None:
    if not argv or argv[0] not in ("chat", "--chat"):
        return None
    return [
        "chat",
        "--expert-slots", FINAL_DEFAULTS["EXPERT_SLOTS"],
        "--spec-slots", FINAL_DEFAULTS["POOL_SPEC_SLOTS"],
        *argv[1:],
    ]


def main() -> None:
    configure()
    chat_argv = _chat_argv(sys.argv[1:])
    if chat_argv is not None:
        from mlx_streaming.cli import main as chat
        raise SystemExit(chat(chat_argv))
    # Configuration modules read several constants while importing, so the
    # final profile must be installed before importing the generic runner.
    from mlx_streaming.runtime.run_mtp_spec import main as run
    run()


if __name__ == "__main__":
    main()
