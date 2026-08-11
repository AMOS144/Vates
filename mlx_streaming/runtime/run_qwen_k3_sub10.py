"""Reproducible Qwen3-Next K=3 streaming profile: ~42 tok/s below 10 GiB."""
import os
import sys


FINAL_DEFAULTS = {
    "STREAM_BLOB_LOADER": "1",
    "EXPERT_SLOTS": "152",
    "EXPERT_POOL_PROFILE": (
        "benchmarks/results/qwen_k3_nosubmit_hot8_profile_tight8.json"
    ),
    "LAYER0_SLOTS": "216",
    "ZEROCOPY_DUAL_SOURCE": "1",
    "POOL_SPEC_SLOTS": "0",
    "POOL_ADMISSION_SLOTS": "32",
    "PREFETCH_DIRECT_SLOTS": "1",
    "DEMAND_ASYNC": "1",
    "DEMAND_WORKERS": "8",
    "MTP_BITS": "2",
    "MTP_GROUP_SIZE": "64",
    "MTP_STREAM_EXPERTS": "1",
    "MTP_EXPERT_SLOTS": "64",
    "MTP_EXPERT_DIR": "models/qn_mtp_experts_2bit_g64",
    "MTP_ADAPTIVE_DEPTH": "1",
    "MTP_CONF_TAU": "0.4",
    "MTP_DEPTH_MAX": "3",
    "NATIVE_FUSED_PREFETCH": "1",
    "NATIVE_NO_SUBMIT": "1",
    "PREFETCH_ADAPTIVE": "1",
    "PREFETCH_ADAPTIVE_FILL": "0.85",
    "PREFETCH_ADAPTIVE_COOLDOWN": "8",
    "PREFETCH_PROGRESSIVE": "1",
    "PREFETCH_PROGRESSIVE_MODE": "k3",
    "PREFETCH_PROGRESSIVE_TARGET_LAYERS": "1,2,5,6,7,17,18,19,30",
    "PREFETCH_PROGRESSIVE_LATE_LAYERS": "5,6,17,18,19,30",
    "PREFETCH_PROGRESSIVE_TARGET_CACHE_LAYERS": "1,2,5,6,7,17,18,19,30",
    "PREFETCH_RERANK": "noisy_or",
    "PREFETCH_RERANK_ROUTER_ALLOW_OVERRIDE": "1",
    "PREFETCH_RERANK_ROUTER_PATHS": (
        "models/qn_forecast_router_fixed64_ridge_l01_12.safetensors,"
        "models/qn_forecast_router_fixed64_ridge_l13_47.safetensors,"
        "models/qn_forecast_router_cap32_k3_diverse_ridge_l01_02.safetensors"
    ),
    "PREFETCH_RERANK_CANDIDATE_WIDTH": "64",
    "PREFETCH_PROGRESSIVE_MAX_WIDTH": "26",
    "K": "3",
}


def configure() -> None:
    for name, value in FINAL_DEFAULTS.items():
        os.environ.setdefault(name, value)


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
