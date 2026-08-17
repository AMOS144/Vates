"""Real-model one-layer correctness probe or all-12 32K throughput run."""

from __future__ import annotations

import argparse
import json
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument(
        "--mode",
        choices=("one", "all", "baseline", "gdn", "dense-one", "dense-all"),
        default="one",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    from mlx_streaming.runtime.run_qwen_k3_sub10 import configure
    configure()
    os.environ["PREFILL_CHUNK"] = str(args.tokens)
    os.environ["K"] = "1"

    import mlx.core as mx
    from mlx_streaming.core.mem import snapshot
    from mlx_streaming.model_builder import build_streaming_model
    from mlx_streaming.mtp.generate import prefill_chunked

    mx.random.seed(args.seed)
    model, _, _ = build_streaming_model()
    ids = mx.random.randint(0, 10000, shape=(1, args.tokens)).astype(mx.int32)

    def run(fused, layers=None):
        os.environ["EXPERT_MAJOR_FUSED_ATTENTION"] = "1" if fused else "0"
        if layers is None:
            os.environ.pop("EXPERT_MAJOR_FUSED_ATTENTION_LAYERS", None)
        else:
            os.environ["EXPERT_MAJOR_FUSED_ATTENTION_LAYERS"] = layers
        mx.reset_peak_memory()
        started = time.perf_counter()
        logits, hidden = prefill_chunked(
            model, ids, model.make_cache(), chunk=args.tokens,
        )
        mx.eval(logits, hidden)
        elapsed = time.perf_counter() - started
        mem = snapshot()
        gdn_calls = sum(
            int(getattr(layer.linear_attn, "_expert_major_fused_gdn_calls", 0))
            for layer in model.model.layers if layer.is_linear
        )
        return logits, hidden, {
            "seconds": elapsed,
            "tok_s": args.tokens / elapsed,
            "active_gb": mem.mlx_active_bytes / 1e9,
            "peak_gb": mem.mlx_peak_bytes / 1e9,
            "fused_gdn_calls": gdn_calls,
        }

    if args.mode == "baseline":
        _, _, stats = run(False)
        print(json.dumps({"mode": "baseline", **stats}, indent=2))
        return
    if args.mode == "all":
        _, _, stats = run(True)
        print(json.dumps({"mode": "all_12_fused", **stats}, indent=2))
        return
    if args.mode == "dense-all":
        os.environ["EXPERT_MAJOR_DENSE_STEEL_ATTENTION"] = "1"
        _, _, stats = run(False)
        print(json.dumps({"mode": "all_12_dense_steel", **stats}, indent=2))
        return

    if args.mode == "gdn":
        os.environ["EXPERT_MAJOR_FUSED_GDN"] = "0"
        baseline_logits, baseline_hidden, baseline = run(False)
        os.environ["EXPERT_MAJOR_FUSED_GDN"] = "1"
        fused_logits, fused_hidden, fused = run(False)
    elif args.mode == "dense-one":
        os.environ["EXPERT_MAJOR_DENSE_STEEL_ATTENTION"] = "0"
        baseline_logits, baseline_hidden, baseline = run(False)
        os.environ["EXPERT_MAJOR_DENSE_STEEL_ATTENTION"] = "1"
        fused_logits, fused_hidden, fused = run(False, "3")
    else:
        baseline_logits, baseline_hidden, baseline = run(False)
        fused_logits, fused_hidden, fused = run(True, "3")

    logit_diff = mx.abs(
        baseline_logits.astype(mx.float32) - fused_logits.astype(mx.float32)
    )
    hidden_diff = mx.abs(
        baseline_hidden.astype(mx.float32) - fused_hidden.astype(mx.float32)
    )
    baseline_logits32 = baseline_logits.astype(mx.float32)
    fused_logits32 = fused_logits.astype(mx.float32)
    baseline_hidden32 = baseline_hidden.astype(mx.float32)
    fused_hidden32 = fused_hidden.astype(mx.float32)
    logit_cos = mx.sum(baseline_logits32 * fused_logits32) / (
        mx.linalg.norm(baseline_logits32) * mx.linalg.norm(fused_logits32)
    )
    hidden_cos = mx.sum(baseline_hidden32 * fused_hidden32) / (
        mx.linalg.norm(baseline_hidden32) * mx.linalg.norm(fused_hidden32)
    )
    mx.eval(logit_diff, hidden_diff, logit_cos, hidden_cos)
    print(json.dumps({
        "mode": (
            "all_36_blocked_gdn" if args.mode == "gdn"
            else (
                "one_full_attention_layer_3_dense_steel"
                if args.mode == "dense-one"
                else "one_full_attention_layer_3"
            )
        ),
        "baseline": baseline,
        "fused": fused,
        "hidden_max_abs": float(mx.max(hidden_diff).item()),
        "hidden_mean_abs": float(mx.mean(hidden_diff).item()),
        "hidden_cosine": float(hidden_cos.item()),
        "logits_max_abs": float(mx.max(logit_diff).item()),
        "logits_mean_abs": float(mx.mean(logit_diff).item()),
        "logits_cosine": float(logit_cos.item()),
        "argmax_equal": bool(mx.array_equal(
            mx.argmax(baseline_logits, axis=-1),
            mx.argmax(fused_logits, axis=-1),
        ).item()),
    }, indent=2))


if __name__ == "__main__":
    main()
