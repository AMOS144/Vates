"""Measure the sole production Expert-major prefill implementation."""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    from mlx_streaming.runtime.run_qwen_k3_sub10 import configure
    configure()

    import mlx.core as mx
    from mlx_streaming.core.mem import snapshot
    from mlx_streaming.model_builder import build_streaming_model
    from mlx_streaming.mtp.generate import prefill_chunked

    mx.random.seed(args.seed)
    model, _, _ = build_streaming_model()
    ids = mx.random.randint(0, 10000, shape=(1, args.tokens)).astype(mx.int32)
    mx.reset_peak_memory()
    started = time.perf_counter()
    logits, hidden = prefill_chunked(model, ids, model.make_cache())
    mx.eval(logits, hidden)
    elapsed = time.perf_counter() - started
    mem = snapshot()
    print(json.dumps({
        "mode": "expert_major_optimal",
        "tokens": args.tokens,
        "seconds": elapsed,
        "tok_s": args.tokens / elapsed,
        "active_gb": mem.mlx_active_bytes / 1e9,
        "peak_gb": mem.mlx_peak_bytes / 1e9,
    }, indent=2))


if __name__ == "__main__":
    main()
