"""Validate repeatability of real Qwen3-Next 80B prefill at a 32K boundary.

This is intentionally an opt-in hardware validation rather than a unit test:
it executes the fixed production path twice and compares the last prompt state
plus a fixed teacher-forced decode suffix.  ``--reference-chunk`` can be set to
a different value for non-gating batch-shape sensitivity analysis.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np


def _metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    a = actual.astype(np.float64).reshape(-1)
    b = reference.astype(np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return {
        "cosine": float(np.dot(a, b) / denom),
        "max_abs": float(np.max(np.abs(a - b))),
        "mean_abs": float(np.mean(np.abs(a - b))),
    }


def _ids(tokenizer, tokens: int, seed: int, prompt_file: str | None):
    import mlx.core as mx

    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        encoded = tokenizer.encode(text)
        if not encoded:
            raise ValueError("prompt file tokenized to an empty sequence")
        repeats = (tokens + len(encoded) - 1) // len(encoded)
        values = (encoded * repeats)[:tokens]
        return mx.array([values], dtype=mx.int32), "prompt_file_repeated"
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 10000, size=tokens, dtype=np.int32)
    return mx.array(values.reshape(1, -1)), "deterministic_ids"


def _run(model, ids, chunk: int, suffix: list[int], *, token_major=False):
    import mlx.core as mx
    from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked

    cache = model.make_cache()
    started = time.perf_counter()
    if token_major:
        logits, hidden = forward_with_hidden(
            model, ids, cache, last_token_only=True,
        )
    else:
        logits, hidden = prefill_chunked(model, ids, cache, chunk=chunk)
    mx.eval(logits, hidden)
    prompt_seconds = time.perf_counter() - started
    logits_rows = [np.asarray(logits[:, -1, :].astype(mx.float32)).copy()]
    hidden_rows = [np.asarray(hidden[:, -1, :].astype(mx.float32)).copy()]
    argmax = [int(mx.argmax(logits[:, -1, :]))]

    for token in suffix:
        logits, hidden = forward_with_hidden(
            model, mx.array([[token]], dtype=mx.int32), cache,
        )
        mx.eval(logits, hidden)
        logits_rows.append(
            np.asarray(logits[:, -1, :].astype(mx.float32)).copy()
        )
        hidden_rows.append(
            np.asarray(hidden[:, -1, :].astype(mx.float32)).copy()
        )
        argmax.append(int(mx.argmax(logits[:, -1, :])))

    offsets = [
        int(item.offset) for item in cache
        if hasattr(item, "offset")
    ]
    result = {
        "chunk": chunk,
        "prompt_seconds": prompt_seconds,
        "prompt_tok_s": int(ids.shape[1]) / prompt_seconds,
        "offset_min": min(offsets) if offsets else None,
        "offset_max": max(offsets) if offsets else None,
        "argmax": argmax,
        "logits": np.concatenate(logits_rows, axis=0),
        "hidden": np.concatenate(hidden_rows, axis=0),
    }
    del cache, logits, hidden
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32769)
    parser.add_argument("--reference-chunk", type=int, default=32767)
    parser.add_argument("--candidate-chunk", type=int, default=32767)
    parser.add_argument("--suffix-tokens", type=int, default=63)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--prompt-file")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--min-token-match", type=float, default=0.95)
    parser.add_argument("--reference-token-major", action="store_true")
    args = parser.parse_args()

    from mlx_streaming.runtime.run_qwen_k3_sub10 import configure
    configure()

    import mlx.core as mx
    from mlx_streaming.core.mem import snapshot
    from mlx_streaming.model_builder import build_streaming_model

    model, tokenizer, _ = build_streaming_model()
    ids, source = _ids(tokenizer, args.tokens, args.seed, args.prompt_file)
    suffix_rng = np.random.default_rng(args.seed + 1)
    suffix = suffix_rng.integers(
        0, 10000, size=args.suffix_tokens, dtype=np.int32,
    ).tolist()

    reference = _run(
        model, ids, args.reference_chunk, suffix,
        token_major=args.reference_token_major,
    )
    candidate = _run(model, ids, args.candidate_chunk, suffix)
    logits_metrics = _metrics(candidate["logits"], reference["logits"])
    hidden_metrics = _metrics(candidate["hidden"], reference["hidden"])
    expected_offset = args.tokens + args.suffix_tokens
    token_matches = sum(
        actual == expected
        for actual, expected in zip(
            candidate["argmax"], reference["argmax"], strict=True,
        )
    )
    token_match_rate = token_matches / len(reference["argmax"])
    passed = (
        token_match_rate >= args.min_token_match
        and logits_metrics["cosine"] >= args.min_cosine
        and hidden_metrics["cosine"] >= args.min_cosine
        and reference["offset_min"] == expected_offset
        and reference["offset_max"] == expected_offset
        and candidate["offset_min"] == expected_offset
        and candidate["offset_max"] == expected_offset
    )
    mem = snapshot()
    output = {
        "passed": passed,
        "model": "qwen3_next_80b_4bit",
        "input_source": source,
        "reference_token_major": args.reference_token_major,
        "tokens": args.tokens,
        "suffix_tokens": args.suffix_tokens,
        "expected_offset": expected_offset,
        "reference": {
            key: value for key, value in reference.items()
            if key not in ("logits", "hidden")
        },
        "candidate": {
            key: value for key, value in candidate.items()
            if key not in ("logits", "hidden")
        },
        "argmax_exact": candidate["argmax"] == reference["argmax"],
        "token_matches": token_matches,
        "token_match_rate": token_match_rate,
        "min_token_match": args.min_token_match,
        "logits": logits_metrics,
        "hidden": hidden_metrics,
        "active_gb": mem.mlx_active_bytes / 1e9,
        "peak_gb": mem.mlx_peak_bytes / 1e9,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
