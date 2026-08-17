"""Compare current tiled K4/V3 attention with block-online reference."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx

from mlx_streaming.core.attention.expert_major import (
    _fused_asymmetric_attention,
    _streaming_asymmetric_attention,
    _tiled_asymmetric_attention,
)


def _timed(fn, warmup, repeats):
    for _ in range(warmup):
        value = fn()
        mx.eval(value)
    samples = []
    value = None
    for _ in range(repeats):
        started = time.perf_counter()
        value = fn()
        mx.eval(value)
        samples.append((time.perf_counter() - started) * 1000.0)
    return value, min(samples), sum(samples) / len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--query-tile", type=int, default=512)
    parser.add_argument("--key-tile", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    mx.random.seed(args.seed)
    records = []
    for length in (int(v) for v in args.lengths.split(",")):
        q = mx.random.normal((1, 16, length, 256)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, length, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, length, 256)).astype(mx.bfloat16)
        qk = mx.quantize(k, group_size=64, bits=4)
        qv = mx.quantize(v, group_size=64, bits=3)
        mx.eval(q, *qk, *qv)
        baseline, base_min, base_mean = _timed(
            lambda: _tiled_asymmetric_attention(
                q, qk, qv, scale=256 ** -0.5, mask="causal",
                group_size=64, k_bits=4, v_bits=3,
            ), args.warmup, args.repeats,
        )
        online, online_min, online_mean = _timed(
            lambda: _streaming_asymmetric_attention(
                q, qk, qv, scale=256 ** -0.5, mask="causal",
                group_size=64, k_bits=4, v_bits=3,
                query_tile=args.query_tile, key_tile=args.key_tile,
            ), args.warmup, args.repeats,
        )
        native = native_min = native_mean = None
        if args.native:
            native, native_min, native_mean = _timed(
                lambda: _fused_asymmetric_attention(
                    q, qk, qv, scale=256 ** -0.5, mask="causal",
                    group_size=64, k_bits=4, v_bits=3,
                ), args.warmup, args.repeats,
            )
        base32 = baseline.astype(mx.float32)
        online32 = online.astype(mx.float32)
        diff = mx.abs(base32 - online32)
        cosine = mx.sum(base32 * online32) / (
            mx.linalg.norm(base32) * mx.linalg.norm(online32)
        )
        mx.eval(diff, cosine)
        record = {
            "length": length,
            "baseline_min_ms": base_min,
            "baseline_mean_ms": base_mean,
            "online_min_ms": online_min,
            "online_mean_ms": online_mean,
            "speedup": base_min / online_min,
            "max_abs": float(mx.max(diff).item()),
            "mean_abs": float(mx.mean(diff).item()),
            "cosine": float(cosine.item()),
        }
        if native is not None:
            native32 = native.astype(mx.float32)
            native_diff = mx.abs(base32 - native32)
            native_cos = mx.sum(base32 * native32) / (
                mx.linalg.norm(base32) * mx.linalg.norm(native32)
            )
            mx.eval(native_diff, native_cos)
            record.update({
                "native_min_ms": native_min,
                "native_mean_ms": native_mean,
                "native_speedup": base_min / native_min,
                "native_max_abs": float(mx.max(native_diff).item()),
                "native_mean_abs": float(mx.mean(native_diff).item()),
                "native_cosine": float(native_cos.item()),
            })
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    print(json.dumps({"results": records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
