"""真实 80B MTP 自投机基准 + 与非投机贪婪逐 token 一致性校验。

环境变量:K / MAXTOK / PROMPT / QN_CONFIG / MTP_OUT(其余主模型路径见 validate_mtp)。
"""
import json
import os
import time

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mem import snapshot, reset_peak
from mlx_streaming.mtp_generate import MTPDrafter, forward_with_hidden, mtp_generate
from mlx_streaming.qwen3_next_mtp import load_mtp
from mlx_streaming.validate_mtp import _build_streaming_model

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "3"))


def _baseline_greedy(model, tok, prompt, n):
    cache = model.make_cache()
    ids = mx.array([tok.encode(prompt)])
    out = []
    t0 = time.perf_counter()
    for _ in range(n):
        logits, _ = forward_with_hidden(model, ids, cache)
        nxt = int(mx.argmax(logits[:, -1, :]))
        out.append(nxt)
        ids = mx.array([[nxt]])
        mx.eval(ids)
    return out, round(n / (time.perf_counter() - t0), 2)


def main():
    reset_peak()
    model, tok, store = _build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens          # 共享主模型 embedding
    drafter = MTPDrafter(mtp, model.lm_head)

    # warmup:跑一遍让专家 LRU 热起来,避免冷启动污染对比
    _baseline_greedy(model, tok, PROMPT, 8)

    store.reset_stats()
    base, base_tps = _baseline_greedy(model, tok, PROMPT, MAXTOK)
    base_miss, base_hit = store.misses, store.hits

    store.reset_stats()
    ids, stats = mtp_generate(model, drafter, tok,
                              mx.array([tok.encode(PROMPT)]),
                              MAXTOK, K=K, ids_mode=True, profile=True)
    spec_miss, spec_hit = store.misses, store.hits
    after = snapshot()
    spec_tps = round(stats["tokens"] / stats["wall_s"], 2)
    proj = stats.get("proj_no_replay_tps", 0.0)
    result = {
        "K": K,
        "max_tokens": MAXTOK,
        "exact_match": ids == base,
        "n_mismatch": sum(1 for a, b in zip(ids, base) if a != b),
        "avg_accept_len": stats["avg_accept_len"],
        "steps": stats["steps"],
        "verify_mode": stats.get("verify_mode"),
        "direct_commits": stats.get("direct_commits"),
        "fallback_replays": stats.get("fallback_replays"),
        "replayed_tokens": stats.get("replayed_tokens"),
        "spec_tok_per_s": spec_tps,
        "baseline_tok_per_s": base_tps,
        "speedup": round(spec_tps / max(base_tps, 1e-6), 2),
        # 分段计时与「重放免费」投机上限
        "t_draft_s": stats.get("t_draft_s"),
        "t_snap_s": stats.get("t_snap_s"),
        "t_verify_s": stats.get("t_verify_s"),
        "t_replay_s": stats.get("t_replay_s"),
        "proj_no_replay_tps": proj,
        "proj_no_replay_speedup": round(proj / max(base_tps, 1e-6), 2),
        "baseline_disk_loads": base_miss,
        "baseline_hit_rate": round(base_hit / max(base_hit + base_miss, 1), 3),
        "spec_disk_loads": spec_miss,
        "spec_hit_rate": round(spec_hit / max(spec_hit + spec_miss, 1), 3),
        "disk_load_ratio": round(spec_miss / max(base_miss, 1), 2),
        "mlx_peak_gb": round(after.mlx_peak_bytes / 1e9, 2),
        "rss_gb": round(after.rss_bytes / 1e9, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
