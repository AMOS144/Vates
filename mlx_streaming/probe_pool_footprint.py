"""测每层专家池的真实高水位驻留数:看 uniform 256/层是否过度配置。

uniform capacity 强迫每个被触碰的层整块预分配 (capacity,*)。若各层实际工作集不均匀
(有些层远 < 256),则按层自适应 / grow-on-demand 可在不掉命中率(吞吐)的前提下省内存。

输出:每层高水位、总驻留专家数、按 0.9375MB/专家折算的池 GB、以及若干"自适应容量上限"
档位下的总内存(无损,因为命中率只取决于工作集而非预留)。
"""
import json
import os

import mlx.core as mx

from mlx_streaming.mtp_generate import MTPDrafter, mtp_generate
from mlx_streaming.qwen3_next_mtp import load_mtp
from mlx_streaming.validate_mtp import _build_streaming_model
from mlx_lm.models.qwen3_next import ModelArgs

MB_PER_EXPERT = 0.9375
QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "2"))
# 产出每层池预算 profile：budget = ceil(高水位 * margin)，上限 capacity。
# margin 给未见过的稍长/不同 prompt 留冗余(冗余不够时会优雅回退 stack，不会崩)。
PROFILE_OUT = os.environ.get("EXPERT_POOL_PROFILE_OUT", "")
MARGIN = float(os.environ.get("EXPERT_POOL_MARGIN", "1.15"))


def main():
    model, tok, store = _build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    mtp_generate(model, drafter, tok, mx.array([tok.encode(PROMPT)]),
                 MAXTOK, K=K, ids_mode=True, profile=False)

    slot_of = store._resident._slot_of
    # 每层高水位(layer_idx -> 驻留专家数)
    per_layer = {int(layer): len(v) for layer, v in slot_of.items()}
    counts = sorted(per_layer.values(), reverse=True)
    n_layers = len(counts)
    total = sum(counts)
    cap = store.capacity

    # 写 profile：每层预算 = min(ceil(高水位*margin), capacity)
    import math
    if PROFILE_OUT:
        profile = {str(l): min(int(math.ceil(c * MARGIN)), cap)
                   for l, c in per_layer.items()}
        with open(PROFILE_OUT, "w") as f:
            json.dump({"capacity": cap, "margin": MARGIN, "layer_caps": profile}, f,
                      ensure_ascii=False, indent=2)
        prof_experts = sum(profile.values())
        prof_gb = round(prof_experts * MB_PER_EXPERT / 1024, 2)

    # 无损自适应:每层只分配 min(cap_limit, 该层高水位)。命中率不变(工作集没变)。
    caps = {}
    for lim in (256, 224, 192, 160, 128, 96):
        alloc = sum(min(lim, c) for c in counts)
        caps[f"cap<= {lim}"] = {
            "experts": alloc, "pool_gb": round(alloc * MB_PER_EXPERT / 1024, 2)}

    print(json.dumps({
        "capacity_uniform": cap,
        "moe_layers_touched": n_layers,
        "uniform_pool_experts": cap * n_layers,
        "uniform_pool_gb": round(cap * n_layers * MB_PER_EXPERT / 1024, 2),
        "actual_highwater_total_experts": total,
        "actual_pool_gb_grow_on_demand": round(total * MB_PER_EXPERT / 1024, 2),
        "per_layer_highwater_sorted": counts,
        "highwater_max": counts[0] if counts else 0,
        "highwater_min": counts[-1] if counts else 0,
        "highwater_mean": round(total / max(n_layers, 1), 1),
        "adaptive_cap_budgets": caps,
        "profile_out": PROFILE_OUT or None,
        "profile_margin": MARGIN if PROFILE_OUT else None,
        "profile_pool_gb": (prof_gb if PROFILE_OUT else None),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
