"""全模型 RTN-from-FP16:直接把 FP16 原始专家权重量化成 2-bit g128(单次量化,无校准)。
逐分片扫描,排除 mtp.* 层,产 per-expert safetensors,与现有流式加载兼容。

环境变量:FP16_DIR / OUT_DIR / META_FROM(借 _split_meta.json 的 g128 2-bit dims)/ BITS / GROUP
"""
import json
import os
from collections import defaultdict

import mlx.core as mx

FP16_DIR = os.environ.get("FP16_DIR", "/tmp/qwen3_next_80b_fp16")
OUT_DIR = os.environ.get("OUT_DIR", os.path.abspath("models/qwen3_next_experts_fp16rtn_g128"))
META_FROM = os.environ.get("META_FROM", os.path.abspath("models/qwen3_next_experts_2bit_g128"))
BITS = int(os.environ.get("BITS", "2"))
GROUP = int(os.environ.get("GROUP", "128"))
PROJS = ["gate_proj", "up_proj", "down_proj"]


def main():
    import time
    os.makedirs(OUT_DIR, exist_ok=True)
    idx = json.load(open(os.path.join(FP16_DIR, "model.safetensors.index.json")))["weight_map"]
    shard_keys = defaultdict(list)
    n_exp = set()
    for k, shard in idx.items():
        if not k.startswith("model.layers.") or ".mlp.experts." not in k:
            continue
        p = k.split(".")
        L, E, proj = int(p[2]), int(p[5]), p[6]
        if proj in PROJS:
            shard_keys[shard].append((L, E, proj, k))
            n_exp.add((L, E))

    pending = defaultdict(dict)
    done = 0
    t0 = time.perf_counter()
    for si, shard in enumerate(sorted(shard_keys)):
        w = mx.load(os.path.join(FP16_DIR, shard))
        for (L, E, proj, k) in shard_keys[shard]:
            t = w[k]
            mx.eval(t)                       # 物化,防 del w 后失效(跨分片专家)
            pending[(L, E)][proj] = t
        for (L, E) in list(pending):
            if len(pending[(L, E)]) == 3:
                out = {}
                for proj in PROJS:
                    q, s, b = mx.quantize(pending[(L, E)][proj], group_size=GROUP, bits=BITS)
                    out[f"{proj}.weight"], out[f"{proj}.scales"], out[f"{proj}.biases"] = q, s, b
                mx.save_safetensors(
                    os.path.join(OUT_DIR, f"layer{L:02d}_expert{E:03d}.safetensors"), out)
                del pending[(L, E)]
                done += 1
        del w
        mx.clear_cache()
        print(f"  shard {si+1}/{len(shard_keys)} 完成,已存专家 {done}/{len(n_exp)} "
              f"({round(time.perf_counter()-t0)}s)", flush=True)

    # meta:借 g128 的 dims(2-bit g128)
    meta = json.load(open(os.path.join(META_FROM, "_split_meta.json")))
    meta["out_dir"] = OUT_DIR
    meta["requantized_from"] = {"dir": FP16_DIR, "bits": "bf16", "single_pass_from_fp16": True}
    json.dump(meta, open(os.path.join(OUT_DIR, "_split_meta.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"完成: 专家 {done} 个, 残留未完成 {len(pending)} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
