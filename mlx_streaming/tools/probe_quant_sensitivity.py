"""de-risk 探针:静态量化误差敏感度(临时)。proxy=2-bit 相对重构误差,按 proj/层 排序。

对采样专家:把 4-bit 源反量化得 W,再按目标(2-bit/g128)重量化→还原得 W_hat,
测相对 L2 误差 ||W-W_hat||/||W||。误差越大 → 2-bit 越伤 → 应多分 bit。
(误差是 PPL 敏感度的廉价代理:标准做法,非精确,但足以指导 bit 分配。)

环境变量:SRC_4BIT(默认 /tmp/qwen3_next_experts)/ DST_BITS(默认 2)/ DST_GROUP(默认 128)
          / EXPERTS_PER_LAYER(每层采样专家数,默认 16)
"""
import json
import os
import statistics as st

import mlx.core as mx

SRC = os.environ.get("SRC_4BIT", "/tmp/qwen3_next_experts")
DST_BITS = int(os.environ.get("DST_BITS", "2"))
DST_GROUP = int(os.environ.get("DST_GROUP", "128"))
EPL = int(os.environ.get("EXPERTS_PER_LAYER", "16"))
PROJS = ["gate_proj", "up_proj", "down_proj"]


def _rel_err(src_path, src_bits, src_group):
    out = {}
    w = mx.load(src_path)
    for p in PROJS:
        W = mx.dequantize(w[f"{p}.weight"], w[f"{p}.scales"], w[f"{p}.biases"],
                          group_size=src_group, bits=src_bits)
        nwq, ns, nb = mx.quantize(W, group_size=DST_GROUP, bits=DST_BITS)
        Wh = mx.dequantize(nwq, ns, nb, group_size=DST_GROUP, bits=DST_BITS)
        err = float(mx.linalg.norm(W - Wh) / (mx.linalg.norm(W) + 1e-9))
        out[p] = err
    return out


def main():
    with open(os.path.join(SRC, "_split_meta.json")) as f:
        meta = json.load(f)
    src_bits = meta["dims"]["bits"]
    src_group = meta["dims"]["group_size"]
    moe_layers = meta["moe_layers"]
    num_experts = meta["dims"]["num_experts"]

    step = max(1, num_experts // EPL)
    sampled_e = list(range(0, num_experts, step))[:EPL]

    by_proj = {p: [] for p in PROJS}
    by_layer = {}            # layer序号 -> 平均(三 proj)误差
    for li in moe_layers:
        layer_errs = {p: [] for p in PROJS}
        for e in sampled_e:
            fn = f"layer{li:02d}_expert{e:03d}.safetensors"
            path = os.path.join(SRC, fn)
            if not os.path.exists(path):
                continue
            r = _rel_err(path, src_bits, src_group)
            for p in PROJS:
                layer_errs[p].append(r[p])
                by_proj[p].append(r[p])
        by_layer[li] = {p: round(st.mean(layer_errs[p]), 4) for p in PROJS if layer_errs[p]}

    print(f"目标 {DST_BITS}-bit / g{DST_GROUP},每层采样 {len(sampled_e)} 专家\n")
    print("== 按 proj 的平均相对重构误差(越大越该多给 bit)==")
    for p in PROJS:
        vals = by_proj[p]
        print(f"  {p:10s} mean={st.mean(vals):.4f}  p90={sorted(vals)[int(len(vals)*0.9)]:.4f}")

    # 层位置效应:看首/中/尾
    n = len(moe_layers)
    def layer_avg(idxs):
        vs = [v for li in idxs for v in by_layer[li].values()]
        return round(st.mean(vs), 4)
    head = moe_layers[:n // 6]
    mid = moe_layers[n // 3: 2 * n // 3]
    tail = moe_layers[-n // 6:]
    print("\n== 按层位置(三 proj 合并)==")
    print(f"  首 {len(head)} 层: {layer_avg(head)}   中 {len(mid)} 层: {layer_avg(mid)}   尾 {len(tail)} 层: {layer_avg(tail)}")

    # 误差最高的 8 个 (层,proj)
    flat = [(li, p, by_layer[li][p]) for li in by_layer for p in by_layer[li]]
    flat.sort(key=lambda x: -x[2])
    print("\n== 误差最高的 10 个 (层, proj) ==")
    for li, p, v in flat[:10]:
        print(f"  layer{li:02d} {p:10s} {v:.4f}")


if __name__ == "__main__":
    main()
