"""逐 proj 量化敏感度探针：在全模型上把某个 proj 降到低 bit、其余保 4-bit，看困惑度涨多少。

不写盘、不接流式：直接 load 全模型(4-bit)，对每个 MoE 层的 switch_mlp 指定 proj 做
「4bit→反量化→低bit 往返→再塞回 4bit 容器」的降质，跑困惑度。涨得越多 = 该 proj 越敏感，
混合精度时越该留高 bit。

环境变量：MODEL、TEXT、DEG_BITS(降到几 bit，默认 2)、PROBE(proj|layer，默认 proj)、
BND(逐层探针的首尾边界层数，默认 4)。
"""
import os
import json

import mlx.core as mx
from mlx_lm import load

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
DEG_BITS = int(os.environ.get("DEG_BITS", "2"))
PROBE = os.environ.get("PROBE", "proj")
BND = int(os.environ.get("BND", "4"))
TEXT = os.environ.get("TEXT", "混合专家模型通过路由器为每个 token 选择少数专家参与计算，"
                                "从而在巨大参数量下保持较低的激活计算成本。"
                                "它的关键在于稀疏激活：虽然总参数量很大，"
                                "但每个 token 只用到其中一小部分。")


def _ppl(model, ids):
    x = ids[None, :-1]
    tgt = ids[1:]
    logits = model(x)[0]
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    return float(mx.exp(mx.mean(nll)))


def _degrade(model, bitmap):
    """按 bitmap={proj:bits} 把每个 MoE 层 switch_mlp 的 proj 降到指定 bit 再塞回原 4-bit 容器。

    bits>=原 bit 时跳过(不降质)。塞回原容器是为了不改 runtime，纯测质量。
    """
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "switch_mlp"):
            continue
        sm = mlp.switch_mlp
        for pn, bits in bitmap.items():
            lin = getattr(sm, pn)
            g, ob = lin.group_size, lin.bits
            if bits >= ob:
                continue
            W = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=g, bits=ob)
            wq, s, b = mx.quantize(W, group_size=g, bits=bits)          # 降质
            Wd = mx.dequantize(wq, s, b, group_size=g, bits=bits)
            nq, ns, nb = mx.quantize(Wd, group_size=g, bits=ob)        # 塞回 4-bit 容器
            lin.update({"weight": nq, "scales": ns, "biases": nb})
    mx.eval(model.parameters())


def _avg_bits(bm):
    return round(sum(bm.values()) / 3, 3)


def _moe_layers(model):
    """按出现顺序返回所有 MoE 层的 switch_mlp（用于按层序定位首/中/尾）。"""
    sms = []
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            sms.append(mlp.switch_mlp)
    return sms


def _degrade_layers(model, deg_idxs, bits):
    """把第 deg_idxs（MoE 层序号集合）层的全部 proj 降到 bits 再塞回 4-bit 容器。"""
    sms = _moe_layers(model)
    for i, sm in enumerate(sms):
        if i not in deg_idxs:
            continue
        for pn in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(sm, pn)
            g, ob = lin.group_size, lin.bits
            if bits >= ob:
                continue
            W = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=g, bits=ob)
            wq, s, b = mx.quantize(W, group_size=g, bits=bits)
            Wd = mx.dequantize(wq, s, b, group_size=g, bits=bits)
            nq, ns, nb = mx.quantize(Wd, group_size=g, bits=ob)
            lin.update({"weight": nq, "scales": ns, "biases": nb})
    mx.eval(model.parameters())


def _degrade_per_layer(model, layer_bitmaps):
    """layer_bitmaps[i]={proj:bits}：把第 i 个 MoE 层各 proj 降到对应 bit（塞回 4-bit 容器）。"""
    sms = _moe_layers(model)
    for i, sm in enumerate(sms):
        bm = layer_bitmaps[i]
        for pn in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(sm, pn)
            g, ob = lin.group_size, lin.bits
            bits = bm[pn]
            if bits >= ob:
                continue
            W = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=g, bits=ob)
            wq, s, b = mx.quantize(W, group_size=g, bits=bits)
            Wd = mx.dequantize(wq, s, b, group_size=g, bits=bits)
            nq, ns, nb = mx.quantize(Wd, group_size=g, bits=ob)
            lin.update({"weight": nq, "scales": ns, "biases": nb})
    mx.eval(model.parameters())


def main_mixlayer():
    """逐层+逐proj 组合：测「首尾层压更狠、中间层保 mixB」能否在更低平均 bit 下守住质量。"""
    model0, tok = load(MODEL)
    n_moe = len(_moe_layers(model0))
    del model0
    ids = mx.array(tok.encode(TEXT))
    G, U, D = "gate_proj", "up_proj", "down_proj"
    mixB = {G: 2, U: 3, D: 3}
    mixA = {G: 2, U: 3, D: 2}
    p2 = {G: 2, U: 2, D: 2}
    bnd = set(range(BND)) | set(range(n_moe - BND, n_moe))

    def per_layer(mid_bm, bnd_bm):
        return [bnd_bm if i in bnd else mid_bm for i in range(n_moe)]

    schemes = {
        "mixA_uniform":        per_layer(mixA, mixA),
        "mixB_uniform":        per_layer(mixB, mixB),
        "mixB_mid_2bit_bnd":   per_layer(mixB, p2),     # 中间 mixB，首尾 2bit
        "mixB_mid_mixA_bnd":   per_layer(mixB, mixA),   # 中间 mixB，首尾 mixA
    }
    out, avgb = {}, {}
    for tag, lbm in schemes.items():
        model, _ = load(MODEL)
        _degrade_per_layer(model, lbm)
        out[tag] = round(_ppl(model, ids), 3)
        avgb[tag] = round(sum(sum(b.values()) for b in lbm) / (3 * n_moe), 3)
        print(f"{tag:22s} avg_bits={avgb[tag]:<6} ppl={out[tag]}", flush=True)
        del model
    print(json.dumps({"n_moe_layers": n_moe, "bnd": BND,
                      "avg_bits": avgb, "ppl": out}, ensure_ascii=False, indent=2))


def main_layer():
    """逐层敏感度：固定 proj 统一 bit，按层位置降质，隔离「层位置」对质量的影响。"""
    model0, tok = load(MODEL)
    n_moe = len(_moe_layers(model0))
    del model0
    ids = mx.array(tok.encode(TEXT))
    bnd = set(range(BND)) | set(range(n_moe - BND, n_moe))      # 首 BND + 尾 BND
    mid = set(range(n_moe)) - bnd                               # 中间层
    allidx = set(range(n_moe))

    configs = {
        "all4_baseline": (set(), None),
        f"all_{DEG_BITS}bit": (allidx, DEG_BITS),
        f"mid_{DEG_BITS}bit_bnd4_keep4": (mid, DEG_BITS),       # 只压中间，首尾留 4bit
        f"bnd4_{DEG_BITS}bit_mid_keep4": (bnd, DEG_BITS),       # 只压首尾，中间留 4bit
    }
    out = {}
    for tag, (idxs, bits) in configs.items():
        model, _ = load(MODEL)
        if idxs:
            _degrade_layers(model, idxs, bits)
        out[tag] = round(_ppl(model, ids), 3)
        print(f"{tag:26s} ppl={out[tag]}", flush=True)
        del model

    base = out["all4_baseline"]
    print(json.dumps({"n_moe_layers": n_moe, "bnd": BND, "deg_bits": DEG_BITS,
                      "ppl": out,
                      "delta_vs_4bit": {k: round(v - base, 3)
                                        for k, v in out.items() if k != "all4_baseline"}},
                     ensure_ascii=False, indent=2))


def main():
    _, tok = load(MODEL)
    ids = mx.array(tok.encode(TEXT))

    G, U, D = "gate_proj", "up_proj", "down_proj"
    # 候选混合方案：键=方案名，值=逐 proj bit 分配
    configs = {
        "all4_baseline":   {G: 4, U: 4, D: 4},
        "all2":            {G: 2, U: 2, D: 2},
        "all3":            {G: 3, U: 3, D: 3},
        "g2_u2_d3_avg2.33": {G: 2, U: 2, D: 3},   # 你提的常识方案
        "g2_u3_d2_avg2.33": {G: 2, U: 3, D: 2},   # 同预算但升 up
        "g2_u3_d3_avg2.67": {G: 2, U: 3, D: 3},   # 数据驱动：gate 留 2，两敏感升 3
        "g3_u3_d3_avg3.0":  {G: 3, U: 3, D: 3},
    }
    out, avgb = {}, {}
    for tag, bm in configs.items():
        model, _ = load(MODEL)
        _degrade(model, bm)
        out[tag] = round(_ppl(model, ids), 3)
        avgb[tag] = _avg_bits(bm)
        print(f"{tag:20s} avg_bits={avgb[tag]:<5} ppl={out[tag]}", flush=True)
        del model

    print(json.dumps({"avg_bits": avgb, "ppl": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if PROBE == "layer":
        main_layer()
    elif PROBE == "mixlayer":
        main_mixlayer()
    else:
        main()
