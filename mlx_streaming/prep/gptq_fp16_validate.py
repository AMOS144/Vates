"""FP16 源小验证:校准用 4-bit 流式模型(代理激活),GPTQ 量化的是 FP16 专家权重。
对目标几层同时产 RTN-from-FP16 与 GPTQ-from-FP16 两版专家目录(其余层 symlink 现有 g128),
供 probe_ppl 测 held-out PPL,隔离"FP16 源"与"GPTQ"各自收益。

环境变量:EXPERT_DIR(4-bit,校准用)/ FP16_DIR / RTN_DIR(现有 g128)/ GPTQ_LAYERS /
          OUT_GPTQ / OUT_RTN / CALIB_REPEAT(校准文本重复次数)
"""
import json
import os

import mlx.core as mx
import numpy as np

from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.generate import forward_with_hidden
from mlx_streaming.prep.gptq import gptq_quantize, rtn_quantize
from mlx_streaming.prep.gptq_calibrate import install_hook_all, CALIB_TEXT

FP16_DIR = os.environ.get("FP16_DIR", "/tmp/qwen3_next_80b_fp16")
PROJS = ["gate_proj", "up_proj", "down_proj"]


def _load_fp16_experts(layers):
    """从 FP16 分片加载目标层全部专家权重 → {(L,E,proj): np.float32 (out,in)}。"""
    idx = json.load(open(os.path.join(FP16_DIR, "model.safetensors.index.json")))["weight_map"]
    need_shards = set()
    keys = {}
    for k, shard in idx.items():
        # 必须是主模型层(排除 mtp.layers.* —— 它的 experts 也会解析出 L=0,会覆盖主模型 layer0)
        if not k.startswith("model.layers.") or ".mlp.experts." not in k:
            continue
        parts = k.split(".")
        L = int(parts[2]); E = int(parts[5]); proj = parts[6]
        if L in layers:
            keys[(L, E, proj)] = (shard, k)
            need_shards.add(shard)
    out = {}
    for shard in sorted(need_shards):
        w = mx.load(os.path.join(FP16_DIR, shard))
        for (L, E, proj), (s, k) in keys.items():
            if s == shard:
                out[(L, E, proj)] = np.array(w[k].astype(mx.float32))
        del w
        mx.clear_cache()
    return out


def _symlink_rest(out_dir, rtn_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(rtn_dir):
        dst = os.path.join(out_dir, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(rtn_dir, fn), dst)


def main():
    layers = [int(x) for x in os.environ.get("GPTQ_LAYERS", "0,1,2").split(",")]
    rtn_dir = os.environ["RTN_DIR"]
    out_gptq = os.environ.get("OUT_GPTQ", os.path.abspath("models/qwen3_next_fp16gptq"))
    out_rtn = os.environ.get("OUT_RTN", os.path.abspath("models/qwen3_next_fp16rtn"))
    rep = int(os.environ.get("CALIB_REPEAT", "6"))
    bits, group = 2, 128

    # 1) 校准(4-bit 流式模型,代理激活)
    model, tok, store = build_streaming_model()
    cap, restore = install_hook_all(model, set(layers))
    ids = mx.array([tok.encode(CALIB_TEXT * rep)])
    mx.eval(forward_with_hidden(model, ids, model.make_cache())[0])
    restore()
    del model, store
    mx.clear_cache()

    # 2) 载入 FP16 专家权重
    print("载入 FP16 专家权重 ...", flush=True)
    fp16 = _load_fp16_experts(set(layers))

    # 3) 两版目录:其余层 symlink g128
    _symlink_rest(out_gptq, rtn_dir)
    _symlink_rest(out_rtn, rtn_dir)
    for d in (out_gptq, out_rtn):
        meta = json.load(open(os.path.join(rtn_dir, "_split_meta.json")))
        json.dump(meta, open(os.path.join(d, "_split_meta.json"), "w"), ensure_ascii=False, indent=2)

    n_gptq = 0
    for L in layers:
        xf = np.concatenate(cap[L]["x"], axis=0)
        idf = np.concatenate(cap[L]["inds"], axis=0)
        experts = sorted(set(int(e) for e in idf.reshape(-1)))
        for e in experts:
            mask = (idf == e).any(axis=1)
            Xe = xf[mask]
            H = (Xe.T @ Xe).astype(np.float64) if Xe.shape[0] else None
            og, orr = {}, {}
            for proj in PROJS:
                W = fp16.get((L, e, proj))
                if W is None:
                    continue
                # GPTQ 版(gate/up 用 H,down 用 RTN);RTN 版(全 RTN)
                if proj in ("gate_proj", "up_proj") and H is not None and Xe.shape[0] >= 32:
                    Wg = gptq_quantize(W, H, bits=bits, group_size=group); n_gptq += 1
                else:
                    Wg = rtn_quantize(W, bits=bits, group_size=group)
                Wr = rtn_quantize(W, bits=bits, group_size=group)
                for tag, Wh, dd in (("g", Wg, og), ("r", Wr, orr)):
                    q, s, b = mx.quantize(mx.array(Wh), group_size=group, bits=bits)
                    dd[f"{proj}.weight"], dd[f"{proj}.scales"], dd[f"{proj}.biases"] = q, s, b
            fn = f"layer{L:02d}_expert{e:03d}.safetensors"
            for dd, d in ((og, out_gptq), (orr, out_rtn)):
                dst = os.path.join(d, fn)
                if os.path.lexists(dst):
                    os.remove(dst)
                mx.save_safetensors(dst, dd)
    print(f"完成: 校准 token={ids.shape[1]}, GPTQ 矩阵={n_gptq}")
    print(f"GPTQ-from-FP16 -> {out_gptq}")
    print(f"RTN-from-FP16  -> {out_rtn}")


if __name__ == "__main__":
    main()
