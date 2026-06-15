"""de-risk 探针:MoE 专家"量化贡献"倾斜度(决定专家粒度混合精度是否值得)。
贡献(e) ≈ 使用频率(e) × 2-bit 重构误差(e)。看 top 20% 专家是否占 ~80% 总贡献。

校准用 4-bit 流式模型取使用频率;误差用 FP16 权重的 2-bit 重构。
环境变量:EXPERT_DIR(4-bit)/ FP16_DIR / GPTQ_LAYERS / CALIB_REPEAT
"""
import json
import os
from collections import defaultdict

import mlx.core as mx
import numpy as np

from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.generate import forward_with_hidden
from mlx_streaming.prep.gptq_calibrate import install_hook_all, CALIB_TEXT
from mlx_streaming.prep.gptq_fp16_validate import _load_fp16_experts

PROJS = ["gate_proj", "up_proj", "down_proj"]


def main():
    layers = [int(x) for x in os.environ.get("GPTQ_LAYERS", "0,1,2").split(",")]
    rep = int(os.environ.get("CALIB_REPEAT", "4"))

    model, tok, store = build_streaming_model()
    cap, restore = install_hook_all(model, set(layers))
    ids = mx.array([tok.encode(CALIB_TEXT * rep)])
    mx.eval(forward_with_hidden(model, ids, model.make_cache())[0])
    restore()
    del model, store
    mx.clear_cache()

    fp16 = _load_fp16_experts(set(layers))

    for L in layers:
        idf = np.concatenate(cap[L]["inds"], axis=0)
        usage = defaultdict(int)
        for e in idf.reshape(-1):
            usage[int(e)] += 1
        # 每专家 2-bit 重构误差(三 proj Frobenius 绝对值之和)
        contrib = {}
        for e in usage:
            err = 0.0
            for proj in PROJS:
                W = fp16.get((L, e, proj))
                if W is None:
                    continue
                q, s, b = mx.quantize(mx.array(W), group_size=128, bits=2)
                deq = np.array(mx.dequantize(q, s, b, group_size=128, bits=2))
                err += float(np.linalg.norm(W - deq))
            contrib[e] = usage[e] * err
        # 倾斜度:按贡献降序,top 20% 占比
        vals = np.array(sorted(contrib.values(), reverse=True))
        if len(vals) == 0:
            continue
        cum = np.cumsum(vals) / vals.sum()
        n = len(vals)
        top20_share = cum[max(0, int(n * 0.2) - 1)]
        top10_share = cum[max(0, int(n * 0.1) - 1)]
        # 纯误差(不乘频率)的倾斜,作对照
        errs = np.array(sorted([contrib[e] / usage[e] for e in usage], reverse=True))
        cume = np.cumsum(errs) / errs.sum()
        err_top20 = cume[max(0, int(n * 0.2) - 1)]
        print(f"层 {L}: 活跃专家 {n}  "
              f"贡献(频率×误差) top10%={top10_share*100:.0f}% top20%={top20_share*100:.0f}%  "
              f"| 纯误差 top20%={err_top20*100:.0f}%")


if __name__ == "__main__":
    main()
