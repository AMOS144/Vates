"""GPTQ 校准 harness:在流式模型上逐专家收集激活、累积 Hessian,再逐专家 GPTQ 量化。

捕获机制(非侵入):monkeypatch 目标 MoE 层的 __call__,用"重算 gate 取路由 + 捕获块输入 x"
累积每个专家 gate/up 输入的 Hessian H_e = Σ x xᵀ(x=路由到该专家的 token 隐藏态)。
只为"见过的专家"惰性累积(dict),控制内存。down_proj 的输入是中间激活(v2 再做),
v1 先 gate/up 走 GPTQ、down 走 RTN。

用法(单层验证):TARGET_LAYER=0 CALIB_TOKENS=... python -m mlx_streaming.prep.gptq_calibrate
"""
import os

import mlx.core as mx
import numpy as np

from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.generate import forward_with_hidden

CALIB_TEXT = (
    "混合专家模型通过路由器为每个 token 选择少数专家参与计算。稀疏激活让超大参数量保持低推理成本。"
    "线性注意力与状态空间模型降低长上下文开销;多 token 预测增强表征并支持自投机解码。"
    "量化把权重压到低 bit 以省内存,GPTQ 用二阶信息逐列补偿误差,优于就近取整。"
) * 4


class _Cap:
    def __init__(self):
        self.H = {}        # expert_id -> (hidden,hidden) float64 Hessian 累积
        self.n = {}        # expert_id -> 样本数


def install_hook(model, target_layer):
    """patch 类方法 __call__(特殊方法走类型查找),只对 target_layer 累积 Hessian。返回 (cap, restore)。"""
    cap = _Cap()
    orig = FileStreamingMoeBlock.__call__

    def hooked(self, x):
        if self.layer_idx == target_layer:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            k = self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            xf = np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32))   # (N, hidden)
            idf = np.array(inds.reshape(-1, k))                            # (N, k)
            for n in range(xf.shape[0]):
                xn = xf[n].astype(np.float64)
                outer = np.outer(xn, xn)
                for e in idf[n]:
                    e = int(e)
                    if e not in cap.H:
                        cap.H[e] = outer.copy()
                        cap.n[e] = 1
                    else:
                        cap.H[e] += outer
                        cap.n[e] += 1
        return orig(self, x)

    FileStreamingMoeBlock.__call__ = hooked
    return cap, lambda: setattr(FileStreamingMoeBlock, "__call__", orig)


def install_hook_all(model, layers):
    """patch 类方法,对 layers 集合里的所有 MoE 层捕获 (x, inds)(存入内存,离线再算 Hessian)。"""
    cap = {L: {"x": [], "inds": []} for L in layers}
    orig = FileStreamingMoeBlock.__call__

    def hooked(self, x):
        if self.layer_idx in cap:
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            inds = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k:]
            cap[self.layer_idx]["x"].append(
                np.array(x.reshape(-1, x.shape[-1]).astype(mx.float32)))
            cap[self.layer_idx]["inds"].append(np.array(inds.reshape(-1, self.top_k)))
        return orig(self, x)

    FileStreamingMoeBlock.__call__ = hooked
    return cap, lambda: setattr(FileStreamingMoeBlock, "__call__", orig)


def _expert_hessian(xf, idf, e, hidden):
    """聚该专家路由到的 token 行 → H = Σ x xᵀ;返回 (H, 样本数)。"""
    mask = (idf == e).any(axis=1)
    Xe = xf[mask]
    if Xe.shape[0] == 0:
        return None, 0
    return (Xe.T @ Xe).astype(np.float64), int(Xe.shape[0])


def build_dir(store, cap, out_dir, gptq_layers, src_bits, src_group,
              bits=2, group=128, min_samples=64):
    """产 GPTQ 专家目录:gptq_layers 的 gate/up 走 GPTQ(样本足够时),其余 + down 走 RTN;
    非 gptq_layers 直接 symlink 现有 RTN(g128)目录,只为目标层写新文件。"""
    import time
    from mlx_streaming.prep.gptq import gptq_quantize, rtn_quantize
    os.makedirs(out_dir, exist_ok=True)
    g128 = os.environ["RTN_DIR"]                       # 既有 2-bit g128 RTN 目录(非目标层 symlink 它)
    # 先把所有专家文件 symlink 自 RTN 目录(瞬时)
    import json
    for fn in os.listdir(g128):
        dst = os.path.join(out_dir, fn)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.join(g128, fn), dst)
    # 目标层:用真文件覆盖(GPTQ gate/up + RTN down)
    PROJ_IN = {"gate_proj": "hidden", "up_proj": "hidden", "down_proj": "moe_inter"}
    t0 = time.perf_counter()
    n_gptq = 0
    for L in gptq_layers:
        xf = np.concatenate(cap[L]["x"], axis=0)
        idf = np.concatenate(cap[L]["inds"], axis=0)
        hidden = xf.shape[1]
        # 该层涉及的专家
        experts = sorted(set(int(e) for e in idf.reshape(-1)))
        for e in experts:
            w = mx.load(store.path(L, e))
            out = {}
            H, ns = _expert_hessian(xf, idf, e, hidden)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                W = np.array(mx.dequantize(w[f"{proj}.weight"], w[f"{proj}.scales"],
                                           w[f"{proj}.biases"], group_size=src_group,
                                           bits=src_bits).astype(mx.float32))
                if proj in ("gate_proj", "up_proj") and H is not None and ns >= min_samples:
                    Wh = gptq_quantize(W, H, bits=bits, group_size=group)
                    n_gptq += 1
                else:
                    Wh = rtn_quantize(W, bits=bits, group_size=group)
                q, s, b = mx.quantize(mx.array(Wh), group_size=group, bits=bits)
                out[f"{proj}.weight"], out[f"{proj}.scales"], out[f"{proj}.biases"] = q, s, b
            dst = os.path.join(out_dir, os.path.basename(store.path(L, e)))
            if os.path.lexists(dst):
                os.remove(dst)
            mx.save_safetensors(dst, out)
    # meta:复制 RTN 的(2-bit g128)
    with open(os.path.join(g128, "_split_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(out_dir, "_split_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"GPTQ 目录产完: gptq_layers={list(gptq_layers)} GPTQ矩阵数={n_gptq} "
          f"耗时={round(time.perf_counter()-t0,1)}s -> {out_dir}")


def main():
    gptq_layers = [int(x) for x in os.environ.get("GPTQ_LAYERS", "0,1,2").split(",")]
    out_dir = os.environ.get("OUT_DIR", os.path.abspath("models/qwen3_next_experts_gptqtest"))
    src_bits = int(os.environ.get("SRC_BITS", "4"))
    src_group = int(os.environ.get("SRC_GROUP", "64"))

    model, tok, store = build_streaming_model()
    cap, restore = install_hook_all(model, set(gptq_layers))
    ids = mx.array([tok.encode(CALIB_TEXT)])
    cache = model.make_cache()
    mx.eval(forward_with_hidden(model, ids, cache)[0])
    restore()
    for L in gptq_layers:
        xf = np.concatenate(cap[L]["x"], axis=0)
        print(f"层 {L}: 校准行={xf.shape[0]}")
    build_dir(store, cap, out_dir, gptq_layers, src_bits, src_group)


if __name__ == "__main__":
    main()
