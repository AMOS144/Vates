"""de-risk 探针:80B-Next 流式路径上的困惑度(PPL)度量(临时,质量评估用)。

确保有效性:
  - teacher-forcing 单次前向(非自回归生成),用项目验证过的 forward_with_hidden(正确建
    GDN ssm_mask + 注意力 mask),对位置 i 用真实 token_{i+1} 求 NLL。
  - logsoftmax 在 fp32 下算(数值稳)。
  - 同一段文本、同一主模型(MODEL),只切 EXPERT_DIR → 隔离"专家量化"这单一变量。
  - 槽数设大(默认 512=全专家可驻),避免单次长前向触发淘汰路径差异。

环境变量:MODEL / EXPERT_DIR / EXPERT_SLOTS(默认 512)/ TEXT
用法:EXPERT_DIR=<dir> EXPERT_SLOTS=512 EXPERT_POOL_PROFILE=none python -m mlx_streaming.tools.probe_ppl
"""
import os

import mlx.core as mx

from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.generate import forward_with_hidden

# 较长、on-distribution 的中文技术文本,token 数足够稳(~数百)
TEXT = os.environ.get("TEXT", (
    "混合专家模型通过路由器为每个 token 选择少数专家参与计算,从而在巨大参数量下保持较低的"
    "激活计算成本。它的关键在于稀疏激活:虽然总参数量很大,但每个 token 只用到其中一小部分。"
    "在 Transformer 中,前馈层被替换为多个并行的专家网络,一个门控网络根据输入决定把 token "
    "分发给哪些专家,再把它们的输出按权重汇总。这样既扩大了模型容量,又不会让单次前向的计算量"
    "线性增长。训练时为避免负载不均,通常会加入辅助的均衡损失,鼓励各专家被均匀使用。"
    "推理时由于每个 token 只激活极少数专家,配合专家权重的按需加载,可以显著降低常驻内存占用。"
    "近年来,线性注意力与状态空间模型被引入混合架构,用以降低长上下文下的显存与计算开销;"
    "但它们的递归状态不可裁剪,给投机解码带来额外挑战。多 token 预测则在训练时让模型同时预测"
    "未来多个位置,既增强表征,又能在推理时充当自投机的草稿头。把这些技术组合起来,工程上需要"
    "在质量、显存与吞吐之间反复权衡,并通过严谨的实测来验证每一步取舍是否真的成立。"
))


def main():
    model, tok, store = build_streaming_model()
    ids = mx.array([tok.encode(TEXT)])
    L = ids.shape[1]

    cache = model.make_cache()
    logits, _ = forward_with_hidden(model, ids, cache)      # (1, L, V),teacher-forcing
    lg = logits[0, :-1, :].astype(mx.float32)               # 预测位置 i→i+1
    tgt = ids[0, 1:]
    logp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).squeeze(-1)
    mx.eval(nll)
    ppl = float(mx.exp(mx.mean(nll)))

    meta_bits = "?"
    try:
        import json
        with open(os.path.join(os.environ["EXPERT_DIR"], "_split_meta.json")) as f:
            d = json.load(f)["dims"]
        meta_bits = f"bits={d.get('bits')},g={d.get('group_size')}" + \
            (f",proj={d.get('proj_bits')}" if d.get("proj_bits") else "") + \
            (",per_layer" if d.get("per_layer_proj_bits") else "")
    except Exception:
        pass

    print(f"EXPERT_DIR={os.environ.get('EXPERT_DIR')}")
    print(f"quant={meta_bits}  n_tokens={L}  PPL={ppl:.4f}  mean_nll={float(mx.mean(nll)):.4f}")


if __name__ == "__main__":
    main()
