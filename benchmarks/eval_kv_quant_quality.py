"""KV 量化质量验收:同一 prompt 下 bf16 基线 vs K4/V3 量化的贪婪生成对比。

判据(轻量自动):
- token 一致率 ≥ 0.95(量化生成的前 N token 与 bf16 基线相同的比例)
- 末步 logits cosine ≥ 0.99

省内存做法:同进程内先用 bf16(KV_QUANT 强制关)跑基线,再 object 级原地 patch 成量化重跑,
不重复加载 80B 权重。

用法(在 worktree 根目录):
  MODEL=models/qwen3_next_80b_4bit EXPERT_DIR=models/qwen3_next_experts_4bit_g64 \
  EXPERT_SLOTS=64 STREAM_BLOB_LOADER=1 N=64 \
  .venv/bin/python -m benchmarks.eval_kv_quant_quality
"""
import os

import mlx.core as mx

# 基线阶段强制关 KV 量化(确保 build 出 bf16 KVCache)
os.environ["KV_QUANT"] = "0"

from mlx_streaming import config                                   # noqa: E402
from mlx_streaming.model_builder import build_streaming_model      # noqa: E402
from mlx_streaming.core.cache.kv_quant_patch import patch_kv_quant  # noqa: E402
from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked  # noqa: E402

N = int(os.environ.get("N", "64"))
PROMPT = os.environ.get("PROMPT", "请用中文简要解释什么是混合专家(MoE)模型,以及它为什么能在不显著增加推理成本的前提下扩大模型容量。")


def greedy(model, tok, prompt, n):
    """贪婪生成 n 个 token,返回 (token 列表, 末步 logits[vocab])。prefill 走分块。"""
    ids = mx.array([tok.encode(prompt)])
    cache = model.make_cache()
    full, _ = prefill_chunked(model, ids, cache)
    logits = full[:, -1, :]
    toks, last = [], logits
    for _ in range(n):
        t = int(mx.argmax(logits, axis=-1).item())
        toks.append(t)
        out, _ = forward_with_hidden(model, mx.array([[t]]), cache)
        logits = out[:, -1, :]
        last = logits
    mx.eval(last)
    return toks, last


def main():
    model, tok, store = build_streaming_model()

    print(f"[基线] bf16 KV 贪婪生成 {N} token …")
    base_toks, base_logits = greedy(model, tok, PROMPT, N)

    # 原地 patch 成 K4/V3(读 config 的位宽/旋转设置,但 KV_QUANT 此处显式启用逻辑)
    print(f"[量化] patch → K{config.kv_k_bits()}/V{config.kv_v_bits()} "
          f"rotate={config.kv_rotate()} g{config.kv_group_size()}")
    patch_kv_quant(model,
                   group_size=config.kv_group_size(),
                   k_bits=config.kv_k_bits(),
                   v_bits=config.kv_v_bits(),
                   rotate=config.kv_rotate(),
                   seed=config.kv_rot_seed())
    if store is not None:
        try:
            store.reset_stats()
        except Exception:
            pass

    print(f"[量化] K4/V3 KV 贪婪生成 {N} token …")
    quant_toks, quant_logits = greedy(model, tok, PROMPT, N)

    agree = sum(1 for a, b in zip(base_toks, quant_toks) if a == b) / max(1, len(base_toks))
    cos = float((base_logits * quant_logits).sum()
                / (mx.linalg.norm(base_logits) * mx.linalg.norm(quant_logits)))

    print("\n==== KV 量化质量验收 ====")
    print(f"  生成 token 数      : {N}")
    print(f"  token 一致率       : {agree:.3f}   (阈值 ≥ 0.95)  {'PASS' if agree >= 0.95 else 'FAIL'}")
    print(f"  末步 logits cosine : {cos:.4f}  (阈值 ≥ 0.99)  {'PASS' if cos >= 0.99 else 'FAIL'}")
    print("\n  基线前 16 token:", base_toks[:16])
    print("  量化前 16 token:", quant_toks[:16])
    print("\n  基线文本:", tok.decode(base_toks))
    print("  量化文本:", tok.decode(quant_toks))


if __name__ == "__main__":
    main()
