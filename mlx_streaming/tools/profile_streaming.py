"""定位流式热路径开销：在 FileStreamingMoeBlock.__call__ 各段间插 mx.eval 屏障计时。

强制同步会扰动绝对速度，但能给出**相对**占比：到底是 Python/同步(routing/unique/fetch)
占大头，还是 GPU 重堆叠+计算(sub.forward)占大头。决定下一步优化方向。

用法：EXPERT_DIR=/tmp/mlx_qwen3_experts_2bit EXPERT_BITS=2 EXPERT_GROUP=64 \
      SLOTS=128 STEPS=60 python3 -m mlx_streaming.tools.profile_streaming
"""
import os
import time
import json
from collections import defaultdict

import mlx.core as mx
from mlx_lm import load, generate

from mlx_streaming.core.moe import block as sm
from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.prefetch.patch import patch_model_filebacked
from mlx_streaming.core.moe.compute import _unique_and_local

EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts_2bit")
EXPERT_BITS = int(os.environ.get("EXPERT_BITS", "2"))
EXPERT_GROUP = int(os.environ.get("EXPERT_GROUP", "64"))
SLOTS = int(os.environ.get("SLOTS", "128"))
STEPS = int(os.environ.get("STEPS", "60"))
MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")

T = defaultdict(float)
N = defaultdict(int)


def _instrumented_call(self, x: mx.array) -> mx.array:
    t0 = time.perf_counter()
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    mx.eval(inds, scores)
    t1 = time.perf_counter()

    uniq, local = _unique_and_local(inds)
    mx.eval(uniq, local)
    t2 = time.perf_counter()

    fetched = self.store.fetch(self.layer_idx, [int(i) for i in uniq.tolist()])
    mx.eval(list(fetched.values()))
    t3 = time.perf_counter()

    y = self._sub.forward(fetched, int(uniq.shape[0]), x, local)
    out = (y * scores[..., None]).sum(axis=-2)
    mx.eval(out)
    t4 = time.perf_counter()

    T["routing"] += t1 - t0
    T["unique"] += t2 - t1
    T["fetch"] += t3 - t2
    T["forward"] += t4 - t3
    N["calls"] += 1
    return out


def main():
    model, tok = load(MODEL, lazy=True)
    gp = None
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp"):
            gp = mlp.switch_mlp.gate_proj
            break
    hidden, moe_inter = gp.input_dims, gp.output_dims
    store = FileExpertStore(EXPERT_DIR, capacity=SLOTS)
    patch_model_filebacked(model, store, hidden, moe_inter, EXPERT_GROUP, EXPERT_BITS)

    # 预热：填满缓存（跑一段生成）
    generate(model, tok, prompt="用三句话解释什么是混合专家模型。", max_tokens=40, verbose=False)

    # 插桩后计时
    sm.FileStreamingMoeBlock.__call__ = _instrumented_call
    store.reset_stats()
    t0 = time.perf_counter()
    generate(model, tok, prompt="用三句话解释什么是混合专家模型。", max_tokens=STEPS, verbose=False)
    wall = time.perf_counter() - t0

    layers = T["routing"] and (N["calls"] / STEPS) or 0
    seg = {k: round(v * 1000, 1) for k, v in T.items()}
    total_seg = sum(T.values())
    pct = {k: round(100 * v / total_seg, 1) for k, v in T.items()}
    print(json.dumps({
        "slots": SLOTS, "bits": EXPERT_BITS, "steps": STEPS,
        "block_calls": N["calls"], "calls_per_token": round(layers, 1),
        "wall_s": round(wall, 2), "hit_rate": round(store.hit_rate(), 3),
        "seg_ms_total": seg, "seg_pct": pct,
        "seg_us_per_call": {k: round(1e6 * v / N["calls"], 1) for k, v in T.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
