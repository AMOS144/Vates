"""de-risk 探针④:把一个 token 的 decode 时间消融拆成 注意力/MoE matmul/编排/读盘(临时)。

方法(消融法,各成分由完整 decode 跑的 tps 差值反推,避免扰动式逐层 eval):
  A 全真路径(host,真读盘)            = 注意力 + MoE matmul + 编排 + 读盘
  B 去读盘(loader 秒返回预载专家)      = 注意力 + MoE matmul + 编排        → 读盘 = A-B
  C 去读盘+去编排(纯 GPU remap,clamp) = 注意力 + MoE matmul              → 编排 = B-C
  D 再去 MoE matmul(跳过 _sub.forward) = 注意力 + 路由gate                → matmul = C-D, 注意力≈D
时间以 秒/token = 1/tps 计。

环境变量:MODEL / EXPERT_DIR / EXPERT_SLOTS / RESIDENT_POOL=1 / PROMPT / WARM / STEPS
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型，并举一个实际应用的例子。")
WARM = int(os.environ.get("WARM", "16"))
STEPS = int(os.environ.get("STEPS", "48"))

_NO_MATMUL = False    # D 变体开关


def _moe_blocks(model):
    return [layer.mlp for layer in model.layers
            if isinstance(getattr(layer, "mlp", None), FileStreamingMoeBlock)]


def _decode_window(model, prompt_ids, warm, steps):
    cache = model.make_cache()
    cur = mx.argmax(model(prompt_ids, cache=cache)[:, -1:, :], axis=-1)
    mx.eval(cur)
    for _ in range(warm):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)
    t0 = time.perf_counter()
    for _ in range(steps):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)
    return steps / (time.perf_counter() - t0)


def _build_slot_tables(model, store):
    for blk in _moe_blocks(model):
        L = blk.layer_idx
        ne = int(blk.gate.weight.shape[0])
        tab = [-1] * ne
        for e, slot in store._resident._slot_of.get(L, {}).items():
            tab[int(e)] = int(slot)
        blk._probe_table = mx.array(tab, dtype=mx.int32)
        blk._probe_pool = store._resident._pools[L]
        blk._probe_cap = store.cap_for(L)
        mx.eval(blk._probe_table)


def _gpu_call(self, x):
    """C/D 变体:纯 GPU remap(clamp miss、零 host 往返)。_NO_MATMUL=True 时跳过量化 matmul。"""
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    local = mx.maximum(mx.take(self._probe_table, inds), 0)
    if _NO_MATMUL:
        y = mx.zeros(x.shape, dtype=x.dtype)            # 跳过 _sub.forward 量化 matmul
    else:
        y = self._sub.forward(self._probe_pool, self._probe_cap, x, local)
        y = (y * scores[..., None]).sum(axis=-2)
    if self.shared_expert is not None:
        y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    return y


def main():
    global _NO_MATMUL
    assert os.environ.get("RESIDENT_POOL", "1") == "1"
    model, tok, store = build_streaming_model()
    prompt_ids = mx.array([tok.encode(PROMPT)])

    # 预热(真读盘灌满池),再建 GPU 查找表
    _decode_window(model, prompt_ids, WARM, 0)
    _build_slot_tables(model, store)

    # ---- A:全真路径(host + 真读盘) ----
    os.environ["GPU_REMAP"] = "0"
    tps_A = _decode_window(model, prompt_ids, WARM, STEPS)

    # ---- B:去读盘(loader 秒返回预载专家,编排照旧) ----
    real_load = store._load_one
    dummy: dict = {}

    def instant_load(layer, e):
        d = dummy.get(layer)
        if d is None:
            d = real_load(layer, e)
            mx.eval(list(d.values()))
            dummy[layer] = d
        return d
    store._load_one = instant_load
    store._resident.loader = instant_load
    tps_B = _decode_window(model, prompt_ids, WARM, STEPS)

    # ---- C:去读盘 + 去编排(纯 GPU remap) ----
    orig_call = FileStreamingMoeBlock.__call__
    FileStreamingMoeBlock.__call__ = _gpu_call
    try:
        _NO_MATMUL = False
        tps_C = _decode_window(model, prompt_ids, WARM, STEPS)
        # ---- D:再去 MoE matmul ----
        _NO_MATMUL = True
        tps_D = _decode_window(model, prompt_ids, WARM, STEPS)
    finally:
        FileStreamingMoeBlock.__call__ = orig_call
        store._load_one = real_load
        store._resident.loader = real_load

    # 秒/token,再做差分归因
    sA, sB, sC, sD = 1/tps_A, 1/tps_B, 1/tps_C, 1/tps_D
    disk = sA - sB
    orch = sB - sC
    matmul = sC - sD
    attn = sD
    comps = {"读盘": disk, "编排": orch, "MoE_matmul": matmul, "注意力+路由": attn}

    out = {
        "tps": {"A_full": round(tps_A, 2), "B_no_disk": round(tps_B, 2),
                "C_no_orch": round(tps_C, 2), "D_no_matmul": round(tps_D, 2)},
        "ms_per_token": {
            "total_A": round(sA * 1e3, 2),
            **{k: round(v * 1e3, 2) for k, v in comps.items()}},
        "pct_of_token": {k: round(100 * v / sA, 1) for k, v in comps.items()},
        "warm": WARM, "steps": STEPS,
        "expert_slots": int(os.environ.get("EXPERT_SLOTS", "256")),
        "steady_hit_rate": round(store.hit_rate(), 4),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
