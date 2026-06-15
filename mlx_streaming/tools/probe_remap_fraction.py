"""de-risk 探针②:定量验证 GPU remap 收益是「线性 vs 阈值」(临时,出结论后删)。

背景:真机 A/B 显示 GPU remap 端到端零收益,怀疑根因是——只有「整层全命中」的层能走 GPU
快路径,而夹在中间的 host 回退层每层插一道同步屏障冲刷 MLX 异步流水线,使 GPU 层无法异步,
收益呈「接近 f≈1 才跳变」的阈值特性,而非随全命中比例 f 线性增长。

做法:人为把比例 p 的 MoE 层(**均匀打散**,镜像真实 miss 的随机分布)路由到纯 GPU 路径
(clamp miss、零 host 往返),其余层走真实 host 路径(.tolist+LRU+读盘)。扫 p∈{0,.25,.5,.75,1},
测 decode tok/s:
  - 若 1/tps 随 p 近似线性下降 → 线性,GPU remap 在部分驻留下也有按比例收益;
  - 若 tps 在 p<1 几乎不动、仅 p=1 跳到上界 → 阈值/流水线冲刷,印证真机零收益的根因。

环境变量:MODEL / EXPERT_DIR / EXPERT_SLOTS(=256)/ RESIDENT_POOL=1 / PROMPT / WARM / STEPS
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型，并举一个实际应用的例子。")
WARM = int(os.environ.get("WARM", "32"))
STEPS = int(os.environ.get("STEPS", "48"))

_GPU_LAYERS: set = set()          # 本轮被路由到纯 GPU 路径的层号
_ORIG_CALL = None                 # 真实 host 路径(原 __call__)


def _moe_blocks(model):
    return [layer.mlp for layer in model.layers
            if isinstance(getattr(layer, "mlp", None), FileStreamingMoeBlock)]


def _timed_decode(model, prompt_ids, warm):
    cache = model.make_cache()
    cur = mx.argmax(model(prompt_ids, cache=cache)[:, -1:, :], axis=-1)
    mx.eval(cur)
    for _ in range(warm):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)
    return cache, cur


def _time_window(model, cache, cur, steps):
    t0 = time.perf_counter()
    for _ in range(steps):
        cur = mx.argmax(model(cur, cache=cache)[:, -1:, :], axis=-1)
        mx.eval(cur)
    return steps / (time.perf_counter() - t0), cur


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


def _gpu_layer_forward(self, x):
    """纯 GPU 路径:slot 重映射 GPU gather,clamp miss,零 host 往返、无同步。"""
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    local = mx.maximum(mx.take(self._probe_table, inds), 0)
    y = self._sub.forward(self._probe_pool, self._probe_cap, x, local)
    y = (y * scores[..., None]).sum(axis=-2)
    if self.shared_expert is not None:
        y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    return y


def _hybrid_call(self, x):
    if self.layer_idx in _GPU_LAYERS and x.shape[1] == 1:
        return _gpu_layer_forward(self, x)
    return _ORIG_CALL(self, x)


def _even_spread(layer_ids, cnt):
    """从 layer_ids 中均匀挑 cnt 个(打散,镜像真实 miss 的随机分布)。"""
    n = len(layer_ids)
    if cnt <= 0:
        return set()
    if cnt >= n:
        return set(layer_ids)
    return {layer_ids[int(round(j * (n - 1) / (cnt - 1)))] if cnt > 1 else layer_ids[0]
            for j in range(cnt)}


def main():
    global _GPU_LAYERS, _ORIG_CALL
    assert os.environ.get("RESIDENT_POOL", "1") == "1"
    model, tok, store = build_streaming_model()
    prompt_ids = mx.array([tok.encode(PROMPT)])

    # warm 灌满池,再建每层 GPU 查找表
    cache, cur = _timed_decode(model, prompt_ids, WARM)
    _build_slot_tables(model, store)
    layer_ids = sorted(blk.layer_idx for blk in _moe_blocks(model))
    n = len(layer_ids)

    _ORIG_CALL = FileStreamingMoeBlock.__call__
    FileStreamingMoeBlock.__call__ = _hybrid_call
    results = []
    try:
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            cnt = round(p * n)
            _GPU_LAYERS = _even_spread(layer_ids, cnt)
            c2, cur2 = _timed_decode(model, prompt_ids, WARM)   # 每轮重新 warm,公平起点
            tps, _ = _time_window(model, c2, cur2, STEPS)
            results.append({"p": p, "gpu_layers": cnt, "tps": round(tps, 2)})
    finally:
        FileStreamingMoeBlock.__call__ = _ORIG_CALL

    base = results[0]["tps"]
    out = {
        "n_moe_layers": n, "warm": WARM, "steps": STEPS,
        "expert_slots": int(os.environ.get("EXPERT_SLOTS", "256")),
        "sweep": results,
        # 线性预期:1/tps(p) 在 1/tps(0) 与 1/tps(1) 间线性内插;实测偏离即阈值证据
        "tps_p0": base, "tps_p1": results[-1]["tps"],
        "speedup_p1_over_p0": round(results[-1]["tps"] / base, 3),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
