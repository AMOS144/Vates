"""de-risk 探针:测「消除每层 host 同步」的单路 decode 吞吐上界(临时,出结论后删)。

动机:`FileStreamingMoeBlock.__call__` 每层都 `inds.tolist()` 把路由拉回 CPU 驱动 LRU 池,
48 层 = 每 token 48 次强制 GPU→CPU 同步,打碎 MLX 异步图。本探针验证:warm 稳态下单路工作集
若已全驻(几乎无 miss),把 slot 重映射搬到 GPU(`local = slot_table[inds]`,零 host 往返)能把
吞吐顶到多少。

做法:
  1) 正常 patch + warm,灌满常驻池;
  2) 读 `store._resident._slot_of[layer]` 建每层 GPU 查找表 `slot_table`(全局 id→slot,-1=不在池);
  3) 把每个 MoE 块的 __call__ 换成纯 GPU 版(不调 .tolist()、不走 Python LRU),计时 decode;
  4) 对照正常路径 tok/s,并用正常路径稳态 store 命中统计估「warm 是否真的 ~0 miss」。

决策门:无同步上界 ≥ 21 tok/s → 值得做真实现;≲ 20 或稳态 miss 多 → NO-GO。

环境变量:MODEL / EXPERT_DIR / EXPERT_SLOTS / RESIDENT_POOL(应=1)/ PROMPT / WARM / STEPS
建议:EXPERT_DIR=…_2bit EXPERT_SLOTS=64 RESIDENT_POOL=1
"""
import json
import os
import time

import mlx.core as mx

from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model

PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型，并举一个实际应用的例子。")
WARM = int(os.environ.get("WARM", "24"))
STEPS = int(os.environ.get("STEPS", "48"))
# MISS_SYNC=1:在 GPU 路径里每层额外 mx.eval 一个 miss 标志,模拟真实现「每层必须知道有没有
# miss(有则读盘)」不可避免的那次同步——但不带 Python LRU 记账。用来分离「同步屏障 vs Python」。
_MISS_SYNC = os.environ.get("MISS_SYNC", "0") == "1"


def _moe_blocks(model):
    return [layer.mlp for layer in model.layers
            if isinstance(getattr(layer, "mlp", None), FileStreamingMoeBlock)]


def _timed_decode(model, prompt_ids, warm, steps):
    """贪心 decode:prefill → warm 步预热 → 计时 steps 步。每 token 一次 mx.eval(与原生同)。"""
    cache = model.make_cache()
    logits = model(prompt_ids, cache=cache)
    cur = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(cur)
    for _ in range(warm):
        logits = model(cur, cache=cache)
        cur = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(cur)
    return cache, cur


def _time_window(model, cache, cur, steps):
    t0 = time.perf_counter()
    for _ in range(steps):
        logits = model(cur, cache=cache)
        cur = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(cur)
    dt = time.perf_counter() - t0
    return steps / dt, cur


def _build_slot_tables(model, store):
    """warm 后读池的 slot 映射,给每个 MoE 块挂上 GPU 查找表与池张量引用。"""
    for blk in _moe_blocks(model):
        L = blk.layer_idx
        ne = int(blk.gate.weight.shape[0])      # gate 输出维 = 专家数
        tab = [-1] * ne
        for e, slot in store._resident._slot_of.get(L, {}).items():
            tab[int(e)] = int(slot)
        blk._slot_table = mx.array(tab, dtype=mx.int32)
        blk._pool_arrays = store._resident._pools[L]
        blk._cap = store.cap_for(L)
        mx.eval(blk._slot_table)


def _nosync_call(self, x: mx.array) -> mx.array:
    """纯 GPU 热路径:slot 重映射用 GPU gather,完全不回 CPU(无 .tolist()/无 LRU)。

    miss 时 slot=-1,clamp 到 0 → 该专家输出不对,但 batch=1/k 专家/48 层的形状不变,
    计时具代表性(稳态是否真有 miss 由正常路径的 store 统计佐证)。
    """
    gates = mx.softmax(self.gate(x), axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    local = mx.take(self._slot_table, inds)         # GPU 重映射,零 host 往返
    if _MISS_SYNC:
        # 模拟真实现每层不可避免的 miss 检查同步(读一个标志回 CPU),不带 Python LRU。
        _ = int(mx.sum((local < 0).astype(mx.int32)))
    local = mx.maximum(local, 0)
    y = self._sub.forward(self._pool_arrays, self._cap, x, local)
    y = (y * scores[..., None]).sum(axis=-2)
    if self.shared_expert is not None:
        y = y + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
    return y


def main():
    assert os.environ.get("RESIDENT_POOL", "1") == "1", "本探针针对常驻池路径,需 RESIDENT_POOL=1"
    model, tok, store = build_streaming_model()
    prompt_ids = mx.array([tok.encode(PROMPT)])

    # ---- run1:正常路径(含每层 .tolist()),并测稳态 miss ----
    cache, cur = _timed_decode(model, prompt_ids, WARM, 0)
    store.reset_stats()                              # 只统计计时窗口的稳态命中
    normal_tps, _ = _time_window(model, cache, cur, STEPS)
    steady_hits, steady_misses = store.hits, store.misses
    steady_hit_rate = store.hit_rate()

    # ---- 建 GPU 查找表,切到无同步路径 ----
    # __call__ 是特殊方法,走类型查找,故 monkeypatch 类方法(所有块共享该类,一起切换;
    # 每块的 _slot_table/_pool_arrays/_cap 在实例上,_nosync_call 通过 self 取)。
    _build_slot_tables(model, store)
    blocks = _moe_blocks(model)
    orig = FileStreamingMoeBlock.__call__
    FileStreamingMoeBlock.__call__ = _nosync_call
    try:
        cache2, cur2 = _timed_decode(model, prompt_ids, WARM, 0)
        nosync_tps, _ = _time_window(model, cache2, cur2, STEPS)
    finally:
        FileStreamingMoeBlock.__call__ = orig

    out = {
        "normal_tps": round(normal_tps, 2),
        "nosync_ceiling_tps": round(nosync_tps, 2),
        "speedup": round(nosync_tps / normal_tps, 3),
        "steady_hit_rate": round(steady_hit_rate, 4),
        "steady_misses": steady_misses,
        "steady_hits": steady_hits,
        "n_moe_layers": len(blocks),
        "warm": WARM, "steps": STEPS,
        "expert_slots": int(os.environ.get("EXPERT_SLOTS", "96")),
        "decision_gate": "≥21 GO / ≲20 NO-GO",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
