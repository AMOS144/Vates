"""运行时内存分块实测:把 decode 稳态的占用拆成各组成块,逐块报字节。

块定义:
  A. 常驻非专家权重   — attention(q/k/v/o + qk_norm)、Gated DeltaNet、router gate、
                        shared expert、embedding、lm_head、各 RMSNorm。流式后这些常驻显存。
  B. 专家常驻池        — ResidentExpertPool 每层物理槽张量(EXPERT_SLOTS×48×单专家)。
  C. staging 预取缓冲  — NativeStagingManager 后台读暂存区(开主动预取时才有)。
  D. MTP drafter       — 投机草稿模型权重(跑 MTP 时才加载;本探针默认不加载)。
  E. KV / 线性 cache   — 全注意力 KV(随上下文增长)+ 线性层递归态(固定)。
  F. MLX 计算缓冲/缓存 — 前向激活、临时量、MLX 可回收缓冲池(active 减去 A~E)。

用法:
  EXPERT_SLOTS=64 STREAM_BLOB_LOADER=1 SEQ=4096 \
  .venv/bin/python -m benchmarks.mem_breakdown
"""
import os
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked
from mlx_streaming.core.cache.quant_kv import AsymmetricQuantizedKVCache
from mlx_streaming import config

GIB = 1024 ** 3
MIB = 1024 ** 2
SEQ = int(os.environ.get("SEQ", "4096"))     # 模拟的上下文长度(KV 增长用)


def _tree_nbytes(obj):
    from mlx.utils import tree_flatten
    tot = 0
    for _, v in tree_flatten(obj):
        if isinstance(v, mx.array):
            tot += v.nbytes
    return tot


def main():
    mx.clear_cache()
    model, tok, store = build_streaming_model()

    # 物化常驻权重(非专家):流式 patch 后,model.parameters() 已不含 40GB 专家。
    mx.eval(model.parameters())
    mx.clear_cache()
    _act_after_build = mx.get_active_memory()
    A = _tree_nbytes(model.parameters())
    print(f"[probe] eval(params)后 active(清缓冲) = {_act_after_build/GIB:.3f} GiB; tree(params) = {A/GIB:.3f} GiB")

    # 跑一段真实 prefill,把线性层递归态填实、专家池按需补起来。
    cache = model.make_cache()
    _rep = int(os.environ.get("PROMPT_REP", "4"))
    ids = mx.array([tok.encode("请解释混合专家模型的稀疏激活原理。" * _rep)])
    print(f"[probe] prefill tokens = {ids.shape[1]}")
    logits, _ = prefill_chunked(model, ids, cache)
    mx.clear_cache(); print(f"[probe] prefill后 active(清缓冲) = {mx.get_active_memory()/GIB:.3f} GiB (chunk={config.prefill_chunk()})")
    # 再 decode 几步,触发 GPU-remap / 预取路径,把专家池补到稳态工作集。
    cur = mx.array([[int(mx.argmax(logits[:, -1, :]))]])
    for _ in range(16):
        logits, _ = forward_with_hidden(model, cur, cache)
        cur = mx.array([[int(mx.argmax(logits[:, -1, :]))]])
    mx.eval(cur)
    mx.clear_cache()
    _act_decode = mx.get_active_memory()
    print(f"[probe] decode后 active(清缓冲) = {_act_decode/GIB:.3f} GiB")

    # B. 专家常驻池:每层池张量物理字节累加
    rp = store._resident
    B = 0
    pool_layers = 0
    for layer, pool in rp._pools.items():
        pool_layers += 1
        for k, v in pool.items():
            B += v.nbytes
    alloc_rows = sum(rp.allocated_slots(l) for l in rp._pools)

    # C. staging 预取缓冲
    C = 0
    stg = getattr(store, "_staging", None)
    if stg is not None:
        C = _tree_nbytes(getattr(stg, "__dict__", {}))

    quant = config.kv_quant()
    FullCache = AsymmetricQuantizedKVCache if quant else KVCache
    lin = [c for c in cache if isinstance(c, ArraysCache)]
    kvs = [c for c in cache if isinstance(c, FullCache)]
    lin_bytes = sum(a.nbytes for c in lin for a in c.cache if a is not None)
    # 当前真实 cache 的 KV 字节(prompt 较短,主要看线性态)
    if quant:
        real_kv = sum(c.nbytes for c in kvs)
    else:
        real_kv = sum((c.keys.nbytes + c.values.nbytes) for c in kvs if c.keys is not None)

    # ---- 测"活跃常驻底":清掉 MLX 可回收缓冲后的 active(= 真正持有的张量)----
    mx.clear_cache()
    mx.eval()
    active_live = mx.get_active_memory()
    peak = mx.get_peak_memory()
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # F. 激活/临时(清缓冲后仍在 active 的非 A/B/C/cache 部分)
    F = max(0, active_live - (A + B + C + real_kv + lin_bytes))

    # E 投影:把全注意力 KV 增长到 SEQ,单独报"上下文每加长会增加多少"(不计入上面的活跃底)
    if quant:
        wq = kvs[0].keys[0]
        H, kb, vb, gs = wq.shape[1], kvs[0].k_bits, kvs[0].v_bits, kvs[0].group_size
        D = wq.shape[-1] * 32 // kb
        dt = kvs[0].keys[1].dtype
        fresh = [AsymmetricQuantizedKVCache(gs, kb, vb) for _ in kvs]
    else:
        _, H, _, D = kvs[0].keys.shape
        dt = kvs[0].keys.dtype
        fresh = [KVCache() for _ in kvs]
    CH = 8192
    for fc in fresh:
        off = 0
        while off < SEQ:
            n = min(CH, SEQ - off)
            fc.update_and_fetch(mx.zeros((1, H, n, D), dtype=dt), mx.zeros((1, H, n, D), dtype=dt))
            mx.eval(list(fc.keys) + list(fc.values)) if quant else mx.eval(fc.keys, fc.values)
            off += n
    kv_at_seq = sum(fc.nbytes for fc in fresh) if quant else \
        sum(fc.keys.nbytes + fc.values.nbytes for fc in fresh)

    def g(x):
        return f"{x / GIB:6.3f} GiB"

    kvlabel = 'K%d/V%d' % (kvs[0].k_bits, kvs[0].v_bits) if quant else 'bf16'
    print("\n==================== 运行时内存分块(decode 稳态, 清缓冲后)====================")
    print(f"配置: EXPERT_SLOTS={store.capacity}  KV量化={kvlabel}")
    print(f"  A. 常驻非专家权重   {g(A)}   (attention/DeltaNet/router/shared/embed/lm_head/norm)")
    print(f"  B. 专家常驻池        {g(B)}   ({pool_layers} 层, 物理槽合计 {alloc_rows} 行, ≈{B/max(alloc_rows,1)/MIB:.3f} MiB/槽)")
    print(f"  C. staging 预取缓冲  {g(C)}   ({'开' if stg is not None else '未开主动预取'})")
    print(f"  D. MTP drafter        (本探针未加载)")
    print(f"  E. cache(当前 prompt) {g(real_kv + lin_bytes)}   (KV {g(real_kv)} + 线性递归态 {lin_bytes/MIB:.1f} MiB)")
    print(f"  F. 激活/分配器水位     {g(F)}   (随 prefill 长度涨;decode 稳态本身很小)")
    print(f"  ------------------------------------------------------------------")
    print(f"  MLX active(清缓冲后) {g(active_live)}   = 实际常驻底(A+B+C+E+F)")
    print(f"  MLX peak(prefill 峰) {g(peak)}")
    print(f"  进程 RSS(ru_maxrss)  {g(rss)}")
    print(f"\n  [E 投影] 上下文 KV 随长度增长:")
    print(f"    SEQ={SEQ:<7} 全注意力 KV = {g(kv_at_seq)}  (线性态固定 {lin_bytes/MIB:.1f} MiB,与长度无关)")
    print(f"    每 token KV ≈ {kv_at_seq/SEQ/1024:.2f} KiB → 128k 时 ≈ {g(kv_at_seq/SEQ*131072)}")


def _last(x):
    return x[:, -1, :]


if __name__ == "__main__":
    main()
