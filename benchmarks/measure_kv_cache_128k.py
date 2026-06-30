"""实测 Qwen3-Next-80B-A3B 在 SEQ(默认128k)上下文下的 cache 占用。

混合架构:(idx+1)%full_attention_interval!=0 的层是线性注意力(Gated DeltaNet,
递归状态固定、与长度无关);其余是全注意力(KVCache 随长度线性增长)。

测法(不跑真 128k prefill,那在 80B 流式上太慢):
1. 建真实模型 + make_cache;
2. 短 prefill 几个 token,把线性层 conv_state/ssm_state 填成真数组 → 读真实 shape/dtype/字节;
3. 新建 12 个全注意力 KVCache,逐层 update_and_fetch 真增长到 SEQ,量 MLX 实际分配 delta;
4. 汇总并与理论值对账。

用法:SEQ=131072 python benchmarks/measure_kv_cache_128k.py
"""
import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.generate import forward_with_hidden
from mlx_streaming.core.cache.quant_kv import AsymmetricQuantizedKVCache

SEQ = int(os.environ.get("SEQ", "131072"))
GIB = 1024 ** 3
MIB = 1024 ** 2

# 全注意力 cache 类型:KV_QUANT=1 时为非对称量化(K4/V3),否则 bf16 KVCache。
_QUANT = bool(getattr(__import__("mlx_streaming.config", fromlist=["kv_quant"]), "kv_quant")())


def main():
    model, tok, store = build_streaming_model()
    cache = model.make_cache()

    # 1) 短 prefill:填充线性层 conv/ssm 真实状态(几 token 即可,状态大小与长度无关)。
    ids = mx.array([tok.encode("你好,世界。请用一句话解释混合专家模型。")])
    forward_with_hidden(model, ids, cache)
    mx.eval(*[a for c in cache if isinstance(c, ArraysCache)
              for a in c.cache if a is not None])
    mx.eval(*[c.keys for c in cache if isinstance(c, KVCache) and c.keys is not None])

    _FullCache = AsymmetricQuantizedKVCache if _QUANT else KVCache
    lin = [c for c in cache if isinstance(c, ArraysCache)]
    kv = [c for c in cache if isinstance(c, _FullCache)]
    print(f"层数:总 {len(cache)} = 线性 {len(lin)} + 全注意力 {len(kv)}"
          f"  | KV 量化={'K%d/V%d' % (kv[0].k_bits, kv[0].v_bits) if _QUANT else 'bf16'}")

    # 2) 线性层递归状态(真实数组字节,固定、与 SEQ 无关)
    c0 = lin[0]
    conv, ssm = c0.cache[0], c0.cache[1]
    print(f"\n[线性层·每层] conv_state {tuple(conv.shape)} {conv.dtype} = {conv.nbytes/MIB:.3f} MiB"
          f" | ssm_state {tuple(ssm.shape)} {ssm.dtype} = {ssm.nbytes/MIB:.3f} MiB")
    lin_bytes = sum(a.nbytes for c in lin for a in c.cache if a is not None)
    print(f"[线性层·合计 {len(lin)} 层] = {lin_bytes/MIB:.2f} MiB  (固定,不随上下文增长)")

    # 3) 全注意力 cache 真增长到 SEQ,量 MLX 实际分配
    if _QUANT:
        # 量化:从已 prefill 的 quant cache 读 H / head_dim(由打包宽度反推)
        wq = kv[0].keys[0]
        H = wq.shape[1]
        kb, vb, gs = kv[0].k_bits, kv[0].v_bits, kv[0].group_size
        D = wq.shape[-1] * 32 // kb
        kdt = kv[0].keys[1].dtype                 # scales/biases dtype(= 原 K dtype)
        print(f"\n[全注意力·每层] n_kv_heads={H} head_dim={D} K{kb}/V{vb} g{gs} | step=256")
        mx.eval()
        before = mx.get_active_memory()
        fresh = [AsymmetricQuantizedKVCache(gs, kb, vb) for _ in kv]
        CH = 16384
        for fc in fresh:
            off = 0
            while off < SEQ:
                n = min(CH, SEQ - off)
                k = mx.zeros((1, H, n, D), dtype=kdt)
                v = mx.zeros((1, H, n, D), dtype=kdt)
                fc.update_and_fetch(k, v)
                mx.eval([a for a in fc.keys] + [a for a in fc.values])
                off += n
        after = mx.get_active_memory()
        kv_nbytes = sum(fc.nbytes for fc in fresh)
        alloc_len = fresh[0].keys[0].shape[2]
        per_layer = fresh[0].nbytes
    else:
        kdt = kv[0].keys.dtype
        _, H, _, D = kv[0].keys.shape
        print(f"\n[全注意力·每层] n_kv_heads={H} head_dim={D} dtype={kdt} | step={KVCache.step}")
        mx.eval()
        before = mx.get_active_memory()
        fresh = [KVCache() for _ in kv]
        CH = 16384                                   # 分块灌,控住瞬时输入内存
        for fc in fresh:
            off = 0
            while off < SEQ:
                n = min(CH, SEQ - off)
                k = mx.zeros((1, H, n, D), dtype=kdt)
                v = mx.zeros((1, H, n, D), dtype=kdt)
                fc.update_and_fetch(k, v)
                mx.eval(fc.keys, fc.values)          # 落定本块,释放输入
                off += n
        after = mx.get_active_memory()
        kv_nbytes = sum(fc.keys.nbytes + fc.values.nbytes for fc in fresh)
        alloc_len = fresh[0].keys.shape[2]
        per_layer = fresh[0].keys.nbytes + fresh[0].values.nbytes
    print(f"[全注意力·每层] K+V 实际分配长度={alloc_len} (请求 {SEQ})"
          f" → {per_layer/MIB:.2f} MiB/层")
    print(f"[全注意力·合计 {len(kv)} 层] nbytes 累加 = {kv_nbytes/GIB:.3f} GiB")
    print(f"[全注意力·合计] MLX active 实测 delta = {(after-before)/GIB:.3f} GiB")

    # 4) 总计 + 理论对账
    total = kv_nbytes + lin_bytes
    print(f"\n==== SEQ={SEQ} cache 总占用 ====")
    print(f"  全注意力 KV : {kv_nbytes/GIB:.3f} GiB")
    print(f"  线性递归态  : {lin_bytes/MIB:.1f} MiB")
    print(f"  合计        : {total/GIB:.3f} GiB")

    if _QUANT:
        # 每 token 每层:K 打包 D*kb/8 + V 打包 D*vb/8 + 各 2 组(scales+biases)元数据
        meta = kdt.size * 2 * (D // gs) * 2        # K、V 各有 scales+biases
        per_tok = len(kv) * (D * kb // 8 + D * vb // 8 + meta) * H
        print(f"\n[对账] 理论每 token KV(K{kb}/V{vb}+元数据)= {per_tok} B = {per_tok/1024:.1f} KiB"
              f" → ×{SEQ} = {per_tok*SEQ/GIB:.3f} GiB")
    else:
        per_tok = len(kv) * 2 * H * D * kdt.size      # K+V, 每 token 每层
        print(f"\n[对账] 理论每 token KV = {per_tok} B = {per_tok/1024:.1f} KiB"
              f" → ×{SEQ} = {per_tok*SEQ/GIB:.3f} GiB")


if __name__ == "__main__":
    main()
