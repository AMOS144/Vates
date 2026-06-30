"""分层时序探针(非侵入 monkey-patch):验证 miss_A_timing 是否集中在晚层、
且晚层 promote 的 take() 读到的"已就绪"专家数是否偏少。

用法(在 worktree 根目录):
  EXPERT_SLOTS=32 RESIDENT_POOL=1 MTP_VERIFY_MODE=batch K=2 NATIVE_FUSED_PREFETCH=1 \
  STREAM_BLOB_LOADER=1 STREAM_BLOB_BG_BUDGET=12 CROSS_LAYER_PREDICT_WIDTH=16 \
  GPU_REMAP=1 VERIFY_GPU_REMAP=1 EVICT_POLICY=lfu MAXTOK=96 GPU_REMAP_PROMOTE_FILTER=1 \
  MISS_ATTRIB=1 CROSS_LAYER_CUTOFF=999 CROSS_LAYER_AHEAD_LO=1 CROSS_LAYER_AHEAD_HI=1 \
  .venv/bin/python -m mlx_streaming.tools.probe_layered_timing

原理:patch note_miss_attrib 按"当前层"桶累加(当前层由 patch FileStreamingMoeBlock.__call__
在入口写入线程局部全局),并在 spec 阶段开始前清零,排除 baseline/warmup 污染。
"""
import sys

# 当前正在构图的层号(单线程图构建,安全)。
_CUR = {"l": -1}
# 分层桶:layer -> dict(routed, miss_A_timing, ready_sum, ready_n)
_BUCKET: "dict[int, dict]" = {}
_ENABLED = {"on": False}


def _bucket(layer: int) -> dict:
    b = _BUCKET.get(layer)
    if b is None:
        b = {"routed": 0, "A_timing": 0, "ready_sum": 0, "ready_n": 0}
        _BUCKET[layer] = b
    return b


def _patched_note(uniq, predicted, resident, is_decode, ready=None):
    # 仅统计 decode/verify 热路径,且仅在 spec 阶段开启后累加。
    if not (_ENABLED["on"] and is_decode):
        return
    layer = _CUR["l"]
    if layer < 0:
        return
    pset = predicted or set()
    rset = ready or set()
    b = _bucket(layer)
    b["ready_sum"] += len(rset)
    b["ready_n"] += 1
    for e in uniq:
        b["routed"] += 1
        # miss_A_timing 口径:本层真实路由专家,既不在常驻、又在预测集里(预测对了)、
        # 但不在已就绪集(pread 没在 take 之前落地) → 纯时序 miss。
        if e not in resident and e in pset and e not in rset:
            b["A_timing"] += 1


def _install_patches():
    import mlx_streaming.core.profiling as prof
    import mlx_streaming.core.moe.block as block

    # 1) patch note_miss_attrib(block 用的是 from-import 的独立绑定,需各自替换)。
    prof.note_miss_attrib = _patched_note
    block.note_miss_attrib = _patched_note

    # 2) patch FileStreamingMoeBlock.__call__:入口写入当前层号。
    _orig_call = block.FileStreamingMoeBlock.__call__

    def _wrapped_call(self, x):
        _CUR["l"] = self.layer_idx
        return _orig_call(self, x)

    block.FileStreamingMoeBlock.__call__ = _wrapped_call

    # 3) patch run_mtp_spec.mtp_generate:spec 阶段开始时清零并开启累加,排除 baseline 污染。
    import mlx_streaming.runtime.run_mtp_spec as r
    _orig_gen = r.mtp_generate

    def _wrapped_gen(*a, **kw):
        _BUCKET.clear()
        _ENABLED["on"] = True
        return _orig_gen(*a, **kw)

    r.mtp_generate = _wrapped_gen


def _report():
    if not _BUCKET:
        print("[probe] 无数据(检查 MISS_ATTRIB=1 是否开启)", file=sys.stderr)
        return
    layers = sorted(_BUCKET)
    print("\n=== 分层时序探针(每层:routed / miss_A_timing / A_timing占比 / 平均就绪数) ===")
    tot_routed = tot_at = 0
    for L in layers:
        b = _BUCKET[L]
        routed = b["routed"]
        at = b["A_timing"]
        avg_ready = b["ready_sum"] / max(1, b["ready_n"])
        share = at / max(1, routed)
        tot_routed += routed
        tot_at += at
        bar = "#" * int(share * 40)
        print(f"  L{L:>2}: routed={routed:>4} A_timing={at:>4} ({share:5.1%}) "
              f"avg_ready={avg_ready:4.1f} {bar}")
    print(f"  --- 合计 routed={tot_routed} A_timing={tot_at} "
          f"({tot_at / max(1, tot_routed):.1%}) ---")

    # 早层 vs 晚层对比(以 cutoff=6 同口径切)。
    early = [L for L in layers if L < len(layers) // 2]
    late = [L for L in layers if L >= len(layers) // 2]

    def _agg(ls):
        r = sum(_BUCKET[L]["routed"] for L in ls)
        a = sum(_BUCKET[L]["A_timing"] for L in ls)
        rs = sum(_BUCKET[L]["ready_sum"] for L in ls)
        rn = sum(_BUCKET[L]["ready_n"] for L in ls)
        return r, a, a / max(1, r), rs / max(1, rn)

    er, ea, esh, erd = _agg(early)
    lr, la, lsh, lrd = _agg(late)
    print(f"\n  前半层(L<{len(layers)//2}): A_timing占比={esh:.1%} 平均就绪={erd:.1f}")
    print(f"  后半层(L>={len(layers)//2}): A_timing占比={lsh:.1%} 平均就绪={lrd:.1f}")


def main():
    _install_patches()
    import mlx_streaming.runtime.run_mtp_spec as r
    r.main()
    _report()


if __name__ == "__main__":
    main()
