"""真实 80B MTP 自投机基准 + 与非投机贪婪逐 token 一致性校验。

环境变量:K / MAXTOK / PROMPT / QN_CONFIG / MTP_OUT(其余主模型路径见 validate_mtp)。
"""
import json
import os
import statistics
import time

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core.moe import native_moe
from mlx_streaming.core.mem import snapshot, reset_peak
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import forward_with_hidden, mtp_generate, prefill_chunked
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp
from mlx_streaming.model_builder import build_streaming_model

from mlx_streaming import config as _cfg
QN_CONFIG = _cfg.qn_config()
MTP_OUT = _cfg.mtp_out()
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "3"))
PIN_HOT = int(os.environ.get("PIN_HOT", "0"))
PIN_CAL_TOK = int(os.environ.get("PIN_CAL_TOK", "32"))
# 稳态测速:warmup 跑满 MAXTOK(热 baseline+spec 两条路径的 Metal kernel 与常驻专家池/预取),
# 再各重复 REPEAT 次取中位数,避免冷启动编译/补池污染绝对 tok/s。WARMUP_TOK=0 关 warmup。
WARMUP_TOK = int(os.environ.get("WARMUP_TOK", str(MAXTOK)))
REPEAT = int(os.environ.get("REPEAT", "3"))


def _spec_once(model, drafter, tok, n, k):
    ids, stats = mtp_generate(model, drafter, tok,
                              mx.array([tok.encode(PROMPT)]),
                              n, K=k, ids_mode=True, profile=True)
    tps = round(stats["tokens"] / stats["wall_s"], 2)
    return ids, stats, tps


def _baseline_greedy(model, tok, prompt, n):
    cache = model.make_cache()
    ids = mx.array([tok.encode(prompt)])
    t0 = time.perf_counter()
    # prefill 分块:把整段 prompt 的激活峰值压到与 decode 同稳态(见 config.prefill_chunk)。
    logits, _ = prefill_chunked(model, ids, cache)
    out = []
    for _ in range(n):
        nxt = int(mx.argmax(logits[:, -1, :]))
        out.append(nxt)
        cur = mx.array([[nxt]])
        mx.eval(cur)
        if len(out) >= n:
            break
        logits, _ = forward_with_hidden(model, cur, cache)
    return out, round(n / (time.perf_counter() - t0), 2)


def main():
    reset_peak()
    model, tok, store = build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens          # 共享主模型 embedding
    drafter = MTPDrafter(mtp, model.lm_head)

    # warmup:同时跑 baseline + spec 两条路径,编译 Metal kernel(含 multistate/batch verify)
    # 并把专家常驻池/预取热起来,确保后续测的是稳态而非冷启动。默认 warmup=MAXTOK 跑满全长。
    if WARMUP_TOK > 0:
        _baseline_greedy(model, tok, PROMPT, WARMUP_TOK)
        _spec_once(model, drafter, tok, WARMUP_TOK, K)

    # ---- baseline 稳态:重复 REPEAT 次取中位数;最后一次清零统计供命中率口径 ----
    base = None
    base_tps_runs = []
    for r in range(REPEAT):
        if r == REPEAT - 1:
            store.reset_stats()
        base, bt = _baseline_greedy(model, tok, PROMPT, MAXTOK)
        base_tps_runs.append(bt)
    base_tps = statistics.median(base_tps_runs)
    base_miss, base_hit = store.misses, store.hits
    base_prefetch_loads = store._resident.prefetch_loads
    base_prefetch_hits = store._resident.prefetch_hits

    # 可选:baseline 之后、spec 之前校准每层热专家，并预取/钉入 resident pool。
    # 这样 disk_load_ratio 的分母仍是未 pin baseline，能直接衡量 pin 是否压低 spec miss。
    if PIN_HOT > 0:
        store.record = True
        _baseline_greedy(model, tok, PROMPT, PIN_CAL_TOK)
        for li in store.recorded_layers():
            store.pin(li, store.hot(li, PIN_HOT))
        store.record = False
        store.reset_stats()

    # 诊断:开 handler 触发时刻探针(仅覆盖 spec 阶段,enable 会清零旧日志)。
    _hprof = bool(os.environ.get("STAGING_HPROF"))
    if _hprof:
        from mlx_streaming import native_moe_ext as _Nhp
        _Nhp.staging_hprof_enable(True)

    # ---- spec 稳态:重复 REPEAT 次取中位数;最后一次清零统计 + reset_peak + 保留其 stats ----
    ids, stats, spec_tps_runs = None, None, []
    for r in range(REPEAT):
        if r == REPEAT - 1:
            store.reset_stats()
            reset_peak()
            from mlx_streaming.core.profiling import tprof_reset, union_reset
            tprof_reset()                     # 探针只统计最终测量轮(与命中率/内存口径一致)
            union_reset()                     # 并集专家数也只统计最终轮
        ids, stats, st = _spec_once(model, drafter, tok, MAXTOK, K)
        spec_tps_runs.append(st)
    spec_tps = statistics.median(spec_tps_runs)
    spec_miss, spec_hit = store.misses, store.hits
    spec_prefetch_loads = store._resident.prefetch_loads
    spec_prefetch_hits = store._resident.prefetch_hits
    after = snapshot()
    proj = stats.get("proj_no_replay_tps", 0.0)
    result = {
        "K": K,
        "max_tokens": MAXTOK,
        "warmup_tok": WARMUP_TOK,
        "repeat": REPEAT,
        "exact_match": ids == base,
        "n_mismatch": sum(1 for a, b in zip(ids, base) if a != b),
        "avg_accept_len": stats["avg_accept_len"],
        "steps": stats["steps"],
        "verify_mode": stats.get("verify_mode"),
        "direct_commits": stats.get("direct_commits"),
        "fallback_replays": stats.get("fallback_replays"),
        "replayed_tokens": stats.get("replayed_tokens"),
        "spec_tok_per_s": spec_tps,
        "baseline_tok_per_s": base_tps,
        "speedup": round(spec_tps / max(base_tps, 1e-6), 2),
        "spec_tps_runs": spec_tps_runs,
        "baseline_tps_runs": base_tps_runs,
        "spec_tps_minmax": [min(spec_tps_runs), max(spec_tps_runs)],
        # 分段计时与「重放免费」投机上限
        "t_draft_s": stats.get("t_draft_s"),
        "t_snap_s": stats.get("t_snap_s"),
        "t_verify_s": stats.get("t_verify_s"),
        "t_commit_s": stats.get("t_commit_s"),
        "t_replay_s": stats.get("t_replay_s"),
        "t_sync_s": stats.get("t_sync_s"),
        "t_finalize_s": stats.get("t_finalize_s"),
        "proj_no_replay_tps": proj,
        "proj_no_replay_speedup": round(proj / max(base_tps, 1e-6), 2),
        "baseline_disk_loads": base_miss,
        "baseline_prefetch_loads": base_prefetch_loads,
        "baseline_prefetch_hits": base_prefetch_hits,
        "baseline_hit_rate": round(base_hit / max(base_hit + base_miss, 1), 3),
        "spec_disk_loads": spec_miss,
        "spec_prefetch_loads": spec_prefetch_loads,
        "spec_prefetch_hits": spec_prefetch_hits,
        "spec_hit_rate": round(spec_hit / max(spec_hit + spec_miss, 1), 3),
        "disk_load_ratio": round(spec_miss / max(base_miss, 1), 2),
        # 双源 acquire 分路计数:n_miss==0 走全 GPU 快路径;任一路由 miss 则整层落 host 慢路径
        # (.tolist 全批同步 + demand 读盘)。fallback 占比高 → 即使 hit 高,慢路径仍按"层"频繁触发。
        "gpu_fastpath": getattr(store._resident, "gpu_fastpath", None),
        "gpu_fallback": getattr(store._resident, "gpu_fallback", None),
        "pin_hot": PIN_HOT,
        "pin_cal_tok": PIN_CAL_TOK,
        "pinned_experts": store.pinned_count(),
        "expert_slots": store.capacity,
        "mlx_active_gb": round(after.mlx_active_bytes / 1e9, 2),
        "mlx_peak_gb": round(after.mlx_peak_bytes / 1e9, 2),
        "rss_gb": round(after.rss_bytes / 1e9, 2),
        # 内存分块(清缓冲后的真实常驻):A 权重 / B 专家池 / C staging / D MTP / F 激活
        "mem_breakdown": _mem_breakdown(model, store, mtp),
        "prefill_chunk": _cfg.prefill_chunk(),
        "native_stage_cache": native_moe.stage_cache_stats(),
        "bg_stats": (store._bg.stats() if getattr(store, "_bg", None) is not None else None),
        "window_prof": _window_prof(),
        "predict_recall": _predict_recall(),
        "miss_attrib": _miss_attrib(),
        "prefetch_tprof": _prefetch_tprof(stats.get("wall_s")),
        "union_experts": _union_prof(),
    }
    # 噪声地板测量口径:DUMP_IDS=1 时把 baseline greedy 与 spec 的完整 token 序列打进日志,
    # 供跨进程 run-to-run 逐位对比(默认关闭,不污染常规输出)。
    if os.environ.get("DUMP_IDS"):
        print("DUMP_BASE_IDS " + json.dumps(list(base)))
        print("DUMP_SPEC_IDS " + json.dumps(list(ids)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if _hprof:
        # dump (gen, layer, t_fire) 原始日志,供离线分析回调触发时刻分布。
        _flat = _Nhp.staging_hprof_get()    # 扁平 [gen,layer,t, ...]
        _path = os.environ.get("STAGING_HPROF_OUT", "/tmp/ab/hprof.jsonl")
        os.makedirs(os.path.dirname(_path), exist_ok=True)
        _n = 0
        with open(_path, "w") as _f:
            for i in range(0, len(_flat), 3):
                _f.write(json.dumps([int(_flat[i]), int(_flat[i + 1]), float(_flat[i + 2])]) + "\n")
                _n += 1
        print(f"[hprof] wrote {_n} handler records to {_path}")


def _tree_nbytes(obj):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(obj) if isinstance(v, mx.array))


def _mem_breakdown(model, store, mtp):
    """把 decode 稳态内存拆成各块(清掉 MLX 可回收缓冲后量真实常驻)。

    A 常驻非专家权重 / B 专家常驻池 / C staging 预取 / D MTP drafter(扣共享 embedding)/
    F 激活+临时(= active底 − A−B−C−D)。返回 GiB 字典。
    """
    GIB = 1024 ** 3
    A = _tree_nbytes(model.parameters())
    rp = store._resident
    B = sum(v.nbytes for pool in rp._pools.values() for v in pool.values())
    C = 0
    stg = getattr(store, "_staging", None)
    if stg is not None:
        from mlx.utils import tree_flatten
        C = sum(v.nbytes for _, v in tree_flatten(getattr(stg, "__dict__", {}))
                if isinstance(v, mx.array))
    # D:MTP drafter 全部权重减去与主模型共享的 embedding(run 里 mtp.embed_tokens = 主模型的)
    D = _tree_nbytes(mtp.parameters())
    emb = getattr(getattr(mtp, "embed_tokens", None), "weight", None)
    shared = 0
    if isinstance(emb, mx.array):
        # 共享 embedding 在 A 已计入;若 mtp 的 embed 与主模型同一对象则从 D 扣除避免重复
        shared = _tree_nbytes(mtp.embed_tokens) if hasattr(mtp, "embed_tokens") else 0
    D = max(0, D - shared)
    mx.clear_cache()
    active = mx.get_active_memory()
    peak = mx.get_peak_memory()
    F = max(0, active - (A + B + C + D))
    pool_rows = sum(rp.allocated_slots(l) for l in rp._pools)
    return {
        "A_weights_gib": round(A / GIB, 3),
        "B_expert_pool_gib": round(B / GIB, 3),
        "B_pool_rows": pool_rows,
        "C_staging_gib": round(C / GIB, 3),
        "D_mtp_drafter_gib": round(D / GIB, 3),
        "F_activation_temp_gib": round(F / GIB, 3),
        "active_live_gib": round(active / GIB, 3),
        "peak_gib": round(peak / GIB, 3),
    }


def _union_prof():
    """按 seq 分桶汇报每层路由专家并集大小(avg/max/p99/分层);seq=K 桶即 MTP verify 的专家并集。

    需 UNION_PROF=1 才有数据。返回 {seq: {avg_union, max_union, p99_union, per_layer...}};
    verify 桶 = seq==K(verify_in=[x, d_1..d_{K-1}] 恰 K 个 token,见 mtp/generate.py)。
    U_max 即该桶各层并集的全局最大值,是池 cap 下限的正确性依据(cap 必须 ≥ U_max 才保证不溢出)。
    """
    from mlx_streaming.core.profiling import UNION_PROF as U, UNION_SAMPLES as S
    if not U:
        return None

    def _p(vals, q):
        # 最近秩法分位数(vals 已升序):q∈[0,1]。样本少时直接给保守上界。
        if not vals:
            return None
        idx = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
        return vals[idx]

    out = {}
    for seq in sorted(U):
        s, n = U[seq]
        rec = {"avg_union": round(s / max(1, n), 2), "n_layer_calls": n}
        # 由分层原始样本汇总出该 seq 桶的 U_max / p99（cap 下限的正确性依据）。
        layer_map = S.get(seq)
        if layer_map:
            flat = sorted(v for lst in layer_map.values() for v in lst)
            rec["max_union"] = flat[-1]
            rec["p99_union"] = _p(flat, 0.99)
            rec["p50_union"] = _p(flat, 0.50)
            rec["n_samples"] = len(flat)
            # 分层分布：每层的 max / avg / 样本数（按层号排序）。
            per_layer = {}
            for li in sorted(layer_map):
                lst = layer_map[li]
                per_layer[li] = {
                    "max": max(lst),
                    "avg": round(sum(lst) / len(lst), 2),
                    "n": len(lst),
                }
            rec["per_layer"] = per_layer
            rec["max_layer_idx"] = max(per_layer, key=lambda li: per_layer[li]["max"])
        out[f"seq{seq}"] = rec
    # 便捷:verify 桶 = seq==K(verify_in=[x, d_1..d_{K-1}] 恰 K 个 token)
    verify_seq = K
    verify = out.get(f"seq{verify_seq}")
    return {"by_seq": out, "verify_seq": verify_seq,
            "verify_avg_union": (verify or {}).get("avg_union"),
            "verify_max_union": (verify or {}).get("max_union"),
            "verify_p99_union": (verify or {}).get("p99_union")}


def _miss_attrib():
    from mlx_streaming.core.profiling import MISS_ATTRIB as M
    if not M["n"]:
        return None
    routed = max(1, M["routed"])
    miss = M["miss_A_predicted"] + M["miss_B_unpredicted"]
    out = {
        "routed": M["routed"],
        "hit_rate": round(M["resident_hit"] / routed, 4),
        "miss_rate": round(miss / routed, 4),
        # A：预测到却没进池（budget/时序/驱逐）；B：没预测到（召回缺口）
        "miss_A_predicted": M["miss_A_predicted"],
        "miss_B_unpredicted": M["miss_B_unpredicted"],
        "A_share_of_miss": round(M["miss_A_predicted"] / max(1, miss), 4),
        "B_share_of_miss": round(M["miss_B_unpredicted"] / max(1, miss), 4),
    }
    if M["dec_n"]:
        # decode/verify 热路径专用（与上面 prefill-主导的全量分开看）：
        dr = max(1, M["dec_routed"])
        dmiss = M["dec_miss_A"] + M["dec_miss_B"]
        out["decode"] = {
            "routed": M["dec_routed"],
            "hit_rate": round(M["dec_resident_hit"] / dr, 4),
            "miss_rate": round(dmiss / dr, 4),
            "miss_A_predicted": M["dec_miss_A"],
            "miss_B_unpredicted": M["dec_miss_B"],
            "A_share_of_miss": round(M["dec_miss_A"] / max(1, dmiss), 4),
            "B_share_of_miss": round(M["dec_miss_B"] / max(1, dmiss), 4),
            # miss_A 细分:时序(pread 没完成) vs 驱逐(就绪过但 acquire 前不在池)
            "miss_A_timing": M["dec_miss_A_timing"],
            "miss_A_evicted": M["dec_miss_A_evicted"],
            "A_timing_share": round(M["dec_miss_A_timing"] / max(1, M["dec_miss_A"]), 4),
            "A_evicted_share": round(M["dec_miss_A_evicted"] / max(1, M["dec_miss_A"]), 4),
        }
    return out


def _predict_recall():
    from mlx_streaming.core.profiling import PREDICT_RECALL_PROF as P
    if not P["n"]:
        return None
    return {"recall": round(P["hit"] / max(1, P["routed"]), 4),
            "avg_routed": round(P["routed"] / P["n"], 2), "n": P["n"]}


def _window_prof():
    from mlx_streaming.core.profiling import WINDOW_PROF
    if not WINDOW_PROF["n"]:
        return None
    return {"avg_ms": round(WINDOW_PROF["sum_s"] / WINDOW_PROF["n"] * 1000, 4),
            "n": WINDOW_PROF["n"]}


def _prefetch_tprof(wall_s=None):
    """预取 host 墙钟探针汇总(PREFETCH_TPROF=1 才有数据)。

    汇报每段:总秒数、占最终测量轮 wall 的百分比、单次调用均值(ms)。
    注意:这是"主线程不可重叠的 host 时间";gate matmul / pool scatter 的 GPU 执行落在
    前向末尾统一 eval,不在此口径,需用消融(NATIVE_NO_SUBMIT/NATIVE_NO_PROMOTE)差值测。
    """
    from mlx_streaming.core.profiling import PREFETCH_TPROF as T
    if not (T["predict_n"] or T["promote_n"] or T["submit_n"]):
        return None

    def _seg(s, n):
        d = {"total_s": round(T[s], 4)}
        if wall_s:
            d["pct_wall"] = round(T[s] / wall_s * 100, 2)
        if n:
            d["avg_ms"] = round(T[s] / n * 1000, 4)
        return d

    host_total = T["predict_s"] + T["submit_s"] + T["promote_s"]
    out = {
        "predict": _seg("predict_s", T["predict_n"]),
        "submit": _seg("submit_s", T["submit_n"]),
        "promote": _seg("promote_s", T["promote_n"]),
        # promote 内部细分(take 锁读 / route 成员同步 / place 切片+scatter 入图)
        "promote_take": _seg("take_s", T["promote_n"]),
        "promote_route": _seg("route_s", T["promote_n"]),
        "promote_place": _seg("place_s", T["promote_n"]),
        "host_total_s": round(host_total, 4),
        "host_total_pct_wall": (round(host_total / wall_s * 100, 2) if wall_s else None),
        "counts": {"predict_n": T["predict_n"], "submit_n": T["submit_n"],
                   "promote_n": T["promote_n"], "place_experts": T["place_experts"]},
    }
    return out


if __name__ == "__main__":
    main()
