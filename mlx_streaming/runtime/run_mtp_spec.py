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
    _dump_margin = bool(os.environ.get("DUMP_MARGIN"))
    out = []
    for _ in range(n):
        lg = logits[:, -1, :]
        nxt = int(mx.argmax(lg))
        out.append(nxt)
        if _dump_margin:
            # 诊断:每步 top-2 logit 与差值,用于判定路径间发散是「FP 近平局」还是「真错」。
            v = lg.reshape(-1)
            top2 = mx.argpartition(-v, 2)[:2]
            t2 = [int(x) for x in top2.tolist()]
            vv = {t: float(v[t]) for t in t2}
            st = sorted(t2, key=lambda t: -vv[t])
            print(f"MARGIN step={len(out)-1} top1={st[0]}({vv[st[0]]:.5f}) "
                  f"top2={st[1]}({vv[st[1]]:.5f}) gap={vv[st[0]]-vv[st[1]]:.6f}", flush=True)
        cur = mx.array([[nxt]])
        mx.eval(cur)
        if len(out) >= n:
            break
        logits, _ = forward_with_hidden(model, cur, cache)
    return out, round(n / (time.perf_counter() - t0), 2)


def _resident_ready_hit_rate(deadline):
    """Return true main-pool readiness, excluding unscheduled layer 0.

    This deliberately does not count completed staging rows: the current
    unified-pool consumer copies those rows into resident slots at demand.
    """
    if not deadline:
        return None
    rows = [
        row for layer, row in deadline.get("per_layer", {}).items()
        if int(layer) > 0
    ]
    actual = sum(int(row.get("actual_unique", 0)) for row in rows)
    resident = sum(int(row.get("real_resident", 0)) for row in rows)
    return round(resident / max(actual, 1), 6) if rows else None


def _read_complete_ceiling(deadline):
    """Resident plus fully read staging rows; not a zero-copy hit rate."""
    if not deadline:
        return None
    rows = [
        row for layer, row in deadline.get("per_layer", {}).items()
        if int(layer) > 0
    ]
    actual = sum(int(row.get("actual_unique", 0)) for row in rows)
    resident = sum(int(row.get("real_resident", 0)) for row in rows)
    staged = sum(
        int(row.get("staging_complete", row.get("side_prefetch_complete", 0)))
        for row in rows
    )
    return round((resident + staged) / max(actual, 1), 6) if rows else None


def _direct_ready_hit_rate(deadline):
    """Rows directly addressable by MoE at target entry.

    In direct-slot mode both the real rows and completely published side rows
    belong to the stable GPU pool allocation.  Unlike global staging, a side
    hit needs no demand-time promotion or memcpy before gather.
    """
    return _read_complete_ceiling(deadline)


def _unified_pool_occupancy(store):
    """Diagnostic snapshot of verified vs speculative rows after the run."""
    if not _cfg.zerocopy_dual_source():
        return None
    try:
        from mlx_streaming import native_moe_ext as native
        rows = {}
        for layer in range(48):
            total = int(native.real_region_count(layer))
            verified = len(native.real_verified_contents(layer)) // 2
            cap = int(store.cap_for(layer))
            rows[str(layer)] = {
                "total": total,
                "verified": verified,
                "speculative": total - verified,
                "free": cap - total,
                "cap": cap,
            }
        return rows
    except Exception:
        return None


def main():
    if (
        _cfg.prefetch_progressive()
        and K >= 3
        and _cfg.prefetch_progressive_mode() != "k3"
    ):
        raise ValueError(
            "K>=3 requires PREFETCH_PROGRESSIVE_MODE=k3; refusing to "
            "silently benchmark the 15-expert K1 rerank width",
        )
    reset_peak()
    route_trace_out = os.environ.get("ROUTE_TRACE_OUT", "").strip()
    if route_trace_out:
        from mlx_streaming.core import route_trace
        route_trace.enable()
    model, tok, store = build_streaming_model()
    with open(QN_CONFIG) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(
        args, MTP_OUT, quantize=True, bits=_cfg.mtp_bits(),
        group_size=_cfg.mtp_group_size(),
        stream_experts=_cfg.mtp_stream_experts(),
        expert_dir=_cfg.mtp_expert_dir(),
        expert_slots=_cfg.mtp_expert_slots(),
    )
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
            if _cfg.demand_async():
                from mlx_streaming import native_moe_ext as _async_native
                _async_native.demand_async_stats_reset()
            from mlx_streaming.core.profiling import rerank_reset
            rerank_reset()
            _prefetch_audits_reset(store)
        base, bt = _baseline_greedy(model, tok, PROMPT, MAXTOK)
        base_tps_runs.append(bt)
    base_async = None
    if _cfg.demand_async():
        from mlx_streaming import native_moe_ext as _async_native
        _async_native.demand_async_check()
        base_async = list(_async_native.demand_async_stats())
    base_tps = statistics.median(base_tps_runs)
    base_miss, base_hit = store.misses, store.hits
    if base_async is not None:
        base_miss = int(base_async[3])
        base_hit = int(base_async[4])
    base_prefetch_loads = store._resident.prefetch_loads
    base_prefetch_hits = store._resident.prefetch_hits
    base_prefetch_audit = _prefetch_audit_prof(primary_seq=1)
    base_prefetch_deadline = _prefetch_deadline_prof()
    base_rerank = _rerank_prof()

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
            if _cfg.demand_async():
                from mlx_streaming import native_moe_ext as _async_native
                _async_native.demand_async_stats_reset()
            reset_peak()
            from mlx_streaming.core.profiling import tprof_reset, union_reset
            tprof_reset()                     # 探针只统计最终测量轮(与命中率/内存口径一致)
            union_reset()                     # 并集专家数也只统计最终轮
            from mlx_streaming.core.profiling import rerank_reset
            rerank_reset()
            _prefetch_audits_reset(store)
            if os.environ.get("SIDEREGION_STATS") == "1":
                # Keep the final-repeat counters occurrence-local.  Draining
                # happens before the timed generate call, so old warmup fills
                # cannot be charged to this measurement.
                import mlx_streaming.native_moe_ext as _side_native
                _side_native.prefetch_staging_drain()
                _side_native.sideregion_prefetch_stats_reset()
        ids, stats, st = _spec_once(model, drafter, tok, MAXTOK, K)
        spec_tps_runs.append(st)
    spec_async = None
    if _cfg.demand_async():
        from mlx_streaming import native_moe_ext as _async_native
        _async_native.demand_async_check()
        spec_async = list(_async_native.demand_async_stats())
    spec_tps = statistics.median(spec_tps_runs)
    spec_miss, spec_hit = store.misses, store.hits
    if spec_async is not None:
        spec_miss = int(spec_async[3])
        spec_hit = int(spec_async[4])
    spec_prefetch_loads = store._resident.prefetch_loads
    spec_prefetch_hits = store._resident.prefetch_hits
    after = snapshot()
    proj = stats.get("proj_no_replay_tps", 0.0)
    spec_deadline = _prefetch_deadline_prof()
    spec_prejoin = _prefetch_prejoin_prof()
    spec_acceptance = _progressive_acceptance_prof(store)
    legacy_base_pool_hit = round(base_hit / max(base_hit + base_miss, 1), 3)
    legacy_spec_pool_hit = round(spec_hit / max(spec_hit + spec_miss, 1), 3)
    # Global staging is not GPU-addressable until demand-time promotion, while
    # direct side rows are part of the stable pool allocation and are true
    # ready hits as soon as native publishes them complete.
    strict_base_hit = _resident_ready_hit_rate(base_prefetch_deadline)
    direct_slots = _cfg.prefetch_direct_slots()
    strict_spec_hit = (
        _direct_ready_hit_rate(spec_deadline)
        if direct_slots
        else _resident_ready_hit_rate(spec_prejoin)
    )
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
        "baseline_hit_rate": strict_base_hit,
        "baseline_pool_access_hit_rate": legacy_base_pool_hit,
        "spec_disk_loads": spec_miss,
        "spec_prefetch_loads": spec_prefetch_loads,
        "spec_prefetch_hits": spec_prefetch_hits,
        "spec_hit_rate": strict_spec_hit,
        "spec_hit_rate_definition": (
            "scheduled layers only: actual route experts already directly "
            "addressable in GPU real+side pool rows at target entry"
            if direct_slots else
            "scheduled layers only: actual route experts already resident in "
            "the GPU-addressable main pool before target demand promotion"
        ),
        "spec_pool_access_hit_rate": legacy_spec_pool_hit,
        "spec_ssd_read_complete_ceiling": (
            strict_spec_hit if direct_slots
            else _read_complete_ceiling(spec_prejoin)
        ),
        "disk_load_ratio": round(spec_miss / max(base_miss, 1), 2),
        # 双源 acquire 分路计数:n_miss==0 走全 GPU 快路径;任一路由 miss 则整层落 host 慢路径
        # (.tolist 全批同步 + demand 读盘)。fallback 占比高 → 即使 hit 高,慢路径仍按"层"频繁触发。
        "gpu_fastpath": (
            int(spec_async[1]) if spec_async is not None
            else getattr(store._resident, "gpu_fastpath", None)
        ),
        "gpu_fallback": (
            int(spec_async[2]) if spec_async is not None
            else getattr(store._resident, "gpu_fallback", None)
        ),
        "async_demand_stats": (
            {
                "calls": int(spec_async[0]),
                "all_hit_layers": int(spec_async[1]),
                "fallback_layers": int(spec_async[2]),
                "unique_disk_loads": int(spec_async[3]),
                "hit_positions": int(spec_async[4]),
                "route_positions": int(spec_async[5]),
                "true_fallback_layers": int(spec_async[6]),
                "pending_rescued_positions": int(spec_async[7]),
                "pending_wait_ms": round(int(spec_async[8]) / 1000.0, 3),
                "true_fallback_wait_ms": round(int(spec_async[9]) / 1000.0, 3),
                # Position coverage alone is not a latency metric.  One miss
                # among a layer's whole route union still gates that layer on
                # the CPU fallback/SSD event.  Report both denominators so a
                # 98% expert-position figure cannot be mistaken for 98% of
                # layer invocations taking the zero-wait path.
                "position_hit_rate": round(
                    int(spec_async[4]) / max(int(spec_async[5]), 1), 6,
                ),
                "all_hit_layer_rate": round(
                    int(spec_async[1]) / max(int(spec_async[0]), 1), 6,
                ),
            }
            if spec_async is not None else None
        ),
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
        "staging_stats": _staging_stats(store),
        "window_prof": _window_prof(),
        "predict_recall": _predict_recall(),
        "baseline_prefetch_rerank": base_rerank,
        "prefetch_rerank": _rerank_prof(),
        "baseline_prefetch_deadline": base_prefetch_deadline,
        "prefetch_deadline": spec_deadline,
        "prefetch_prejoin_deadline": spec_prejoin,
        "prefetch_join": _prefetch_join_prof(),
        # 逐 occurrence 的逻辑 rerank 验收：selected 不超过真实并集 150%，
        # 且每层 selected recall 至少保留 raw top64 recall 的 95%。
        "progressive_acceptance": spec_acceptance,
        "rerank_physical_alignment": _rerank_physical_alignment(
            spec_acceptance, spec_deadline,
        ),
        "baseline_prefetch_audit": base_prefetch_audit,
        "prefetch_audit": _prefetch_audit_prof(primary_seq=K),
        "miss_attrib": _miss_attrib(),
        "prefetch_tprof": _prefetch_tprof(stats.get("wall_s")),
        "union_experts": _union_prof(),
        "sideregion_prefetch": _sideregion_prefetch_prof(),
        "unified_pool_occupancy": _unified_pool_occupancy(store),
    }
    if route_trace_out:
        result["route_trace_events"] = route_trace.dump_jsonl(route_trace_out)
    # 噪声地板测量口径:DUMP_IDS=1 时把 baseline greedy 与 spec 的完整 token 序列打进日志,
    # 供跨进程 run-to-run 逐位对比(默认关闭,不污染常规输出)。
    if os.environ.get("DUMP_IDS"):
        print("DUMP_BASE_IDS " + json.dumps(list(base)))
        print("DUMP_SPEC_IDS " + json.dumps(list(ids)))
    # 字节真值校验自证口径:开 STG_VERIFY 时,把两处校验器的累计计数
    # (ok/bad/calls)打进日志。关键:让「0 BAD」可判真伪——若 calls==0 说明本配置根本
    # 没触发该校验器(如 STG_VERIFY 在 zerocopy_dual 路径不接线),此时 0 BAD 是空结论。
    if os.environ.get("STG_VERIFY"):
        from mlx_streaming.core.cache import resident_pool as _rp_mod
        from mlx_streaming.core.cache import virtual_pool as _vp_mod
        _vsum = {
            "STG_VERIFY.resident(verify_acquire_bytes)": dict(_rp_mod._stg_verify_state),
            "STG_VERIFY.virtual(_verify_native_bytes)": dict(_vp_mod._stg_verify_state),
        }
        print("VERIFY_SUMMARY " + json.dumps(_vsum, ensure_ascii=False))
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    result_out = os.environ.get("RESULT_OUT")
    if result_out:
        os.makedirs(os.path.dirname(result_out) or ".", exist_ok=True)
        with open(result_out, "w", encoding="utf-8") as file:
            file.write(serialized + "\n")
        print(f"[result] wrote {result_out}")
    if os.environ.get("RESULT_STDOUT", "1") == "1":
        print(serialized)
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


def _rerank_prof():
    """汇总动态候选宽度与保留概率质量。"""
    from mlx_streaming.core.profiling import RERANK_PROF as R
    if not R["n"]:
        return None

    def percentile(values, q):
        ordered = sorted(float(v) for v in values)
        pos = (len(ordered) - 1) * q
        lo = int(pos)
        hi = min(len(ordered) - 1, lo + 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

    n = R["n"]
    widths = R["effective_width_samples"]
    return {
        "avg_candidate_width": round(R["candidate_width"] / n, 2),
        "avg_effective_width": round(R["effective_width"] / n, 2),
        "p50_effective_width": round(percentile(widths, 0.5), 2),
        "p95_effective_width": round(percentile(widths, 0.95), 2),
        "avg_resident_count": round(R["resident_count"] / n, 2),
        "avg_retained_mass": round(R["retained_mass_sum"] / n, 4),
        "n": n,
    }


def _sideregion_prefetch_prof():
    """Return native side-cache work counters when explicitly requested."""
    if os.environ.get("SIDEREGION_STATS") != "1":
        return None
    import mlx_streaming.native_moe_ext as native
    values = native.sideregion_prefetch_stats()
    names = (
        "input_ids", "candidates", "side_hits", "reserved_reads",
        "evictions", "pread_ok", "pread_fail", "unique_layer_expert_reads",
    )
    return {name: int(value) for name, value in zip(names, values, strict=True)}


def _prefetch_audits_reset(store=None):
    """在测量轮边界排空旧 fill，再开启独立的 deadline/audit 统计。"""
    if not (_cfg.prefetch_deadline_prof() or _cfg.prefetch_audit_prof()):
        return
    from mlx_streaming import native_moe_ext as N
    # reset 时若旧 callback 仍在途，它会污染新一轮 key；先排空才有严格边界。
    # Drain either backend before resetting counters so the next measurement
    # cannot inherit completion publications from the preceding warm-up.
    N.prefetch_staging_drain()
    N.sideregion_drain()
    if _cfg.prefetch_deadline_prof():
        N.demand_deadline_stats_reset()
        N.demand_prejoin_stats_reset()
        N.prefetch_staging_wait_stats_reset()
    if _cfg.prefetch_audit_prof():
        N.prefetch_audit_stats_reset()
        from mlx_streaming.core.prefetch import progressive_acceptance
        progressive_acceptance.reset()
        vpool = getattr(store, "_vpool", None)
        if vpool is not None:
            vpool.reset_progressive_acceptance_pending()


def _progressive_acceptance_prof(store=None):
    """Return the requested per-layer top64-retention/150%-width contract."""
    if not _cfg.prefetch_audit_prof():
        return None
    vpool = getattr(store, "_vpool", None)
    if vpool is not None:
        vpool.flush_progressive_acceptance()
    from mlx_streaming.core.prefetch import progressive_acceptance
    return progressive_acceptance.report(threshold=0.95)


def _rerank_physical_alignment(acceptance, deadline):
    """Reconcile logical selected hits with post-join physical availability.

    Layer 0 has no cross-layer prediction and must not be mixed into this
    denominator.  Exact integer equality per scheduled layer is stronger than
    comparing two rounded aggregate rates and prevents the legacy mixed-unit
    hit rate from masquerading as this contract.
    """
    if acceptance is None or deadline is None:
        return None
    logical = acceptance.get("per_layer", {})
    physical = deadline.get("per_layer", {})
    per_layer = {}
    logical_actual = logical_hits = 0
    physical_actual = physical_hits = 0
    mismatches = []
    for layer, selected in sorted(logical.items(), key=lambda item: int(item[0])):
        available = physical.get(str(layer))
        if available is None:
            mismatches.append({"layer": int(layer), "reason": "missing_physical"})
            continue
        la = int(selected["actual_routes"])
        lh = int(selected["selected_hits"])
        pa = int(available["actual_unique"])
        ph = int(available["real_resident"]) + int(
            available["side_prefetch_complete"]
        )
        exact = la == pa and lh == ph
        per_layer[str(layer)] = {
            "logical_actual": la,
            "logical_selected_hits": lh,
            "physical_actual": pa,
            "physical_available_after_join": ph,
            "exact_integer_match": exact,
        }
        logical_actual += la
        logical_hits += lh
        physical_actual += pa
        physical_hits += ph
        if not exact:
            mismatches.append({
                "layer": int(layer), "logical": [lh, la],
                "physical": [ph, pa],
            })
    return {
        "scope": "scheduled_rerank_layers_only",
        "excludes_unscheduled_layer0": True,
        "logical_selected_coverage": round(
            logical_hits / max(logical_actual, 1), 9,
        ),
        "physical_available_after_join_coverage": round(
            physical_hits / max(physical_actual, 1), 9,
        ),
        "logical_counts": [logical_hits, logical_actual],
        "physical_counts": [physical_hits, physical_actual],
        "all_layers_exact_integer_match": not mismatches,
        "mismatches": mismatches,
        "per_layer": per_layer,
    }


def _prefetch_audit_prof(primary_seq: int):
    """逐次 source-time rerank、I/O 与目标 demand 的严格配对审计。"""
    if not _cfg.prefetch_audit_prof():
        return None
    from mlx_streaming import native_moe_ext as N
    # demand 判定已固定，不会因这里等待而改善；排空只为收齐晚到 callback/publish
    # 的时间戳，区分“预测正确但字节迟到”和“根本没预测到”。
    N.prefetch_staging_drain()
    columns = (
        "forward_id", "source_layer", "target_layer", "physical_gen",
        "sequence_length", "submit_eval_us", "callback_us", "pread_start_us",
        "publish_end_us", "demand_us", "candidate_width",
        "source_resident_count", "actual_unique", "candidate_hits",
        "source_resident_hits", "system_prediction_hits",
        "candidate_complete_hits", "system_complete_hits", "deadline_real",
        "deadline_side", "fallback", "pread_requested", "pread_completed",
        "submission_count", "demand_count", "callback_before_demand",
    )
    flat = [int(value) for value in N.prefetch_audit_stats()]
    width = len(columns)
    if len(flat) % width:
        raise RuntimeError(
            f"prefetch audit 不是 {width} 列: values={len(flat)}",
        )
    records = [
        dict(zip(columns, flat[offset:offset + width], strict=True))
        for offset in range(0, len(flat), width)
    ]

    def percentile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        index = int(round((len(ordered) - 1) * q))
        return round(ordered[index], 3)

    def summarize(rows):
        expected_submissions = 2 if _cfg.prefetch_progressive() else 1
        per_layer = {}
        failures = []
        for row in rows:
            layer = row["target_layer"]
            rec = per_layer.setdefault(layer, {
                "calls": 0,
                "submissions": 0,
                "actual_unique": 0,
                "candidate_width": 0,
                "max_candidate_width": 0,
                "candidate_hits": 0,
                "system_prediction_hits": 0,
                "candidate_complete_hits": 0,
                "system_complete_hits": 0,
                "deadline_real": 0,
                "deadline_side": 0,
                "fallback": 0,
                "width_violations": 0,
                "over_top64": 0,
                "missing_submissions": 0,
                "duplicate_submissions": 0,
                "duplicate_demands": 0,
                "schedule_mismatches": 0,
                "callback_late": 0,
                "publish_late": 0,
                "pread_requested": 0,
                "pread_completed": 0,
                "submit_to_demand_ms": [],
                "callback_to_demand_ms": [],
                "publish_slack_ms": [],
            })
            actual = row["actual_unique"]
            limit = (actual * 3) // 2
            candidate_width = row["candidate_width"]
            submission_count = row["submission_count"]
            demand_count = row["demand_count"]
            rec["calls"] += demand_count
            rec["submissions"] += submission_count
            rec["actual_unique"] += actual
            rec["candidate_width"] += candidate_width
            rec["max_candidate_width"] = max(
                rec["max_candidate_width"], candidate_width,
            )
            for key in (
                "candidate_hits", "system_prediction_hits",
                "candidate_complete_hits", "system_complete_hits",
                "deadline_real", "deadline_side", "fallback",
                "pread_requested", "pread_completed",
            ):
                rec[key] += row[key]
            violation = submission_count > 0 and candidate_width > limit
            over_top64 = candidate_width > 64
            # Progressive mode deliberately has two submissions for one
            # target/generation: the unchanged original-source early core and
            # a T-1 exact fill.  Fewer means one stage is absent; more means a
            # real duplicate.  Legacy mode keeps the original exactly-one
            # contract.
            missing = (
                demand_count > 0
                and layer > 0
                and submission_count < expected_submissions
            )
            duplicate_submission = submission_count > expected_submissions
            duplicate_demand = demand_count > 1
            ahead_profile = _cfg.cross_layer_ahead_profile()
            expected_ahead = ahead_profile.get(
                layer,
                _cfg.cross_layer_ahead_lo()
                if layer <= _cfg.cross_layer_cutoff()
                else _cfg.cross_layer_ahead_hi(),
            )
            schedule_mismatch = (
                submission_count > 0
                and layer > 0
                and layer - row["source_layer"] != expected_ahead
            )
            callback_late = (
                demand_count > 0
                and (row["callback_us"] < 0
                     or row["callback_us"] > row["demand_us"])
            )
            publish_late = (
                row["pread_requested"] > 0
                and demand_count > 0
                and (row["publish_end_us"] < 0
                     or row["publish_end_us"] > row["demand_us"])
            )
            rec["width_violations"] += int(violation)
            rec["over_top64"] += int(over_top64)
            rec["missing_submissions"] += int(missing)
            rec["duplicate_submissions"] += int(duplicate_submission)
            rec["duplicate_demands"] += int(duplicate_demand)
            rec["schedule_mismatches"] += int(schedule_mismatch)
            rec["callback_late"] += int(callback_late)
            rec["publish_late"] += int(publish_late)
            if row["submit_eval_us"] >= 0 and row["demand_us"] >= 0:
                rec["submit_to_demand_ms"].append(
                    (row["demand_us"] - row["submit_eval_us"]) / 1000,
                )
            if row["callback_us"] >= 0 and row["demand_us"] >= 0:
                rec["callback_to_demand_ms"].append(
                    (row["demand_us"] - row["callback_us"]) / 1000,
                )
            if row["publish_end_us"] >= 0 and row["demand_us"] >= 0:
                rec["publish_slack_ms"].append(
                    (row["demand_us"] - row["publish_end_us"]) / 1000,
                )
            reasons = [
                name for name, failed in (
                    ("width>1.5x", violation),
                    ("width>top64", over_top64),
                    (f"submissions<{expected_submissions}", missing),
                    ("duplicate_submission", duplicate_submission),
                    ("duplicate_demand", duplicate_demand),
                    ("schedule_mismatch", schedule_mismatch),
                    ("callback_after_demand", callback_late),
                    ("publish_after_demand", publish_late),
                ) if failed
            ]
            if reasons and len(failures) < 64:
                failures.append({
                    "forward_id": row["forward_id"],
                    "source_layer": row["source_layer"],
                    "target_layer": layer,
                    "actual_unique": actual,
                    "candidate_width": candidate_width,
                    "width_limit": limit,
                    "reasons": reasons,
                })

        output_layers = {}
        for layer in sorted(per_layer):
            rec = per_layer[layer]
            actual = max(rec["actual_unique"], 1)
            calls = max(rec["calls"], 1)
            rec["avg_actual_unique"] = round(rec["actual_unique"] / calls, 4)
            rec["avg_candidate_width"] = round(rec["candidate_width"] / calls, 4)
            rec["candidate_recall"] = round(rec["candidate_hits"] / actual, 6)
            rec["system_prediction_recall"] = round(
                rec["system_prediction_hits"] / actual, 6,
            )
            rec["candidate_complete_recall"] = round(
                rec["candidate_complete_hits"] / actual, 6,
            )
            rec["system_complete_recall"] = round(
                rec["system_complete_hits"] / actual, 6,
            )
            rec["deadline_byte_coverage"] = round(
                (rec["deadline_real"] + rec["deadline_side"]) / actual, 6,
            )
            submit_values = rec.pop("submit_to_demand_ms")
            rec["submit_to_demand_p50_ms"] = percentile(submit_values, 0.50)
            # 最小侧更接近最危险窗口；P05 比单次极值更稳。
            rec["submit_to_demand_p05_ms"] = percentile(submit_values, 0.05)
            callback_values = rec.pop("callback_to_demand_ms")
            publish_values = rec.pop("publish_slack_ms")
            rec["callback_to_demand_p50_ms"] = percentile(callback_values, 0.50)
            rec["callback_to_demand_p05_ms"] = percentile(callback_values, 0.05)
            rec["publish_slack_p50_ms"] = percentile(publish_values, 0.50)
            rec["publish_slack_p05_ms"] = percentile(publish_values, 0.05)
            rec["strict_pass"] = (
                layer > 0
                and rec["system_prediction_recall"] >= 0.98
                and rec["system_complete_recall"] >= 0.98
                and rec["deadline_byte_coverage"] >= 0.98
                and rec["width_violations"] == 0
                and rec["over_top64"] == 0
                and rec["missing_submissions"] == 0
                and rec["duplicate_submissions"] == 0
                and rec["duplicate_demands"] == 0
                and rec["schedule_mismatches"] == 0
            )
            output_layers[str(layer)] = rec

        scheduled = [rec for layer, rec in output_layers.items() if int(layer) > 0]
        all_layers = list(output_layers.values())
        total_actual = sum(rec["actual_unique"] for rec in all_layers)
        overall = {
            "expected_submissions_per_demand": expected_submissions,
            "calls": sum(rec["calls"] for rec in all_layers),
            "actual_unique": total_actual,
            "candidate_width": sum(rec["candidate_width"] for rec in all_layers),
            "system_prediction_recall": round(
                sum(rec["system_prediction_hits"] for rec in all_layers)
                / max(total_actual, 1), 6,
            ),
            "system_complete_recall": round(
                sum(rec["system_complete_hits"] for rec in all_layers)
                / max(total_actual, 1), 6,
            ),
            "deadline_byte_coverage": round(
                sum(rec["deadline_real"] + rec["deadline_side"] for rec in all_layers)
                / max(total_actual, 1), 6,
            ),
            "width_violations": sum(rec["width_violations"] for rec in all_layers),
            "over_top64": sum(rec["over_top64"] for rec in all_layers),
            "missing_submissions": sum(
                rec["missing_submissions"] for rec in all_layers
            ),
            "schedule_mismatches": sum(
                rec["schedule_mismatches"] for rec in all_layers
            ),
            "callback_late": sum(rec["callback_late"] for rec in all_layers),
            "publish_late": sum(rec["publish_late"] for rec in all_layers),
            "strict_pass_layers": sum(rec["strict_pass"] for rec in scheduled),
            "scheduled_layers": len(scheduled),
        }
        return {
            "records": len(rows),
            "overall": overall,
            "per_layer": output_layers,
            "scheduled_layers_all_strict_pass": bool(scheduled) and all(
                rec["strict_pass"] for rec in scheduled
            ),
            "all_observed_layers_strict_pass": bool(all_layers) and all(
                rec["strict_pass"] for rec in all_layers
            ),
            "failing_layers": [
                int(layer) for layer, rec in output_layers.items()
                if not rec["strict_pass"]
            ],
            "failure_examples": failures,
        }

    by_seq = {}
    for seq in sorted({row["sequence_length"] for row in records}):
        by_seq[str(seq)] = summarize([
            row for row in records if row["sequence_length"] == seq
        ])
    return {
        "schema": "prefetch-audit-v1",
        "columns": list(columns),
        "record_count": len(records),
        "primary_seq": int(primary_seq),
        "primary": by_seq.get(str(int(primary_seq))),
        "by_seq": by_seq,
    }


def _prefetch_deadline_prof():
    """目标 MoE 消费边界的逐层完整字节覆盖率（需显式开启）。"""
    if not _cfg.prefetch_deadline_prof():
        return None
    from mlx_streaming import native_moe_ext as N
    flat = list(N.demand_deadline_stats())
    if len(flat) % 6:
        raise RuntimeError("demand deadline stats 不是 6 列")
    per_layer = {}
    totals = {
        "calls": 0,
        "actual_unique": 0,
        "real_resident": 0,
        "side_prefetch_complete": 0,
        "demand_fallback": 0,
    }
    for offset in range(0, len(flat), 6):
        layer, calls, actual, resident, prefetched, fallback = map(
            int, flat[offset:offset + 6],
        )
        if resident + prefetched + fallback != actual:
            raise RuntimeError(
                f"layer {layer} deadline 分类不闭合: "
                f"{resident}+{prefetched}+{fallback}!={actual}",
            )
        per_layer[str(layer)] = {
            "calls": calls,
            "actual_unique": actual,
            "real_resident": resident,
            "side_prefetch_complete": prefetched,
            "demand_fallback": fallback,
            "resident_coverage": round(resident / max(actual, 1), 6),
            "prefetch_coverage": round(prefetched / max(actual, 1), 6),
            "deadline_byte_coverage": round(
                (resident + prefetched) / max(actual, 1), 6,
            ),
        }
        for key, value in zip(
            ("calls", "actual_unique", "real_resident",
             "side_prefetch_complete", "demand_fallback"),
            (calls, actual, resident, prefetched, fallback),
            strict=True,
        ):
            totals[key] += value
    totals["resident_coverage"] = round(
        totals["real_resident"] / max(totals["actual_unique"], 1), 6,
    )
    totals["prefetch_coverage"] = round(
        totals["side_prefetch_complete"] / max(totals["actual_unique"], 1), 6,
    )
    totals["deadline_byte_coverage"] = round(
        (totals["real_resident"] + totals["side_prefetch_complete"])
        / max(totals["actual_unique"], 1),
        6,
    )
    return {"overall": totals, "per_layer": per_layer}


def _prefetch_prejoin_prof():
    """Blocking join 前的真实 target deadline；可直接与全驻比较。"""
    if not _cfg.prefetch_deadline_prof():
        return None
    from mlx_streaming import native_moe_ext as N
    flat = list(N.demand_prejoin_stats())
    if len(flat) % 6:
        raise RuntimeError("demand prejoin stats 不是 6 列")
    per_layer = {}
    totals = {
        "calls": 0, "actual_unique": 0, "real_resident": 0,
        "staging_complete": 0, "not_ready": 0,
    }
    for offset in range(0, len(flat), 6):
        layer, calls, actual, resident, complete, not_ready = map(
            int, flat[offset:offset + 6],
        )
        if resident + complete + not_ready != actual:
            raise RuntimeError(
                f"layer {layer} prejoin 分类不闭合: "
                f"{resident}+{complete}+{not_ready}!={actual}",
            )
        per_layer[str(layer)] = {
            "calls": calls,
            "actual_unique": actual,
            "real_resident": resident,
            "staging_complete": complete,
            "not_ready": not_ready,
            "prejoin_byte_coverage": round(
                (resident + complete) / max(actual, 1), 6,
            ),
        }
        for key, value in zip(totals, (calls, actual, resident, complete, not_ready), strict=True):
            totals[key] += value
    totals["prejoin_byte_coverage"] = round(
        (totals["real_resident"] + totals["staging_complete"])
        / max(totals["actual_unique"], 1), 6,
    )
    return {"overall": totals, "per_layer": per_layer}


def _prefetch_join_prof():
    """逐专家 join 的阻塞成本；不把等待完成后的行伪装成 deadline hit。"""
    if not _cfg.prefetch_deadline_prof():
        return None
    from mlx_streaming import native_moe_ext as N
    flat = list(N.prefetch_staging_wait_stats())
    if len(flat) % 6:
        raise RuntimeError("staging wait stats 不是 6 列")
    per_layer = {}
    totals = {
        "calls": 0, "actual_unique": 0, "staging_complete_at_entry": 0,
        "pending_at_entry": 0, "wait_us": 0,
    }
    for offset in range(0, len(flat), 6):
        layer, calls, actual, complete, pending, wait_us = map(
            int, flat[offset:offset + 6],
        )
        per_layer[str(layer)] = {
            "calls": calls,
            "actual_unique": actual,
            "staging_complete_at_entry": complete,
            "pending_at_entry": pending,
            "total_wait_ms": round(wait_us / 1000, 3),
            "avg_wait_ms": round(wait_us / max(calls, 1) / 1000, 6),
        }
        for key, value in zip(totals, (calls, actual, complete, pending, wait_us), strict=True):
            totals[key] += value
    totals["total_wait_ms"] = round(totals.pop("wait_us") / 1000, 3)
    totals["avg_wait_ms"] = round(
        totals["total_wait_ms"] / max(totals["calls"], 1), 6,
    )
    return {"overall": totals, "per_layer": per_layer}


def _window_prof():
    from mlx_streaming.core.profiling import WINDOW_PROF
    if not WINDOW_PROF["n"]:
        return None
    return {"avg_ms": round(WINDOW_PROF["sum_s"] / WINDOW_PROF["n"] * 1000, 4),
            "n": WINDOW_PROF["n"]}


def _staging_stats(store):
    staging = getattr(store, "_staging", None)
    if staging is None:
        return None
    return {
        "submitted": int(getattr(staging, "submitted", 0)),
        "skipped_no_bank": int(getattr(staging, "skipped_no_bank", 0)),
        "max_busy_banks": int(getattr(staging, "max_busy_banks", 0)),
        "bank_count": int(getattr(staging, "ring", 0)),
    }


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
