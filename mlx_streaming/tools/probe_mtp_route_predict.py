"""评估“draft tokens -> cheap router probe -> future experts”的可行性。

这个 probe 不改热路径：在每个 MTP step 里，用便宜代理 hidden 喂各层 router，
预测 verify K token 会用到的专家集合，再和真实主模型 verify 路由集合比较。

代理:
- mtp_hidden: MTP 单层产生的 hidden 序列（便宜、已有）
- embedding: token embedding（更便宜，但通常更弱）
"""
import json
import os
from collections import Counter, defaultdict

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.core import route_trace
from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.kv_cache import _restore, _snapshot, begin_speculative_checkpoints, commit_verified_prefix, enable_qwen3next_speculative_checkpoints
from mlx_streaming.mtp.generate import accept_prefix, forward_with_hidden
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp, mtp_step

QN_CONFIG = os.environ.get("QN_CONFIG", "/tmp/qn_orig_config.json")
MTP_OUT = os.environ.get("MTP_OUT", "/tmp/qn_mtp_weights.safetensors")
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "3"))
MULTS = [int(x) for x in os.environ.get("ROUTE_PRED_MULTS", "1,2,4").split(",")]
HOT_LIST = [int(x) for x in os.environ.get("ROUTE_HYBRID_HOT", "8,16,32").split(",")]
TRANS_LIST = [int(x) for x in os.environ.get("ROUTE_TRANS_TOP", "4,8,16").split(",")]
HYBRID_TRACE_OUT = os.environ.get("ROUTE_HYBRID_TRACE_OUT")
HYBRID_TRACE_MULT = int(os.environ.get("ROUTE_HYBRID_TRACE_MULT", "2"))
HYBRID_TRACE_HOT = int(os.environ.get("ROUTE_HYBRID_TRACE_HOT", "16"))


def _moe_blocks(model) -> list[FileStreamingMoeBlock]:
    return [layer.mlp for layer in model.model.layers
            if isinstance(getattr(layer, "mlp", None), FileStreamingMoeBlock)]


def _predict_for_proxy(blocks, proxy_h: mx.array, mult: int) -> dict[int, set[int]]:
    out = {}
    for blk in blocks:
        gates = mx.softmax(blk.gate(proxy_h), axis=-1, precise=True)
        k = min(gates.shape[-1], blk.top_k * mult)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        mx.eval(inds)
        out[blk.layer_idx] = {int(e) for e in inds.reshape(-1).tolist()}
    return out


def _actual_from_events(events: list[dict]) -> dict[int, set[int]]:
    out = {}
    for rec in events:
        out[int(rec["layer"])] = {int(e) for e in rec["experts"]}
    return out


def _score(pred: dict[int, set[int]], actual: dict[int, set[int]]) -> dict:
    rec_sum = prec_sum = layers = 0
    pred_n = actual_n = hit_n = 0
    missed_layers = 0
    for layer, a in actual.items():
        p = pred.get(layer, set())
        hit = len(p & a)
        pred_n += len(p)
        actual_n += len(a)
        hit_n += hit
        rec_sum += hit / max(1, len(a))
        prec_sum += hit / max(1, len(p))
        missed_layers += int(hit < len(a))
        layers += 1
    return {
        "layers": layers,
        "recall": rec_sum / max(1, layers),
        "precision": prec_sum / max(1, layers),
        "global_recall": hit_n / max(1, actual_n),
        "global_precision": hit_n / max(1, pred_n),
        "avg_pred_experts": pred_n / max(1, layers),
        "avg_actual_experts": actual_n / max(1, layers),
        "missed_layers": missed_layers,
    }


def _hot_sets(history: dict[int, Counter[int]], hot_n: int) -> dict[int, set[int]]:
    return {layer: {e for e, _ in counter.most_common(hot_n)}
            for layer, counter in history.items()}


def _transition_sets(
    transitions: dict[int, dict[int, Counter[int]]],
    prev_actual: dict[int, set[int]],
    top_n: int,
) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for layer, prev_experts in prev_actual.items():
        cand = out.setdefault(layer, set())
        layer_trans = transitions.get(layer, {})
        for e in prev_experts:
            cand.update(n for n, _ in layer_trans.get(e, Counter()).most_common(top_n))
    return out


def _update_transitions(
    transitions: dict[int, dict[int, Counter[int]]],
    prev_actual: dict[int, set[int]],
    actual: dict[int, set[int]],
) -> None:
    for layer, prev_experts in prev_actual.items():
        cur = actual.get(layer)
        if not cur:
            continue
        layer_trans = transitions.setdefault(layer, defaultdict(Counter))
        for p in prev_experts:
            layer_trans[p].update(cur)


def _merge_preds(*items: dict[int, set[int]]) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for item in items:
        for layer, experts in item.items():
            out.setdefault(layer, set()).update(experts)
    return out


def main():
    os.environ["ROUTE_TRACE"] = "1"
    enable_qwen3next_speculative_checkpoints()
    model, tok, store = build_streaming_model()
    blocks = _moe_blocks(model)
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    main_cache = model.make_cache()
    mtp_cache = drafter.make_cache()
    ids = mx.array([tok.encode(PROMPT)])
    logits, H = forward_with_hidden(model, ids, main_cache)
    x = int(mx.argmax(logits[:, -1, :]))
    H_last = H[:, -1:, :]
    produced = [x]
    mx.eval(H_last)

    agg = defaultdict(lambda: defaultdict(float))
    n_steps = 0
    total_actual_loads = 0
    prev_actual: dict[int, set[int]] = {}
    hot_history: dict[int, Counter[int]] = defaultdict(Counter)
    transitions: dict[int, dict[int, Counter[int]]] = {}
    hybrid_trace: list[dict] = []
    while len(produced) < MAXTOK:
        snap_d = _snapshot(mtp_cache)
        prev_H_last = H_last

        drafts = []
        mtp_hiddens = []
        h, cur = H_last, mx.array([[x]])
        for _ in range(K):
            dlogits, mh = mtp_step(mtp, h, cur, model.lm_head, mtp_cache[0])
            d = int(mx.argmax(dlogits[0]))
            drafts.append(d)
            mtp_hiddens.append(mh)
            h, cur = mh, mx.array([[d]])

        verify_in = mx.array([[x] + drafts[:K - 1]])
        # 位置对齐代理：当前 token 用主模型 H_last，后续 draft token 用 MTP hidden。
        proxy_mtp = mx.concatenate([H_last] + mtp_hiddens[:K - 1], axis=1)
        proxy_emb = model.model.embed_tokens(verify_in)
        mx.eval(proxy_mtp, proxy_emb)

        preds = {"mtp_hidden": {}, "embedding": {}}
        for mult in MULTS:
            preds["mtp_hidden"][mult] = _predict_for_proxy(blocks, proxy_mtp, mult)
            preds["embedding"][mult] = _predict_for_proxy(blocks, proxy_emb, mult)

        snap_m = _snapshot(main_cache)
        route_trace.enable()
        begin_speculative_checkpoints(main_cache)
        vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
        mx.eval(vlogits, vH)
        events = route_trace.events()
        route_trace.disable()
        actual = _actual_from_events(events)
        total_actual_loads += sum(len(v) for v in actual.values())

        for proxy_name, by_mult in preds.items():
            for mult, pred in by_mult.items():
                s = _score(pred, actual)
                key = f"{proxy_name}_x{mult}"
                for k, v in s.items():
                    agg[key][k] += float(v)

        # 规则 hybrid：MTP proxy top-M ∪ 上一步同层真实路由 ∪ 在线 hot experts。
        # hot 只使用之前步骤的真实路由，不看当前 actual，避免数据泄漏。
        for mult, pred in preds["mtp_hidden"].items():
            for hot_n in HOT_LIST:
                hot = _hot_sets(hot_history, hot_n)
                hybrid = _merge_preds(pred, prev_actual, hot)
                s = _score(hybrid, actual)
                key = f"hybrid_mtp_x{mult}_prev_hot{hot_n}"
                for k, v in s.items():
                    agg[key][k] += float(v)

        # Online adaptive predictor：MTP proxy top-M ∪ hot-H ∪ transition(prev experts)->top-T。
        # transition 只由历史 step 更新，不看当前 actual。
        for mult, pred in preds["mtp_hidden"].items():
            for hot_n in HOT_LIST:
                hot = _hot_sets(hot_history, hot_n)
                for trans_n in TRANS_LIST:
                    trans = _transition_sets(transitions, prev_actual, trans_n)
                    adaptive = _merge_preds(pred, hot, trans)
                    s = _score(adaptive, actual)
                    key = f"adaptive_mtp_x{mult}_hot{hot_n}_trans{trans_n}"
                    for k, v in s.items():
                        agg[key][k] += float(v)

        if HYBRID_TRACE_OUT:
            hot = _hot_sets(hot_history, HYBRID_TRACE_HOT)
            hybrid = _merge_preds(preds["mtp_hidden"][HYBRID_TRACE_MULT], prev_actual, hot)
            for layer, experts in actual.items():
                hybrid_trace.append({
                    "layer": int(layer),
                    "experts": sorted(int(e) for e in experts),
                    "avoid": sorted(int(e) for e in hybrid.get(layer, set())),
                })

        _update_transitions(transitions, prev_actual, actual)
        prev_actual = actual
        for layer, experts in actual.items():
            hot_history[layer].update(experts)

        token_preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]
        matched = accept_prefix(drafts, token_preds)
        accepted_len = min(matched + 1, K)
        committed = commit_verified_prefix(main_cache, verified_len=K, accepted_len=accepted_len)
        if not committed:
            raise RuntimeError("probe 期望 MTP_ARRAY_COMMIT=1 的 direct commit 路径")
        _restore(mtp_cache, snap_d)
        accepted_in = verify_in[:, :accepted_len]
        rH = vH[:, :accepted_len, :]
        if matched == K:
            new_tokens = drafts[:K]
            x = drafts[-1]
        else:
            new_tokens = drafts[:matched] + [token_preds[matched]]
            x = token_preds[matched]
        produced.extend(new_tokens)
        produced = produced[:MAXTOK]
        H_last = rH[:, -1:, :]
        drafter.sync(prev_H_last, rH, accepted_in, mtp_cache)
        mx.eval(x, H_last)
        n_steps += 1
        _ = snap_m  # 保留变量，便于和 mtp_generate 对齐阅读。

    rows = []
    for key, vals in sorted(agg.items()):
        rows.append({
            "predictor": key,
            **{k: round(v / max(1, n_steps), 4) for k, v in vals.items()},
        })
    print(json.dumps({
        "K": K,
        "tokens": len(produced),
        "steps": n_steps,
        "avg_accept_len": round(len(produced) / max(1, n_steps), 3),
        "avg_actual_expert_union_per_step": round(total_actual_loads / max(1, n_steps), 2),
        "hybrid_trace_out": HYBRID_TRACE_OUT,
        "hybrid_trace_events": len(hybrid_trace),
        "hybrid_trace_mult": HYBRID_TRACE_MULT,
        "hybrid_trace_hot": HYBRID_TRACE_HOT,
        "rows": rows,
        "store_misses": store.misses,
        "store_hits": store.hits,
    }, ensure_ascii=False, indent=2))
    if HYBRID_TRACE_OUT:
        with open(HYBRID_TRACE_OUT, "w") as f:
            for rec in hybrid_trace:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
