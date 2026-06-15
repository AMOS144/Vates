"""采集 layer-wise router predictor 训练数据。

每个样本对应一次 MTP verify step 的某个 MoE 层：
- x: MTP proxy hidden 的均值 (hidden,)
- layer: 层号
- y: 真实 verify 路由专家 union 的 multi-hot (num_experts,)

这只是离线数据采集，不接入推理热路径。
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
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
PROMPTS_FILE = os.environ.get("PROMPTS_FILE")
MAXTOK = int(os.environ.get("MAXTOK", "96"))
K = int(os.environ.get("K", "3"))
OUT = os.environ.get("ROUTER_PRED_DATA_OUT", "/tmp/router_pred_data.npz")
FEATURE = os.environ.get("ROUTER_PRED_FEATURE", "mean")  # mean | flat
HOT_N = int(os.environ.get("ROUTER_PRED_HOT_N", "32"))
TRANS_N = int(os.environ.get("ROUTER_PRED_TRANS_N", "16"))
PROXY_MULT = int(os.environ.get("ROUTER_PRED_PROXY_MULT", "4"))
TARGET = os.environ.get("ROUTER_PRED_TARGET", "actual")  # actual | miss | future_actual | future_miss
FUTURE_STEPS = int(os.environ.get("ROUTER_PRED_FUTURE_STEPS", "0"))


def _prompts() -> list[str]:
    if PROMPTS_FILE:
        return [l.strip() for l in Path(PROMPTS_FILE).read_text().splitlines() if l.strip()]
    return [PROMPT]


def _moe_blocks(model) -> list[FileStreamingMoeBlock]:
    return [layer.mlp for layer in model.model.layers
            if isinstance(getattr(layer, "mlp", None), FileStreamingMoeBlock)]


def _actual_from_events(events: list[dict]) -> dict[int, set[int]]:
    out = {}
    for rec in events:
        out[int(rec["layer"])] = {int(e) for e in rec["experts"]}
    return out


def _miss_from_events(events: list[dict]) -> dict[int, set[int]]:
    out = {}
    for rec in events:
        out[int(rec["layer"])] = {int(e) for e in rec.get("miss", rec["experts"])}
    return out


def _resident_from_events(events: list[dict]) -> dict[int, set[int]]:
    out = {}
    for rec in events:
        out[int(rec["layer"])] = {int(e) for e in rec.get("resident", [])}
    return out


def _resident_rank_from_events(events: list[dict], num_experts: int) -> dict[int, np.ndarray]:
    out = {}
    for rec in events:
        v = np.zeros((num_experts,), dtype=np.float16)
        for e, score in rec.get("resident_rank", []):
            v[int(e)] = float(score)
        out[int(rec["layer"])] = v
    return out


def _multi_hot(experts: set[int], num_experts: int) -> np.ndarray:
    v = np.zeros((num_experts,), dtype=np.uint8)
    if experts:
        v[list(experts)] = 1
    return v


def _future_union_targets(y: np.ndarray, prompt_id: np.ndarray, layer: np.ndarray,
                          step: np.ndarray, horizon: int) -> np.ndarray:
    """把当前 step 标签转换成未来 horizon 步同 prompt/同层的 union 标签。"""
    yb = y.astype(np.uint8)
    out = np.zeros_like(yb)
    index = {}
    for i, key in enumerate(zip(prompt_id, layer, step, strict=False)):
        index[key] = i
    for i, (p, l, s) in enumerate(zip(prompt_id, layer, step, strict=False)):
        acc = np.zeros_like(yb[i])
        for dt in range(1, horizon + 1):
            j = index.get((p, l, s + dt))
            if j is not None:
                acc |= yb[j]
        out[i] = acc
    return out


def _hot_multi_hot(counter: Counter[int], num_experts: int, hot_n: int) -> np.ndarray:
    return _multi_hot({e for e, _ in counter.most_common(hot_n)}, num_experts)


def _freq_score(counter: Counter[int], num_experts: int) -> np.ndarray:
    v = np.zeros((num_experts,), dtype=np.float16)
    if not counter:
        return v
    max_count = max(counter.values())
    for e, c in counter.items():
        v[int(e)] = c / max_count
    return v


def _transition_multi_hot(transitions, prev_experts: set[int],
                          num_experts: int, trans_n: int) -> np.ndarray:
    cand = set()
    for e in prev_experts:
        cand.update(n for n, _ in transitions.get(e, Counter()).most_common(trans_n))
    return _multi_hot(cand, num_experts)


def _proxy_router_features(blocks, proxy_h: mx.array, mult: int):
    top_sets = {}
    scores = {}
    sums = {}
    counts = {}
    for blk in blocks:
        gates = mx.softmax(blk.gate(proxy_h), axis=-1, precise=True)
        k = min(gates.shape[-1], blk.top_k * mult)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        # 每个专家取 K 个位置中的最大 proxy gate，保留排序强度而不只保留 top-M 集合。
        flat_gates = gates.reshape(-1, gates.shape[-1])
        score = mx.max(flat_gates, axis=0).astype(mx.float16)
        sum_score = mx.sum(flat_gates, axis=0).astype(mx.float16)
        mx.eval(inds, score, sum_score)
        inds_np = np.array(inds.reshape(-1), dtype=np.int64)
        count = np.bincount(inds_np, minlength=gates.shape[-1]).astype(np.float16)
        top_sets[blk.layer_idx] = {int(e) for e in inds_np.tolist()}
        scores[blk.layer_idx] = np.array(score)
        sums[blk.layer_idx] = np.array(sum_score)
        counts[blk.layer_idx] = count
    return top_sets, scores, sums, counts


def _collect_prompt(model, tok, drafter, blocks, prompt: str, max_tokens: int,
                    num_experts: int, prompt_id: int, xs, layers, steps_out,
                    prompt_ids, proxy_scores, proxy_sums, proxy_counts, proxy_feats, resident_feats,
                    resident_ranks, freq_scores, prev_same, hot_feats,
                    prev_layer_feats, trans_feats, ys) -> dict:
    main_cache = model.make_cache()
    mtp_cache = drafter.make_cache()
    ids = mx.array([tok.encode(prompt)])
    logits, H = forward_with_hidden(model, ids, main_cache)
    x = int(mx.argmax(logits[:, -1, :]))
    H_last = H[:, -1:, :]
    produced = [x]
    mx.eval(H_last)
    steps = 0
    prev_actual: dict[int, set[int]] = {}
    hot_history: dict[int, Counter[int]] = defaultdict(Counter)
    transitions: dict[int, dict[int, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))

    while len(produced) < max_tokens:
        snap_d = _snapshot(mtp_cache)
        prev_H_last = H_last
        drafts, mtp_hiddens = [], []
        h, cur = H_last, mx.array([[x]])
        for _ in range(K):
            dlogits, mh = mtp_step(drafter.mtp, h, cur, model.lm_head, mtp_cache[0])
            d = int(mx.argmax(dlogits[0]))
            drafts.append(d)
            mtp_hiddens.append(mh)
            h, cur = mh, mx.array([[d]])

        verify_in = mx.array([[x] + drafts[:K - 1]])
        proxy_mtp = mx.concatenate([H_last] + mtp_hiddens[:K - 1], axis=1)
        if FEATURE == "flat":
            proxy_vec = proxy_mtp.reshape(-1).astype(mx.float16)
        else:
            proxy_vec = mx.mean(proxy_mtp, axis=1)[0].astype(mx.float16)
        mx.eval(proxy_vec)
        proxy_np = np.array(proxy_vec)
        proxy_router, proxy_score, proxy_sum, proxy_count = _proxy_router_features(
            blocks, proxy_mtp, PROXY_MULT)

        route_trace.enable()
        begin_speculative_checkpoints(main_cache)
        vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
        mx.eval(vlogits, vH)
        events = route_trace.events()
        actual = _actual_from_events(events)
        miss = _miss_from_events(events)
        resident = _resident_from_events(events)
        resident_rank = _resident_rank_from_events(events, num_experts)
        route_trace.disable()

        for layer, experts in actual.items():
            target_experts = miss.get(layer, set()) if TARGET in ("miss", "future_miss") else experts
            target = _multi_hot(target_experts, num_experts)
            prev_layer = actual.get(layer - 1, set())
            xs.append(proxy_np)
            layers.append(layer)
            steps_out.append(steps)
            prompt_ids.append(prompt_id)
            proxy_scores.append(proxy_score[layer])
            proxy_sums.append(proxy_sum[layer])
            proxy_counts.append(proxy_count[layer])
            proxy_feats.append(_multi_hot(proxy_router.get(layer, set()), num_experts))
            resident_feats.append(_multi_hot(resident.get(layer, set()), num_experts))
            resident_ranks.append(resident_rank.get(
                layer, np.zeros((num_experts,), dtype=np.float16)))
            freq_scores.append(_freq_score(hot_history[layer], num_experts))
            prev_same.append(_multi_hot(prev_actual.get(layer, set()), num_experts))
            hot_feats.append(_hot_multi_hot(hot_history[layer], num_experts, HOT_N))
            prev_layer_feats.append(_multi_hot(prev_layer, num_experts))
            trans_feats.append(_transition_multi_hot(
                transitions[layer], prev_actual.get(layer, set()), num_experts, TRANS_N))
            ys.append(target)

        token_preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]
        matched = accept_prefix(drafts, token_preds)
        accepted_len = min(matched + 1, K)
        committed = commit_verified_prefix(main_cache, verified_len=K, accepted_len=accepted_len)
        if not committed:
            raise RuntimeError("采集脚本需要 MTP_ARRAY_COMMIT=1 的 direct commit 路径")
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
        produced = produced[:max_tokens]
        H_last = rH[:, -1:, :]
        drafter.sync(prev_H_last, rH, accepted_in, mtp_cache)
        mx.eval(x, H_last)
        for layer, prev in prev_actual.items():
            cur = actual.get(layer)
            if cur:
                for e in prev:
                    transitions[layer][e].update(cur)
        prev_actual = actual
        for layer, experts in actual.items():
            hot_history[layer].update(experts)
        steps += 1
    return {"tokens": len(produced), "steps": steps}


def main():
    os.environ["ROUTE_TRACE"] = "1"
    os.environ.setdefault("MTP_ARRAY_COMMIT", "1")
    enable_qwen3next_speculative_checkpoints()
    model, tok, _store = build_streaming_model()
    blocks = _moe_blocks(model)
    num_experts = int(blocks[0].gate.weight.shape[0])
    args = ModelArgs.from_dict(json.load(open(QN_CONFIG)))
    mtp = load_mtp(args, MTP_OUT, quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    xs, layers, steps_out, prompt_ids = [], [], [], []
    proxy_scores, proxy_sums, proxy_counts = [], [], []
    proxy_feats, resident_feats, resident_ranks, freq_scores = [], [], [], []
    prev_same, hot_feats, prev_layer_feats, trans_feats, ys = [], [], [], [], []
    prompt_stats = []
    for prompt_id, prompt in enumerate(_prompts()):
        prompt_stats.append(_collect_prompt(
            model, tok, drafter, blocks, prompt, MAXTOK, num_experts, prompt_id,
            xs, layers, steps_out, prompt_ids, proxy_scores, proxy_sums, proxy_counts,
            proxy_feats, resident_feats,
            resident_ranks, freq_scores,
            prev_same, hot_feats, prev_layer_feats, trans_feats, ys))

    x_arr = np.stack(xs).astype(np.float16)
    layer_arr = np.array(layers, dtype=np.int16)
    step_arr = np.array(steps_out, dtype=np.int16)
    prompt_arr = np.array(prompt_ids, dtype=np.int16)
    proxy_score_arr = np.stack(proxy_scores).astype(np.float16)
    proxy_sum_arr = np.stack(proxy_sums).astype(np.float16)
    proxy_count_arr = np.stack(proxy_counts).astype(np.float16)
    proxy_arr = np.stack(proxy_feats).astype(np.uint8)
    resident_arr = np.stack(resident_feats).astype(np.uint8)
    resident_rank_arr = np.stack(resident_ranks).astype(np.float16)
    freq_score_arr = np.stack(freq_scores).astype(np.float16)
    prev_same_arr = np.stack(prev_same).astype(np.uint8)
    hot_arr = np.stack(hot_feats).astype(np.uint8)
    prev_layer_arr = np.stack(prev_layer_feats).astype(np.uint8)
    trans_arr = np.stack(trans_feats).astype(np.uint8)
    y_arr = np.stack(ys).astype(np.uint8)
    if TARGET in ("future_actual", "future_miss"):
        horizon = FUTURE_STEPS or 4
        y_arr = _future_union_targets(y_arr, prompt_arr, layer_arr, step_arr, horizon)
    np.savez_compressed(
        OUT, x=x_arr, layer=layer_arr, step=step_arr, prompt_id=prompt_arr,
        proxy=proxy_arr, prev_same=prev_same_arr,
        proxy_score=proxy_score_arr, proxy_sum=proxy_sum_arr, proxy_count=proxy_count_arr,
        resident=resident_arr,
        resident_rank=resident_rank_arr, freq_score=freq_score_arr, hot=hot_arr,
        prev_layer=prev_layer_arr, transition=trans_arr, y=y_arr,
        meta=json.dumps({
            "k": K,
            "max_tokens": MAXTOK,
            "num_prompts": len(_prompts()),
            "num_experts": num_experts,
            "hidden": int(x_arr.shape[1]),
            "feature": FEATURE,
            "target": TARGET,
            "future_steps": FUTURE_STEPS,
            "hot_n": HOT_N,
            "trans_n": TRANS_N,
            "proxy_mult": PROXY_MULT,
            "prompt_stats": prompt_stats,
        }, ensure_ascii=False))
    print(json.dumps({
        "out": OUT,
        "samples": int(x_arr.shape[0]),
        "hidden": int(x_arr.shape[1]),
        "num_experts": num_experts,
        "prompts": len(_prompts()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
