"""Qwen3-Next MTP 自投机贪婪解码循环。

见 spec/plan: docs/superpowers/{specs,plans}/2026-06-07-qwen3next-mtp-self-speculation*。

核心:每步主模型前向出 hidden -> MTP 自回归抽 K 草稿 -> 主模型并行验证 ->
接受最长命中前缀 -> cache 快照/恢复/重放回滚(统一处理 ArraysCache 与 KVCache),
保证输出与非投机贪婪逐 token 等价。

cache 校验/提交机制见 mtp/kv_cache.py；drafter 接口见 mtp/drafter.py。
"""
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from mlx_streaming import config
from mlx_streaming.mtp.kv_cache import (
    enable_qwen3next_speculative_checkpoints, begin_speculative_checkpoints,
    _snapshot, _restore, commit_verified_prefix, commit_verified_snapshot,
    tile_caches, commit_tree_row)


def _batch_direct_commit_guaranteed(model) -> bool:
    """判断该模型的 batch verify 是否「必定」能直接提交(commit_verified_prefix 恒成功)。

    commit 成功的充要条件是每个 cache 都满足:可裁剪(KVCache)或已捕获 per-token checkpoint。
    - 非线性层 → KVCache,可裁剪,必成功;
    - 线性层 → 必须是被 patch 过的 Qwen3NextGatedDeltaNet,verify 前向才会写 `_spec_checkpoints`。
    全部满足才返回 True——此时每步的 `snap_m` 回退快照永远用不上,可安全省略以压低在途峰值。
    通用/玩具递归层(未 patch)返回 False,保留 snap_m 走安全 replay 兜底(与原行为逐 token 等价)。
    """
    import mlx_streaming.mtp.kv_cache as _kv
    if not getattr(_kv, "_QWEN3NEXT_CHECKPOINTS_PATCHED", False):
        return False
    try:
        from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet
        layers = model.model.layers
    except (ImportError, AttributeError):
        return False
    for l in layers:
        if getattr(l, "is_linear", False) and not isinstance(
                getattr(l, "linear_attn", None), Qwen3NextGatedDeltaNet):
            return False
    return True


def forward_with_hidden(model, ids, cache, compute_logits: bool = True):
    """跑主模型层循环 + 最终 norm,返回 (logits(1,L,V), hidden(1,L,H))。

    logits 恒由 final-norm 后的 H 经 lm_head 得到(保证与贪婪一致)。
    返回给 MTP drafter 的 hidden:**默认用 final-norm 之前的 h**——这与 MTP 训练/验证
    (`capture_prenorm_hidden`)消费的输入一致;喂 norm 后的 hidden 会双重归一化、显著压低
    草稿接受率(不影响正确性,因 verify 走主模型)。`MTP_HIDDEN=post_norm` 可切回旧行为对照。

    compute_logits=False:跳过 lm_head(2048×151936 的大投影),只更新 cache 并返回 hidden。
    供分块 prefill 的中间块用(中间块不需要 logits,只需 cache 因果累积)。
    """
    inner = model.model
    h = inner.embed_tokens(ids)
    layers = inner.layers
    has_full = any(not l.is_linear for l in layers)
    has_linear = any(l.is_linear for l in layers)
    fa_idx = next((i for i, l in enumerate(layers) if not l.is_linear), 0)
    ssm_idx = next((i for i, l in enumerate(layers) if l.is_linear), 0)
    fa_mask = create_attention_mask(h, cache[fa_idx]) if has_full else None
    ssm_mask = create_ssm_mask(h, cache[ssm_idx]) if has_linear else None
    for layer, c in zip(layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)
    H = inner.norm(h)
    hidden = H if config.mtp_hidden() == "post_norm" else h
    logits = model.lm_head(H) if compute_logits else None
    return logits, hidden


def prefill_chunked(model, ids, cache, chunk: "int | None" = None):
    """分块 prefill:把 ids 按 chunk 切片逐块喂入,增量更新 cache,返回最后一块的 (logits, hidden)。

    整段 prefill 一次前向的激活峰值 ∝ prompt 长度(每 MoE 层瞬时物化大量唯一专家 + 长序列激活)。
    分块后每块只算 chunk 个 token,逐块 eval 释放上一块瞬时图,峰值压回 ∝chunk,与 decode 同稳态。

    数值等价:全注意力层逐块因果累积与整段完全等价;线性层(gated-delta)分块在 chunk 边界处有
    重结合的末位浮点差异(与 decode 本身逐 token 同源,可忽略)。中间块跳过 lm_head 省大投影。
    chunk<=0 或 prompt 不超过一块时,退化为整段 prefill。
    """
    if chunk is None:
        chunk = config.prefill_chunk()
    L = ids.shape[1]
    if chunk <= 0 or L <= chunk:
        return forward_with_hidden(model, ids, cache)
    last_start = ((L - 1) // chunk) * chunk
    for s in range(0, last_start, chunk):
        _, h = forward_with_hidden(model, ids[:, s:s + chunk], cache, compute_logits=False)
        mx.eval(h)                      # 物化本块、写 cache、释放上一块瞬时图
    return forward_with_hidden(model, ids[:, last_start:], cache)


def forward_with_hidden_stepwise(model, ids, cache, capture_snapshots: bool = False):
    """逐 token 走主模型解码路径,可在每个 token 后保存 cache 快照。

    这是 exact direct-commit 的保守路径:verify 与 baseline 都走 N=1 cache 更新,
    接受前缀时直接恢复到对应 token 后的快照,不再 replay accepted prefix。
    """
    logits_parts, hidden_parts, snaps = [], [], []
    for i in range(ids.shape[1]):
        logits, H = forward_with_hidden(model, ids[:, i:i + 1], cache)
        mx.eval(logits, H)
        logits_parts.append(logits)
        hidden_parts.append(H)
        if capture_snapshots and i < ids.shape[1] - 1:
            snaps.append(_snapshot(cache))
    logits = mx.concatenate(logits_parts, axis=1)
    H = mx.concatenate(hidden_parts, axis=1)
    if capture_snapshots:
        return logits, H, snaps
    return logits, H


# ----------------------------------------------------------------- 接受判定
def accept_prefix(drafts, preds):
    """drafts: MTP 抽的 K 个草稿; preds: 主模型对应位置真实下一 token(len==K)。

    返回命中草稿数 matched(最长使 preds[j]==drafts[j] 的前缀长度)。
    发射 token 由调用方组装为 drafts[:matched] + [重放末位 bonus],统一处理纠正/全命中,
    保证贪婪下逐 token 与非投机等价。
    """
    matched = 0
    for d, p in zip(drafts, preds):
        if int(p) == int(d):
            matched += 1
        else:
            break
    return matched


# --------------------------------------------------------- 完整树形验证(batch-of-paths)
def tree_verify_step(model, drafter, x, H_last, x_ids, mtp_cache, main_cache, K, P, snap_d):
    """一次 batched 前向并行验证 P 条候选路径,提交接受最长的赢家路径。

    返回 (new_tokens, x_new, rH, accepted_in, matched)。

    原理:每条路径 `[x, d_1..d_{K-1}]` 是普通线性序列,拍到 batch 维后线性层/全注意力层都走成熟
    batch 前向(逐 row 与 batch=1 等价,见 benchmarks/test_layerwise.py)。batch=P 的计算量加宽了
    每层预取窗口(hit_rate),多路径又提升接受长度(accept_len)。验证后按赢家 row 提取 checkpoint
    /trim 提交回 batch=1 主 cache,后续解码逐 token 从赢家路径续,与非投机贪婪等价。
    """
    paths = drafter.draft_paths(H_last, x_ids, mtp_cache, K, P)
    verify_in = mx.array([[x] + p[: K - 1] for p in paths])       # (P, K)

    snap_m = _snapshot(main_cache)                                # tile 前的 batch=1 快照(回退用)
    tile_caches(main_cache, len(paths))
    begin_speculative_checkpoints(main_cache)
    vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
    mx.eval(vlogits, vH)

    best_w, matched, preds = 0, -1, None
    for wi, p in enumerate(paths):
        preds_i = [int(t) for t in mx.argmax(vlogits[wi], axis=-1)]
        m = accept_prefix(p, preds_i)
        if m > matched:                                          # 平局取更靠前的路径(top-1 优先)
            best_w, matched, preds = wi, m, preds_i
    drafts = paths[best_w]
    accepted_len = min(matched + 1, K)

    committed = commit_tree_row(main_cache, verified_len=K,
                                accepted_len=accepted_len, row=best_w)
    if snap_d is not None:
        _restore(mtp_cache, snap_d)

    if not committed:
        # 理论上 checkpoint 齐全不会走到;稳妥回退:恢复 batch=1 主 cache 后重放赢家 accepted prefix。
        _restore(main_cache, snap_m)
        accepted_in = mx.array([[x] + drafts[:matched]])
        rlogits, rH = forward_with_hidden(model, accepted_in, main_cache)
        mx.eval(rlogits)
        bonus = int(mx.argmax(rlogits[:, -1, :]))
        return drafts[:matched] + [bonus], bonus, rH, accepted_in, matched

    rH = vH[best_w:best_w + 1, :accepted_len, :]
    accepted_in = verify_in[best_w:best_w + 1, :accepted_len]
    if matched == K:
        return drafts[:K], drafts[-1], rH, accepted_in, matched
    return drafts[:matched] + [preds[matched]], preds[matched], rH, accepted_in, matched


# ----------------------------------------------------------------- 主循环
def mtp_generate(model, drafter, tok, prompt, max_tokens, K=3, ids_mode=False,
                 profile=False, on_tokens=None, main_cache=None, cached_len=0):
    """贪婪 MTP 自投机。

    drafter 需提供 draft(H_last(1,1,H), x_ids(1,1), mtp_cache, K) -> list[int](长度 K);
    可选 make_cache()->list 与 sync(rH, replay_in, mtp_cache)。
    ids_mode=True 时 prompt 已是 ids(1,L) 且返回 token id 列表(测试用)。
    on_tokens:可选流式钩子,每步用「本步真正写入 produced 的 token id 列表」回调一次
    (prefill 的首 token 作为第一次回调);返回 True 表示请求尽快停止生成。

    跨轮复用(main_cache/cached_len):传入已持有前 cached_len 个 prompt token 的 main_cache,
    则只 prefill `prompt[:, cached_len:]`(等价于一次多 token decode,复用历史 KV/递归态,不重算)。
    调用方须保证 prompt[:cached_len] 与 cache 中已有 token 完全一致(否则结果错误)。
    不变式:返回时 main_cache 恰好持有 `prompt + produced[:-1]`(produced[-1] 为 pending,未入 cache),
    调用方可据此拼下轮的 cached prefix。main_cache=None 时内部新建(默认,整段 prefill)。
    """
    enable_qwen3next_speculative_checkpoints()
    if main_cache is None:
        main_cache = model.make_cache()
        cached_len = 0
    mtp_cache = drafter.make_cache() if hasattr(drafter, "make_cache") else None
    ids = prompt if ids_mode else mx.array([tok.encode(prompt)])
    if cached_len < 0 or cached_len >= ids.shape[1]:
        raise ValueError(
            f"cached_len={cached_len} 必须落在 [0, prompt_len={ids.shape[1]}) 内,至少留 1 个 token 供 prefill")

    # prefill(分块):只喂尚未在 cache 中的后缀 ids[:, cached_len:](cached_len==0 即整段);
    # 得到第 1 个 pending token x 与其 hidden;分块把激活峰值压到与 decode 同稳态。
    logits, H = prefill_chunked(model, ids[:, cached_len:], main_cache)
    x = int(mx.argmax(logits[:, -1, :]))
    H_last = H[:, -1:, :]
    produced = [x]
    _stop_from_initial = on_tokens is not None and on_tokens([x])
    mx.eval(H_last)
    t0 = time.perf_counter()
    n_steps = 0
    # profile 分段累加(秒):把 sync/finalize 拆开,避免隐藏在 replay 名下。
    t_draft = t_verify = t_replay = t_snap = 0.0
    t_commit = t_sync = t_finalize = 0.0
    direct_commits = fallback_replays = replayed_tokens = 0
    # matched 直方图:accept_hist[j] = 恰好命中 j 个草稿(j∈0..K)的步数。用于实测每位置接受率,
    # 不做几何分布假设。emitted/step = min(matched+1,K)，故上限为 K(本实现 verify 只喂 K-1 草稿)。
    accept_hist = [0] * (K + 1)
    # top-k 覆盖探针:每位置统计模型真实 token 落在 MTP top-1/2/3 的步数(树形救回上界)。
    topk_probe = config.accept_topk() if profile else 0
    tk_n = [0] * K
    tk_cover1 = [0] * K
    tk_cover2 = [0] * K
    tk_cover3 = [0] * K
    verify_mode = config.mtp_verify_mode()
    tree_mode = config.tree_top2()
    tree_verify_mode = config.tree_verify()     # 完整树形验证(batch-of-paths),优先级最高
    tree_P = max(1, config.tree_branches())
    tree_rescues = 0                            # 位置1 top-2 成功救回(B 链首命中)的步数
    # 纯 batch 直接提交路径 + 模型保证 commit 恒成功时,跳过每步一次「全 cache 深拷贝 + eval」
    # (snap_m 只用于 tree 救回 / step / replay 回退,这些路径都不满足下方条件)。省 ~72MiB 在途
    # 峰值和一次同步栅栏,数值完全不变(bit-exact,只改内存调度)。模型结构恒定,循环外算一次即可。
    _skip_snap = (verify_mode != "step") and (not tree_mode) and (not tree_verify_mode) \
        and _batch_direct_commit_guaranteed(model)

    while len(produced) < max_tokens and not _stop_from_initial:
        x0 = x
        x_ids = mx.array([[x]])

        snap_d = _snapshot(mtp_cache) if mtp_cache else None
        prev_H_last = H_last

        # 完整树形验证:一次 batched 前向验证 P 条路径,提交赢家。自带 step 尾处理后 continue。
        if tree_verify_mode and verify_mode != "step":
            if profile:
                _tic = time.perf_counter()
            new_tokens, x, rH, accepted_in, matched = tree_verify_step(
                model, drafter, x, H_last, x_ids, mtp_cache, main_cache, K, tree_P, snap_d)
            accept_hist[matched] += 1
            direct_commits += 1
            if profile:
                t_verify += time.perf_counter() - _tic
                _tic = time.perf_counter()
            _stop = False
            _n_before = len(produced)
            for t in new_tokens:
                produced.append(t)
                if len(produced) >= max_tokens:
                    break
            # 只把「本步真正写入 produced」的 token 交给回调:多 token 步命中 max_tokens
            # 上限时,截断掉的尾巴不应上报给流式消费者(避免超报被丢弃的 token)。
            if on_tokens is not None and on_tokens(produced[_n_before:]):
                _stop = True
            H_last = rH[:, -1:, :]
            if mtp_cache is not None and hasattr(drafter, "sync"):
                drafter.sync(prev_H_last, rH, accepted_in, mtp_cache)
            if profile:
                t_sync += time.perf_counter() - _tic
            n_steps += 1
            mx.eval(x, H_last)
            if _stop:
                break
            continue

        if profile:
            _tic = time.perf_counter()
        tree_b = None
        if tree_mode and verify_mode != "step":
            drafts, tree_b = drafter.draft_tree(H_last, x_ids, mtp_cache, K)  # chainA, chainB
            draft_cands = None
        elif topk_probe > 0:
            drafts, draft_cands = drafter.draft(H_last, x_ids, mtp_cache, K, topk=topk_probe)
        else:
            drafts, draft_cands = drafter.draft(H_last, x_ids, mtp_cache, K), None  # 长度 K
        verify_in = mx.array([[x] + drafts[: K - 1]])              # [x, d_1..d_{K-1}]
        if profile:
            mx.eval(verify_in)
            t_draft += time.perf_counter() - _tic
            _tic = time.perf_counter()

        snap_m = None if _skip_snap else _snapshot(main_cache)
        if profile:
            t_snap += time.perf_counter() - _tic
            _tic = time.perf_counter()

        verify_snaps = None
        if verify_mode == "step":
            vlogits, vH, verify_snaps = forward_with_hidden_stepwise(
                model, verify_in, main_cache, capture_snapshots=True)
        else:
            begin_speculative_checkpoints(main_cache)
            vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
            mx.eval(vlogits, vH)
        preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]   # 长度 K
        matched = accept_prefix(drafts, preds)
        # 最小树救回:A 链首草稿被拒(matched==0)且 B 候选=模型真实 token(=preds[0])时,
        # 恢复 cache、改验 B 链。preds[0] 只依赖 [x],两链一致;故 d1b==preds[0] 即 B 首命中。
        if tree_b is not None and matched == 0 and tree_b[0] == preds[0]:
            _restore(main_cache, snap_m)
            begin_speculative_checkpoints(main_cache)
            # 重建 verify_in 为 B 链:后续 accepted_in=verify_in[:, :accepted_len] 才会取到 chainB,
            # 否则残留 chainA 的错误首草稿会经 sync 污染 MTP cache、拉低后续草稿质量(低估救回收益)。
            verify_in = mx.array([[x] + tree_b[:K - 1]])
            vlogits, vH = forward_with_hidden(model, verify_in, main_cache)
            mx.eval(vlogits, vH)
            preds = [int(t) for t in mx.argmax(vlogits[0], axis=-1)]
            drafts = tree_b
            matched = accept_prefix(drafts, preds)
            tree_rescues += 1
        accept_hist[matched] += 1
        if draft_cands is not None:
            # preds[i]=模型在位置 i 的真实下一 token;查它是否在 MTP 该位置 top-1/2/3 候选里。
            for i in range(K):
                p = preds[i]
                c = draft_cands[i]
                tk_n[i] += 1
                if p == c[0]:
                    tk_cover1[i] += 1
                if p in c[:2]:
                    tk_cover2[i] += 1
                if p in c[:3]:
                    tk_cover3[i] += 1
        if profile:
            t_verify += time.perf_counter() - _tic
            _tic = time.perf_counter()

        accepted_len = min(matched + 1, K)

        if verify_snaps is not None:
            # step 模式：逐 token 解码路径产生的快照可精确 direct commit。
            committed = commit_verified_snapshot(main_cache, verify_snaps,
                                                 accepted_len, verified_len=K)
        else:
            # batch 模式：一次并行验证后按接受长度直接提交，零 replay。
            # - 可裁剪 cache（KVCache）：trim 掉 rejected 后缀即精确。
            # - 递归状态 cache（Qwen3-Next gated-delta 的 conv/ssm）：verify 前向里用
            #   multistate kernel 捕获的 per-token checkpoint 与 baseline kernel 逐 bit 等价
            #   （见 mtp/kv_cache.py 与 tests/test_gated_delta_multistate.py），可精确直提交。
            # commit_verified_prefix 一次处理混合 cache：KV 走 trim、Arrays 走 checkpoint 直提交。
            # 若任一递归 cache 缺 checkpoint（异常情况）则返回 False，自动回退到下方 replay。
            committed = commit_verified_prefix(main_cache, verified_len=K,
                                               accepted_len=accepted_len)
        if snap_d is not None:
            _restore(mtp_cache, snap_d)

        accepted_in = verify_in[:, :accepted_len]
        if profile:
            t_commit += time.perf_counter() - _tic
            _tic = time.perf_counter()
        if committed:
            direct_commits += 1
            # 直接使用验证前向产生的 hidden 同步 MTP cache,避免主模型 replay。
            rH = vH[:, :accepted_len, :]
            if matched == K:
                new_tokens = drafts[:K]
                x = drafts[-1]
            else:
                new_tokens = drafts[:matched] + [preds[matched]]
                x = preds[matched]
        else:
            fallback_replays += 1
            if snap_m is None:
                # 理论不可达:batch 直接提交路径 checkpoint 齐全,commit 必成功。若真走到这里,
                # 说明前提被破坏(如未捕获 checkpoint),此时无快照可回退,直接抛错定位而非静默错算。
                raise RuntimeError("batch verify commit failed but snapshot was skipped")
            # fallback:回滚到验证前,重放 accepted prefix,保证 ArraysCache 正确。
            _restore(main_cache, snap_m)
            accepted_in = mx.array([[x0] + drafts[:matched]])
            replayed_tokens += accepted_in.shape[1]
            rlogits, rH = forward_with_hidden(model, accepted_in, main_cache)
            mx.eval(rlogits)
            bonus = int(mx.argmax(rlogits[:, -1, :]))
            new_tokens = drafts[:matched] + [bonus]
            x = bonus
        if profile:
            t_replay += time.perf_counter() - _tic
            _tic = time.perf_counter()

        _n_before = len(produced)
        for t in new_tokens:
            produced.append(t)
            if len(produced) >= max_tokens:
                break
        # 只把「本步真正写入 produced」的 token 交给回调:多 token 步命中 max_tokens
        # 上限时,截断掉的尾巴不应上报给流式消费者(避免超报被丢弃的 token)。
        _stop = on_tokens is not None and on_tokens(produced[_n_before:])

        H_last = rH[:, -1:, :]
        if mtp_cache is not None and hasattr(drafter, "sync"):
            drafter.sync(prev_H_last, rH, accepted_in, mtp_cache)
        if profile:
            t_sync += time.perf_counter() - _tic
            _tic = time.perf_counter()
        n_steps += 1
        mx.eval(x, H_last)
        if profile:
            t_finalize += time.perf_counter() - _tic
        if _stop:
            break

    produced = produced[:max_tokens]
    wall = time.perf_counter() - t0
    # cache 实际驻留 token 数(以可裁剪 cache 的 offset 为准),供跨轮复用精确对账:
    # 末步多 token 跨 max_tokens 会 over-commit(cache 领先于 produced),据此识别并禁用复用。
    resident_tokens = None
    for c in main_cache:
        if getattr(c, "is_trimmable", None) and c.is_trimmable() and hasattr(c, "offset"):
            resident_tokens = int(c.offset)
            break
    stats = {
        "steps": n_steps,
        "tokens": len(produced),
        "resident_tokens": resident_tokens,
        "avg_accept_len": round(len(produced) / max(n_steps, 1), 3),
        "wall_s": round(wall, 3),
        "verify_mode": verify_mode,
        "direct_commits": direct_commits,
        "fallback_replays": fallback_replays,
        "replayed_tokens": replayed_tokens,
        "accept_hist": accept_hist,               # [恰好命中0,1,...,K 个草稿的步数]
        "tree_rescues": tree_rescues,             # 最小树位置1 成功救回步数(tree_top2 开时)
    }
    if topk_probe > 0:
        stats["topk_probe"] = {
            "topk": topk_probe, "n": tk_n,
            "cover_top1": tk_cover1, "cover_top2": tk_cover2, "cover_top3": tk_cover3,
        }
    if profile:
        seg = t_draft + t_snap + t_verify + t_commit + t_replay + t_sync + t_finalize
        stats.update({
            "t_draft_s": round(t_draft, 3),
            "t_snap_s": round(t_snap, 3),
            "t_verify_s": round(t_verify, 3),
            "t_commit_s": round(t_commit, 3),
            "t_replay_s": round(t_replay, 3),
            "t_sync_s": round(t_sync, 3),
            "t_finalize_s": round(t_finalize, 3),
            # 「重放免费」上限:草稿+验证(不含快照,因 per-token 检查点设计无需全 cache 快照)
            "proj_no_replay_tps": round(len(produced) / max(t_draft + t_verify, 1e-6), 2),
            "measured_seg_tps": round(len(produced) / max(seg, 1e-6), 2),
        })
    if ids_mode:
        return produced, stats
    return tok.decode(produced), stats
