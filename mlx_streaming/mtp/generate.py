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
    _snapshot, _restore, commit_verified_prefix, commit_verified_snapshot)


def forward_with_hidden(model, ids, cache):
    """跑主模型层循环 + 最终 norm,返回 (logits(1,L,V), H(1,L,hidden))。H 为 norm 后。"""
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
    return model.lm_head(H), H


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


# ----------------------------------------------------------------- 主循环
def mtp_generate(model, drafter, tok, prompt, max_tokens, K=3, ids_mode=False,
                 profile=False):
    """贪婪 MTP 自投机。

    drafter 需提供 draft(H_last(1,1,H), x_ids(1,1), mtp_cache, K) -> list[int](长度 K);
    可选 make_cache()->list 与 sync(rH, replay_in, mtp_cache)。
    ids_mode=True 时 prompt 已是 ids(1,L) 且返回 token id 列表(测试用)。
    """
    enable_qwen3next_speculative_checkpoints()
    main_cache = model.make_cache()
    mtp_cache = drafter.make_cache() if hasattr(drafter, "make_cache") else None
    ids = prompt if ids_mode else mx.array([tok.encode(prompt)])

    # prefill:得到第 1 个 pending token x 与其 hidden
    logits, H = forward_with_hidden(model, ids, main_cache)
    x = int(mx.argmax(logits[:, -1, :]))
    H_last = H[:, -1:, :]
    produced = [x]
    mx.eval(H_last)
    t0 = time.perf_counter()
    n_steps = 0
    # profile 分段累加(秒):把 sync/finalize 拆开,避免隐藏在 replay 名下。
    t_draft = t_verify = t_replay = t_snap = 0.0
    t_commit = t_sync = t_finalize = 0.0
    direct_commits = fallback_replays = replayed_tokens = 0
    verify_mode = config.mtp_verify_mode()

    while len(produced) < max_tokens:
        x0 = x
        x_ids = mx.array([[x]])

        snap_d = _snapshot(mtp_cache) if mtp_cache else None
        prev_H_last = H_last
        if profile:
            _tic = time.perf_counter()
        drafts = drafter.draft(H_last, x_ids, mtp_cache, K)        # 长度 K
        verify_in = mx.array([[x] + drafts[: K - 1]])              # [x, d_1..d_{K-1}]
        if profile:
            mx.eval(verify_in)
            t_draft += time.perf_counter() - _tic
            _tic = time.perf_counter()

        snap_m = _snapshot(main_cache)
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
        if profile:
            t_verify += time.perf_counter() - _tic
            _tic = time.perf_counter()

        accepted_len = min(matched + 1, K)

        has_recurrent_cache = any(not c.is_trimmable() for c in main_cache)
        if verify_snaps is not None:
            # step 模式：逐 token 解码路径产生的快照可精确 direct commit。
            committed = commit_verified_snapshot(main_cache, verify_snaps,
                                                 accepted_len, verified_len=K)
        elif has_recurrent_cache:
            # 递归状态 cache（Qwen3-Next gated-delta 的 conv/ssm）一律回滚重放，保证与非投机
            # 贪婪逐位等价。原因：batch verify 捕获的 per-token checkpoint 走手写
            # _gated_delta_step_ops，与 baseline 的 kernel 化 gated_delta_update 不逐 bit 等价，
            # 直接提交会产生不忠实输出（实测 MAXTOK=96 时 37/96 token 偏离贪婪）；且其更低的接受率
            # 反让端到端更慢（实测 13.6 vs 重放 14.5 tok/s）——既不正确也不更快，故不再提供该捷径。
            committed = False
        else:
            # 可裁剪 cache（KVCache）：trim 掉 rejected 后缀即精确，直接提交省一次重放。
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

        for t in new_tokens:
            produced.append(t)
            if len(produced) >= max_tokens:
                break

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

    produced = produced[:max_tokens]
    wall = time.perf_counter() - t0
    stats = {
        "steps": n_steps,
        "tokens": len(produced),
        "avg_accept_len": round(len(produced) / max(n_steps, 1), 3),
        "wall_s": round(wall, 3),
        "verify_mode": verify_mode,
        "direct_commits": direct_commits,
        "fallback_replays": fallback_replays,
        "replayed_tokens": replayed_tokens,
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
