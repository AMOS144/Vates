"""Qwen3-Next MTP 自投机贪婪解码循环。

见 spec/plan: docs/superpowers/{specs,plans}/2026-06-07-qwen3next-mtp-self-speculation*。

核心:每步主模型前向出 hidden -> MTP 自回归抽 K 草稿 -> 主模型并行验证 ->
接受最长命中前缀 -> cache 快照/恢复/重放回滚(统一处理 ArraysCache 与 KVCache),
保证输出与非投机贪婪逐 token 等价。
"""
import time
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from mlx_streaming.qwen3_next_mtp import mtp_step


_QWEN3NEXT_CHECKPOINTS_PATCHED = False
_EMPTY_CACHE = object()


def enable_qwen3next_speculative_checkpoints():
    """给 Qwen3-Next 线性注意力层加 verify-time per-token cache checkpoint。

    普通前向仍走 mlx-lm 原实现;只有 `begin_speculative_checkpoints()` 标记过的
    ArraysCache 会走逐 token gated-delta ops,并把每个 prefix 后的 `[conv, ssm]`
    状态写入 `cache._spec_checkpoints`。
    """
    global _QWEN3NEXT_CHECKPOINTS_PATCHED
    if _QWEN3NEXT_CHECKPOINTS_PATCHED:
        return

    from mlx_lm.models.gated_delta import compute_g, _gated_delta_step_ops
    from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet

    orig_call = Qwen3NextGatedDeltaNet.__call__

    def patched_call(self, inputs, mask=None, cache=None):
        capture = bool(getattr(cache, "_capture_spec_checkpoints", False))
        if not capture:
            return orig_call(self, inputs, mask=mask, cache=cache)

        B, S, _ = inputs.shape
        q, k, v, z, b, a = self.fix_query_key_value_ordering(
            self.in_proj_qkvz(inputs), self.in_proj_ba(inputs)
        )

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        mixed_qkv = mx.concatenate(
            [q.reshape(B, S, -1), k.reshape(B, S, -1), v.reshape(B, S, -1)], axis=-1
        )
        if mask is not None:
            mixed_qkv = mx.where(mask[..., None], mixed_qkv, 0)
        conv_input = mx.concatenate([conv_state, mixed_qkv], axis=1)

        n_keep = self.conv_kernel_size - 1
        conv_checkpoints = [
            mx.contiguous(conv_input[:, i:i + n_keep, :])
            for i in range(1, S + 1)
        ]
        if cache is not None:
            cache[0] = conv_checkpoints[-1]

        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]

        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        beta = mx.sigmoid(b)
        g = compute_g(self.A_log, a, self.dt_bias)
        if state is None:
            Hk, Dk = q.shape[2:]
            Hv, Dv = v.shape[2:]
            state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        if (repeat_factor := v.shape[2] // q.shape[2]) > 1:
            q_step = mx.repeat(q, repeat_factor, -2)
            k_step = mx.repeat(k, repeat_factor, -2)
        else:
            q_step, k_step = q, k

        ys, ssm_checkpoints = [], []
        for i in range(S):
            y_i, state = _gated_delta_step_ops(
                q_step[:, i],
                k_step[:, i],
                v[:, i],
                g[:, i],
                beta[:, i],
                state,
                None if mask is None else mask[:, i],
            )
            ys.append(y_i)
            ssm_checkpoints.append(mx.array(state))
        out = mx.stack(ys, axis=1)

        if cache is not None:
            cache[1] = state
            cache.advance(S)
            cache._spec_checkpoints = [
                [conv_checkpoints[i], ssm_checkpoints[i]]
                for i in range(S)
            ]
            cache._capture_spec_checkpoints = False

        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))

    Qwen3NextGatedDeltaNet.__call__ = patched_call
    _QWEN3NEXT_CHECKPOINTS_PATCHED = True


def begin_speculative_checkpoints(caches):
    """标记 ArraysCache 在下一次 verify forward 中记录 per-token checkpoint。"""
    for c in caches:
        if not c.is_trimmable():
            c._capture_spec_checkpoints = True
            c._spec_checkpoints = None


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


# ----------------------------------------------------------------- cache 快照
def _copy_state(st):
    """深拷贝 cache.state(支持 None / array / list / tuple 嵌套)。"""
    if st is None:
        return None
    if isinstance(st, (list, tuple)):
        return type(st)(_copy_state(s) for s in st)
    return mx.array(st)


def _iter_arrays(st):
    if st is None:
        return
    if isinstance(st, (list, tuple)):
        for s in st:
            yield from _iter_arrays(s)
    else:
        yield st


def _snapshot(caches):
    """对每个 cache 深拷贝 state + meta_state(强制 eval,防 update_and_fetch 原地改写)。"""
    snaps = []
    arrays = []
    for c in caches:
        if hasattr(c, "keys") and hasattr(c, "empty") and c.empty():
            snaps.append((_EMPTY_CACHE, c.meta_state))
            continue
        st_copy = _copy_state(c.state)
        arrays.extend(_iter_arrays(st_copy))
        snaps.append((st_copy, c.meta_state))
    if arrays:
        mx.eval(arrays)
    return snaps


def _restore(caches, snaps):
    for c, (st, meta) in zip(caches, snaps):
        if st is _EMPTY_CACHE:
            if hasattr(c, "keys") and hasattr(c, "values"):
                c.keys = None
                c.values = None
                c.offset = 0
            elif hasattr(c, "cache"):
                c.cache = [None] * len(c.cache)
            c.meta_state = meta
            continue
        c.state = st
        c.meta_state = meta


def commit_verified_prefix(caches, verified_len: int, accepted_len: int) -> bool:
    """把验证前向产生的 cache 直接提交到 accepted prefix。

    vLLM 语义下,验证 K 个 token 后只保留 accepted_len 个 token 对 cache 的贡献。
    对 KVCache 这类可裁剪 cache,直接 trim 掉 rejected 后缀即可;对 ArraysCache 这类
    递归状态 cache,必须有 per-token checkpoint 才能精确提交,否则返回 False 让调用方
    走 fallback replay。
    """
    rejected = verified_len - accepted_len
    if rejected < 0:
        raise ValueError("accepted_len cannot exceed verified_len")
    def can_commit(c):
        if c.is_trimmable():
            return True
        checkpoints = getattr(c, "_spec_checkpoints", None)
        return checkpoints is not None and len(checkpoints) >= accepted_len

    if not all(can_commit(c) for c in caches):
        return False
    for c in caches:
        if c.is_trimmable():
            if rejected:
                c.trim(rejected)
        else:
            c.state = c._spec_checkpoints[accepted_len - 1]
        if hasattr(c, "_spec_checkpoints"):
            c._spec_checkpoints = None
        if hasattr(c, "_capture_spec_checkpoints"):
            c._capture_spec_checkpoints = False
    return True


def commit_verified_snapshot(caches, snapshots, accepted_len: int,
                             verified_len: int | None = None) -> bool:
    """把 cache 恢复到 stepwise verify 的 accepted_len 后快照。"""
    if verified_len is not None and accepted_len == verified_len:
        return True
    if accepted_len <= 0 or accepted_len > len(snapshots):
        return False
    _restore(caches, snapshots[accepted_len - 1])
    return True


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
    t_draft = t_verify = t_replay = t_snap = 0.0   # profile 分段累加(秒)
    direct_commits = fallback_replays = replayed_tokens = 0
    verify_mode = os.environ.get("MTP_VERIFY_MODE", "batch")

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
        allow_array_commit = os.environ.get("MTP_ARRAY_COMMIT", "0") == "1"
        if verify_snaps is not None:
            committed = commit_verified_snapshot(main_cache, verify_snaps,
                                                 accepted_len, verified_len=K)
        elif has_recurrent_cache and not allow_array_commit:
            # MLX 中 verify 长度 K 与 replay 长度 accepted_len 的 prefix hidden 不逐 bit 等价;
            # 直接提交 ArraysCache checkpoint 会破坏贪婪精确等价,默认一律 replay。
            committed = False
        else:
            committed = commit_verified_prefix(main_cache, verified_len=K,
                                               accepted_len=accepted_len)
        if snap_d is not None:
            _restore(mtp_cache, snap_d)

        accepted_in = verify_in[:, :accepted_len]
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

        for t in new_tokens:
            produced.append(t)
            if len(produced) >= max_tokens:
                break

        H_last = rH[:, -1:, :]
        if mtp_cache is not None and hasattr(drafter, "sync"):
            drafter.sync(prev_H_last, rH, accepted_in, mtp_cache)
        n_steps += 1
        mx.eval(x, H_last)
        if profile:
            t_replay += time.perf_counter() - _tic

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
        seg = t_draft + t_snap + t_verify + t_replay
        stats.update({
            "t_draft_s": round(t_draft, 3),
            "t_snap_s": round(t_snap, 3),
            "t_verify_s": round(t_verify, 3),
            "t_replay_s": round(t_replay, 3),
            # 「重放免费」上限:草稿+验证(不含快照,因 per-token 检查点设计无需全 cache 快照)
            "proj_no_replay_tps": round(len(produced) / max(t_draft + t_verify, 1e-6), 2),
            "measured_seg_tps": round(len(produced) / max(seg, 1e-6), 2),
        })
    if ids_mode:
        return produced, stats
    return tok.decode(produced), stats


# ----------------------------------------------------------------- 真实 MTP drafter
class MTPDrafter:
    """把 Qwen3NextMTP 包成 mtp_generate 需要的 drafter 接口。"""

    def __init__(self, mtp, lm_head):
        self.mtp = mtp
        self.lm_head = lm_head
        self.embed_tokens = mtp.embed_tokens

    def make_cache(self):
        from mlx_lm.models import cache as kc
        return [kc.KVCache()]            # MTP 单层全注意力

    def draft(self, H_last, x_ids, mtp_cache, K):
        drafts = []
        h, cur = H_last, x_ids
        for _ in range(K):
            logits, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
            d = int(mx.argmax(logits[0]))
            drafts.append(d)
            h, cur = mh, mx.array([[d]])
        return drafts

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        """用已接受 token 的真实主模型 hidden 推进 MTP KV cache。

        MTP 在位置 i 消费 (H_i, t_{i+1}) 预测 t_{i+2};因此提交 accepted prefix
        `[t_{i+1}, ..., t_{i+n}]` 时,hidden 序列应为 `[H_i, ..., H_{i+n-1}]`。
        """
        h_seq = mx.concatenate([prev_H, rH[:, :-1, :]], axis=1)
        for i in range(replay_in.shape[1]):
            mtp_step(self.mtp, h_seq[:, i:i + 1, :], replay_in[:, i:i + 1],
                     self.lm_head, mtp_cache[0])
