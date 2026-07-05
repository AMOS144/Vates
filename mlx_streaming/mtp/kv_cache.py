"""MTP 投机解码的 KV/递归状态 cache 校验机制：per-token checkpoint + 快照/恢复/提交。

speculative decoding 需要在「验证 K 个草稿」后只保留 accepted_len 个 token 对 cache 的
贡献。两类 cache 处理方式不同：
- 可裁剪 cache（KVCache）：直接 trim 掉 rejected 后缀。
- 递归状态 cache（Qwen3-Next 线性注意力 ArraysCache）：必须在 verify 前向中逐 token 记录
  `[conv, ssm]` checkpoint 才能精确提交，否则回退到「快照 + replay accepted prefix」。
"""
import mlx.core as mx
import mlx.nn as nn

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

    from mlx_lm.models.qwen3_next import Qwen3NextGatedDeltaNet
    from mlx_streaming.core.linear_attn.gated_delta_multistate import (
        gated_delta_update_multistate,
    )

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

        # 关键：用 multistate kernel 一次前向算出每个 token 处理后的 ssm 递归态。
        # 它与 baseline 解码走的上游 gated_delta_kernel 是同一份 kernel、同序、fp32，
        # 因此 states_out[:, i] 与「逐 token 单步解码」逐 bit 等价（见
        # tests/test_gated_delta_multistate.py）——这是验证后能直接提交、零 replay 的根基。
        # 注意：kernel 路径内部处理 GQA（hk_idx 映射），不在 Python 侧 repeat q/k，
        # 与 baseline kernel 路径对齐（旧 ops 路径的 repeat 会引入数值差异）。
        out, state, states_out = gated_delta_update_multistate(
            q, k, v, a, b, self.A_log, self.dt_bias, state, mask
        )

        if cache is not None:
            cache[1] = state
            cache.advance(S)
            cache._spec_checkpoints = [
                [conv_checkpoints[i], states_out[:, i]]
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
        # 必须装入副本:ArraysCache.state setter 是别名赋值(self.cache=v),后续前向
        # 的 cache[idx]=new 会原地改写这个 list 的元素,从而污染快照本体。救回路径会对
        # 同一 snap_m 恢复两次(救回时一次 + fallback 一次),若不复制,第二次恢复到的将是
        # 被中间前向污染的状态 → 递归 cache 损坏、输出发散。
        c.state = _copy_state(st)
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


# ----------------------------------------------------------- 树形验证 batch 工具
def _tile_state(st, P: int):
    """把 cache.state 沿 batch 轴(axis0)复制 P 份((1,...) → (P,...))。"""
    if st is None:
        return None
    if isinstance(st, (list, tuple)):
        return type(st)(_tile_state(s, P) for s in st)
    return mx.contiguous(mx.repeat(st, P, axis=0))


def _row_state(st, w: int):
    """取 cache.state 的第 w 行((P,...) → (1,...)),保持 batch 维。"""
    if st is None:
        return None
    if isinstance(st, (list, tuple)):
        return type(st)(_row_state(s, w) for s in st)
    return mx.contiguous(st[w:w + 1])


def tile_caches(caches, P: int):
    """把 batch=1 的 cache 全部平铺到 batch=P,供 batch-of-paths 树验证并行前向。

    KVCache 的 state setter 会按数组 shape 复位 offset;ArraysCache 无 offset。两者 meta
    (lengths/left_padding) 在贪婪解码里恒为 None,tile 后语义不变。前提:所有 cache 当前 batch=1。
    """
    for c in caches:
        c.state = _tile_state(c.state, P)


def commit_tree_row(caches, verified_len: int, accepted_len: int, row: int) -> bool:
    """把 batched(batch=P)验证前向的第 `row` 条路径,按 accepted_len 提交回 batch=1 主 cache。

    等价于「先按接受长度裁剪 rejected 后缀,再抽出赢家路径那一行」:
    - 可裁剪 cache(KVCache):trim(rejected) 后取第 row 行 KV。
    - 递归状态 cache(ArraysCache):取 verify 前向捕获的 per-token checkpoint[accepted_len-1]
      的第 row 行 [conv, ssm]。要求 batched 前向前调过 begin_speculative_checkpoints。
    提交后主 cache 变回 batch=1,后续解码逐 token 从赢家路径续。
    """
    rejected = verified_len - accepted_len
    if rejected < 0:
        raise ValueError("accepted_len cannot exceed verified_len")

    def can_commit(c):
        if c.is_trimmable():
            return True
        cks = getattr(c, "_spec_checkpoints", None)
        return cks is not None and len(cks) >= accepted_len

    if not all(can_commit(c) for c in caches):
        return False
    for c in caches:
        if c.is_trimmable():
            if rejected:
                c.trim(rejected)
            c.state = _row_state(c.state, row)
        else:
            ck = c._spec_checkpoints[accepted_len - 1]     # [conv(P,...), ssm(P,...)]
            c.state = [_row_state(ck[0], row), _row_state(ck[1], row)]
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
