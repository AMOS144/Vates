"""MTP 自投机:模块增强 + cache 快照/恢复 + 接受判定 + 贪婪等价性。"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as kvcache
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mtp.qwen3_next_mtp import Qwen3NextMTP, mtp_advance, mtp_step


def _tiny_args():
    return ModelArgs(
        model_type="qwen3_next", hidden_size=32, num_hidden_layers=1,
        intermediate_size=64, num_attention_heads=4, linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, num_experts=4, num_experts_per_tok=2,
        decoder_sparse_step=1, shared_expert_intermediate_size=64, mlp_only_layers=[],
        moe_intermediate_size=64, rms_norm_eps=1e-6, vocab_size=50,
        num_key_value_heads=2, rope_theta=10000.0, partial_rotary_factor=0.5,
        max_position_embeddings=128, head_dim=8, full_attention_interval=4)


# ---------------------------------------------------------------- Task 1
def test_return_hidden_matches_lm_head():
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    head = mx.random.normal((args.hidden_size, args.vocab_size))
    lm = lambda h: h @ head
    hidden = mx.random.normal((1, 4, args.hidden_size))
    next_ids = mx.array([[1, 2, 3, 4]])
    logits, H = mtp(hidden, next_ids, lm, return_hidden=True)
    assert H.shape == (1, 4, args.hidden_size)
    assert float(mx.abs(logits - (H @ head)).max()) < 1e-4


def test_mtp_step_single_token():
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    head = mx.random.normal((args.hidden_size, args.vocab_size))
    lm = lambda h: h @ head
    c = kvcache.KVCache()
    h = mx.random.normal((1, 1, args.hidden_size))
    logits, mh = mtp_step(mtp, h, mx.array([[7]]), lm, c)
    assert logits.shape == (1, args.vocab_size)
    assert mh.shape == (1, 1, args.hidden_size)


def test_mtp_advance_batch_matches_stepwise_hidden():
    """批量推进 cache 的 hidden 应与逐 token 推进一致,且不依赖 logits。"""
    mx.random.seed(6)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    head = mx.random.normal((args.hidden_size, args.vocab_size))
    lm = lambda h: h @ head
    hidden = mx.random.normal((1, 3, args.hidden_size))
    token = mx.array([[3, 5, 7]])

    batch_cache = kvcache.KVCache()
    batch_H = mtp_advance(mtp, hidden, token, batch_cache)

    step_cache = kvcache.KVCache()
    parts = []
    for i in range(token.shape[1]):
        _, H = mtp_step(mtp, hidden[:, i:i + 1, :], token[:, i:i + 1], lm, step_cache)
        parts.append(H)
    step_H = mx.concatenate(parts, axis=1)

    assert batch_cache.offset == step_cache.offset == 3
    assert float(mx.abs(batch_H - step_H).max()) < 2e-3


# ---------------------------------------------------------------- Task 2
def test_snapshot_restore_kvcache_idempotent():
    from mlx_streaming.mtp.kv_cache import _snapshot, _restore
    mx.random.seed(0)
    c = kvcache.KVCache()
    k0 = mx.random.normal((1, 2, 3, 8))
    c.update_and_fetch(k0, k0)            # offset=3
    snap = _snapshot([c])
    k1 = mx.random.normal((1, 2, 2, 8))
    c.update_and_fetch(k1, k1)            # offset=5
    assert c.offset == 5
    _restore([c], snap)
    assert c.offset == 3
    ks, vs = c.state
    assert ks.shape[2] == 3


def test_snapshot_restore_empty_kvcache():
    """空 KVCache 也要能快照/恢复,真实 MTP cache 首步就是空的。"""
    from mlx_streaming.mtp.kv_cache import _snapshot, _restore

    c = kvcache.KVCache()
    snap = _snapshot([c])
    k0 = mx.random.normal((1, 2, 1, 8))
    c.update_and_fetch(k0, k0)
    assert c.offset == 1
    _restore([c], snap)
    assert c.empty()
    assert c.offset == 0


def test_snapshot_restore_arrayscache_idempotent():
    from mlx_streaming.mtp.kv_cache import _snapshot, _restore
    c = kvcache.ArraysCache(size=2)
    c.cache = [mx.ones((1, 3, 4)), mx.ones((1, 5))]
    snap = _snapshot([c])
    c.cache = [mx.zeros((1, 3, 4)), mx.zeros((1, 5))]
    _restore([c], snap)
    assert float(c.cache[0].sum()) == 12.0 and float(c.cache[1].sum()) == 5.0


# ---------------------------------------------------------------- Task 3
def test_accept_prefix_partial():
    from mlx_streaming.mtp.generate import accept_prefix
    assert accept_prefix([11, 22, 33], [11, 22, 99]) == 2


def test_accept_prefix_first_miss():
    from mlx_streaming.mtp.generate import accept_prefix
    assert accept_prefix([11, 22, 33], [99, 22, 33]) == 0


def test_accept_prefix_all_match():
    from mlx_streaming.mtp.generate import accept_prefix
    assert accept_prefix([11, 22, 33], [11, 22, 33]) == 3


# ---------------------------------------------------------------- Task 4/5 共用玩具模型
class _FAOnlyLayer(nn.Module):
    """真用 KV cache 的极简因果自注意力(增量解码 == 全前缀解码)。"""
    is_linear = False

    def __init__(self, h, n_heads=2):
        super().__init__()
        self.h, self.nh, self.hd = h, n_heads, h // n_heads
        self.q = nn.Linear(h, h, bias=False)
        self.k = nn.Linear(h, h, bias=False)
        self.v = nn.Linear(h, h, bias=False)
        self.o = nn.Linear(h, h, bias=False)
        self.norm = nn.RMSNorm(h)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q(x).reshape(B, L, self.nh, self.hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, L, self.nh, self.hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, L, self.nh, self.hd).transpose(0, 2, 1, 3)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.hd ** -0.5, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.h)
        return self.norm(x + self.o(out))


class _RecurLayer(nn.Module):
    is_linear = True

    def __init__(self, h):
        super().__init__()
        self.proj = nn.Linear(h, h, bias=False)

    def __call__(self, x, mask=None, cache=None):
        B, L, H = x.shape
        state = None
        if cache is not None and cache[1] is not None:
            state = cache[1]
        if state is None:
            state = mx.zeros((B, H))
        outs = []
        for t in range(L):
            state = state + x[:, t, :]
            outs.append(state)
        if cache is not None:
            cache[1] = state
        return self.proj(mx.stack(outs, axis=1))


class _Inner(nn.Module):
    def __init__(self, vocab, h, nl):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.layers = [_FAOnlyLayer(h) for _ in range(nl)]
        self.norm = nn.RMSNorm(h)


class _ToyModel(nn.Module):
    """仅全注意力的玩具自回归模型,接口契合 forward_with_hidden。"""

    def __init__(self, vocab=40, h=32, nl=2):
        super().__init__()
        self.model = _Inner(vocab, h, nl)
        self.lm_head = nn.Linear(h, vocab, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [kvcache.ArraysCache(2) if l.is_linear else kvcache.KVCache()
                for l in self.layers]


class _SelfDraft:
    """测试用:用主模型自身产生草稿(贪婪下与验证一致),仅为验证循环骨架。"""

    def __init__(self, model):
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def draft(self, H_last, x_ids, mtp_cache, K):
        from mlx_streaming.mtp.generate import forward_with_hidden
        # 用独立 cache 喂当前 token,取 argmax 作为单草稿(K 限定为 1)
        logits, _ = forward_with_hidden(self.model, x_ids, self.model.make_cache())
        return [int(mx.argmax(logits[:, -1, :]))]


class _RandDraft:
    """测试用:返回 K 个伪随机草稿(与主模型无关),专门压 matched=0/部分/全中 各路径。
    等价性必须与草稿质量无关。"""

    def __init__(self, vocab, seed=0):
        self.vocab = vocab
        self._s = seed

    def draft(self, H_last, x_ids, mtp_cache, K):
        out = []
        x = int(x_ids[0, 0])
        for i in range(K):
            self._s = (self._s * 1103515245 + 12345 + x + i) % 2147483647
            out.append(self._s % self.vocab)
        return out


def _naive_greedy(model, prompt, n):
    from mlx_streaming.mtp.generate import forward_with_hidden
    cache = model.make_cache()
    ids, out = prompt, []
    for _ in range(n):
        logits, _ = forward_with_hidden(model, ids, cache)
        nxt = int(mx.argmax(logits[:, -1, :]))
        out.append(nxt)
        ids = mx.array([[nxt]])
        mx.eval(ids)
    return out


def test_mtp_generate_greedy_equiv_kv():
    from mlx_streaming.mtp.generate import mtp_generate
    mx.random.seed(0)
    model = _ToyModel(nl=2)            # 全 KVCache(无线性层)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    prompt = mx.array([[1, 5, 9]])
    ref = _naive_greedy(model, prompt, 12)
    got, stats = mtp_generate(model, _SelfDraft(model), tok=None, prompt=prompt,
                              max_tokens=12, K=1, ids_mode=True)
    assert got == ref


def test_mtp_generate_greedy_equiv_arrayscache():
    from mlx_streaming.mtp.generate import mtp_generate
    mx.random.seed(1)
    model = _ToyModel(nl=2)
    model.model.layers[0] = _RecurLayer(32)   # 第 0 层换递归(ArraysCache)
    mx.eval(model.parameters())
    prompt = mx.array([[2, 7, 1]])
    ref = _naive_greedy(model, prompt, 10)
    got, _ = mtp_generate(model, _SelfDraft(model), None, prompt, 10, K=1, ids_mode=True)
    assert got == ref


def test_mtp_generate_greedy_equiv_kv_k3():
    """K=3 多草稿并行验证 + 部分命中路径,等价性与草稿质量无关。"""
    from mlx_streaming.mtp.generate import mtp_generate
    mx.random.seed(0)
    model = _ToyModel(nl=2, vocab=40)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    prompt = mx.array([[1, 5, 9]])
    ref = _naive_greedy(model, prompt, 16)
    got, stats = mtp_generate(model, _RandDraft(40), None, prompt, 16, K=3, ids_mode=True)
    assert got == ref


def test_mtp_generate_greedy_equiv_arrayscache_k3():
    """K=3 + 含 ArraysCache 递归层,验证多 token 重放下递归状态回滚正确。"""
    from mlx_streaming.mtp.generate import mtp_generate
    mx.random.seed(3)
    model = _ToyModel(nl=3, vocab=40)
    model.model.layers[1] = _RecurLayer(32)
    mx.eval(model.parameters())
    prompt = mx.array([[2, 7, 1]])
    ref = _naive_greedy(model, prompt, 16)
    got, _ = mtp_generate(model, _RandDraft(40, seed=5), None, prompt, 16, K=3, ids_mode=True)
    assert got == ref


def test_mtp_generate_arrayscache_default_uses_safe_replay(monkeypatch):
    """正确版:block-align exact 默认不直接提交递归 cache,而是重放 unsafe tail。"""
    from mlx_streaming.mtp.generate import mtp_generate

    monkeypatch.delenv("MTP_VERIFY_MODE", raising=False)
    monkeypatch.delenv("MTP_ARRAY_COMMIT", raising=False)
    mx.random.seed(5)
    model = _ToyModel(nl=3, vocab=40)
    model.model.layers[1] = _RecurLayer(32)
    mx.eval(model.parameters())
    prompt = mx.array([[2, 7, 1]])
    ref = _naive_greedy(model, prompt, 12)
    got, stats = mtp_generate(model, _RandDraft(40, seed=7), None, prompt, 12,
                              K=3, ids_mode=True)

    assert got == ref
    assert stats["fallback_replays"] > 0
    assert stats["direct_commits"] == 0


def test_mtp_generate_arrayscache_step_verify_direct_commit_exact(monkeypatch):
    """第二步:verify 逐 token 走解码路径时,可 exact 地 direct commit。"""
    from mlx_streaming.mtp.generate import mtp_generate

    monkeypatch.setenv("MTP_VERIFY_MODE", "step")
    monkeypatch.delenv("MTP_ARRAY_COMMIT", raising=False)
    mx.random.seed(6)
    model = _ToyModel(nl=3, vocab=40)
    model.model.layers[1] = _RecurLayer(32)
    mx.eval(model.parameters())
    prompt = mx.array([[2, 7, 1]])
    ref = _naive_greedy(model, prompt, 12)
    got, stats = mtp_generate(model, _RandDraft(40, seed=9), None, prompt, 12,
                              K=3, ids_mode=True)

    assert got == ref
    assert stats["direct_commits"] > 0
    assert stats["fallback_replays"] == 0


def test_mtp_drafter_sync_advances_cache_with_accepted_main_hidden():
    """vLLM 语义:MTP KV cache 随已接受 token 持久前进,不能每步清空。"""
    from mlx_streaming.mtp.drafter import MTPDrafter

    mx.random.seed(4)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    lm = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    drafter = MTPDrafter(mtp, lm)
    mtp_cache = drafter.make_cache()

    prev_H = mx.random.normal((1, 1, args.hidden_size))
    replay_in = mx.array([[5, 7]])
    rH = mx.random.normal((1, 2, args.hidden_size))
    drafter.sync(prev_H, rH, replay_in, mtp_cache)

    assert mtp_cache[0].offset == 2


def test_mtp_drafter_sync_does_not_compute_logits():
    """同步 MTP cache 只需推进状态,不能额外跑 lm_head 白算 logits。"""
    from mlx_streaming.mtp.drafter import MTPDrafter

    mx.random.seed(5)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())

    def fail_lm_head(_):
        raise AssertionError("sync 不应调用 lm_head")

    drafter = MTPDrafter(mtp, fail_lm_head)
    mtp_cache = drafter.make_cache()

    prev_H = mx.random.normal((1, 1, args.hidden_size))
    replay_in = mx.array([[5, 7]])
    rH = mx.random.normal((1, 2, args.hidden_size))
    drafter.sync(prev_H, rH, replay_in, mtp_cache)

    assert mtp_cache[0].offset == 2


def test_commit_verified_prefix_trims_kvcache_without_replay():
    """vLLM 语义:验证 K 个 token 后,KVCache 可直接裁掉 rejected 后缀完成 commit。"""
    from mlx_streaming.mtp.kv_cache import commit_verified_prefix

    c = kvcache.KVCache()
    keys = mx.random.normal((1, 2, 5, 4))
    vals = mx.random.normal((1, 2, 5, 4))
    c.update_and_fetch(keys, vals)

    committed = commit_verified_prefix([c], verified_len=5, accepted_len=3)

    assert committed is True
    assert c.offset == 3


def test_commit_verified_prefix_uses_arrayscache_checkpoint():
    """线性注意力递归 cache 有 per-token checkpoint 时,应能直接提交 accepted prefix。"""
    from mlx_streaming.mtp.kv_cache import commit_verified_prefix

    c = kvcache.ArraysCache(size=2)
    c.cache = [mx.full((1, 3, 4), 99.0), mx.full((1, 2, 2), 99.0)]
    c._spec_checkpoints = [
        [mx.full((1, 3, 4), 1.0), mx.full((1, 2, 2), 10.0)],
        [mx.full((1, 3, 4), 2.0), mx.full((1, 2, 2), 20.0)],
        [mx.full((1, 3, 4), 3.0), mx.full((1, 2, 2), 30.0)],
    ]

    committed = commit_verified_prefix([c], verified_len=3, accepted_len=2)

    assert committed is True
    assert float(c.cache[0].sum()) == 24.0
    assert float(c.cache[1].sum()) == 80.0


# ---------------------------------------------------------------- 最小树 top-2 救回
class _OracleTreeDraft:
    """玩具 drafter：用贪婪参考 ref 当"主模型真实 preds"的 oracle。

    每步 chainA 首草稿故意置错(必触发 matched==0)、chainB = 从当前位置起的真实贪婪续,
    故 chainB[0]==preds[0] 必触发救回且全命中。用于确定性地覆盖 tree_top2 救回分支,
    验证主输出仍逐 token 等于贪婪(lossless)。pos 靠 sync 每步按接受长度前进。
    """

    def __init__(self, ref, vocab):
        self.ref = list(ref)      # 完整贪婪续(主模型真值)
        self.pos = 0              # 当前 x 在 ref 中的下标
        self.vocab = vocab

    def make_cache(self):
        # 返回非 None,使 generate 触发 snap_d 与 sync(sync 用于推进 pos)。
        from mlx_lm.models import cache as kvcache
        return [kvcache.KVCache()]

    def draft_tree(self, H_last, x_ids, mtp_cache, K):
        nxt = self.ref[self.pos + 1: self.pos + 1 + K]   # 真实 preds[0..K-1]
        while len(nxt) < K:                              # 末尾补齐(会被拒,不破坏 lossless)
            nxt.append(0)
        wrong0 = (nxt[0] + 1) % self.vocab               # 保证 != preds[0]
        chain_a = [wrong0] + nxt[1:]                     # 首错 → matched==0
        chain_b = list(nxt)                              # 首=真实 → 救回且全命中
        return chain_a, chain_b

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        self.pos += int(replay_in.shape[1])              # 按本步接受长度前进


def test_mtp_generate_tree_top2_lossless_and_rescues(monkeypatch):
    """最小树 top-2:强制每步走救回分支,主输出必须逐 token 等于贪婪,且救回确有触发。"""
    from mlx_streaming.mtp.generate import mtp_generate

    monkeypatch.setenv("TREE_TOP2", "1")
    monkeypatch.delenv("MTP_VERIFY_MODE", raising=False)
    monkeypatch.delenv("TREE_VERIFY", raising=False)
    mx.random.seed(0)
    model = _ToyModel(nl=2, vocab=40)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    prompt = mx.array([[1, 5, 9]])
    ref = _naive_greedy(model, prompt, 16)

    drafter = _OracleTreeDraft(ref, vocab=40)
    got, stats = mtp_generate(model, drafter, None, prompt, 16, K=3, ids_mode=True)

    assert got == ref                       # lossless
    assert stats["tree_rescues"] > 0        # 救回分支确实被覆盖
