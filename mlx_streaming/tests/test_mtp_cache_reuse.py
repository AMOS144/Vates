"""跨轮复用 main_cache 的正确性:只 prefill 新增后缀,产出必须与每轮全量 prefill 逐 token 一致。

主模型 verify 保证输出与贪婪逐 token 等价,与草稿质量无关;因此只要 cache 复用正确,
「复用后缀 prefill」与「全量 prefill」必然产出完全相同的 token。任何 cache 污染都会让本测试变红。

关键不变式(mtp_generate 内部保证):一轮结束时 main_cache 恰好持有 `ids + produced[:-1]`,
最后一个 pending token(produced[-1])尚未写入 cache。跨轮据此拼 cached_ids 即可精确对齐。
"""
import mlx.core as mx

from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.tests.test_mtp_generate import _RandDraft, _ToyModel
from mlx_streaming.tests.test_mtp_stream_hook import _AllAcceptDraft, _kv_toy_k3


def _gen(model, ids, max_tokens, *, main_cache=None, cached_len=0):
    # 每次用新 drafter 实例;草稿质量不影响正确性,仅影响接受率。
    return mtp_generate(model, _RandDraft(40), None, mx.array([ids]), max_tokens,
                        K=3, ids_mode=True, main_cache=main_cache, cached_len=cached_len)


def test_reuse_turn1_equals_fresh_turn1():
    """显式传入空 cache 的首轮结果,应与默认(内部建 cache)完全一致。"""
    model = _kv_toy_k3()
    p1 = [1, 5, 9, 2, 7]
    fresh, _ = _gen(model, p1, 6)
    cache = model.make_cache()
    reuse, _ = _gen(model, p1, 6, main_cache=cache, cached_len=0)
    assert reuse == fresh


def test_second_turn_cache_reuse_matches_full_prefill():
    """第二轮复用首轮 cache 只 prefill 后缀,产出必须与第二轮全量 prefill 逐 token 一致。"""
    model = _kv_toy_k3()
    p1 = [1, 5, 9, 2, 7]

    # 首轮:显式 cache,记下产出;不变式 → cache 现持有 p1 + prod1[:-1]
    cache = model.make_cache()
    prod1, _ = _gen(model, p1, 6, main_cache=cache, cached_len=0)
    cached_ids = p1 + list(prod1[:-1])

    # 构造第二轮完整输入:首轮全部 token(p1 + prod1)后再接一段「新用户消息」
    tail = [3, 8, 4]
    new_ids = p1 + list(prod1) + tail
    assert new_ids[:len(cached_ids)] == cached_ids  # cache 是新序列的严格前缀

    # 复用路径:只 prefill new_ids[len(cached_ids):]
    prod2_reuse, _ = _gen(model, new_ids, 6,
                          main_cache=cache, cached_len=len(cached_ids))

    # 全量路径:全新 cache,整段 prefill new_ids
    prod2_fresh, _ = _gen(model, new_ids, 6)

    assert prod2_reuse == prod2_fresh


def test_reuse_matches_fresh_on_mixed_kv_and_recurrent_cache():
    """混合 cache(全注意力 KVCache + 递归 ArraysCache)下,复用后缀 prefill 仍与全量一致。

    _ToyModel 默认全 FA 层;这里手动把一层换成递归层,确保「递归态延续(不回退)」也被覆盖。
    """
    mx.random.seed(0)
    import mlx.nn as nn
    from mlx_streaming.tests.test_mtp_generate import _FAOnlyLayer, _RecurLayer
    model = _ToyModel(vocab=40, nl=2)
    # 第 2 层换成递归层,make_cache 会给它 ArraysCache
    model.model.layers[1] = _RecurLayer(32)
    mx.eval(model.parameters())

    p1 = [1, 5, 9, 2, 7]
    cache = model.make_cache()
    prod1, _ = _gen(model, p1, 5, main_cache=cache, cached_len=0)
    cached_ids = p1 + list(prod1[:-1])
    new_ids = p1 + list(prod1) + [3, 8, 4]

    prod2_reuse, _ = _gen(model, new_ids, 5,
                          main_cache=cache, cached_len=len(cached_ids))
    prod2_fresh, _ = _gen(model, new_ids, 5)
    assert prod2_reuse == prod2_fresh


def test_stats_resident_tokens_equals_cache_offset_and_detects_overcommit():
    """stats['resident_tokens'] 必须等于 cache 真实 offset(供后端精确对账);
    且在「每步满 K、末步跨 max_tokens」时确实发生 over-commit(resident > 不变式预期),
    这正是后端据此禁用复用、避免错算的场景。"""
    model = _kv_toy_k3()
    p1 = [1, 5, 9]
    cache = model.make_cache()
    produced, stats = mtp_generate(model, _AllAcceptDraft(model), None, mx.array([p1]), 5,
                                   K=3, ids_mode=True, main_cache=cache, cached_len=0)
    assert stats["resident_tokens"] == cache[0].offset
    # over-commit:cache 驻留数超过「len(ids)+len(produced)-1」这一无截断不变式
    assert stats["resident_tokens"] > len(p1) + len(produced) - 1
