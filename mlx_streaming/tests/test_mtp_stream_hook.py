"""mtp_generate 的 on_tokens 钩子:流式回调 + 中断,且默认行为不变。"""
import mlx.core as mx
from mlx_lm.models import cache as kvcache

from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.tests.test_mtp_generate import _SelfDraft, _ToyModel


def _kv_toy():
    mx.random.seed(0)
    model = _ToyModel(nl=2)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    return model


def test_on_tokens_receives_all_produced_in_order():
    """所有回调收到的 token 拼起来应等于最终 produced。"""
    model = _kv_toy()
    prompt = mx.array([[1, 5, 9]])
    seen = []

    def on_tokens(new_ids):
        seen.extend(new_ids)
        return False

    produced, _ = mtp_generate(
        model,
        _SelfDraft(model),
        None,
        prompt,
        12,
        K=1,
        ids_mode=True,
        on_tokens=on_tokens,
    )
    assert seen == produced
    assert len(produced) == 12


def test_on_tokens_true_requests_stop():
    """回调返回 True 后应尽快停止,产出远少于 max_tokens。"""
    model = _kv_toy()
    prompt = mx.array([[1, 5, 9]])
    calls = {"n": 0}

    def on_tokens(new_ids):
        calls["n"] += 1
        return True

    produced, _ = mtp_generate(
        model,
        _SelfDraft(model),
        None,
        prompt,
        100,
        K=1,
        ids_mode=True,
        on_tokens=on_tokens,
    )
    assert calls["n"] == 1
    assert len(produced) < 100


def test_default_none_unchanged():
    """不传 on_tokens 时行为与之前一致(能正常产出 max_tokens 个)。"""
    model = _kv_toy()
    prompt = mx.array([[1, 5, 9]])
    produced, _ = mtp_generate(model, _SelfDraft(model), None, prompt, 12, K=1, ids_mode=True)
    assert len(produced) == 12
