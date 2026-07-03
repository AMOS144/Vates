"""mtp_generate 的 on_tokens 钩子:流式回调 + 中断,且默认行为不变。"""
import mlx.core as mx
from mlx_lm.models import cache as kvcache

from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.tests.test_mtp_generate import _RandDraft, _SelfDraft, _ToyModel


def _kv_toy():
    mx.random.seed(0)
    model = _ToyModel(nl=2)
    model.make_cache = lambda: [kvcache.KVCache() for _ in model.layers]
    mx.eval(model.parameters())
    return model


def _kv_toy_k3():
    mx.random.seed(0)
    model = _ToyModel(nl=2, vocab=40)
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


def test_on_tokens_k3_receives_all_produced_no_overreport():
    """K=3 多 token 步:回调拼接必须恰好等于最终 produced(不超报被截断的尾 token)。"""
    model = _kv_toy_k3()
    prompt = mx.array([[1, 5, 9]])
    seen = []

    def on_tokens(new_ids):
        seen.extend(new_ids)
        return False

    produced, _ = mtp_generate(model, _RandDraft(40), None, prompt, 16,
                               K=3, ids_mode=True, on_tokens=on_tokens)
    assert seen == produced
    assert len(produced) == 16


def test_on_tokens_stop_in_loop_k3():
    """K=3:首 token 不停,循环内第一步返回 True,验证 in-loop break 路径。"""
    model = _kv_toy_k3()
    prompt = mx.array([[1, 5, 9]])
    calls = {"n": 0}

    def on_tokens(new_ids):
        calls["n"] += 1
        return calls["n"] >= 2  # 首 token(第1次)不停,循环内第一步(第2次)请求停止

    produced, _ = mtp_generate(model, _RandDraft(40), None, prompt, 100,
                               K=3, ids_mode=True, on_tokens=on_tokens)
    assert calls["n"] == 2
    assert len(produced) < 100
