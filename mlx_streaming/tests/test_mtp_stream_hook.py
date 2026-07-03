"""mtp_generate 的 on_tokens 钩子:流式回调 + 中断,且默认行为不变。"""
import mlx.core as mx
from mlx_lm.models import cache as kvcache

from mlx_streaming.mtp.generate import forward_with_hidden, mtp_generate
from mlx_streaming.tests.test_mtp_generate import _RandDraft, _SelfDraft, _ToyModel


class _AllAcceptDraft:
    """测试用草稿器:贪婪跑主模型自身产出后续 K 个 token 作为草稿。

    因草稿恰为主模型贪婪续写,验证时全部命中(matched==K),故每步发满 K 个 token。
    专门用来构造「多 token 步跨越 max_tokens 上限」的截断场景,复现 on_tokens 超报缺陷。
    接口对齐 _SelfDraft.draft(H_last, x_ids, mtp_cache, K) -> list[int]。
    """

    def __init__(self, model):
        self.model = model

    def draft(self, H_last, x_ids, mtp_cache, K):
        cache = self.model.make_cache()
        ids = x_ids
        out = []
        for _ in range(K):
            logits, _ = forward_with_hidden(self.model, ids, cache)
            nxt = int(mx.argmax(logits[:, -1, :]))
            out.append(nxt)
            ids = mx.array([[nxt]])
            mx.eval(ids)
        return out


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


def test_on_tokens_k3_no_overreport_when_step_crosses_cap():
    """草稿全接受(每步发满 K)且 max_tokens 让末步跨越上限:回调不得超报被截断的尾 token。

    这是真正能在修复前变红的回归:pre-fix 把整段 new_tokens(长度 K)交给回调,末步跨
    max_tokens 时 seen 会比截断后的 produced 多出被丢弃的尾 token(seen != produced)。
    """
    model = _kv_toy_k3()
    prompt = mx.array([[1, 5, 9]])
    seen = []

    def on_tokens(new_ids):
        seen.extend(new_ids)
        return False

    # max_tokens=10:在该 seed 下末步为多 token 步且会跨越上限,触发截断路径。
    produced, _ = mtp_generate(model, _AllAcceptDraft(model), None, prompt, 10,
                               K=3, ids_mode=True, on_tokens=on_tokens)
    assert seen == produced
    assert len(produced) == 10
