import os, pytest, mlx.core as mx

pytestmark = pytest.mark.skipif(
    not os.path.exists("models/qn_mtp_weights.safetensors"),
    reason="需要 Qwen MTP 权重")

GOLDEN_PROMPT = "用三句话解释什么是混合专家模型。"

def _run_qwen_mtp(max_tokens=32, K=3):
    from mlx_streaming.runtime.run_mtp_spec import build_qwen_spec  # Task 1.5 暴露
    from mlx_streaming.mtp.generate import mtp_generate
    spec_model, drafter, tok = build_qwen_spec()
    ids = mx.array([tok.encode(GOLDEN_PROMPT)])
    produced, _ = mtp_generate(spec_model, drafter, tok, ids,
                               max_tokens, K=K, ids_mode=True)
    return produced

def test_qwen_mtp_matches_greedy_baseline():
    from mlx_streaming.runtime.run_mtp_spec import build_qwen_spec
    spec_model, drafter, tok = build_qwen_spec()
    ids = mx.array([tok.encode(GOLDEN_PROMPT)])
    cache = spec_model.make_cache()
    logits, _ = spec_model.forward_with_hidden(ids, cache)
    ref = [int(mx.argmax(logits[:, -1, :]))]
    cur = mx.array([[ref[0]]])
    for _ in range(31):
        logits, _ = spec_model.forward_with_hidden(cur, cache)
        ref.append(int(mx.argmax(logits[:, -1, :])))
        cur = mx.array([[ref[-1]]])
    spec = _run_qwen_mtp(max_tokens=32)
    assert spec[:32] == ref[:32]
