import mlx.core as mx

from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.extract_mtp import stack_mtp_experts, bump_mtp_norms
from mlx_streaming.qwen3_next_mtp import Qwen3NextMTP
from mlx_streaming.validate_mtp import capture_prenorm_hidden


def _tiny_args():
    return ModelArgs(
        model_type="qwen3_next",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        shared_expert_intermediate_size=64,
        mlp_only_layers=[],
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=50,
        num_key_value_heads=2,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        max_position_embeddings=128,
        head_dim=8,
        full_attention_interval=4,
    )


def test_stack_mtp_experts_shapes_and_order():
    # 2 个专家、小维度的伪权重
    w = {
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((4, 3)),
        "mtp.layers.0.mlp.experts.1.gate_proj.weight": mx.ones((4, 3)),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((4, 3)),
        "mtp.layers.0.mlp.experts.1.up_proj.weight": mx.ones((4, 3)),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((3, 4)),
        "mtp.layers.0.mlp.experts.1.down_proj.weight": mx.ones((3, 4)),
        "mtp.fc.weight": mx.zeros((3, 6)),
    }
    out = stack_mtp_experts(w, num_experts=2)
    # 专家被堆叠且原 per-expert 键被移除
    g = out["mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]
    assert g.shape == (2, 4, 3)
    # 升序:expert 0 全 0、expert 1 全 1
    assert float(g[0].sum()) == 0.0 and float(g[1].sum()) == 12.0
    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" not in out
    # 非专家键原样保留
    assert "mtp.fc.weight" in out


def test_bump_mtp_norms_adds_one_only_to_norms():
    w = {
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((4,)),
        "mtp.pre_fc_norm_embedding.weight": mx.zeros((4,)),
        "mtp.norm.weight": mx.zeros((4,)),
        "mtp.layers.0.input_layernorm.weight": mx.zeros((4,)),
        "mtp.layers.0.post_attention_layernorm.weight": mx.zeros((4,)),
        "mtp.layers.0.self_attn.q_norm.weight": mx.zeros((2,)),
        "mtp.layers.0.self_attn.k_norm.weight": mx.zeros((2,)),
        "mtp.fc.weight": mx.zeros((3, 6)),  # 非 norm,不应被改
    }
    out = bump_mtp_norms(w)
    assert float(out["mtp.norm.weight"][0]) == 1.0
    assert float(out["mtp.pre_fc_norm_hidden.weight"][0]) == 1.0
    assert float(out["mtp.layers.0.self_attn.q_norm.weight"][0]) == 1.0
    assert float(out["mtp.fc.weight"].sum()) == 0.0  # 未被 +1


def test_mtp_forward_shape_and_causal():
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    mx.eval(mtp.parameters())
    head_w = mx.random.normal((args.hidden_size, args.vocab_size))
    lm_head = lambda h: h @ head_w
    B, L = 1, 6
    hidden = mx.random.normal((B, L, args.hidden_size))
    next_ids = mx.array([[3, 7, 1, 9, 2, 5]])
    logits = mtp(hidden, next_ids, lm_head)
    assert logits.shape == (B, L, args.vocab_size)
    # 因果:改最后一位输入,前面 logits 不变
    hidden2 = mx.concatenate(
        [hidden[:, :-1, :], hidden[:, -1:, :] + 10.0], axis=1
    )
    logits2 = mtp(hidden2, next_ids, lm_head)
    assert float(mx.abs(logits[:, :-1] - logits2[:, :-1]).max()) < 1e-3


def test_mtp_fc_concat_order_is_emb_then_hidden():
    # 验证 concat 顺序为 [emb, hidden]:关掉 emb 通路、令 fc=[0 | I],
    # 则 fc 输入 layer 前的张量应等于 rms_norm(hidden)。
    mx.random.seed(0)
    args = _tiny_args()
    mtp = Qwen3NextMTP(args)
    h = args.hidden_size
    mtp.pre_fc_norm_embedding.weight = mx.zeros((h,))  # emb -> 0
    mtp.pre_fc_norm_hidden.weight = mx.ones((h,))      # hidden 通过
    mtp.fc.weight = mx.concatenate([mx.zeros((h, h)), mx.eye(h)], axis=1)  # (h, 2h)
    mx.eval(mtp.parameters())

    captured = {}
    orig_layer = mtp.layer

    def spy(x, mask=None, cache=None):
        captured["x"] = x
        return x

    mtp.layer = spy
    hidden = mx.random.normal((1, 3, h))
    next_ids = mx.array([[1, 2, 3]])
    mtp(hidden, next_ids, lambda z: z)
    mtp.layer = orig_layer

    expected = mx.fast.rms_norm(hidden, mx.ones((h,)), args.rms_norm_eps)
    # 顺序若反(=[hid, emb])会让 emb 通路=0、误差约 1.0;1e-3 量级仅为 fc 矩阵乘浮点噪声
    assert float(mx.abs(captured["x"] - expected).max()) < 5e-3


class _FakeNorm:
    def __call__(self, x):
        return x * 0.0  # norm 后必为 0,便于区分


class _FakeInner:
    """模拟 Qwen3NextModel:embed -> layers -> norm。"""

    def __init__(self):
        self.norm = _FakeNorm()

    def embed_tokens(self, ids):
        return mx.ones((ids.shape[0], ids.shape[1], 4))

    @property
    def layers(self):
        return []

    def make_cache(self):
        return []


class _FakeModel:
    def __init__(self):
        self.model = _FakeInner()

    def make_cache(self):
        return []


def test_capture_prenorm_hidden_is_before_norm():
    m = _FakeModel()
    ids = mx.array([[1, 2, 3]])
    pre = capture_prenorm_hidden(m, ids)
    # norm 会把它清零;捕获的是 norm 之前 => 全 1
    assert float(pre.sum()) == 12.0  # 1*3*4
