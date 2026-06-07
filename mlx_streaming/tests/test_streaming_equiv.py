import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from mlx_streaming.streaming_moe import streaming_switch_glu_forward


def test_streaming_matches_switchglu():
    mx.random.seed(0)
    dim, hidden, E, k = 16, 32, 8, 2
    glu = SwitchGLU(dim, hidden, E)
    mx.eval(glu.parameters())
    x = mx.random.normal((1, 5, dim))
    inds = mx.argpartition(mx.random.normal((1, 5, E)), kth=-k, axis=-1)[..., -k:]

    ref = glu(x, inds)
    got = streaming_switch_glu_forward(glu, x, inds)   # 只算选中专家，结果须等价
    assert mx.allclose(ref, got, atol=1e-4).item()


def test_streaming_matches_quantized_switchglu():
    # 真实 Qwen3 MoE 用的是量化专家（QuantizedSwitchLinear），这里覆盖量化路径
    mx.random.seed(0)
    dim, hidden, E, k = 64, 128, 8, 2
    glu = SwitchGLU(dim, hidden, E)
    mx.eval(glu.parameters())
    nn.quantize(glu, group_size=64, bits=4)            # 原地量化三组 SwitchLinear
    mx.eval(glu.parameters())

    x = mx.random.normal((1, 5, dim))
    inds = mx.argpartition(mx.random.normal((1, 5, E)), kth=-k, axis=-1)[..., -k:]

    ref = glu(x, inds)
    got = streaming_switch_glu_forward(glu, x, inds)
    assert mx.allclose(ref, got, atol=1e-4).item()
