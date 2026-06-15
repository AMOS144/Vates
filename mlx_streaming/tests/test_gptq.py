"""验证 GPTQ 核心:在相关性激活下,GPTQ 的输出误差应低于 RTN(就近取整)。"""
import numpy as np

from mlx_streaming.prep.gptq import gptq_quantize, rtn_quantize


def _output_err(W, Wq, X):
    """||X Wᵀ - X Wqᵀ|| / ||X Wᵀ|| —— 量化对层输出的相对偏差。"""
    Y = X @ W.T
    Yq = X @ Wq.T
    return float(np.linalg.norm(Y - Yq) / (np.linalg.norm(Y) + 1e-9))


def test_gptq_beats_rtn_on_correlated_activations():
    rng = np.random.default_rng(0)
    out_dim, in_dim, T = 64, 128, 512
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float32)

    # 相关性激活:GPTQ 靠 Hessian 的非对角结构补偿误差,iid 时退化为 RTN,故需相关性
    A = rng.standard_normal((in_dim, in_dim))
    cov = A @ A.T / in_dim
    L = np.linalg.cholesky(cov + 1e-3 * np.eye(in_dim))
    X = (rng.standard_normal((T, in_dim)) @ L.T).astype(np.float32)
    H = (X.T @ X).astype(np.float32)

    bits, gs = 2, 32
    Wq_rtn = rtn_quantize(W, bits=bits, group_size=gs)
    Wq_gptq = gptq_quantize(W, H, bits=bits, group_size=gs)

    e_rtn = _output_err(W, Wq_rtn, X)
    e_gptq = _output_err(W, Wq_gptq, X)
    print(f"输出相对误差: RTN={e_rtn:.4f}  GPTQ={e_gptq:.4f}  改善={1 - e_gptq / e_rtn:.1%}")

    assert e_gptq < e_rtn, f"GPTQ({e_gptq}) 未优于 RTN({e_rtn})"


def test_gptq_output_is_mlx_affine_compatible_and_keeps_gain():
    """GPTQ 产出 → mx.quantize 打包 → mx.dequantize 还原:round-trip 误差应极小,
    且打包后的权重仍比 RTN 低输出误差(证明 from-FP16 的 GPTQ 产物能直接跑 MLX gather_qmm)。"""
    import mlx.core as mx

    rng = np.random.default_rng(1)
    out_dim, in_dim, T = 64, 128, 512
    W = rng.standard_normal((out_dim, in_dim)).astype(np.float32)
    A = rng.standard_normal((in_dim, in_dim))
    L = np.linalg.cholesky(A @ A.T / in_dim + 1e-3 * np.eye(in_dim))
    X = (rng.standard_normal((T, in_dim)) @ L.T).astype(np.float32)
    H = (X.T @ X).astype(np.float32)
    bits, gs = 2, 64

    W_gptq = gptq_quantize(W, H, bits=bits, group_size=gs)

    # 打包成 MLX affine 格式再还原(模型 runtime 实际用的就是这个还原值)
    wq, scales, biases = mx.quantize(mx.array(W_gptq), group_size=gs, bits=bits)
    W_packed = np.array(mx.dequantize(wq, scales, biases, group_size=gs, bits=bits))

    roundtrip = float(np.linalg.norm(W_packed - W_gptq) / (np.linalg.norm(W_gptq) + 1e-9))
    e_rtn = _output_err(W, rtn_quantize(W, bits=bits, group_size=gs), X)
    e_packed = _output_err(W, W_packed, X)
    print(f"round-trip 误差={roundtrip:.4f}  RTN={e_rtn:.4f}  GPTQ(打包后)={e_packed:.4f}")

    assert roundtrip < 0.05, f"打包 round-trip 误差过大 {roundtrip}"
    assert e_packed < e_rtn, f"打包后 GPTQ({e_packed}) 未优于 RTN({e_rtn})"


if __name__ == "__main__":
    test_gptq_beats_rtn_on_correlated_activations()
    test_gptq_output_is_mlx_affine_compatible_and_keeps_gain()
    print("OK")
