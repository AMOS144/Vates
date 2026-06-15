"""GPTQ 核心(离线 PTQ):用校准激活的二阶信息(Hessian)逐列量化 + 误差补偿,
最小化输出偏差 ||WX - ŴX||,优于就近取整(RTN)。输出仍是仿射量化(weight/scales/biases),
可直接跑在 MLX 现有 gather_qmm 上,无需自定义 Metal kernel。

参考:Frantar et al., GPTQ(2022);OBQ 的逐列误差补偿。numpy 实现(离线,不求快)。
"""
import numpy as np
import mlx.core as mx

# MLX affine 精确约定(已探针验证):deq = q*scale + bias,q=clip(round((w-bias)/scale),0,2^b-1),
# 其中 scale/bias 由 mx.quantize 内部算法给出(scale ≠ (max-min)/qmax),故直接调 mx.quantize 取。


def _mx_group_scale_bias(wg, bits, group_size):
    """(out, group_size) 的一个 group:用 mx.quantize 取 MLX 的 scale/bias(逐输出行,长 out)。"""
    _, s, b = mx.quantize(mx.array(wg.astype(np.float32)), group_size=group_size, bits=bits)
    return np.array(s).reshape(-1).astype(np.float64), np.array(b).reshape(-1).astype(np.float64)


def _quant_col(w, scale, bias, qmax):
    """逐输出行用该 group 的 scale/bias 量化单列(输入维),返回反量化值(落在 MLX 网格上)。"""
    q = np.clip(np.round((w - bias) / scale), 0, qmax)
    return q * scale + bias


def gptq_quantize(W, H, bits=2, group_size=128, percdamp=0.01):
    """对单个权重矩阵 W(out, in)做 GPTQ。H(in, in)=校准激活的 Hessian(X^T X)。
    返回 W_hat(out, in) 反量化后的浮点权重(其元素落在 affine 网格上,可再用 mx.quantize 打包)。

    逐列:量化第 i 列 → 用 Hinv 把该列的量化误差补偿到尚未量化的右侧列,group 内共享 scale。
    """
    W = W.astype(np.float64).copy()
    out_dim, in_dim = W.shape
    H = H.astype(np.float64).copy()

    # 死列(对角为 0)阻尼,避免 Cholesky 失败
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0

    damp = percdamp * np.mean(np.diag(H))
    H[np.diag_indices_from(H)] += damp

    # OBQ/GPTQ 所需的上三角因子:H^-1 = U^T U,U 上三角(Hinv[i,i]、Hinv[i,i+1:] 用于误差补偿)
    Hinv_full = np.linalg.inv(H)
    Hinv = np.linalg.cholesky(Hinv_full).T            # 上三角 U,满足 Hinv_full = U^T U

    # 提速:一次 mx.quantize 预算所有 group 的 MLX scale/bias(静态分组),消除逐 group GPU 同步
    _, S, Bz = mx.quantize(mx.array(W.astype(np.float32)), group_size=group_size, bits=bits)
    S = np.array(S).astype(np.float64)                # (out, n_groups)
    Bz = np.array(Bz).astype(np.float64)
    qmax = (1 << bits) - 1

    # 分块(block-wise)GPTQ:块内逐列补偿,块后一次性批量更新剩余列(标准提速,减少全宽更新次数)
    Q = np.zeros_like(W)
    B = 128
    for b0 in range(0, in_dim, B):
        b1 = min(b0 + B, in_dim)
        Wb = W[:, b0:b1].copy()
        Eb = np.zeros((W.shape[0], b1 - b0), dtype=np.float64)   # 块内每列误差
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            i = b0 + j
            g = i // group_size
            scale, bias = S[:, g], Bz[:, g]
            w_col = Wb[:, j]
            q = np.clip(np.round((w_col - bias) / scale), 0, qmax)
            deq = q * scale + bias
            Q[:, i] = deq
            err = (w_col - deq) / Hb[j, j]
            Eb[:, j] = err
            if j + 1 < b1 - b0:                                   # 仅更新块内剩余列
                Wb[:, j + 1:] -= np.outer(err, Hb[j, j + 1:])
        if b1 < in_dim:                                           # 块后:一次性更新块外所有列
            W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
    return Q.astype(np.float32)


def rtn_quantize(W, bits=2, group_size=128):
    """就近取整基线 = 直接 mx.quantize/dequantize(真正的 MLX RTN),用于对比。"""
    q, s, b = mx.quantize(mx.array(W.astype(np.float32)), group_size=group_size, bits=bits)
    return np.array(mx.dequantize(q, s, b, group_size=group_size, bits=bits)).astype(np.float32)
