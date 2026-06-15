"""无 Hessian 的 scale-only 量化(逐组最优裁剪)。

思路:RTN 用 min/max 定 scale,对高斯型权重把尾巴也包进去 → scale 偏大、低 bit 误差大。
本方法逐组逐行搜一个裁剪系数 c,把区间裁到 [mid-c*half, mid+c*half],最小化组内 MSE,
得到更优的 scale/bias。不算 Hessian、不动权重、无校准、无秩亏问题;输出标准仿射 (q,scale,bias),
自打包成 MLX uint32 布局,mx.dequantize 可无损还原(已验证 round-trip < 1e-6)。

⚠️ 实测负面结论(2026-06-09,Qwen3-Next-80B,2-bit g128):
  本方法把权重重构误差降了 18.6%(0.443→0.361),但 PPL 反而从 14.67 升到 16.66。
  原因:纯权重-MSE 是错的目标——最优裁剪会裁掉分布尾巴(salient 大权重),而这些权重
  恰是对输出最重要的(正是 Hessian/激活感知要保护的)。降平均权重误差 ≠ 降输出误差。
  → 若要救:目标须换成激活感知 ‖WX-ŴX‖(AWQ 式 clip,仍只调 scale、无 Hessian 秩亏)。
  此处保留打包逻辑供该激活感知版复用;纯 MSE 版请勿用于生产。
"""
import numpy as np
import mlx.core as mx

# 裁剪系数网格:1.0=不裁(等价裸 min/max),越小裁得越狠
_CLIP_GRID = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6], dtype=np.float64)


def clip_search_quantize(W, group_size=128, bits=2):
    """对 W(out, in) 逐组(沿 in)逐行搜最优裁剪,返回 (packed_q, scales, biases) MLX 数组。

    与 mx.quantize 同接口:packed_q 为打包后的 uint32,scales/biases 形状 (out, in//group_size)。
    """
    if isinstance(W, mx.array):                               # bf16 不能直转 numpy,先 float32
        Wn = np.array(W.astype(mx.float32)).astype(np.float64)
    else:
        Wn = np.asarray(W, dtype=np.float64)
    out_dim, in_dim = Wn.shape
    qmax = (1 << bits) - 1
    n_groups = in_dim // group_size
    g = Wn.reshape(out_dim, n_groups, group_size)             # (out, G, gs)
    mn = g.min(axis=2, keepdims=True)
    mx_ = g.max(axis=2, keepdims=True)
    mid = (mn + mx_) / 2.0
    half0 = (mx_ - mn) / 2.0

    best_q = None
    best_lo = None
    best_sc = None
    best_e = None
    for c in _CLIP_GRID:
        half = half0 * c
        lo = mid - half
        sc = (2.0 * half) / qmax
        sc = np.where(sc == 0, 1.0, sc)
        gc = np.clip(g, lo, mid + half)
        q = np.clip(np.round((gc - lo) / sc), 0, qmax)
        deq = q * sc + lo
        e = ((g - deq) ** 2).sum(axis=2, keepdims=True)        # (out, G, 1) 组内 MSE
        if best_e is None:
            best_e = e + 1e30
            best_q = q.copy(); best_lo = np.broadcast_to(lo, g.shape).copy(); best_sc = np.broadcast_to(sc, g.shape).copy()
        m = e < best_e
        best_q = np.where(m, q, best_q)
        best_lo = np.where(m, lo, best_lo)
        best_sc = np.where(m, sc, best_sc)
        best_e = np.where(m, e, best_e)

    q_int = best_q.reshape(out_dim, in_dim).astype(np.uint32)
    scales = best_sc[:, :, 0].reshape(out_dim, n_groups).astype(np.float32)
    biases = best_lo[:, :, 0].reshape(out_dim, n_groups).astype(np.float32)

    # 打包:每 uint32 装 per=32//bits 个值(小端,低位在前),沿 in 维
    per = 32 // bits
    qf = q_int.reshape(out_dim, in_dim // per, per)
    packed = np.zeros((out_dim, in_dim // per), dtype=np.uint32)
    for j in range(per):
        packed |= (qf[:, :, j] << (bits * j))
    return mx.array(packed), mx.array(scales), mx.array(biases)
