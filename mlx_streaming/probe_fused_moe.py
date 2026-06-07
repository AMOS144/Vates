"""Phase 0 spike：fused gate+up+SwiGLU Metal kernel vs mlx gather_qmm。

只算 gate/up 投影 + SwiGLU 激活(不含 down)。先验数值 allclose，再比 ms。
GO 条件：fused ≥ 1.5× 于 mlx 同段(decode T=1,k=10 与 verify T=3,k=10)。

2-bit affine, group_size=64, hidden(IN)=2048, moe_inter(OUT)=512。
weight (cap,512,128)uint32; scales/biases (cap,512,32)bf16(本 spike 转 float32 喂 kernel)。
"""
import glob
import os
import time

import mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.models.activations import swiglu

EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts_2bit")
CAP = int(os.environ.get("CAP", "16"))
K = int(os.environ.get("K", "10"))
IN, OUT, PIN, NG, GS = 2048, 512, 128, 32, 64

_SRC = f"""
    const uint OUT = {OUT}u, IN = {IN}u, PIN = {PIN}u, NG = {NG}u, GS = {GS}u, K = {K}u;
    uint gid = thread_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tgsize = threads_per_threadgroup.x;
    // 同一 threadgroup 内所有线程共享同一 (t,k)（每 (t,k) 512 输出，tgsize|512）→ 协作把 x[t] 载入共享内存
    uint tk = gid / OUT;
    uint t  = tk / K;
    threadgroup float xs[{IN}];
    for (uint i = lid; i < IN; i += tgsize) xs[i] = x[t * IN + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = T_TIMES_K * OUT;
    if (gid >= total) return;
    uint o  = gid % OUT;
    uint j  = tk % K;
    int slot = slots[t * K + j];
    uint wbase = ((uint)slot * OUT + o) * PIN;
    uint sbase = ((uint)slot * OUT + o) * NG;
    float accg = 0.0f, accu = 0.0f;
    for (uint p = 0; p < PIN; p++) {{
        uint wg = gate_w[wbase + p];
        uint wu = up_w[wbase + p];
        uint g = (p * 16u) / GS;
        float gsv = gate_s[sbase + g], gbv = gate_b[sbase + g];
        float usv = up_s[sbase + g], ubv = up_b[sbase + g];
        for (uint e = 0; e < 16u; e++) {{
            float xv = xs[p * 16u + e];
            uint sh = e * 2u;
            accg += xv * (float((wg >> sh) & 0x3u) * gsv + gbv);
            accu += xv * (float((wu >> sh) & 0x3u) * usv + ubv);
        }}
    }}
    float sg = accg / (1.0f + exp(-accg));
    out[(t * K + j) * OUT + o] = sg * accu;
"""


def _load_pool(cap):
    files = sorted(glob.glob(os.path.join(EXPERT_DIR, "layer00_expert*.safetensors")))[:cap]
    ws = [mx.load(f) for f in files]
    pool = {k: mx.stack([w[k] for w in ws]) for k in ws[0].keys()}
    mx.eval(list(pool.values()))
    return pool


def _ref_gate_up_swiglu(pool, x, idx):
    g = QuantizedSwitchLinear(IN, OUT, CAP, bias=False, group_size=GS, bits=2, mode="affine")
    u = QuantizedSwitchLinear(IN, OUT, CAP, bias=False, group_size=GS, bits=2, mode="affine")
    g.update({"weight": pool["gate_proj.weight"], "scales": pool["gate_proj.scales"], "biases": pool["gate_proj.biases"]})
    u.update({"weight": pool["up_proj.weight"], "scales": pool["up_proj.scales"], "biases": pool["up_proj.biases"]})
    xx = mx.expand_dims(x, (-2, -3))            # (1,T,1,1,IN)
    xg = g(xx, idx)                             # (1,T,1,K,OUT)
    xu = u(xx, idx)
    return swiglu(xg, xu).squeeze(-2)           # silu(gate)*up → (1,T,K,OUT)


def _make_kernel(T):
    src = _SRC.replace("T_TIMES_K", str(T * K))
    return mx.fast.metal_kernel(
        name=f"fused_gus_{T}",
        input_names=["x", "slots", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"],
        output_names=["out"],
        source=src, header="")


def _fused(kern, inputs, T):
    (out,) = kern(inputs=inputs, grid=(T * K * OUT, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(T * K * OUT,)], output_dtypes=[mx.float32])
    return out.reshape(1, T, K, OUT)


def _bench(fn, n=200):
    for _ in range(20):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    return (time.perf_counter() - t) / n * 1000


def main():
    mx.random.seed(0)
    pool = _load_pool(CAP)
    import json
    results = {}
    for T in (1, 3):
        x = mx.random.normal((1, T, IN)).astype(mx.float32)
        idx = mx.broadcast_to(mx.arange(K, dtype=mx.uint32)[None, None], (1, T, K))
        x_flat = x.reshape(T, IN)
        slots_flat = idx.reshape(T, K).astype(mx.int32)
        mx.eval(x, idx, x_flat, slots_flat)

        # 预转换 scales/biases + 预建 kernel(提到计时外，模拟真实路径常驻池已是 float32)
        g_s = pool["gate_proj.scales"].astype(mx.float32)
        g_b = pool["gate_proj.biases"].astype(mx.float32)
        u_s = pool["up_proj.scales"].astype(mx.float32)
        u_b = pool["up_proj.biases"].astype(mx.float32)
        inputs = [x_flat, slots_flat, pool["gate_proj.weight"], g_s, g_b,
                  pool["up_proj.weight"], u_s, u_b]
        mx.eval(inputs)
        kern = _make_kernel(T)

        ref = _ref_gate_up_swiglu(pool, x, idx); mx.eval(ref)
        got = _fused(kern, inputs, T); mx.eval(got)
        diff = float(mx.abs(ref - got).max())
        rel = float(mx.abs(ref - got).max() / (mx.abs(ref).max() + 1e-6))

        t_ref = _bench(lambda: _ref_gate_up_swiglu(pool, x, idx))
        t_fused = _bench(lambda: _fused(kern, inputs, T))
        results[f"T={T}"] = {
            "max_abs_diff": round(diff, 4), "max_rel_diff": round(rel, 4),
            "allclose_1e-2": diff < 1e-2 or rel < 2e-2,
            "ms_mlx": round(t_ref, 4), "ms_fused": round(t_fused, 4),
            "speedup": round(t_ref / max(t_fused, 1e-9), 2),
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
