"""de-risk：MLX 能否把后台线程的专家加载(mx.load+eval)与主线程 GPU 计算重叠。

动机：流式 MoE 的 miss 加载当前是同步阻塞的。若后台线程的 load+eval 能和主线程
matmul 重叠，则异步预取/重叠有意义；若不能（GIL 或单队列串行），则 b 的异步路无效。

测法：
  baseline_sum = 顺序(主线程 load N 个专家 + 主线程跑 M 次 matmul)
  overlapped   = 后台线程 load N 个专家 同时 主线程跑 M 次 matmul
若 overlapped 明显 < baseline_sum（趋近 max(load, compute)）→ 可重叠。
"""
import os
import time
import threading

import mlx.core as mx

ROOT = os.environ.get("EXPERT_DIR", "/tmp/qwen3_next_experts_2bit")
LAYER = int(os.environ.get("LAYER", "0"))
N = int(os.environ.get("N_LOAD", "16"))          # 模拟 miss 加载的专家数
M = int(os.environ.get("N_MATMUL", "40"))        # 主线程 GPU 计算量
DIM = int(os.environ.get("DIM", "2048"))
REPEAT = int(os.environ.get("REPEAT", "5"))


def _expert_path(e: int) -> str:
    return os.path.join(ROOT, f"layer{LAYER:02d}_expert{e:03d}.safetensors")


def load_experts(ids):
    """模拟 miss：逐个 mx.load + eval（与 _load_one 一致）。"""
    outs = []
    for e in ids:
        w = mx.load(_expert_path(e))
        mx.eval(w)
        outs.append(w)
    return outs


def compute(a, b, m):
    """主线程 GPU 计算负载：m 次矩阵乘 + eval。"""
    x = a
    for _ in range(m):
        x = mx.matmul(x, b)
        x = x - x.mean()
    mx.eval(x)
    return x


def main():
    ids = list(range(N))
    a = mx.random.normal((DIM, DIM))
    b = mx.random.normal((DIM, DIM))
    mx.eval(a, b)

    # warm：让文件进页缓存 + 编译 matmul
    load_experts(ids)
    compute(a, b, M)

    # 1) 单独测 load 与 compute
    load_ts, comp_ts = [], []
    for _ in range(REPEAT):
        t = time.perf_counter(); load_experts(ids); load_ts.append(time.perf_counter() - t)
        t = time.perf_counter(); compute(a, b, M); comp_ts.append(time.perf_counter() - t)
    t_load = min(load_ts) * 1000
    t_comp = min(comp_ts) * 1000

    # 2) 顺序基线
    seq_ts = []
    for _ in range(REPEAT):
        t = time.perf_counter()
        load_experts(ids)
        compute(a, b, M)
        seq_ts.append(time.perf_counter() - t)
    t_seq = min(seq_ts) * 1000

    # 3) 重叠：后台线程 load，主线程 compute
    ov_ts = []
    for _ in range(REPEAT):
        t = time.perf_counter()
        holder = {}
        th = threading.Thread(target=lambda: holder.setdefault("r", load_experts(ids)))
        th.start()
        compute(a, b, M)
        th.join()
        ov_ts.append(time.perf_counter() - t)
    t_ov = min(ov_ts) * 1000

    import json
    print(json.dumps({
        "N_load": N, "N_matmul": M, "dim": DIM,
        "t_load_ms": round(t_load, 1),
        "t_compute_ms": round(t_comp, 1),
        "t_sequential_ms": round(t_seq, 1),
        "t_overlapped_ms": round(t_ov, 1),
        "ideal_max_ms": round(max(t_load, t_comp), 1),
        "overlap_saving_ms": round(t_seq - t_ov, 1),
        "overlap_efficiency": round((t_seq - t_ov) / max(1e-6, min(t_load, t_comp)), 3),
        "note": "efficiency≈1 → 完全重叠(可做异步);≈0 → 不重叠(GIL/单队列串行,b异步无效)",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
