"""合成专家张量探针：回答方案最核心的问题——

  「lazy 加载一个堆叠量化专家张量后，只对选中的 k 个专家做 gather_qmm，
   到底会只占用 k 个专家的内存，还是会把整张量都物化进内存？」

这决定走路线 A（便宜：lazy + 内存上限，靠按需换页）还是路线 B（自定义流式：
显式只加载选中专家）。用 mlx-lm 真实的 QuantizedSwitchLinear（Qwen3 MoE 同款）
构造一个约 1GB 的堆叠专家权重，存盘后用不同方式重载实测 RSS。

不依赖任何模型下载。

用法：
  python -m mlx_streaming.probe_gather_paging build         # 造数据存盘（含整体 + 每专家文件）
  python -m mlx_streaming.probe_gather_paging run <mode>    # mode: evalall | lazy | mmap | perexpert
  python -m mlx_streaming.probe_gather_paging drive         # 子进程逐个跑各 mode 并汇总
"""
import os
import sys
import json
import time
import glob
import subprocess

import mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from mlx_streaming.mem import snapshot, reset_peak

DIR = os.environ.get("PROBE_DIR", "/tmp/mlx_probe_experts")
E = int(os.environ.get("E", "256"))      # 专家数
O = int(os.environ.get("O", "4096"))     # 输出维
I = int(os.environ.get("I", "2048"))     # 输入维
K = int(os.environ.get("K", "8"))        # 单次激活专家数
GROUP = 64
BITS = 4
STACK_PATH = os.path.join(DIR, "stacked.safetensors")


def _quantized_params():
    """造一个 (E,O,I) 的随机权重并量化，返回 mlx-lm 风格的参数 dict。"""
    w = mx.random.uniform(low=-0.05, high=0.05, shape=(E, O, I))
    qw, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS, mode="affine")
    return {"weight": qw, "scales": scales, "biases": biases}


def build():
    os.makedirs(DIR, exist_ok=True)
    params = _quantized_params()
    mx.eval(params)
    mx.save_safetensors(STACK_PATH, params)
    # 每专家单独存一个小文件（路线 B 的 per-expert 后端）
    for e in range(E):
        sub = {k: v[e] for k, v in params.items()}
        mx.eval(sub)
        mx.save_safetensors(os.path.join(DIR, f"expert_{e:04d}.safetensors"), sub)
    sz = os.path.getsize(STACK_PATH)
    print(json.dumps({
        "built": True, "dir": DIR, "E": E, "O": O, "I": I, "bits": BITS,
        "stacked_gb": round(sz / 1e9, 3),
        "per_expert_files": E,
    }, ensure_ascii=False, indent=2))


def _make_linear():
    lin = QuantizedSwitchLinear(I, O, E, bias=False, group_size=GROUP, bits=BITS, mode="affine")
    return lin


def _gather_inputs():
    # x: (tokens, 1, I)，indices: (tokens, K) —— 模拟 K 个激活专家
    x = mx.random.normal((1, 1, I))
    inds = mx.array([list(range(K))], dtype=mx.uint32)  # 选前 K 个专家
    return x, inds


def run(mode: str):
    reset_peak()
    before_init = snapshot()

    if mode == "perexpert":
        # 路线 B：只从磁盘加载选中的 K 个专家小文件
        x, inds = _gather_inputs()
        picked = []
        for e in range(K):
            w = mx.load(os.path.join(DIR, f"expert_{e:04d}.safetensors"))
            picked.append(w)
        stacked = {k: mx.stack([p[k] for p in picked]) for k in picked[0].keys()}
        lin = QuantizedSwitchLinear(I, O, K, bias=False, group_size=GROUP, bits=BITS, mode="affine")
        lin.update(stacked)
        after_load = snapshot()
        local_inds = mx.array([list(range(K))], dtype=mx.uint32)
        y = lin(x, local_inds)
        mx.eval(y)
        after_gather = snapshot()
    else:
        # evalall / lazy / mmap：加载整堆叠张量
        load_kwargs = {}
        if mode == "mmap":
            load_kwargs["use_mmap"] = True
        try:
            weights = mx.load(STACK_PATH, **load_kwargs)
        except TypeError as ex:
            print(json.dumps({"mode": mode, "error": f"mx.load 不支持参数: {ex!r}"}, ensure_ascii=False))
            return
        if mode == "evalall":
            mx.eval(weights)  # 强制整张量物化 —— 全量驻留上界
        lin = _make_linear()
        lin.update(weights)
        after_load = snapshot()
        x, inds = _gather_inputs()
        y = lin(x, inds)
        mx.eval(y)
        after_gather = snapshot()

    stacked_gb = round(os.path.getsize(STACK_PATH) / 1e9, 3)
    out = {
        "mode": mode, "E": E, "K": K, "stacked_gb": stacked_gb,
        "rss_gb_before": round(before_init.rss_bytes / 1e9, 3),
        "rss_gb_after_load": round(after_load.rss_bytes / 1e9, 3),
        "rss_gb_after_gather": round(after_gather.rss_bytes / 1e9, 3),
        "mlx_active_gb_after_gather": round(after_gather.mlx_active_bytes / 1e9, 3),
        "mlx_peak_gb": round(after_gather.mlx_peak_bytes / 1e9, 3),
    }
    print(json.dumps(out, ensure_ascii=False))


def drive():
    if not os.path.exists(STACK_PATH):
        print("先 build...")
        subprocess.run([sys.executable, "-m", "mlx_streaming.probe_gather_paging", "build"], check=True)
    rows = []
    for mode in ["evalall", "lazy", "mmap", "perexpert"]:
        p = subprocess.run(
            [sys.executable, "-m", "mlx_streaming.probe_gather_paging", "run", mode],
            capture_output=True, text=True,
        )
        line = ""
        for ln in p.stdout.splitlines():
            if ln.strip().startswith("{"):
                line = ln.strip()
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"mode": mode, "raw": p.stdout[-300:], "err": p.stderr[-300:]})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    # 结论判定
    by = {r.get("mode"): r for r in rows if "rss_gb_after_gather" in r}
    if "evalall" in by and "lazy" in by:
        full = by["evalall"]["rss_gb_after_gather"]
        lazy = by["lazy"]["rss_gb_after_gather"]
        print(f"\n判定：evalall(全量)={full}GB, lazy={lazy}GB, "
              f"stacked={by['evalall']['stacked_gb']}GB")
        if lazy < full * 0.5:
            print("→ lazy 显著低于全量：按需换页有效，路线 A 可行")
        else:
            print("→ lazy ≈ 全量：gather 物化了整张量，需走路线 B（per-expert）")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "drive"
    if cmd == "build":
        build()
    elif cmd == "run":
        run(sys.argv[2])
    else:
        drive()
