"""⑤ 探针：每层单文件 + 单专家切片，lazy 加载后只切 8 个专家，实测 RSS。

回答：mx.load(per_layer_file, lazy) 后对堆叠张量切 arr[e]+eval，是只读这几个专家
的页（mmap 按需），还是把整层文件物化进内存？决定 ⑤ 是否真能省读放大。

用法：
  python -m mlx_streaming.probe_layerfile build   # 用 per-expert 文件拼一个每层单文件
  python -m mlx_streaming.probe_layerfile run <mode>   # mode: wholelayer | slice8
  python -m mlx_streaming.probe_layerfile drive
"""
import os
import sys
import json
import glob
import subprocess

import mlx.core as mx

from mlx_streaming.mem import snapshot, reset_peak

PER_EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts")
LAYER_DIR = os.environ.get("LAYER_DIR", "/tmp/mlx_qwen3_layerfiles")
LAYER = int(os.environ.get("PROBE_LAYER", "0"))
LAYER_PATH = os.path.join(LAYER_DIR, f"layer{LAYER:02d}.safetensors")


def build():
    os.makedirs(LAYER_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(PER_EXPERT_DIR, f"layer{LAYER:02d}_expert*.safetensors")))
    assert files, f"找不到 per-expert 文件于 {PER_EXPERT_DIR}"
    per = [mx.load(f) for f in files]
    keys = per[0].keys()
    stacked = {k: mx.stack([p[k] for p in per]) for k in keys}
    mx.eval(stacked)
    mx.save_safetensors(LAYER_PATH, stacked)
    sz = os.path.getsize(LAYER_PATH)
    print(json.dumps({"built": True, "experts": len(files),
                      "layer_file_mb": round(sz / 1e6, 1),
                      "keys": list(keys)}, ensure_ascii=False))


def run(mode: str):
    reset_peak()
    before = snapshot()
    w = mx.load(LAYER_PATH)            # 默认（lazy 与否取决于 MLX；safetensors 通常 mmap）
    after_load = snapshot()
    if mode == "wholelayer":
        mx.eval(w)                     # 整层物化 → 上界
        tag = w
    else:  # slice8：只切 8 个专家
        picked = []
        for e in range(8):
            sub = {k: arr[e] for k, arr in w.items()}
            mx.eval(sub)
            picked.append(sub)
        tag = picked
    after = snapshot()
    sz_mb = round(os.path.getsize(LAYER_PATH) / 1e6, 1)
    print(json.dumps({
        "mode": mode, "layer_file_mb": sz_mb,
        "rss_mb_before": round(before.rss_bytes / 1e6, 1),
        "rss_mb_after_load": round(after_load.rss_bytes / 1e6, 1),
        "rss_mb_after": round(after.rss_bytes / 1e6, 1),
        "mlx_active_mb": round(after.mlx_active_bytes / 1e6, 1),
        "mlx_peak_mb": round(after.mlx_peak_bytes / 1e6, 1),
    }, ensure_ascii=False))


def drive():
    if not os.path.exists(LAYER_PATH):
        subprocess.run([sys.executable, "-m", "mlx_streaming.probe_layerfile", "build"], check=True)
    rows = []
    for mode in ["wholelayer", "slice8"]:
        p = subprocess.run([sys.executable, "-m", "mlx_streaming.probe_layerfile", "run", mode],
                           capture_output=True, text=True)
        line = ""
        for ln in p.stdout.splitlines():
            if ln.strip().startswith("{"):
                line = ln.strip()
        rows.append(json.loads(line) if line else {"mode": mode, "err": p.stderr[-300:]})
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    by = {r.get("mode"): r for r in rows if "rss_mb_after" in r}
    if "wholelayer" in by and "slice8" in by:
        whole = by["wholelayer"]["rss_mb_after"]
        s8 = by["slice8"]["rss_mb_after"]
        base = by["slice8"]["rss_mb_after_load"]
        print(f"\n判定：整层物化 RSS={whole}MB, 切8专家 RSS={s8}MB（载入后基线 {base}MB），"
              f"文件 {by['slice8']['layer_file_mb']}MB")
        if (s8 - base) < (whole - base) * 0.5:
            print("→ 切片只读了部分：mmap 按需有效，⑤ 能省读放大")
        else:
            print("→ 切片≈整层：lazy 也会读整层，⑤ 只省文件打开数、不省读量")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "drive"
    if cmd == "build":
        build()
    elif cmd == "run":
        run(sys.argv[2])
    else:
        drive()
