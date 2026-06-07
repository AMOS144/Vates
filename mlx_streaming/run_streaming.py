"""路线 B 端到端：lazy 加载 + 离线拆分专家 + 文件后端流式 + generate，量内存/速度/命中率。

环境变量：
  MODEL         模型 repo 或本地路径（默认 mlx-community/Qwen3-30B-A3B-4bit）
  EXPERT_DIR    per-expert 拆分输出目录（默认 /tmp/mlx_qwen3_experts）
  EXPERT_SLOTS  每层 LRU 专家槽数（默认 8=top_k），worst-case 常驻≈槽数×MoE层数
  PROMPT/MAXTOK 提示与生成 token 数
  WIRED_GB      可选，set_wired_limit 上限（GB）
  CACHE_GB      可选，set_cache_limit 上限（GB），控制 MLX 缓冲复用上界
"""
import os
import time
import json

import mlx.core as mx
from mlx_lm import load, generate

from mlx_streaming.mem import snapshot, reset_peak, clear_cache
from mlx_streaming.expert_store import FileExpertStore
from mlx_streaming.split_experts import split_model
from mlx_streaming.streaming_moe import patch_model_filebacked

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts")
EXPERT_SLOTS = int(os.environ.get("EXPERT_SLOTS", "8"))   # 每层槽数
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
WIRED_GB = os.environ.get("WIRED_GB")
CACHE_GB = os.environ.get("CACHE_GB")
CLEAR_ON_EVICT = os.environ.get("CLEAR_ON_EVICT", "0") == "1"
PIN_HOT = int(os.environ.get("PIN_HOT", "0"))        # ③ 每层钉住的热专家数（0=关）
CAL_TOK = int(os.environ.get("CAL_TOK", "32"))       # 校准生成 token 数
# 专家量化格式覆盖：当 EXPERT_DIR 指向重量化目录(2/3-bit)时，文件后端 QSL 要用对应 bit/group
EXPERT_BITS = os.environ.get("EXPERT_BITS")
EXPERT_GROUP = os.environ.get("EXPERT_GROUP")
EXPERT_ROT = os.environ.get("EXPERT_ROT", "0") == "1"   # 专家为 Hadamard 旋转重量化版时置 1


def _first_moe_dims(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            return {
                "hidden": gp.input_dims, "moe_inter": gp.output_dims,
                "group_size": getattr(gp, "group_size", 64),
                "bits": getattr(gp, "bits", 4),
            }
    raise RuntimeError("模型里没有找到 MoE 层")


def main():
    if WIRED_GB:
        fn = getattr(mx, "set_wired_limit", None)
        if fn:
            print("set_wired_limit 旧值=", fn(int(float(WIRED_GB) * 1e9)))
    if CACHE_GB:
        fn = getattr(mx, "set_cache_limit", None)
        if fn:
            print("set_cache_limit 旧值=", fn(int(float(CACHE_GB) * 1e9)))

    # 1. 拆分专家到磁盘（若尚未拆分）
    if not os.path.exists(os.path.join(EXPERT_DIR, "_split_meta.json")):
        print("拆分专家到", EXPERT_DIR, "...")
        t = time.perf_counter()
        meta = split_model(MODEL, EXPERT_DIR)
        print("拆分完成", round(time.perf_counter() - t, 1), "s; MoE 层=", len(meta["moe_layers"]))

    reset_peak()
    # 2. lazy 加载（不强制 eval 全部）
    t0 = time.perf_counter()
    model, tok = load(MODEL, lazy=True)
    dims = _first_moe_dims(model)
    # 专家目录 meta 是 bit/group/proj_bits 的权威来源（重量化目录都会写）；先用它覆盖，
    # 再让 EXPERT_BITS/EXPERT_GROUP 环境变量做最高优先级手动覆盖。proj_bits 非空走混合精度。
    proj_bits = None
    layer_proj_bits = None
    meta_path = os.path.join(EXPERT_DIR, "_split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ed = json.load(f).get("dims", {})
        dims["bits"] = ed.get("bits", dims["bits"])
        dims["group_size"] = ed.get("group_size", dims["group_size"])
        proj_bits = ed.get("proj_bits")
        if "per_layer_proj_bits" in ed:  # 逐层混合：键转回 int 层号
            layer_proj_bits = {int(k): v for k, v in ed["per_layer_proj_bits"].items()}
    if EXPERT_BITS:
        dims["bits"] = int(EXPERT_BITS)
    if EXPERT_GROUP:
        dims["group_size"] = int(EXPERT_GROUP)
    # 3. 文件后端 patch，丢弃常驻堆叠 switch_mlp
    store = FileExpertStore(EXPERT_DIR, capacity=EXPERT_SLOTS,
                            clear_on_evict=CLEAR_ON_EVICT, record=PIN_HOT > 0)
    n = patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                               dims["group_size"], dims["bits"], rotated=EXPERT_ROT,
                               proj_bits=proj_bits, layer_proj_bits=layer_proj_bits)
    t1 = time.perf_counter()
    after_patch = snapshot()

    # 3.5 ③ 热专家常驻：先校准跑一遍统计激活频率，钉住每层最热的 PIN_HOT 个专家
    cal_s = 0.0
    if PIN_HOT > 0:
        tc = time.perf_counter()
        generate(model, tok, prompt=PROMPT, max_tokens=CAL_TOK, verbose=False)
        for li in store.recorded_layers():
            store.pin(li, store.hot(li, PIN_HOT))
        store.record = False
        store.reset_stats()
        cal_s = round(time.perf_counter() - tc, 2)

    # 4. 生成
    reset_peak()
    text = generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK, verbose=False)
    t2 = time.perf_counter()
    clear_cache()
    after_gen = snapshot()

    out = {
        "mode": "streaming_filebacked", "model": MODEL,
        "expert_dir": EXPERT_DIR,
        "expert_bits": dims["bits"], "expert_group": dims["group_size"],
        "proj_bits": proj_bits, "layered": layer_proj_bits is not None,
        "per_layer_slots": EXPERT_SLOTS, "wired_gb": WIRED_GB, "cache_gb": CACHE_GB,
        "clear_on_evict": CLEAR_ON_EVICT,
        "pin_hot": PIN_HOT, "cal_s": cal_s,
        "patched_moe_layers": n,
        "resident_experts": store.resident_count(),
        "pinned_experts": store.pinned_count(),
        "load_patch_s": round(t1 - t0, 2), "gen_s": round(t2 - t1, 2),
        "tok_per_s": round(MAXTOK / (t2 - t1), 2),
        "rss_gb_after_patch": round(after_patch.rss_bytes / 1e9, 2),
        "rss_gb_after_gen": round(after_gen.rss_bytes / 1e9, 2),
        "mlx_active_gb_after_gen": round(after_gen.mlx_active_bytes / 1e9, 2),
        "mlx_peak_gb": round(after_gen.mlx_peak_bytes / 1e9, 2),
        "expert_hit_rate": round(store.hit_rate(), 4),
        "expert_hits": store.hits, "expert_misses": store.misses,
        "sample": text[:240],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
