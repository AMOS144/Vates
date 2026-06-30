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
import statistics

import mlx.core as mx
from mlx_lm import load, generate

from mlx_streaming import config
from mlx_streaming.core.mem import snapshot, reset_peak, clear_cache
from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.prep.split_experts import split_model
from mlx_streaming.core.prefetch.patch import patch_model_filebacked
from mlx_streaming.model_builder import load_pool_profile

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts")
EXPERT_SLOTS = int(os.environ.get("EXPERT_SLOTS", "8"))   # 每层槽数
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")
MAXTOK = int(os.environ.get("MAXTOK", "128"))
# 稳态测速:warmup 跑满 MAXTOK(把整段会路由到的专家都装进常驻池 + 编译 Metal kernel),
# 再重复 REPEAT 次取中位数,避免冷启动/补池污染绝对 tok/s。WARMUP_TOK=0 关 warmup。
WARMUP_TOK = int(os.environ.get("WARMUP_TOK", str(MAXTOK)))
REPEAT = int(os.environ.get("REPEAT", "3"))
WIRED_GB = os.environ.get("WIRED_GB")
CACHE_GB = os.environ.get("CACHE_GB")
CLEAR_ON_EVICT = os.environ.get("CLEAR_ON_EVICT", "0") == "1"
PIN_HOT = int(os.environ.get("PIN_HOT", "0"))        # ③ 每层钉住的热专家数（0=关）
CAL_TOK = int(os.environ.get("CAL_TOK", "32"))       # 校准生成 token 数
# 专家量化格式覆盖：当 EXPERT_DIR 指向重量化目录(2/3-bit)时，文件后端 QSL 要用对应 bit/group
EXPERT_BITS = os.environ.get("EXPERT_BITS")
EXPERT_GROUP = os.environ.get("EXPERT_GROUP")

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
    # 每层池预算 profile:默认从 {EXPERT_DIR}/pool_profile.json 自动启用(无损省内存)，
    # EXPERT_POOL_PROFILE 可显式指定路径或设 none 关闭。命中率/输出/吞吐不变。
    layer_caps = load_pool_profile(EXPERT_DIR)
    # 3. 文件后端 patch，丢弃常驻堆叠 switch_mlp
    store = FileExpertStore(EXPERT_DIR, capacity=EXPERT_SLOTS, layer_caps=layer_caps,
                            clear_on_evict=CLEAR_ON_EVICT, record=PIN_HOT > 0)
    n = patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                               dims["group_size"], dims["bits"],
                               proj_bits=proj_bits, layer_proj_bits=layer_proj_bits)
    t1 = time.perf_counter()
    after_patch = snapshot()

    # 分块 prefill:把 mlx_lm.generate 内部 prefill 步长压到 config.prefill_chunk()(默认 2),
    # 整段 prefill 的激活峰值 ∝prompt 长度 → ∝chunk,使 prefill 与 decode 同稳态。
    # PREFILL_CHUNK=0 时不传,回退 mlx_lm 默认 2048。
    _ps = config.prefill_chunk()
    _gen_kw = {"prefill_step_size": _ps} if _ps > 0 else {}

    # 3.5 ③ 热专家常驻：先校准跑一遍统计激活频率，钉住每层最热的 PIN_HOT 个专家
    cal_s = 0.0
    if PIN_HOT > 0:
        tc = time.perf_counter()
        generate(model, tok, prompt=PROMPT, max_tokens=CAL_TOK, verbose=False, **_gen_kw)
        for li in store.recorded_layers():
            store.pin(li, store.hot(li, PIN_HOT))
        store.record = False
        store.reset_stats()
        cal_s = round(time.perf_counter() - tc, 2)

    # 3.6 warmup:跑满 MAXTOK 把整段专家装进常驻池并编译 Metal kernel(PIN_HOT 已校准时
    # 这里仍补满全长,确保第一次正式测量即稳态)。
    if WARMUP_TOK > 0:
        generate(model, tok, prompt=PROMPT, max_tokens=WARMUP_TOK, verbose=False, **_gen_kw)

    # 4. 生成:重复 REPEAT 次取中位数;最后一次清零专家统计供命中率口径对应稳态。
    reset_peak()
    text = None
    gen_runs = []
    for r in range(REPEAT):
        if r == REPEAT - 1:
            store.reset_stats()
        tg = time.perf_counter()
        text = generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK, verbose=False, **_gen_kw)
        gen_runs.append(round(MAXTOK / (time.perf_counter() - tg), 2))
    tok_per_s = statistics.median(gen_runs)
    t2 = time.perf_counter()
    clear_cache()
    after_gen = snapshot()

    out = {
        "mode": "streaming_filebacked", "model": MODEL,
        "expert_dir": EXPERT_DIR,
        "expert_bits": dims["bits"], "expert_group": dims["group_size"],
        "proj_bits": proj_bits, "layered": layer_proj_bits is not None,
        "per_layer_slots": EXPERT_SLOTS, "pool_profile": bool(layer_caps),
        "wired_gb": WIRED_GB, "cache_gb": CACHE_GB,
        "clear_on_evict": CLEAR_ON_EVICT,
        "pin_hot": PIN_HOT, "cal_s": cal_s,
        "warmup_tok": WARMUP_TOK, "repeat": REPEAT,
        "patched_moe_layers": n,
        "resident_experts": store.resident_count(),
        "pinned_experts": store.pinned_count(),
        "load_patch_s": round(t1 - t0, 2),
        "tok_per_s": tok_per_s,
        "tok_per_s_runs": gen_runs,
        "tok_per_s_minmax": [min(gen_runs), max(gen_runs)],
        "rss_gb_after_patch": round(after_patch.rss_bytes / 1e9, 2),
        "rss_gb_after_gen": round(after_gen.rss_bytes / 1e9, 2),
        "mlx_active_gb_after_gen": round(after_gen.mlx_active_bytes / 1e9, 2),
        "mlx_peak_gb": round(after_gen.mlx_peak_bytes / 1e9, 2),
        "expert_hit_rate": round(store.hit_rate(), 4),
        "expert_hits": store.hits, "expert_misses": store.misses,
        # GPU remap 取证:整层全命中走 GPU 快路径 vs 有 miss 回退 host 的层调用占比
        "gpu_fastpath": store._resident.gpu_fastpath,
        "gpu_fallback": store._resident.gpu_fallback,
        "gpu_fastpath_frac": round(
            store._resident.gpu_fastpath
            / max(1, store._resident.gpu_fastpath + store._resident.gpu_fallback), 4),
        "sample": text[:240],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
