"""投机解码 + 流式专家 验证：target=流式 Qwen3-30B，draft=常驻 Qwen3-0.6B。

测「独立 draft 投机」在 I/O 流式 MoE 上到底有没有加速：对比不投机 vs 不同
num_draft_tokens 的 decode tok/s 与草稿接受率。

环境变量：
  MODEL/EXPERT_DIR/EXPERT_BITS/EXPERT_GROUP/EXPERT_SLOTS 同 run_streaming
  DRAFT       draft 模型路径（默认 /tmp/qwen3_0.6b_4bit）
  NDRAFTS     要扫的 num_draft_tokens 列表（逗号分隔，默认 "2,3,4"）
  MAXTOK/PROMPT
"""
import os
import time
import json

import mlx.core as mx
from mlx_lm import load, stream_generate

from mlx_streaming.core.mem import snapshot, reset_peak
from mlx_streaming.core.cache.expert_store import FileExpertStore
from mlx_streaming.core.prefetch.patch import patch_model_filebacked

MODEL = os.environ.get("MODEL", "mlx-community/Qwen3-30B-A3B-4bit")
EXPERT_DIR = os.environ.get("EXPERT_DIR", "/tmp/mlx_qwen3_experts_2bit")
EXPERT_BITS = int(os.environ.get("EXPERT_BITS", "2"))
EXPERT_GROUP = int(os.environ.get("EXPERT_GROUP", "64"))
EXPERT_SLOTS = int(os.environ.get("EXPERT_SLOTS", "64"))
EXPERT_ROT = os.environ.get("EXPERT_ROT", "0") == "1"   # 专家为 Hadamard 旋转重量化版时置 1
DRAFT = os.environ.get("DRAFT", "/tmp/qwen3_0.6b_4bit")
NDRAFTS = [int(x) for x in os.environ.get("NDRAFTS", "2,3,4").split(",")]
MAXTOK = int(os.environ.get("MAXTOK", "128"))
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")


def _first_moe_dims(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            gp = mlp.switch_mlp.gate_proj
            return {"hidden": gp.input_dims, "moe_inter": gp.output_dims}
    raise RuntimeError("无 MoE 层")


def _build_target():
    model, tok = load(MODEL, lazy=True)
    dims = _first_moe_dims(model)
    # 混合精度：从专家目录 meta 读 proj_bits/bits/group（有则优先于环境变量默认）
    bits, group, proj_bits, layer_proj_bits = EXPERT_BITS, EXPERT_GROUP, None, None
    meta_path = os.path.join(EXPERT_DIR, "_split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            ed = json.load(f).get("dims", {})
        bits = ed.get("bits", bits)
        group = ed.get("group_size", group)
        proj_bits = ed.get("proj_bits")
        if "per_layer_proj_bits" in ed:
            layer_proj_bits = {int(k): v for k, v in ed["per_layer_proj_bits"].items()}
    store = FileExpertStore(EXPERT_DIR, capacity=EXPERT_SLOTS)
    patch_model_filebacked(model, store, dims["hidden"], dims["moe_inter"],
                           group, bits, rotated=EXPERT_ROT, proj_bits=proj_bits,
                           layer_proj_bits=layer_proj_bits)
    return model, tok, store, {"bits": bits, "group": group, "proj_bits": proj_bits,
                               "layered": layer_proj_bits is not None}


def _run(model, tok, draft_model, num_draft, store):
    store.reset_stats() if hasattr(store, "reset_stats") else None
    n_tok = 0
    n_draft_acc = 0
    last_tps = 0.0
    kwargs = {}
    if draft_model is not None:
        kwargs["num_draft_tokens"] = num_draft
    t0 = time.perf_counter()
    for r in stream_generate(model, tok, prompt=PROMPT, max_tokens=MAXTOK,
                             draft_model=draft_model, **kwargs):
        n_tok = r.generation_tokens
        last_tps = r.generation_tps
        if getattr(r, "from_draft", False):
            n_draft_acc += 1
    wall = time.perf_counter() - t0
    return {
        "num_draft_tokens": num_draft if draft_model is not None else None,
        "gen_tokens": n_tok,
        "tok_per_s": round(last_tps, 2),
        "wall_s": round(wall, 2),
        "draft_accepted_frac": round(n_draft_acc / n_tok, 3) if n_tok else 0.0,
        "expert_hit_rate": round(store.hit_rate(), 3),
    }


def main():
    reset_peak()
    model, tok, store, qinfo = _build_target()
    draft_model, _ = load(DRAFT)            # 常驻小 draft
    mx.eval(draft_model.parameters())

    results = []
    # 1) 不投机基线
    results.append({"mode": "no_spec", **_run(model, tok, None, 0, store)})
    # 2) 投机：扫 num_draft_tokens
    for nd in NDRAFTS:
        results.append({"mode": "spec", **_run(model, tok, draft_model, nd, store)})

    after = snapshot()
    print(json.dumps({
        "target": MODEL, "draft": DRAFT,
        "expert_bits": qinfo["bits"], "expert_proj_bits": qinfo["proj_bits"],
        "expert_slots": EXPERT_SLOTS,
        "rss_gb": round(after.rss_bytes / 1e9, 2),
        "mlx_peak_gb": round(after.mlx_peak_bytes / 1e9, 2),
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
