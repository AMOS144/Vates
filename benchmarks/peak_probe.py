"""THROWAWAY 探针（一次性，用完即弃）：分段定位 MTP 解码的 in-flight 峰值来源。

复用 run_mtp_spec 的模型/drafter 搭建，用 reset_peak()/get_peak_memory() 在各阶段之间
清零高水位，逐段量峰值（分配量，不受 swap/墙钟影响）：
  - prefill（分块）
  - decode 单步 forward（seq=1，含 lm_head）
  - verify 前向（seq=K，含 lm_head）vs 同 seq 不含 lm_head → 差值即 lm_head 边际
  - lm_head 单独投影

用法：EXPERT_SLOTS=32 ZEROCOPY_DUAL_SOURCE=1 SIDEREGION_LFU=1 POOL_SPEC_SLOTS=32 \
     STREAM_BLOB_LOADER=1 NATIVE_FUSED_PREFETCH=1 K=3 \
     .venv/bin/python -m benchmarks.peak_probe
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming import config as _cfg
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.generate import forward_with_hidden, prefill_chunked, mtp_generate
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp

GIB = 1024 ** 3
K = int(os.environ.get("K", "3"))
PROMPT = os.environ.get("PROMPT", "用三句话解释什么是混合专家模型。")


def _peak():
    return mx.get_peak_memory() / GIB


def _active():
    mx.clear_cache()
    return mx.get_active_memory() / GIB


def main():
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = MTPDrafter(mtp, model.lm_head)

    # 先跑一轮完整 MTP 把专家池/侧区/kernel 热到稳态（与 run_mtp_spec 测量轮一致）。
    _ids, _ = mtp_generate(model, drafter, tok, mx.array([tok.encode(PROMPT)]),
                           48, K=K, ids_mode=True, profile=False)

    out = {}

    # ---- 真实测量轮：整轮 mtp_generate 的绝对峰值 / 清缓冲后 active（口径同 run_mtp_spec）----
    mx.reset_peak_memory()
    mtp_generate(model, drafter, tok, mx.array([tok.encode(PROMPT)]),
                 48, K=K, ids_mode=True, profile=False)
    _round_peak = round(mx.get_peak_memory() / 1e9, 3)
    mx.clear_cache()
    out["real_round"] = {"round_peak_gb": _round_peak,
                         "active_after_clear_gb": round(mx.get_active_memory() / 1e9, 3)}

    # ---- 阶段 1：prefill（分块）----
    cache = model.make_cache()
    ids = mx.array([tok.encode(PROMPT)])
    mx.reset_peak_memory()
    logits, H = prefill_chunked(model, ids, cache)
    mx.eval(logits, H)
    out["prefill"] = {"peak_gib": round(_peak(), 3), "prompt_tokens": int(ids.shape[1]),
                      "chunk": _cfg.prefill_chunk(), "active_after_gib": round(_active(), 3)}

    x = int(mx.argmax(logits[:, -1, :]))

    # ---- 阶段 2：decode 单步（seq=1，含 lm_head）----
    cur = mx.array([[x]])
    mx.reset_peak_memory()
    l1, h1 = forward_with_hidden(model, cur, cache)
    mx.eval(l1, h1)
    out["decode_seq1_full"] = {"peak_gib": round(_peak(), 3)}

    # ---- 阶段 3：verify 前向（seq=K，含/不含 lm_head）----
    verify_in = mx.array([[x] + [x] * (K - 1)])              # (1,K)，内容无关紧要（只量内存）
    c2 = model.make_cache()
    _fill = prefill_chunked(model, ids, c2)                  # 先把 cache 填到与 decode 同上下文（分块）
    mx.eval(_fill[1])
    mx.reset_peak_memory()
    vl, vH = forward_with_hidden(model, verify_in, c2, compute_logits=True)
    mx.eval(vl, vH)
    out["verify_seqK_full"] = {"peak_gib": round(_peak(), 3), "K": K,
                               "logits_shape": list(vl.shape)}

    c3 = model.make_cache()
    _fill3 = prefill_chunked(model, ids, c3)
    mx.eval(_fill3[1])
    mx.reset_peak_memory()
    _vl2, vH2 = forward_with_hidden(model, verify_in, c3, compute_logits=False)
    mx.eval(vH2)
    out["verify_seqK_nolmhead"] = {"peak_gib": round(_peak(), 3)}

    # ---- 阶段 4：lm_head 单独投影（H → V）----
    Hn = model.model.norm(vH2)                               # (1,K,hidden)
    mx.eval(Hn)
    mx.reset_peak_memory()
    lg = model.lm_head(Hn)
    mx.eval(lg)
    out["lmhead_alone"] = {"peak_gib": round(_peak(), 3), "out_shape": list(lg.shape),
                           "vocab": int(lg.shape[-1])}

    out["lmhead_marginal_gib"] = round(
        out["verify_seqK_full"]["peak_gib"] - out["verify_seqK_nolmhead"]["peak_gib"], 3)

    # ---- 单步解剖：量 snap_m / _spec_checkpoints 的字节 + 各段增量峰值 ----
    from mlx_streaming.mtp.kv_cache import (
        _snapshot, begin_speculative_checkpoints, enable_qwen3next_speculative_checkpoints)
    from mlx.utils import tree_flatten
    enable_qwen3next_speculative_checkpoints()

    def _nbytes(o):
        if o is None:
            return 0
        tot = 0
        stack = [o]
        while stack:
            v = stack.pop()
            if isinstance(v, mx.array):
                tot += v.nbytes
            elif isinstance(v, (list, tuple)):
                stack.extend(v)
        return tot

    ca = model.make_cache()
    prefill_chunked(model, ids, ca)
    mx.eval([v for c in ca for v in (getattr(c, "state", None) or []) if isinstance(v, mx.array)]
            if False else [])
    step = {}
    mx.reset_peak_memory()
    snap_m = _snapshot(ca)
    step["snap_m_bytes_gib"] = round(_nbytes(snap_m) / GIB, 4)
    step["peak_after_snap_gib"] = round(_peak(), 3)

    mx.reset_peak_memory()
    begin_speculative_checkpoints(ca)
    vl3, vH3 = forward_with_hidden(model, verify_in, ca, compute_logits=True)
    mx.eval(vl3, vH3)
    step["peak_after_verify_ckpt_gib"] = round(_peak(), 3)
    ck_bytes = 0
    for c in ca:
        cks = getattr(c, "_spec_checkpoints", None)
        ck_bytes += _nbytes(cks)
    step["spec_checkpoints_bytes_gib"] = round(ck_bytes / GIB, 4)
    out["step_anatomy"] = step

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
