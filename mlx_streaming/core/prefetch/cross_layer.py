"""跨层专家预取：在 attention/GDN 之前用本层输入预测目标层 MoE 路由并提前取专家。

通过 monkeypatch `Qwen3NextDecoderLayer.__call__`，在真正进入 MoE 计算前用
gate_L(post_attention_layernorm_L(h)) 预测目标层专家（对齐 recall≈0.95 的探针口径），
把 top-(top_k*mult) 专家在当前层计算窗口内异步预取，藏住读盘/物化延迟。

支持多条预取后端（按环境开关择一）：STREAM_BLOB_BG（后台物化进池）、STREAM_BLOB
（blob 字节预读）、STREAM_BLOB_LOADER（blob-loader 字节预热）、native stage 预取。
"""
import time

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

from mlx_streaming import config
from mlx_streaming.core.moe import native_moe
from mlx_streaming.core.moe.gate import _predict_layer_experts
from mlx_streaming.core.moe.block import FileStreamingMoeBlock

# 是否已对 decoder layer 打过预取补丁（全局只打一次）。
_CROSS_LAYER_PREFETCH_PATCHED = False
# 每次「整模型前向」内的全局预取预算（按层序回绕重置），避免无界预取撑爆窗口。
_FILE_PREFETCH_REMAINING = 0
_FILE_PREFETCH_LAST_LAYER = -1
_STAGE_PREFETCH_REMAINING = 0
_STAGE_PREFETCH_LAST_LAYER = -1


def _legacy_decoder_prefetch_enabled(native_staging) -> bool:
    """Legacy decoder predictor must not duplicate native fused staging."""
    return (
        config.cross_layer_prefetch()
        and config.resident_pool_enabled()
        and native_staging is None
    )


def _target_cache_gate_logits(
    target, state, cache, *, move_for_reuse: bool = False,
):
    """把相邻目标层的真实 attention/gate 提前执行并交给目标层复用。

    当 ``state`` 是目标 ``T`` 的真实 ``T-1`` decoder 输出时，这与随后目标层
    实际执行的 gate 完全同构；较早 source 把近似 hidden 传进来时才只是近似值。
    """
    if not move_for_reuse:
        raise RuntimeError(
            "non-adjacent target-cache replay was removed because it "
            "duplicates target attention/gate",
        )
    if cache is None or (hasattr(cache, "empty") and cache.empty()):
        return None
    # The adjacent target is the very next decoder call. Execute on its real
    # cache and consume the saved result exactly once: this is movement, not a
    # shadow replay or a clone followed by checkpoint/state copying.
    normalized = target.input_layernorm(state)
    if target.is_linear:
        mask = create_ssm_mask(state, cache)
        attention = target.linear_attn(normalized, mask, cache)
    else:
        mask = create_attention_mask(state, cache)
        attention = target.self_attn(normalized, mask, cache)
    post_attention = state + attention
    gate_input = target.post_attention_layernorm(post_attention)
    logits = target.mlp.gate(gate_input)
    # Only this adjacent moved computation may replace the real decoder call.
    if move_for_reuse:
        shared = None
        target_mlp = getattr(target, "mlp", None)
        if (
            isinstance(target_mlp, FileStreamingMoeBlock)
            and target_mlp.shared_expert is not None
        ):
            shared = (
                mx.sigmoid(target_mlp.shared_expert_gate(gate_input))
                * target_mlp.shared_expert(gate_input)
            )
            # The shared expert is resident and independent of routed-expert
            # SSD readiness. Start the exact computation on the progressive
            # stream and reuse the same output in the real target call.
            mx.async_eval(shared)
        object.__setattr__(target, "_prefetch_moved_result", (
            state, cache, post_attention, gate_input, logits, shared,
        ))
    return logits


def _progressive_decoder_call(layer, x, *, mask=None, cache=None):
    """严格复现 decoder，并在真实 T-1 输出后追加目标层精确补位。

    第一次 early-core submit 仍由 ``FileStreamingMoeBlock`` 在原 source gate
    demand 的 completed callback 发起。这里不改动、也不等待那次提交；只在目标
    ``T`` 的前一层 ``T-1`` 完整算完后，把目标 attention/GDN 的真实计算提前，
    再向同一个 generation 补交剩余合法槽位；目标层直接复用，不再重算。

    第二个 dummy 在独立 stream 上 async-eval，保证立即发起而不阻塞主线程。只有
    ``PREFETCH_PROGRESSIVE_WAIT=1`` 的诊断兼容模式才同步等待 tail 字节。
    """
    # 保存同一个 cache 对象。MTP commit/restore 在对象内换 state；下一个 forward
    # 的 T-1 moved call 因而读取已经提交的目标 cache，而不是 speculative 脏快照。
    object.__setattr__(layer, "_target_cache_ref", cache)

    reuse = getattr(layer, "_prefetch_adjacent_reuse", None)
    object.__setattr__(layer, "_prefetch_adjacent_reuse", None)
    if reuse is not None and reuse[0] is x and reuse[1] is cache:
        (
            _input, _cache, post_attention, gate_input, logits, shared,
        ) = reuse
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, FileStreamingMoeBlock):
            object.__setattr__(mlp, "_prefetch_reuse_raw_gates", logits)
            if shared is not None:
                object.__setattr__(
                    mlp, "_prefetch_reuse_shared", (gate_input, shared),
                )
    else:
        normalized = layer.input_layernorm(x)
        if layer.is_linear:
            attention = layer.linear_attn(normalized, mask, cache)
        else:
            attention = layer.self_attn(normalized, mask, cache)
        post_attention = x + attention
        gate_input = layer.post_attention_layernorm(post_attention)

    mlp = getattr(layer, "mlp", None)
    if isinstance(mlp, FileStreamingMoeBlock):
        # 与原 wrapper 一样，为 source-time proxy 保留未归一化 decoder 输入。
        mlp._unnormed_input = x
    out = post_attention + layer.mlp(gate_input)

    if not isinstance(mlp, FileStreamingMoeBlock):
        return out
    if int(x.shape[1]) > config.prefetch_target_cache_max_seq():
        return out

    source_idx = int(getattr(mlp, "layer_idx", getattr(layer, "_layer_idx", -1)))
    vpool = getattr(mlp, "_vpool", None)
    if source_idx < 0 or vpool is None or not hasattr(vpool, "has_progressive"):
        return out

    # Only the adjacent target can move its real decoder computation here and
    # reuse it later. Non-adjacent/post-MoE shadow gates are deliberately not
    # constructed; config.prefetch_progressive_for routes those targets through
    # the ordinary one-shot early rerank.
    target_indices = [
        source_idx + 1
        if vpool.has_progressive(source_idx + 1) else None
    ]
    target_indices = [target for target in target_indices if target is not None]
    if not target_indices:
        return out

    model = getattr(mlp, "_prefetch_model_ref", None)
    layers = getattr(model, "layers", ())

    # Build and submit the tail on its own device stream. The original early
    # callback remains in the main graph; target demand never waits for tail
    # SSD I/O unless the explicit diagnostic compatibility switch is enabled.
    with mx.stream(vpool.progressive_stream()):
        for target_idx in target_indices:
            if not 0 <= target_idx < len(layers):
                continue
            target = layers[target_idx]
            target_mlp = getattr(target, "mlp", None)
            target_cache = getattr(target, "_target_cache_ref", None)
            if not isinstance(target_mlp, FileStreamingMoeBlock):
                continue
            refinement_logits = _target_cache_gate_logits(
                target,
                out,
                target_cache,
                move_for_reuse=True,
            )
            moved = getattr(target, "_prefetch_moved_result", None)
            object.__setattr__(target, "_prefetch_moved_result", None)
            exact_route = False
            if (
                moved is not None
                and moved[0] is out
                and moved[1] is target_cache
            ):
                object.__setattr__(
                    target, "_prefetch_adjacent_reuse", moved,
                )
                exact_route = True
            if refinement_logits is None:
                # First prefill has no committed cache. The state is discarded
                # at the next forward boundary.
                continue
            refinement_kwargs = {"source_layer": source_idx}
            if exact_route:
                refinement_kwargs["exact_route"] = True
            refinement = vpool.refine_progressive(
                target_idx, refinement_logits, **refinement_kwargs,
            )
            if refinement is not None:
                dummy, exact_routes = refinement
                if config.prefetch_progressive_wait():
                    mx.eval(dummy, exact_routes)
                    vpool.wait_progressive(target_idx, exact_routes)
                else:
                    mx.async_eval(dummy)
    return out


def _cross_layer_prefetch_mult() -> int:
    return config.cross_layer_mult()


def _submit_missing_prefetch(store, layer: int, predicted) -> "list[int]":
    """只把"预测∩非常驻"的专家提交给后台预取器，避免重载常驻、撑爆 attention 窗口。

    返回实际提交的缺失专家列表（供测试/统计）。
    """
    bg = getattr(store, "_bg", None)
    if bg is None:
        return []
    resident = store.resident_experts(layer) if hasattr(store, "resident_experts") else set()
    missing = [int(e) for e in predicted if int(e) not in resident]
    if missing:
        bg.submit(int(layer), missing)
    return missing


def _prefetch_budget(layer_idx: int, kind: str) -> int:
    """按一次层序前向重置全局预算；layer 序号回绕说明进入新 forward。"""
    global _FILE_PREFETCH_REMAINING, _FILE_PREFETCH_LAST_LAYER
    global _STAGE_PREFETCH_REMAINING, _STAGE_PREFETCH_LAST_LAYER
    if kind == "file":
        total = config.file_prefetch_global_budget()
        if layer_idx <= _FILE_PREFETCH_LAST_LAYER:
            _FILE_PREFETCH_REMAINING = total
        _FILE_PREFETCH_LAST_LAYER = layer_idx
        return max(0, _FILE_PREFETCH_REMAINING)
    total = config.stage_prefetch_global_budget()
    if layer_idx <= _STAGE_PREFETCH_LAST_LAYER:
        _STAGE_PREFETCH_REMAINING = total
    _STAGE_PREFETCH_LAST_LAYER = layer_idx
    return max(0, _STAGE_PREFETCH_REMAINING)


def _consume_prefetch_budget(n: int, kind: str) -> None:
    global _FILE_PREFETCH_REMAINING, _STAGE_PREFETCH_REMAINING
    if kind == "file":
        _FILE_PREFETCH_REMAINING = max(0, _FILE_PREFETCH_REMAINING - n)
    else:
        _STAGE_PREFETCH_REMAINING = max(0, _STAGE_PREFETCH_REMAINING - n)


def _prefetch_layers(kind: str) -> "set[int] | None":
    env = "FILE_PREFETCH_LAYERS" if kind == "file" else "STAGE_PREFETCH_LAYERS"
    return config.parse_layers_env(env)


def enable_cross_layer_prefetch():
    """给 Qwen3NextDecoderLayer 加跨层专家预取。

    在 attention/GDN 之前，用当前层输入 hidden 的 post_attention_layernorm 近似预测
    本层 MoE 路由，并把 top-(top_k*mult) 专家预取进 resident pool。
    """
    global _CROSS_LAYER_PREFETCH_PATCHED
    if _CROSS_LAYER_PREFETCH_PATCHED:
        return
    from mlx_lm.models.qwen3_next import Qwen3NextDecoderLayer
    orig_call = Qwen3NextDecoderLayer.__call__

    def patched_call(self, x, mask=None, cache=None):
        if config.prefetch_progressive():
            return _progressive_decoder_call(self, x, mask=mask, cache=cache)
        if config.prefetch_target_cache():
            raise RuntimeError(
                "PREFETCH_TARGET_CACHE clone/replay was removed because it "
                "duplicates target attention/gate; use PREFETCH_PROGRESSIVE "
                "with adjacent reuse",
            )
        mlp = getattr(self, "mlp", None)
        native_staging = (
            getattr(getattr(mlp, "store", None), "_staging", None)
            if mlp is not None else None
        )
        if native_staging is not None and not config.predict_use_x():
            # 存本层「未归一化」decoder 输入：供 native-fused-prefetch 用目标层 norm 正确预测
            # （对齐 0.95-recall 探针；MoE forward 拿到的 x 已被本层 norm 过，norm 用错会掉点）。
            mlp._unnormed_input = x
        # FileStreamingMoeBlock already submits the native fused prediction
        # through ``_native_fused_prefetch`` when staging is attached.  The
        # legacy decoder-level path below computes another target gate and,
        # for STREAM_BLOB_LOADER, launches a second byte prefetch.  Enabling
        # CROSS_LAYER_PREFETCH together with NATIVE_FUSED_PREFETCH therefore
        # used to duplicate the whole predictor and cut K=3 throughput almost
        # in half.  Keep this branch only for stores without native staging.
        if _legacy_decoder_prefetch_enabled(native_staging):
            ahead = config.cross_layer_ahead(default=0)
            target_layer = getattr(self, "_layer_idx", None)
            if target_layer is not None:
                target_layer += ahead
            target_mlp = None
            target_decoder = None
            if ahead == 0 and isinstance(mlp, FileStreamingMoeBlock):
                target_mlp = mlp
                target_decoder = self
            elif target_layer is not None:
                layers = getattr(getattr(self, "_prefetch_model_ref", None), "layers", [])
                if 0 <= target_layer < len(layers):
                    target_decoder = layers[target_layer]
                    target_mlp = getattr(target_decoder, "mlp", None)
            if isinstance(target_mlp, FileStreamingMoeBlock):
                if config.probe_predict_only():
                    # 诊断：只做预测 gate 前向 + eval(同步在预测上)，不 .tolist、不预取。
                    # 用于拆分 32% 税：≈S(+9%)→税是 Python 编排(native 可救)；≈B(+32%)→税是
                    # gate前向被 barrier 暴露在关键路径(简单 native 派发救不了)。
                    g = target_mlp.gate(target_decoder.post_attention_layernorm(x))
                    kk = min(g.shape[-1], 4)
                    ip = mx.argpartition(g, kth=-kk, axis=-1)[..., -kk:]
                    mx.eval(ip)
                    return orig_call(self, x, mask=mask, cache=cache)
                # 关键：用**目标层**的 post_attention_layernorm（匹配 probe 验证的
                # gate_L(post_norm_L(h_{L-ahead})) 配置，recall_miss≈0.95）。
                best, num_experts = _predict_layer_experts(
                    target_decoder.post_attention_layernorm, target_mlp.gate,
                    target_mlp.top_k, x, _cross_layer_prefetch_mult())
                bg = getattr(target_mlp.store, "_bg", None)
                if config.stream_blob_bg() and bg is not None:
                    # 同层(AHEAD=0)预测 + 只预取"预测∩非常驻"，藏进 attention/GDN 窗口。
                    budget = config.stream_blob_bg_budget(
                        default=target_mlp.top_k * _cross_layer_prefetch_mult())
                    picked = [e for e, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:budget]
                    _submit_missing_prefetch(target_mlp.store, target_mlp.layer_idx, picked)
                    if config.window_prof():
                        target_mlp._submit_t = time.perf_counter()
                    return orig_call(self, x, mask=mask, cache=cache)
                if config.stream_blob() and target_mlp._blob is not None:
                    # 全流式：后台预读下一层预测专家的字节，与当前层计算重叠。
                    budget = config.stream_blob_prefetch_budget(default=target_mlp.top_k * 2)
                    picked = [e for e, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:budget]
                    target_mlp._blob.prefetch_async(target_mlp.layer_idx, picked)
                    return orig_call(self, x, mask=mask, cache=cache)
                blob_loader = getattr(target_mlp.store, "_blob_loader", None)
                if config.stream_blob_loader() and blob_loader is not None:
                    # blob-loader 路径：后台只预读字节（与当前层计算重叠），
                    # demand 时主线程从预热字节快速物化进常驻池，不在主线程同步阻塞读盘。
                    budget = config.stage_prefetch_per_layer_budget(default=12)
                    picked = [e for e, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)][:budget]
                    blob_loader.prefetch_async(target_mlp.layer_idx, picked)
                    return orig_call(self, x, mask=mask, cache=cache)
                allowed_layers = _prefetch_layers("stage")
                if allowed_layers is not None and target_mlp.layer_idx not in allowed_layers:
                    return orig_call(self, x, mask=mask, cache=cache)
                min_score = config.stage_prefetch_min_score()
                per_layer = config.stage_prefetch_per_layer_budget(default=4)
                remaining = _prefetch_budget(target_mlp.layer_idx, "stage")
                budget = max(0, min(per_layer, remaining))
                expert_ids = [
                    e for e, s in sorted(best.items(), key=lambda kv: kv[1], reverse=True)
                    if s >= min_score
                ][:budget]
                _consume_prefetch_budget(len(expert_ids), "stage")
                proj_bits = getattr(target_mlp._sub, "proj_bits", {})
                bits_set = {
                    int(proj_bits.get(name, target_mlp.bits))
                    for name in ("gate_proj", "up_proj", "down_proj")
                }
                if len(bits_set) == 1 and native_moe.prefetch_native_moe_stage(
                    target_mlp.layer_idx,
                    expert_ids,
                    target_mlp.hidden,
                    target_mlp.moe_inter,
                    target_mlp.group_size,
                    bits_set.pop(),
                    num_experts,
                ):
                    pass
                else:
                    target_mlp.store.prefetch(target_mlp.layer_idx, expert_ids)
        return orig_call(self, x, mask=mask, cache=cache)

    Qwen3NextDecoderLayer.__call__ = patched_call
    _CROSS_LAYER_PREFETCH_PATCHED = True
