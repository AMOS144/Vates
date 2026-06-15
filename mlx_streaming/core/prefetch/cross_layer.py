"""跨层专家预取：在 attention/GDN 之前用本层输入预测目标层 MoE 路由并提前取专家。

通过 monkeypatch `Qwen3NextDecoderLayer.__call__`，在真正进入 MoE 计算前用
gate_L(post_attention_layernorm_L(h)) 预测目标层专家（对齐 recall≈0.95 的探针口径），
把 top-(top_k*mult) 专家在当前层计算窗口内异步预取，藏住读盘/物化延迟。

支持多条预取后端（按环境开关择一）：STREAM_BLOB_BG（后台物化进池）、STREAM_BLOB
（blob 字节预读）、STREAM_BLOB_LOADER（blob-loader 字节预热）、native stage 预取。
"""
import time

import mlx.core as mx

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
        mlp = getattr(self, "mlp", None)
        if mlp is not None and getattr(getattr(mlp, "store", None), "_staging", None) is not None:
            # 存本层「未归一化」decoder 输入：供 native-fused-prefetch 用目标层 norm 正确预测
            # （对齐 0.95-recall 探针；MoE forward 拿到的 x 已被本层 norm 过，norm 用错会掉点）。
            mlp._unnormed_input = x
        if config.cross_layer_prefetch() and config.resident_pool_enabled():
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
