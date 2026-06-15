"""模型 patch：把原生 MoE 块替换成流式块（文件后端 / 常驻切片版），并按需挂跨层预取。"""
from mlx_streaming import config
from mlx_streaming.core.moe.block import FileStreamingMoeBlock, StreamingMoeBlock
from mlx_streaming.core.prefetch.cross_layer import enable_cross_layer_prefetch


def patch_model_filebacked(model, store, hidden, moe_inter, group_size, bits,
                           rotated: bool = False, proj_bits: dict | None = None,
                           layer_proj_bits: dict | None = None):
    """把每个 MoE 块替换为 FileStreamingMoeBlock，并丢弃常驻的堆叠 switch_mlp。

    store：FileExpertStore（所有 MoE 层共用，按 (layer,expert) 缓存）。
    rotated：是否用 Hadamard 旋转前向（专家权重须为 rotate_requantize 产出的旋转版）。
    proj_bits：非空时走混合精度（逐 proj 不同 bit），专家须为对应混合重量化产出。
    layer_proj_bits：{绝对层号: {proj:bits}}，非空时逐层用各自 proj_bits（优先于 proj_bits），
        对应 requantize_dir_layered 产出。各层 QSL 用该层 bit，与流式存盘文件一一对应。
    返回被替换的层数。被替换后原 switch_mlp 不再被引用，惰性权重不会被物化。
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        layer._layer_idx = i
        layer._prefetch_model_ref = model
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            pb = layer_proj_bits.get(i, proj_bits) if layer_proj_bits else proj_bits
            # 捕获共享专家引用使其常驻（Qwen3-Next 有，Qwen3-MoE 无）
            layer.mlp = FileStreamingMoeBlock(
                gate=mlp.gate, top_k=mlp.top_k, norm_topk_prob=mlp.norm_topk_prob,
                store=store, layer_idx=i, hidden=hidden, moe_inter=moe_inter,
                group_size=group_size, bits=bits, rotated=rotated, proj_bits=pb,
                shared_expert=getattr(mlp, "shared_expert", None),
                shared_expert_gate=getattr(mlp, "shared_expert_gate", None),
            )
            layer.mlp._prefetch_model_ref = model  # 供 native-fused-prefetch 取下层 gate
            patched += 1
    if config.cross_layer_prefetch() or getattr(store, "_staging", None) is not None:
        enable_cross_layer_prefetch()
    return patched


def patch_model(model, store_factory=None):
    """把模型里每个 MoE 块（含 switch_mlp 与 gate 的块）替换成 StreamingMoeBlock。

    store_factory(layer_idx)->LruExpertStore，可为 None（仅做 uniq 切片、不接磁盘后端）。
    返回被替换的层数，便于校验确实命中了 MoE 层。
    """
    patched = 0
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "switch_mlp") and hasattr(mlp, "gate"):
            store = store_factory(i) if store_factory is not None else None
            layer.mlp = StreamingMoeBlock(mlp, layer_idx=i, store=store)
            patched += 1
    return patched
