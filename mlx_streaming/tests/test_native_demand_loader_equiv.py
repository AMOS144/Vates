"""native demand loader 等价性:C++ load_experts_native 与 numpy load_experts_stacked
在同一份 blob 字节上必须逐字节产出一致的专家张量。

落地 NATIVE_DEMAND_LOADER 默认开的正确性闸门:native 把字节物化从 numpy(frombuffer+stack)
换成 C++ blob_load(pread 直进 MLX 数组),数值必须完全等价。需 native 扩展;未编译则 skip。
"""
import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_streaming.core.cache.blob_loader import BlobExpertSource

native_ext = pytest.importorskip("mlx_streaming.native_moe_ext")


def _write_synthetic_blob(blob_dir, num_experts, stride, seed=0):
    """写一个 layer00.blob:num_experts × stride 随机字节,两条 loader 路径读同一文件。"""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=num_experts * stride, dtype=np.uint8).tobytes()
    with open(os.path.join(blob_dir, "layer00.blob"), "wb") as f:
        f.write(data)


def test_native_matches_numpy_stacked_mxfp4(tmp_path):
    ne = 8
    src = BlobExpertSource(str(tmp_path), hidden=64, inter=64, group=32, bits=4,
                           num_experts=ne, nocache=True, quant_mode="mxfp4")
    try:
        _write_synthetic_blob(str(tmp_path), ne, src.stride)
        ids = [3, 5, 1]
        stacked = src.load_experts_stacked(0, ids, view_bf16=True)   # {k:(N,*shape)}
        native = src.load_experts_native(0, ids, view_bf16=True)     # {e:{k:arr}}
        mx.eval(stacked, native)
        for i, e in enumerate(ids):
            for k in stacked:
                a = stacked[k][i]
                b = native[e][k]
                assert a.dtype == b.dtype, f"dtype 不一致 {k}: {a.dtype} vs {b.dtype}"
                assert a.shape == b.shape, f"shape 不一致 {k}: {a.shape} vs {b.shape}"
                assert mx.array_equal(a, b).item(), f"字节不一致 expert={e} key={k}"
    finally:
        src.close()
