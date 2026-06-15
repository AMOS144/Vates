import os

import numpy as np
import pytest

from mlx_streaming.prep.repack_expert_blobs import _layout, repack_layer

SRC = "/tmp/cb_2bit_g128"


@pytest.mark.skipif(not os.path.exists(os.path.join(SRC, "layer15.gate_proj.weight.bin")),
                    reason="需要 /tmp/cb_2bit_g128 源 compute buffer")
def test_blob_roundtrip_matches_compute_buffer(tmp_path, monkeypatch):
    monkeypatch.setenv("BLOB_DIR", str(tmp_path))
    # repack_layer 读 BLOB_DIR 环境变量在 import 时已固定，故直接传 OUT 通过 monkeypatch 重载
    import importlib
    import mlx_streaming.prep.repack_expert_blobs as mod
    importlib.reload(mod)
    idx = mod.repack_layer(15)
    segs, stride = mod._layout()
    assert idx["stride"] == stride
    assert idx["page_aligned"]

    blob_path = os.path.join(str(tmp_path), "layer15.blob")
    assert os.path.getsize(blob_path) == stride * mod.NUM_EXPERTS

    # 第一段是 gate.weight：blob 里 expert 0 的前 nbytes 应等于源 compute buffer 前 nbytes
    gate_w_nbytes = segs[0][2]
    with open(blob_path, "rb") as f:
        raw = f.read(gate_w_nbytes)
    src = np.memmap(os.path.join(SRC, "layer15.gate_proj.weight.bin"), dtype=np.uint8, mode="r")
    assert raw == src[:gate_w_nbytes].tobytes()
