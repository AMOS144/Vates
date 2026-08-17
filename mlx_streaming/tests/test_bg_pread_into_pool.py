import os
import tempfile

import mlx.core as mx
import numpy as np

import mlx_streaming.native_moe_ext as N


def test_bg_pread_into_pool_writes_segments_to_slot():
    # 两段：seg0=1024B、seg1=512B，stride=1536；blob 含 4 个专家，专家 e 的字节全为 e
    nb = [1024, 512]
    stride = sum(nb)
    ne = 4
    blob = b"".join(bytes([e]) * stride for e in range(ne))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".blob") as f:
        f.write(blob)
        path = f.name
    try:
        cap = 5
        # 池两段张量（uint8 简化；真实池 dtype 不影响字节拷贝）
        t0 = mx.zeros((cap, nb[0]), dtype=mx.uint8)
        t1 = mx.zeros((cap, nb[1]), dtype=mx.uint8)
        mx.eval(t0, t1)
        N.bg_reader_start(1)
        # 把专家 3 写进 slot 2
        N.bg_pread_into_pool([t0, t1], [0, nb[0]], nb, 2, 3, path, stride, 777)
        N.bg_reader_wait(777)
        assert N.bg_reader_ready(777) is False
        a0, a1 = np.array(t0), np.array(t1)
        assert int(a0[2, 0]) == 3 and int(a0[2, -1]) == 3   # slot2 段0 全是 3
        assert int(a1[2, 0]) == 3 and int(a1[2, -1]) == 3   # slot2 段1 全是 3
        assert int(a0[0, 0]) == 0 and int(a1[1, 0]) == 0    # 其它槽不动
    finally:
        N.bg_reader_stop()
        os.unlink(path)
