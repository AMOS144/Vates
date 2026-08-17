import os
import tempfile

import mlx.core as mx
import numpy as np

import mlx_streaming.native_moe_ext as N


def test_bg_reader_reads_correct_bytes():
    # 造一个已知字节的临时 blob：4 个 "专家"，每个 stride 字节、内容为其编号
    stride = 4096
    ne = 4
    blob = b"".join(bytes([e]) * stride for e in range(ne))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".blob") as f:
        f.write(blob)
        path = f.name
    try:
        dst = mx.zeros((3, stride), dtype=mx.uint8)
        mx.eval(dst)                                  # 物化，.data() 可用
        N.bg_reader_start(1)
        # 把专家 2→行0、专家0→行1、专家3→行2 读进 dst
        N.bg_reader_submit(dst, [2, 0, 3], [0, 1, 2], path, stride, 100)
        N.bg_reader_wait(100)
        # wait consumes the completion ticket; ready only reports an
        # unconsumed completion and must be false after wait returns.
        assert N.bg_reader_ready(100) is False
        arr = np.array(dst)                           # 读 dst 当前 buffer（应见 C++ 写入的字节）
        assert int(arr[0, 0]) == 2 and int(arr[0, -1]) == 2
        assert int(arr[1, 0]) == 0
        assert int(arr[2, 0]) == 3
    finally:
        N.bg_reader_stop()
        os.unlink(path)
