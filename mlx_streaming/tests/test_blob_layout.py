from mlx_streaming.prep.blob_layout import layout_for, BLOB_V1_AFFINE, BLOB_V2_MXFP4


def test_v2_mxfp4_stride_page_aligned():
    segs, stride = layout_for(BLOB_V2_MXFP4, hidden=4096, inter=2048, bits=4, group=32)
    assert stride == 13_369_344
    assert stride % 16384 == 0
    assert len(segs) == 6  # gate/up/down × (weight, scales)
    assert all(s[1] != "biases" for s in segs)


def test_v1_affine_has_biases():
    # v1 = affine：每 proj 3 段(weight/scales/biases)，共 9 段；不写死 stride 魔数，
    # 只校验结构 + 与手算一致（weight 4B/字 + scales/biases 各 2B/组）。
    segs, stride = layout_for(BLOB_V1_AFFINE, hidden=2048, inter=512, bits=4, group=64)
    assert len(segs) == 9
    assert sum(1 for s in segs if s[1] == "biases") == 3
    # 手算参考：gate/up(out=512,in=2048) + down(out=2048,in=512)
    def proj_nb(out_d, in_d):
        w = out_d * (in_d * 4 // 32) * 4
        sb = out_d * (in_d // 64) * 2
        return w + sb + sb
    expect = proj_nb(512, 2048) * 2 + proj_nb(2048, 512)
    assert stride == expect
