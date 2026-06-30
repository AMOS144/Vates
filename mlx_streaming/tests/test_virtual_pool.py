from mlx_streaming.core.cache.virtual_pool import VirtualPool


def test_ahead_for_cutoff():
    vp = VirtualPool(num_layers=48, cutoff=6, ahead_lo=1, ahead_hi=3)
    assert vp.ahead_for(0) == 1
    assert vp.ahead_for(5) == 1
    assert vp.ahead_for(6) == 3
    assert vp.ahead_for(40) == 3


def test_target_for_skip_no_clamp():
    vp = VirtualPool(num_layers=48, cutoff=6, ahead_lo=1, ahead_hi=3)
    assert vp.target_for(0) == 1        # 0+1
    assert vp.target_for(10) == 13      # 10+3
    assert vp.target_for(44) == 47      # 44+3=47 恰好命中末层 → 预读
    assert vp.target_for(45) == 0       # 45+3=48 越界 → 跳过（不再 clamp 堆叠到 47）
    assert vp.target_for(46) == 0       # 46+3=49 越界 → 跳过
    assert vp.target_for(47) == 0       # 末层无可预读 → 0
