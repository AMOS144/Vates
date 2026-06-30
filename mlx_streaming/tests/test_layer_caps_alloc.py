from mlx_streaming.tools.profile_layer_caps import allocate_caps


def test_allocate_prefers_high_value_layer():
    # 层0 加槽收益大,层1 收益小;预算 3 应优先喂层0,达到最优总 hit=1.0
    htab = {0: [0.0, 0.5, 0.9, 1.0], 1: [0.0, 0.1, 0.15, 0.2]}
    caps = allocate_caps(htab, budget=3, floor=0)
    assert caps[0] + caps[1] <= 3
    assert htab[0][caps[0]] + htab[1][caps[1]] == 1.0      # (3,0) 或 (2,1) 都达 1.0
    assert caps[0] >= 2                                     # 预算明显偏向高价值层


def test_allocate_respects_floor():
    # floor=1 时每层至少 1 槽,即便层1 几乎无收益
    htab = {0: [0.0, 0.5, 0.9, 1.0], 1: [0.0, 0.1, 0.15, 0.2]}
    caps = allocate_caps(htab, budget=3, floor=1)
    assert caps[0] >= 1 and caps[1] >= 1
    assert caps[0] + caps[1] <= 3


def test_allocate_caps_at_layer_cmax():
    # 某层工作集小(cmax=1),不应分到超过其曲线长度的 cap
    htab = {0: [0.0, 0.3], 1: [0.0, 0.2, 0.6, 0.95]}
    caps = allocate_caps(htab, budget=4, floor=0)
    assert caps[0] <= 1                                     # 层0 曲线只到 c=1
    assert caps[0] + caps[1] <= 4
