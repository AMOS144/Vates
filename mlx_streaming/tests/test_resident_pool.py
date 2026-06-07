import mlx.core as mx
import pytest

from mlx_streaming.expert_store import ResidentExpertPool


def _loader_factory():
    # 每个专家是可区分的小张量：weight 全 e
    def load(layer, e):
        return {"weight": mx.full((4, 3), float(e))}
    return load


def test_acquire_miss_then_hit_slots_stable():
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    arrs1, slots1 = pool.acquire(0, [2, 5])      # 2 miss
    arrs2, slots2 = pool.acquire(0, [2, 5])      # 2 hit，槽位不变
    assert slots1 == slots2
    assert pool.misses == 2 and pool.hits == 2
    # 池里对应槽位内容 == 专家值
    assert mx.array_equal(arrs1["weight"][slots1[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs1["weight"][slots1[1]], mx.full((4, 3), 5.0)).item()


def test_acquire_lru_evicts_least_recent():
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.acquire(0, [0])          # slot for 0
    pool.acquire(0, [1])          # slot for 1
    pool.acquire(0, [0])          # touch 0 -> 1 是最久未用
    pool.acquire(0, [2])          # 淘汰 1，复用其槽
    assert pool.resident_count(0) == 2
    # 专家 1 应已被淘汰：再取触发 miss
    m0 = pool.misses
    pool.acquire(0, [1])
    assert pool.misses == m0 + 1


def test_per_layer_isolation():
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.acquire(0, [0, 1])
    pool.acquire(1, [0, 1])       # 不挤掉 layer0
    assert pool.resident_count(0) == 2 and pool.resident_count(1) == 2


def test_capacity_must_cover_topk():
    pool = ResidentExpertPool(capacity=1, loader=_loader_factory())
    with pytest.raises(ValueError):
        pool.acquire(0, [0, 1])   # 一次请求 2 个专家但容量 1


def test_layer_caps_allocate_per_layer_budget():
    # 全局容量 256，但 profile 给 layer0 只 32 槽 → 该层池物理只分配 32(无损省内存)
    pool = ResidentExpertPool(capacity=256, loader=_loader_factory(),
                              layer_caps={0: 32})
    arrs, slots = pool.acquire(0, [1, 2, 3])
    assert pool.cap_for(0) == 32
    assert pool.allocated_slots(0) == 32          # 一次性按预算分配，非 256
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 1.0)).item()


def test_layer_without_cap_uses_global_capacity():
    pool = ResidentExpertPool(capacity=64, loader=_loader_factory(),
                              layer_caps={0: 16})
    pool.acquire(1, [0, 1])                       # layer1 无 profile → 用全局 64
    assert pool.cap_for(1) == 64
    assert pool.allocated_slots(1) == 64


def test_layer_cap_evicts_within_its_budget():
    pool = ResidentExpertPool(capacity=256, loader=_loader_factory(),
                              layer_caps={0: 2})
    pool.acquire(0, [0])
    pool.acquire(0, [1])
    pool.acquire(0, [0])                          # 触摸 0 → 1 最久未用
    pool.acquire(0, [2])                          # 超预算(2) → 淘汰 1
    assert pool.resident_count(0) == 2
    m0 = pool.misses
    pool.acquire(0, [1])                          # 1 已淘汰 → miss
    assert pool.misses == m0 + 1


def test_layer_cap_caps_at_global_capacity():
    # profile 给的预算超过全局 capacity 时，以 capacity 为准
    pool = ResidentExpertPool(capacity=8, loader=_loader_factory(),
                              layer_caps={0: 999})
    assert pool.cap_for(0) == 8
    with pytest.raises(ValueError):
        pool.acquire(0, list(range(9)))           # 9 > 该层容量 8


from mlx_streaming.expert_store import FileExpertStore


def test_file_store_acquire_matches_disk(tmp_path):
    d = str(tmp_path)
    for e in range(4):
        mx.save_safetensors(f"{d}/layer00_expert{e:03d}.safetensors",
                            {"weight": mx.full((4, 3), float(e))})
    store = FileExpertStore(d, capacity=4)
    arrs, slots = store.acquire(0, [1, 3])
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 1.0)).item()
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 3.0)).item()
    arrs2, slots2 = store.acquire(0, [1, 3])   # 命中
    assert slots == slots2
    assert store.hits >= 2


from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU


def _quant_glu(num_experts, hidden, inter, group=64, bits=4):
    glu = SwitchGLU(hidden, inter, num_experts)
    glu.gate_proj = QuantizedSwitchLinear(hidden, inter, num_experts, bias=False, group_size=group, bits=bits)
    glu.up_proj = QuantizedSwitchLinear(hidden, inter, num_experts, bias=False, group_size=group, bits=bits)
    glu.down_proj = QuantizedSwitchLinear(inter, hidden, num_experts, bias=False, group_size=group, bits=bits)
    return glu


def test_pool_forward_bit_equiv_to_stack():
    hidden, inter, E = 64, 128, 8
    full = _quant_glu(E, hidden, inter)
    mx.eval(full.parameters())
    x = mx.random.normal((1, 1, hidden))
    routed = [2, 5]                      # 该 token 路由的专家

    # stack 路径：把专家 2,5 切出来，local=[0,1]
    def slice_qsl(lin, ids):
        new = QuantizedSwitchLinear(lin.input_dims, lin.output_dims, len(ids),
                                    bias=False, group_size=lin.group_size, bits=lin.bits)
        new.update({k: v[mx.array(ids)] for k, v in lin.parameters().items()})
        return new
    stack = SwitchGLU(hidden, inter, len(routed))
    stack.gate_proj = slice_qsl(full.gate_proj, routed)
    stack.up_proj = slice_qsl(full.up_proj, routed)
    stack.down_proj = slice_qsl(full.down_proj, routed)
    y_stack = stack(mx.expand_dims(x, -2), mx.array([[[0, 1]]]))

    # pool 路径：直接用 full 作为「池」(num_experts=E)，slot=路由 id
    y_pool = full(mx.expand_dims(x, -2), mx.array([[routed]]))

    assert mx.allclose(y_stack, y_pool, atol=1e-6).item()
