import mlx.core as mx
import pytest

from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _loader_factory():
    # 每个专家是可区分的小张量：weight 全 e
    def load(layer, e):
        return {"weight": mx.full((4, 3), float(e))}
    return load


def test_direct_slot_layer0_uses_same_unified_capacity(monkeypatch):
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "1")
    monkeypatch.setenv("LAYER0_SLOTS", "64")
    pool = ResidentExpertPool(
        capacity=32,
        loader=_loader_factory(),
        spec_slots=32,
        spec_gens=1,
    )
    assert pool.native_real_cap_for(0) == 32
    assert pool.native_real_cap_for(1) == 32


def test_non_direct_layer0_keeps_real_capacity(monkeypatch):
    monkeypatch.setenv("PREFETCH_DIRECT_SLOTS", "0")
    pool = ResidentExpertPool(
        capacity=32,
        loader=_loader_factory(),
        spec_slots=32,
        spec_gens=1,
    )
    assert pool.native_real_cap_for(0) == 32


def test_batch_loader_equiv_to_serial():
    # batch_loader 路径应与逐专家串行加载得到完全相同的槽位与内容
    calls = []

    def batch(layer, ids):
        calls.append(list(ids))                       # 记录一次批量调用收到哪些 miss
        return {int(e): {"weight": mx.full((4, 3), float(e))} for e in ids}

    pool = ResidentExpertPool(capacity=8, loader=_loader_factory(), batch_loader=batch)
    arrs, slots = pool.acquire(0, [2, 5, 5, 9])        # 唯一 miss = {2,5,9}，一次批量读
    assert calls == [[2, 5, 9]]                        # 三个 miss 收成一批，仅一次调用
    assert pool.misses == 3 and pool.hits == 0         # uniq 去重后重复的 5 不单独计 hit（与串行一致）
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 5.0)).item()
    assert slots[1] == slots[2]                        # 重复专家共享同一槽
    assert mx.array_equal(arrs["weight"][slots[3]], mx.full((4, 3), 9.0)).item()
    # 第二次全命中，不再批量读
    pool.acquire(0, [2, 5, 9])
    assert calls == [[2, 5, 9]]


def test_write_slots_batch_writes_all_in_one_scatter():
    # 新增批量写:一次 stacked scatter 把多个专家写进各自槽位，内容正确、旧槽不动。
    pool = ResidentExpertPool(capacity=8, loader=_loader_factory())
    pool.acquire(0, [0])                                # 建池，槽 0 = 专家 0
    s7, _ = pool._alloc_slot(0, 7, {0, 7, 8})
    s8, _ = pool._alloc_slot(0, 8, {0, 7, 8})
    pool._write_slots_batch(0, [s7, s8],
                            [{"weight": mx.full((4, 3), 7.0)},
                             {"weight": mx.full((4, 3), 8.0)}])
    assert mx.array_equal(pool._pools[0]["weight"][s7], mx.full((4, 3), 7.0)).item()
    assert mx.array_equal(pool._pools[0]["weight"][s8], mx.full((4, 3), 8.0)).item()
    assert mx.array_equal(pool._pools[0]["weight"][0], mx.full((4, 3), 0.0)).item()


def test_acquire_batches_miss_writes():
    # demand 多 miss 走批量写：不再逐槽 _write_slot，每 key 仅一次 scatter。
    pool = ResidentExpertPool(capacity=8, loader=_loader_factory())
    calls = {"n": 0}
    orig = pool._write_slot

    def counted(layer, slot, expert):
        calls["n"] += 1
        return orig(layer, slot, expert)

    pool._write_slot = counted
    arrs, slots = pool.acquire(0, [2, 5, 9])           # 3 miss
    assert calls["n"] == 0                              # 批量路径不调用逐槽 _write_slot
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 5.0)).item()
    assert mx.array_equal(arrs["weight"][slots[2]], mx.full((4, 3), 9.0)).item()


def test_stacked_batch_loader_equiv_to_serial():
    # stacked_batch_loader 返回 {k:(N,*)}（批量读+批量物化），写池结果与逐专家等价。
    calls = []

    def stacked(layer, ids):
        calls.append(list(ids))
        return {"weight": mx.stack([mx.full((4, 3), float(e)) for e in ids], axis=0)}

    pool = ResidentExpertPool(capacity=8, loader=_loader_factory(),
                              stacked_batch_loader=stacked)
    arrs, slots = pool.acquire(0, [2, 5, 5, 9])        # 唯一 miss = {2,5,9}
    assert calls == [[2, 5, 9]]                        # 三个 miss 收成一批，一次调用
    assert pool.misses == 3 and pool.hits == 0
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 5.0)).item()
    assert slots[1] == slots[2]
    assert mx.array_equal(arrs["weight"][slots[3]], mx.full((4, 3), 9.0)).item()
    pool.acquire(0, [2, 5, 9])                          # 全命中，不再批量读
    assert calls == [[2, 5, 9]]


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


def test_layer_caps_cap_for_is_ceiling():
    # 全局容量 256，但 profile 给 layer0 上限 32 槽。grow-on-demand 下 cap_for 是「天花板」，
    # 物理分配按需增长且不超过它(只用 3 个专家 → 远小于 32)。
    pool = ResidentExpertPool(capacity=256, loader=_loader_factory(),
                              layer_caps={0: 32})
    arrs, slots = pool.acquire(0, [1, 2, 3])
    assert pool.cap_for(0) == 32                   # 天花板仍是 32
    assert pool.allocated_slots(0) <= 32           # 懒分配,不超天花板
    assert pool.allocated_slots(0) >= 3            # 至少够放本次 3 个
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 1.0)).item()


def test_layer_without_cap_uses_global_capacity():
    pool = ResidentExpertPool(capacity=64, loader=_loader_factory(),
                              layer_caps={0: 16})
    pool.acquire(1, [0, 1])                       # layer1 无 profile → 天花板=全局 64
    assert pool.cap_for(1) == 64
    assert pool.allocated_slots(1) <= 64          # 只用 2 个 → 物理远小于 64
    assert pool.allocated_slots(1) >= 2


def test_pool_allocates_lazily_and_grows():
    # 核心:容量 256,但只用少量专家时物理分配应远小于 256,工作集扩大时按需增长。
    pool = ResidentExpertPool(capacity=256, loader=_loader_factory())
    pool.acquire(0, [0, 1])                       # 仅 2 个专家
    assert pool.allocated_slots(0) < 256          # 不一次性预留 256
    assert pool.allocated_slots(0) >= 2
    for e in range(2, 50):                        # 工作集扩到 50
        pool.acquire(0, [e])
    assert pool.resident_count(0) == 50           # 50 < 256 → 全驻,零淘汰
    assert 50 <= pool.allocated_slots(0) <= 256   # 增长到够用,封顶 256


def test_grow_preserves_resident_data_and_slots():
    # 增长(扩容池张量)后,已驻专家的 slot 与数据必须不变
    pool = ResidentExpertPool(capacity=256, loader=_loader_factory())
    arrs0, sl0 = pool.acquire(0, [3])             # 装入专家 3
    slot3 = sl0[0]
    for e in range(10, 60):                       # 触发多次增长
        pool.acquire(0, [e])
    arrs1, sl1 = pool.acquire(0, [3])             # 命中
    assert sl1[0] == slot3                         # slot 不变
    assert mx.array_equal(arrs1["weight"][slot3], mx.full((4, 3), 3.0)).item()


def test_grows_before_evicting_within_capacity():
    # 容量内绝不淘汰:装满 capacity 个不同专家后重取应全命中
    pool = ResidentExpertPool(capacity=16, loader=_loader_factory())
    for e in range(16):
        pool.acquire(0, [e])
    assert pool.resident_count(0) == 16
    m = pool.misses
    for e in range(16):
        pool.acquire(0, [e])                      # 全在池里
    assert pool.misses == m                        # 零新 miss(未淘汰)


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


def test_pinned_expert_survives_lru_eviction():
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.pin(0, [0])
    pool.acquire(0, [1])
    pool.acquire(0, [2])                          # 只能淘汰非 pinned 的 1
    assert pool.resident_count(0) == 2
    m0 = pool.misses
    arrs, slots = pool.acquire(0, [0])            # pinned 的 0 仍在池里
    assert pool.misses == m0
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 0.0)).item()


def test_prefetch_populates_pool_without_counting_miss():
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    pool.prefetch(0, [2, 5])
    assert pool.misses == 0 and pool.hits == 0
    m0 = pool.misses
    arrs, slots = pool.acquire(0, [2, 5])
    assert pool.misses == m0
    assert pool.hits == 2
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 2.0)).item()


def test_lfu_policy_keeps_frequent_expert_over_lru_oldest(monkeypatch):
    monkeypatch.setenv("EVICT_POLICY", "lfu")
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.acquire(0, [0])
    pool.acquire(0, [1])
    pool.acquire(0, [0])
    pool.acquire(0, [0])
    pool.acquire(0, [1])                          # 此时 0 更高频，但比 1 更老
    pool.acquire(0, [2])                          # LFU 应淘汰低频的 1，而不是 LRU 最老的 0
    m0 = pool.misses
    pool.acquire(0, [0])
    assert pool.misses == m0


def test_lfu_decay_off_by_default(monkeypatch):
    # 默认 LFU_DECAY_INTERVAL=0 → 频率持续累加,绝不减半
    monkeypatch.setenv("EVICT_POLICY", "lfu")
    monkeypatch.delenv("LFU_DECAY_INTERVAL", raising=False)
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    assert pool.lfu_decay_interval == 0
    for _ in range(20):
        pool.acquire(0, [0])
    assert pool._freq[0][0] == 20


def test_lfu_decay_when_interval_positive(monkeypatch):
    # 显式开启 interval=4 时,第 4 次访问触发减半 4→2
    monkeypatch.setenv("EVICT_POLICY", "lfu")
    monkeypatch.setenv("LFU_DECAY_INTERVAL", "4")
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    for _ in range(4):
        pool.acquire(0, [0])
    assert pool._freq[0][0] == 2


def test_acquire_gpu_counts_frequency_for_lfu(monkeypatch):
    # 全命中 GPU 快路径也要计频(否则 LFU 在稳态 decode 失效)
    monkeypatch.setenv("EVICT_POLICY", "lfu")
    monkeypatch.delenv("LFU_DECAY_INTERVAL", raising=False)
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    pool.acquire(0, [2, 5])                       # 驻留 2,5,freq 各 1
    for _ in range(3):                            # 3 次全命中快路径
        pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert pool._freq[0][2] == 4 and pool._freq[0][5] == 4


def test_acquire_gpu_no_freq_when_lru(monkeypatch):
    # LRU 默认路径不应触碰 _freq(零开销、行为不变)
    monkeypatch.setenv("EVICT_POLICY", "lru")
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    pool.acquire(0, [2, 5])
    pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert sum(pool._freq.get(0, {}).values()) == 0


def test_layer_cap_caps_at_global_capacity():
    # profile 给的预算超过全局 capacity 时，以 capacity 为准
    pool = ResidentExpertPool(capacity=8, loader=_loader_factory(),
                              layer_caps={0: 999})
    assert pool.cap_for(0) == 8
    with pytest.raises(ValueError):
        pool.acquire(0, list(range(9)))           # 9 > 该层容量 8


from mlx_streaming.core.cache.expert_store import FileExpertStore


def test_acquire_gpu_matches_host_slots():
    # 预热后,GPU 侧 slot 重映射须与 host acquire 给出相同 slot、且不产生新 miss
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    _, slots = pool.acquire(0, [2, 5])
    m0 = pool.misses
    arrs, local = pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert pool.misses == m0
    assert local.reshape(-1).tolist() == slots
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 2.0)).item()


def test_acquire_gpu_loads_on_miss_then_hits():
    # 首次全 miss → acquire_gpu 回退加载并更新表;再取全命中
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    arrs, local = pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert pool.misses == 2
    sl = local.reshape(-1).tolist()
    assert mx.array_equal(arrs["weight"][sl[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs["weight"][sl[1]], mx.full((4, 3), 5.0)).item()
    m0 = pool.misses
    pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert pool.misses == m0


def test_acquire_gpu_counts_fastpath_and_fallback():
    # 计数器:全 miss → 回退 host(fallback);随后全命中 → GPU 快路径(fastpath)
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert pool.gpu_fastpath == 0 and pool.gpu_fallback == 1
    pool.acquire_gpu(0, mx.array([[[2, 5]]]), num_experts=8)
    assert pool.gpu_fastpath == 1 and pool.gpu_fallback == 1


def test_acquire_gpu_reflects_host_eviction():
    # host 驱逐某专家后,GPU 查找表对应项须失效(再 GPU 取触发 miss)
    pool = ResidentExpertPool(capacity=2, loader=_loader_factory())
    pool.acquire_gpu(0, mx.array([[[0, 1]]]), num_experts=8)   # 装 0、1,建表
    pool.acquire(0, [1])          # host 触摸 1 → 0 最久未用
    pool.acquire(0, [2])          # 淘汰 0 → 表中 0 应失效
    m0 = pool.misses
    pool.acquire_gpu(0, mx.array([[[0]]]), num_experts=8)      # 0 已淘汰 → miss
    assert pool.misses == m0 + 1


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


def test_file_store_pin_populates_resident_gpu_path(tmp_path):
    d = str(tmp_path)
    for e in range(4):
        mx.save_safetensors(f"{d}/layer00_expert{e:03d}.safetensors",
                            {"weight": mx.full((4, 3), float(e))})
    store = FileExpertStore(d, capacity=2)
    store.pin(0, [0])
    arrs, local = store.acquire_gpu(0, mx.array([[[0]]]), num_experts=4)
    slot = int(local.reshape(-1).tolist()[0])
    assert store.misses == 0
    assert mx.array_equal(arrs["weight"][slot], mx.full((4, 3), 0.0)).item()


def test_file_store_prefetch_populates_resident_gpu_path(tmp_path):
    d = str(tmp_path)
    for e in range(4):
        mx.save_safetensors(f"{d}/layer00_expert{e:03d}.safetensors",
                            {"weight": mx.full((4, 3), float(e))})
    store = FileExpertStore(d, capacity=4)
    store.prefetch(0, [2])
    arrs, local = store.acquire_gpu(0, mx.array([[[2]]]), num_experts=4)
    slot = int(local.reshape(-1).tolist()[0])
    assert store.misses == 0
    assert mx.array_equal(arrs["weight"][slot], mx.full((4, 3), 2.0)).item()


def test_file_store_async_prefetch_buffer_hits_without_resident_pollution(tmp_path, monkeypatch):
    monkeypatch.setenv("ASYNC_PREFETCH", "1")
    monkeypatch.setenv("PREFETCH_WORKERS", "1")
    d = str(tmp_path)
    for e in range(4):
        mx.save_safetensors(f"{d}/layer00_expert{e:03d}.safetensors",
                            {"weight": mx.full((4, 3), float(e))})
    store = FileExpertStore(d, capacity=4)
    store.prefetch(0, [2])
    assert store._resident.resident_count(0) == 0
    store.wait_prefetch()
    arrs, local = store.acquire_gpu(0, mx.array([[[2]]]), num_experts=4)
    slot = int(local.reshape(-1).tolist()[0])
    assert store.misses == 0
    assert store.prefetch_buffer_hits == 1
    assert mx.array_equal(arrs["weight"][slot], mx.full((4, 3), 2.0)).item()


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
