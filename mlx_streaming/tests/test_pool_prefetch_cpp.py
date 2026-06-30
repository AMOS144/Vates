"""prefetch_cpp / _alloc_slot：C++ 直写预取与 demand fallback 共用槽分配器。"""
import mlx.core as mx

from mlx_streaming.core.cache.resident_pool import ResidentExpertPool


def _loader_factory():
    def load(layer, e):
        return {"weight": mx.full((4, 3), float(e))}
    return load


def _prebuilt_pool(cap=4):
    """建好满 cap 的池（模拟暖池：preallocate 后池张量已 eval、不再 grow）。"""
    pool = ResidentExpertPool(capacity=cap, loader=_loader_factory())
    pool.preallocate(0, {"weight": mx.zeros((4, 3))}, cap)
    pool._ensure_table(0, num_experts=16)
    return pool


def test_prefetch_cpp_allocates_distinct_slots_and_sets_table():
    pool = _prebuilt_pool(cap=4)
    placed = []
    pool.prefetch_cpp(0, [2, 5, 7], lambda slot, e: placed.append((slot, e)))
    slots = [s for s, _ in placed]
    assert sorted(e for _, e in placed) == [2, 5, 7]      # 三个未驻都被提交
    assert len(set(slots)) == 3                            # 槽互不冲突
    table = pool._slot_table[0]
    for slot, e in placed:
        assert int(table[e]) == slot                       # slot_table 已更新指向各自槽


def test_prefetch_cpp_skips_resident_and_touches_lru():
    pool = _prebuilt_pool(cap=4)
    pool.prefetch_cpp(0, [2], lambda slot, e: None)        # 先放 2
    placed = []
    pool.prefetch_cpp(0, [2, 5], lambda slot, e: placed.append((slot, e)))
    assert placed == [(pool._slot_of[0][5], 5)]            # 2 已驻被跳过，只提交 5
    assert pool._slot_of[0][2] is not None                 # 2 仍在池


def test_prefetch_cpp_then_place_expert_no_slot_conflict():
    # 预取占了一批槽后，demand fallback(_place_expert/acquire) 必须从剩余槽分配，绝不串台
    pool = _prebuilt_pool(cap=4)
    pool.prefetch_cpp(0, [2, 5], lambda slot, e: None)     # 占 2 个槽
    arrs, slots = pool.acquire(0, [2, 9])                  # 2 命中预取槽，9 走 fallback 取新槽
    assert slots[0] == pool._slot_of[0][2]                 # 2 用预取时的槽
    assert slots[1] != slots[0]                            # 9 不和 2 抢同一槽
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 9.0)).item()


def test_prefetch_cpp_evicts_lru_when_full_excluding_current():
    pool = _prebuilt_pool(cap=2)
    pool.prefetch_cpp(0, [0], lambda slot, e: None)        # slot for 0
    pool.prefetch_cpp(0, [1], lambda slot, e: None)        # slot for 1，池满
    pool.prefetch_cpp(0, [0], lambda slot, e: None)        # 触摸 0 → 1 最久未用
    placed = []
    pool.prefetch_cpp(0, [2], lambda slot, e: placed.append((slot, e)))  # 满 → 驱逐 LRU 最旧(1)
    assert 1 not in pool._slot_of[0]                        # 1 被驱逐
    assert 0 in pool._slot_of[0] and 2 in pool._slot_of[0]
    assert int(pool._slot_table[0][1]) == -1               # 表中 1 失效


def test_place_expert_refactor_regression():
    # _place_expert 改用 _alloc_slot 后，既有 acquire 行为不变
    pool = ResidentExpertPool(capacity=4, loader=_loader_factory())
    arrs, slots = pool.acquire(0, [2, 5])
    assert mx.array_equal(arrs["weight"][slots[0]], mx.full((4, 3), 2.0)).item()
    assert mx.array_equal(arrs["weight"][slots[1]], mx.full((4, 3), 5.0)).item()
    arrs2, slots2 = pool.acquire(0, [2, 5])
    assert slots == slots2 and pool.hits == 2
