import mlx.core as mx

from mlx_streaming.expert_store import LruExpertStore, FileExpertStore


def _fake_stacked(num_experts=8, out=16, inp=12):
    return {"weight": mx.arange(num_experts * out * inp).reshape(num_experts, out, inp).astype(mx.float32)}


def test_fetch_returns_only_requested_experts():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=4)
    got = store.fetch(layer=0, expert_ids=[2, 5])
    assert got["weight"].shape[0] == 2
    assert mx.array_equal(got["weight"][0], _fake_stacked()["weight"][2]).item()


def test_lru_hits_and_misses():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=4)
    store.fetch(layer=0, expert_ids=[1, 2])     # 2 misses
    store.fetch(layer=0, expert_ids=[1, 2])     # 2 hits
    assert store.misses == 2
    assert store.hits == 2


def test_fetch_reuses_recent_stacked_tuple():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=4, stack_cache_size=1)
    got1 = store.fetch(layer=0, expert_ids=[1, 2])
    got2 = store.fetch(layer=0, expert_ids=[1, 2])
    got3 = store.fetch(layer=0, expert_ids=[2, 3])

    assert got1 is got2
    assert got3 is not got1


def test_capacity_evicts():
    store = LruExpertStore(stacked={0: _fake_stacked()}, capacity=2)
    store.fetch(layer=0, expert_ids=[0, 1])
    store.fetch(layer=0, expert_ids=[2, 3])     # 同层触发驱逐
    assert store.resident_count() <= 2          # 每层槽数=2


def test_per_layer_isolation():
    # 每层独立 LRU：不同层互不驱逐；同层重复取应命中
    store = LruExpertStore(stacked={0: _fake_stacked(), 1: _fake_stacked()}, capacity=2)
    store.fetch(layer=0, expert_ids=[0, 1])     # 2 miss
    store.fetch(layer=1, expert_ids=[0, 1])     # 2 miss（不挤掉 layer0）
    assert store.resident_count() == 4          # 两层各 2 个
    store.fetch(layer=0, expert_ids=[0, 1])     # layer0 仍在 → 2 hit
    assert store.hits == 2
    assert store.misses == 4


def test_record_hot_and_pin(tmp_path):
    d = str(tmp_path)
    for e in range(4):
        mx.save_safetensors(f"{d}/layer00_expert{e:03d}.safetensors",
                            {"weight": mx.full((2, 2), float(e))})
    store = FileExpertStore(d, capacity=1, record=True)
    store.fetch(0, [0, 1, 0, 1, 0])              # 频率：0->3, 1->2
    assert store.hot(0, 1) == [0]                 # 最热是专家 0

    store.pin(0, store.hot(0, 1))                 # 钉住专家 0
    store.record = False
    store.reset_stats()

    store.fetch(0, [0])                           # 命中 pinned，不读盘
    assert store.hits == 1 and store.misses == 0
    store.fetch(0, [2])                            # 冷专家走 LRU miss
    assert store.misses == 1
    assert store.pinned_count() == 1              # pinned 永不驱逐
