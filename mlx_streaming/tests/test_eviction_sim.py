from mlx_streaming.tools.simulate_eviction import simulate_layer_events


def test_belady_evicts_farthest_future_use():
    events = [[0], [1], [2], [0], [1], [2]]
    lru = simulate_layer_events(events, capacity=2, policy="lru")
    belady = simulate_layer_events(events, capacity=2, policy="belady")
    assert lru["misses"] == 6
    assert belady["misses"] == 4


def test_window_policy_avoids_near_future_experts():
    events = [[0], [1], [2], [0], [3], [1]]
    lru = simulate_layer_events(events, capacity=2, policy="lru")
    win = simulate_layer_events(events, capacity=2, policy="window", window=2)
    assert lru["misses"] == 6
    assert win["misses"] == 5


def test_avoid_policy_preserves_hint_candidates():
    events = [[0], [1], [2], [0]]
    avoid = [set(), set(), {0}, set()]
    lru = simulate_layer_events(events, capacity=2, policy="lru")
    hinted = simulate_layer_events(events, capacity=2, policy="avoid", avoid_events=avoid)
    assert lru["misses"] == 4
    assert hinted["misses"] == 3


def test_2q_promotes_reused_experts_to_protected_queue():
    events = [[0], [1], [0], [2], [3], [0]]
    lru = simulate_layer_events(events, capacity=2, policy="lru")
    twoq = simulate_layer_events(events, capacity=2, policy="2q")
    assert lru["misses"] == 5
    assert twoq["misses"] == 4


def test_pinned_experts_are_never_evicted():
    events = [[0], [1], [2], [0]]
    pinned = simulate_layer_events(events, capacity=2, policy="lru", pinned={0})
    assert pinned["misses"] == 2
    assert pinned["hits"] == 2
