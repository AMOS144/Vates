import numpy as np

from mlx_streaming.tools.simulate_prefetch_budget import simulate_budgeted_prefetch_arrays


def test_budgeted_prefetch_picks_highest_scored_candidates():
    prompt_id = np.array([0, 0], dtype=np.int16)
    step = np.array([0, 0], dtype=np.int16)
    layer = np.array([0, 1], dtype=np.int16)
    y = np.array([
        [0, 1, 0, 1],
        [1, 0, 0, 0],
    ], dtype=np.uint8)
    candidate = np.array([
        [0, 1, 1, 1],
        [1, 1, 0, 0],
    ], dtype=np.uint8)
    score = np.array([
        [0.0, 0.9, 0.8, 0.1],
        [0.7, 0.6, 0.0, 0.0],
    ], dtype=np.float32)

    out = simulate_budgeted_prefetch_arrays(
        prompt_id, step, layer, y, candidate, score,
        global_budget=2, per_layer_budget=2)
    assert out["miss_total"] == 3
    assert out["prefetch_total"] == 2
    assert out["hit_total"] == 1
    assert out["waste_total"] == 1


def test_per_layer_budget_limits_single_layer_prefetches():
    prompt_id = np.array([0, 0], dtype=np.int16)
    step = np.array([0, 0], dtype=np.int16)
    layer = np.array([0, 1], dtype=np.int16)
    y = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=np.uint8)
    candidate = y.copy()
    score = np.array([
        [0.9, 0.8, 0.0, 0.0],
        [0.7, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    out = simulate_budgeted_prefetch_arrays(
        prompt_id, step, layer, y, candidate, score,
        global_budget=3, per_layer_budget=1)
    assert out["prefetch_total"] == 2
    assert out["hit_total"] == 2
    assert out["covered_miss_ratio"] == round(2 / 3, 4)
