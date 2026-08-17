import mlx.core as mx

from mlx_streaming.core.moe.block import FileStreamingMoeBlock


class _ReusingStore:
    """Tiny mutable-slot store which deliberately reuses rows per group."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.loads = []

    def acquire(self, layer, experts):
        del layer
        experts = [int(v) for v in experts]
        self.loads.append(tuple(experts))
        slot_experts = experts + [0] * (self.capacity - len(experts))
        return {"slot_experts": mx.array(slot_experts)}, list(range(len(experts)))


class _DeterministicSub:
    def forward(self, fetched, n, x, local):
        del n
        # One easily checked expert function: y_e(x) = x + (e + 1).
        expert = fetched["slot_experts"][local]
        return x[..., None, :] + (expert[..., None] + 1).astype(x.dtype)


def test_expert_major_matches_token_major_weighted_sum(monkeypatch):
    monkeypatch.setenv("EXPERT_MAJOR_GROUP_EXPERTS", "2")
    block = object.__new__(FileStreamingMoeBlock)
    block.layer_idx = 0
    block.store = _ReusingStore(capacity=2)
    block._sub = _DeterministicSub()

    x = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    inds = mx.array([[[0, 3], [1, 0], [3, 2]]], dtype=mx.uint32)
    scores = mx.array([[[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]]])

    actual = block._expert_major_forward(
        x, inds, scores, num_experts=4, layer_cap=2,
    )
    mx.eval(actual)

    expected = []
    for token, token_routes, token_scores in zip(
        x[0].tolist(), inds[0].tolist(), scores[0].tolist(),
    ):
        expected.append([
            sum((value + expert + 1) * score
                for expert, score in zip(token_routes, token_scores))
            for value in token
        ])
    expected = mx.array([expected])

    assert bool(mx.allclose(actual, expected, rtol=1e-6, atol=1e-6))
    assert block.store.loads == [(0, 1), (2, 3)]

