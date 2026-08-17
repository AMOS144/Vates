import mlx.core as mx

from mlx_streaming.core.moe.block import FileStreamingMoeBlock
from mlx_streaming.core.moe.custom_kernel import deterministic_route_reduce


class _ReusingStore:
    """Tiny mutable-slot store which deliberately reuses rows per group."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.loads = []
        self._blob_loader = self

    def load_experts_stacked(self, layer, experts):
        del layer
        experts = [int(v) for v in experts]
        self.loads.append(tuple(experts))
        return {"slot_experts": mx.array(experts)}


class _DeterministicSub:
    def forward(self, fetched, n, x, local):
        del n
        # One easily checked expert function: y_e(x) = x + (e + 1).
        expert = fetched["slot_experts"][local]
        return x[..., None, :] + (expert[..., None] + 1).astype(x.dtype)

    def release_bound(self):
        pass


def test_expert_major_matches_token_major_weighted_sum():
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


def test_expert_major_duplicate_destinations_are_bit_deterministic():
    """Multiple routed experts for one token must not race in one atomic add."""
    block = object.__new__(FileStreamingMoeBlock)
    block.layer_idx = 0
    block.store = _ReusingStore(capacity=2)
    block._sub = _DeterministicSub()

    tokens, routes = 257, 4
    x = mx.arange(tokens * 2, dtype=mx.float32).reshape(1, tokens, 2) / 17
    base = mx.arange(tokens * routes, dtype=mx.uint32).reshape(
        1, tokens, routes,
    )
    inds = base % 4
    scores = mx.array([0.1, 0.2, 0.3, 0.4]).reshape(1, 1, routes)
    scores = mx.broadcast_to(scores, inds.shape)

    outputs = []
    for _ in range(4):
        actual = block._expert_major_forward(
            x, inds, scores, num_experts=4, layer_cap=2,
        )
        mx.eval(actual)
        outputs.append(actual)
    assert all(bool(mx.array_equal(outputs[0], value)) for value in outputs[1:])


def test_deterministic_route_reduce_uses_fixed_rank_order():
    previous = mx.array([[1.0, -1.0], [2.0, -2.0]], dtype=mx.float32)
    # Group-local rows arrive in expert-major order, while assignment positions
    # identify their original token-major route rank.
    assignment_pos = mx.array([3, 0, 2], dtype=mx.int32)
    weighted = mx.array(
        [[30.0, 300.0], [10.0, 100.0], [20.0, 200.0]],
        dtype=mx.float32,
    )
    actual = deterministic_route_reduce(
        previous, weighted, assignment_pos, routes_per_token=2,
    )
    expected = mx.array([[11.0, 99.0], [52.0, 498.0]], dtype=mx.float32)
    mx.eval(actual)
    assert bool(mx.array_equal(actual, expected))


def test_deterministic_route_reduce_matches_bfloat16_scatter_chain_bits():
    tokens, routes, hidden = 17, 4, 8
    previous = (
        mx.arange(tokens * hidden, dtype=mx.uint32).reshape(tokens, hidden)
        % 31
    ).astype(mx.bfloat16) / 31
    assignment_pos = (
        mx.arange(37, dtype=mx.int32) * 11
    ) % (tokens * routes)
    weighted = (
        mx.arange(37 * hidden, dtype=mx.uint32).reshape(37, hidden) % 43
    ).astype(mx.bfloat16) / 43

    host = [int(value) for value in assignment_pos.tolist()]
    reference = previous
    for rank in range(routes):
        local = [i for i, value in enumerate(host) if value % routes == rank]
        if local:
            reference = reference.at[
                mx.array([host[i] // routes for i in local], dtype=mx.int32)
            ].add(weighted[mx.array(local, dtype=mx.int32)])

    actual = deterministic_route_reduce(
        previous, weighted, assignment_pos, routes_per_token=routes,
    )
    mx.eval(reference, actual)
    assert bool(mx.array_equal(actual, reference))
