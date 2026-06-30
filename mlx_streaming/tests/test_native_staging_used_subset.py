import mlx.core as mx
from mlx_streaming.core.prefetch.native_staging import route_used_subset


def test_basic_subset():
    # 预读候选 [1,5,9]，真实路由 [5,9,3] → 命中 {5,9}
    assert route_used_subset([1, 5, 9], mx.array([[5, 9, 3]]), num_experts=16) == {5, 9}


def test_empty_cand_returns_empty():
    # 空候选不触发 GPU op，返回空集
    assert route_used_subset([], mx.array([1, 2]), num_experts=8) == set()


def test_no_hit_returns_empty():
    # 候选 [2] 不在路由 [7,8] 中 → 空集
    assert route_used_subset([2], mx.array([7, 8]), num_experts=16) == set()


def test_duplicate_route_idempotent():
    # 路由含重复 id，mask 幂等，不影响结果
    assert route_used_subset([3, 4], mx.array([3, 3, 3]), num_experts=8) == {3}


def test_multidim_route_inds():
    # route_inds 多维(decode 形如 (1,1,k)),内部 reshape(-1)
    assert route_used_subset([0, 6], mx.array([[[6, 6, 0]]]), num_experts=8) == {0, 6}


import sys
import types
from mlx_streaming.core.prefetch.native_staging import NativeStagingManager


class _FakeSrc:
    # 单段 weight uint32 (1,1)=4 字节,stride=4,使 _slice 能跑
    def __init__(self):
        self._segs = [("gate_proj", "weight", "uint32", (1, 1), 4)]
        self.stride = 4
        self.dir = "/tmp"


class _FakeResident:
    def __init__(self):
        self.placed = []

    def _ensure_layer(self, layer):
        pass

    def _place_expert(self, layer, e, d, current=None):
        self.placed.append(int(e))
        return 0


class _FakeStore:
    def __init__(self):
        self._resident = _FakeResident()

    def resident_experts(self, layer):
        return set()


def _patch_take(monkeypatch, flat):
    import mlx_streaming
    fake = types.ModuleType("mlx_streaming.native_moe_ext")
    fake.prefetch_staging_take = lambda layer: flat
    monkeypatch.setitem(sys.modules, "mlx_streaming.native_moe_ext", fake)
    monkeypatch.setattr(mlx_streaming, "native_moe_ext", fake, raising=False)


def test_promote_gpu_path_filters_false_positives(monkeypatch):
    mgr = NativeStagingManager(_FakeSrc(), budget=4)
    gen = 7
    mgr._gen_buf[gen] = mx.zeros((4, 4), dtype=mx.uint8)
    # 预读 {5,9,2}（flat = [gen, e,row, e,row, ...]）
    _patch_take(monkeypatch, [gen, 5, 0, 9, 1, 2, 2])
    store = _FakeStore()
    # 本层真实路由只含 5、9（2 是假阳性，应被过滤）
    mgr.promote(0, store, route_inds=mx.array([[5, 9]]), num_experts=16)
    assert set(store._resident.placed) == {5, 9}


def test_promote_explicit_used_takes_precedence(monkeypatch):
    mgr = NativeStagingManager(_FakeSrc(), budget=4)
    gen = 3
    mgr._gen_buf[gen] = mx.zeros((4, 4), dtype=mx.uint8)
    _patch_take(monkeypatch, [gen, 5, 0, 9, 1])
    store = _FakeStore()
    # 显式 used={5} 优先,即使 route_inds 含 9
    mgr.promote(0, store, used={5}, route_inds=mx.array([[5, 9]]), num_experts=16)
    assert set(store._resident.placed) == {5}
