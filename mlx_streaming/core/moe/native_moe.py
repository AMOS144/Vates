"""CPython native MoE backend 的 Python 包装层。"""
import importlib
import json
import os
import time
from collections import OrderedDict
from functools import lru_cache

import mlx.core as mx
import numpy as np

from mlx_streaming import config


_EXPERT_STAGE_CACHE: "OrderedDict[tuple, tuple[mx.array, mx.array, mx.array]]" = OrderedDict()
_BUNDLE_STAGE_CACHE: "OrderedDict[tuple, tuple[mx.array, ...]]" = OrderedDict()
_SLOT_POOLS: dict[tuple, "NativeComputeSlotPool"] = {}
_STAGE_STATS = {
    "expert_hits": 0,
    "expert_misses": 0,
    "bundle_hits": 0,
    "bundle_misses": 0,
    "evictions": 0,
    "can_checks": 0,
    "can_true": 0,
    "can_false": 0,
    "calls": 0,
    "route_sync_s": 0.0,
    "stage_s": 0.0,
    "enqueue_s": 0.0,
}


class NativeComputeSlotPool:
    """按层维护 compute-buffer 常驻 slot，hit 时复用 [cap,...] MLX arrays。"""

    def __init__(self, compute_dir: str, layer: int, hidden: int, inter: int,
                 group: int, bits: int, num_experts: int, cap: int):
        self.compute_dir = compute_dir
        self.layer = int(layer)
        self.hidden = int(hidden)
        self.inter = int(inter)
        self.group = int(group)
        self.bits = int(bits)
        self.num_experts = int(num_experts)
        self.cap = int(cap)
        self.slot_of: "OrderedDict[int, int]" = OrderedDict()
        self.expert_of: list[int | None] = [None] * self.cap
        self.slot_arrays = [self._empty_slot() for _ in range(self.cap)]
        self.pool_arrays: tuple[mx.array, ...] | None = None
        self.dirty = True
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rebuilds = 0

    def _empty_slot(self) -> tuple[mx.array, ...]:
        gu_words = self.hidden * self.bits // 32
        gu_groups = self.hidden // self.group
        down_words = self.inter * self.bits // 32
        down_groups = self.inter // self.group
        return (
            mx.zeros((self.inter, gu_words), dtype=mx.uint32),
            mx.zeros((self.inter, gu_groups), dtype=mx.uint16),
            mx.zeros((self.inter, gu_groups), dtype=mx.uint16),
            mx.zeros((self.inter, gu_words), dtype=mx.uint32),
            mx.zeros((self.inter, gu_groups), dtype=mx.uint16),
            mx.zeros((self.inter, gu_groups), dtype=mx.uint16),
            mx.zeros((self.hidden, down_words), dtype=mx.uint32),
            mx.zeros((self.hidden, down_groups), dtype=mx.uint16),
            mx.zeros((self.hidden, down_groups), dtype=mx.uint16),
        )

    def _load_slot(self, expert: int) -> tuple[mx.array, ...]:
        gate = _stage_one_projection_uncached(
            self.compute_dir, self.layer, "gate_proj", expert,
            self.inter, self.hidden, self.group, self.bits, self.num_experts)
        up = _stage_one_projection_uncached(
            self.compute_dir, self.layer, "up_proj", expert,
            self.inter, self.hidden, self.group, self.bits, self.num_experts)
        down = _stage_one_projection_uncached(
            self.compute_dir, self.layer, "down_proj", expert,
            self.hidden, self.inter, self.group, self.bits, self.num_experts)
        return (*gate, *up, *down)

    def _assign_slot(self, expert: int) -> int:
        cached = self.slot_of.get(expert)
        if cached is not None:
            self.slot_of.move_to_end(expert)
            self.hits += 1
            return cached
        self.misses += 1
        if len(self.slot_of) < self.cap:
            slot = next(i for i, e in enumerate(self.expert_of) if e is None)
        else:
            old_expert, slot = self.slot_of.popitem(last=False)
            self.expert_of[slot] = None
            self.evictions += 1
        self.slot_arrays[slot] = self._load_slot(expert)
        self.expert_of[slot] = expert
        self.slot_of[expert] = slot
        self.dirty = True
        return slot

    def _rebuild_pool_arrays(self) -> None:
        self.pool_arrays = tuple(
            mx.stack([slot[i] for slot in self.slot_arrays], axis=0)
            for i in range(9)
        )
        self.rebuilds += 1
        self.dirty = False

    def acquire(self, expert_ids: list[int]) -> tuple[list[int], tuple[mx.array, ...]]:
        local = [self._assign_slot(int(e)) for e in expert_ids]
        if self.pool_arrays is None or self.dirty:
            self._rebuild_pool_arrays()
        assert self.pool_arrays is not None
        return local, self.pool_arrays

    def stats(self) -> dict:
        return {
            "cap": self.cap,
            "entries": len(self.slot_of),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "rebuilds": self.rebuilds,
        }


def _compute_dir() -> str:
    return config.compute_buffer_dir() or os.path.join(
        config.expert_dir(default=""), "compute_buffers")


@lru_cache(maxsize=1)
def _load_ext():
    try:
        return importlib.import_module("mlx_streaming.native_moe_ext")
    except Exception:
        return None


@lru_cache(maxsize=256)
def _has_layer_buffers(compute_dir: str, layer: int) -> bool:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for name in ("weight", "scales", "biases"):
            path = os.path.join(compute_dir, f"layer{layer:02d}.{proj}.{name}.bin")
            if not os.path.exists(path):
                return False
    return True


@lru_cache(maxsize=1024)
def _buffer_index_matches(compute_dir: str, layer: int, proj: str,
                          out_dim: int, in_dim: int, group: int, bits: int,
                          num_experts: int) -> bool:
    path = os.path.join(compute_dir, f"layer{layer:02d}.{proj}.index.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            index = json.load(f)
    except Exception:
        return False
    tensors = index.get("tensors", {})
    weight_shape = tensors.get("weight", {}).get("shape_per_expert")
    scale_shape = tensors.get("scales", {}).get("shape_per_expert")
    bias_shape = tensors.get("biases", {}).get("shape_per_expert")
    expected_weight = [out_dim, in_dim * bits // 32]
    expected_grouped = [out_dim, in_dim // group]
    return (
        int(index.get("num_experts", -1)) == int(num_experts)
        and weight_shape == expected_weight
        and scale_shape == expected_grouped
        and bias_shape == expected_grouped
    )


@lru_cache(maxsize=512)
def _has_matching_compute_buffers(compute_dir: str, layer: int,
                                  hidden: int, inter: int, group: int,
                                  bits: int, num_experts: int) -> bool:
    if not _has_layer_buffers(compute_dir, layer):
        return False
    # index.json 是防呆线：bit/group 或目录错配时宁可回退 MLX，不冒险读错 mmap shape。
    return (
        _buffer_index_matches(
            compute_dir, layer, "gate_proj", inter, hidden, group, bits, num_experts)
        and _buffer_index_matches(
            compute_dir, layer, "up_proj", inter, hidden, group, bits, num_experts)
        and _buffer_index_matches(
            compute_dir, layer, "down_proj", hidden, inter, group, bits, num_experts)
    )


@lru_cache(maxsize=256)
def _projection_memmaps(compute_dir: str, layer: int, proj: str,
                        out_dim: int, in_dim: int, group: int, bits: int,
                        num_experts: int):
    words = in_dim * bits // 32
    groups = in_dim // group
    base = os.path.join(compute_dir, f"layer{layer:02d}.{proj}")
    weight = np.memmap(
        base + ".weight.bin",
        dtype=np.uint32,
        mode="r",
        shape=(num_experts, out_dim, words),
    )
    scales = np.memmap(
        base + ".scales.bin",
        dtype=np.uint16,
        mode="r",
        shape=(num_experts, out_dim, groups),
    )
    biases = np.memmap(
        base + ".biases.bin",
        dtype=np.uint16,
        mode="r",
        shape=(num_experts, out_dim, groups),
    )
    return weight, scales, biases


def _stage_projection(compute_dir: str, layer: int, proj: str,
                      expert_ids: list[int], out_dim: int, in_dim: int,
                      group: int, bits: int, num_experts: int):
    if config.native_moe_stage_cache():
        staged = [
            _stage_one_projection(
                compute_dir, layer, proj, int(e), out_dim, in_dim, group, bits, num_experts)
            for e in expert_ids
        ]
        return tuple(mx.stack([entry[i] for entry in staged], axis=0) for i in range(3))
    return _stage_projection_uncached(
        compute_dir, layer, proj, expert_ids, out_dim, in_dim, group, bits, num_experts)


def _stage_projection_uncached(compute_dir: str, layer: int, proj: str,
                              expert_ids: list[int], out_dim: int, in_dim: int,
                              group: int, bits: int, num_experts: int):
    weight, scales, biases = _projection_memmaps(
        compute_dir, layer, proj, out_dim, in_dim, group, bits, num_experts)
    ids = np.asarray(expert_ids, dtype=np.int64)
    # advanced indexing 已经产出 numpy 临时数组；mx.array 再接管成 MLX-managed staging arrays。
    return (
        mx.array(np.asarray(weight[ids])),
        mx.array(np.asarray(scales[ids])),
        mx.array(np.asarray(biases[ids])),
    )


def _stage_one_projection_uncached(compute_dir: str, layer: int, proj: str, expert: int,
                                  out_dim: int, in_dim: int, group: int, bits: int,
                                  num_experts: int):
    weight, scales, biases = _projection_memmaps(
        compute_dir, layer, proj, out_dim, in_dim, group, bits, num_experts)
    return (
        mx.array(np.asarray(weight[int(expert)])),
        mx.array(np.asarray(scales[int(expert)])),
        mx.array(np.asarray(biases[int(expert)])),
    )


def _stage_one_projection(compute_dir: str, layer: int, proj: str, expert: int,
                          out_dim: int, in_dim: int, group: int, bits: int,
                          num_experts: int):
    key = (compute_dir, int(layer), proj, int(expert), out_dim, in_dim, group, bits, num_experts)
    cached = _EXPERT_STAGE_CACHE.get(key)
    if cached is not None:
        _EXPERT_STAGE_CACHE.move_to_end(key)
        _STAGE_STATS["expert_hits"] += 1
        return cached
    _STAGE_STATS["expert_misses"] += 1
    staged = _stage_one_projection_uncached(
        compute_dir, layer, proj, expert, out_dim, in_dim, group, bits, num_experts)
    _EXPERT_STAGE_CACHE[key] = staged
    _EXPERT_STAGE_CACHE.move_to_end(key)
    limit = max(0, config.native_moe_stage_cache_experts())
    while limit and len(_EXPERT_STAGE_CACHE) > limit:
        _EXPERT_STAGE_CACHE.popitem(last=False)
        _STAGE_STATS["evictions"] += 1
    return staged


def _slot_pool(compute_dir: str, layer: int, hidden: int, inter: int, group: int,
               bits: int, num_experts: int) -> NativeComputeSlotPool:
    cap = max(1, config.native_moe_slot_cap())
    key = (compute_dir, int(layer), hidden, inter, group, bits, num_experts, cap)
    pool = _SLOT_POOLS.get(key)
    if pool is None:
        pool = NativeComputeSlotPool(
            compute_dir, int(layer), hidden, inter, group, bits, num_experts, cap)
        _SLOT_POOLS[key] = pool
    return pool


def _stage_bundle(compute_dir: str, layer: int, expert_ids: list[int],
                  hidden: int, inter: int, group: int, bits: int,
                  num_experts: int):
    key = (
        compute_dir, int(layer), tuple(int(e) for e in expert_ids),
        hidden, inter, group, bits, num_experts,
    )
    if config.native_moe_stage_bundle_cache():
        cached = _BUNDLE_STAGE_CACHE.get(key)
        if cached is not None:
            _BUNDLE_STAGE_CACHE.move_to_end(key)
            _STAGE_STATS["bundle_hits"] += 1
            return cached
    _STAGE_STATS["bundle_misses"] += 1
    gate = _stage_projection(
        compute_dir, int(layer), "gate_proj", expert_ids, inter, hidden, group, bits, num_experts)
    up = _stage_projection(
        compute_dir, int(layer), "up_proj", expert_ids, inter, hidden, group, bits, num_experts)
    down = _stage_projection(
        compute_dir, int(layer), "down_proj", expert_ids, hidden, inter, group, bits, num_experts)
    staged = (*gate, *up, *down)
    if config.native_moe_stage_bundle_cache():
        _BUNDLE_STAGE_CACHE[key] = staged
        _BUNDLE_STAGE_CACHE.move_to_end(key)
        limit = max(0, config.native_moe_stage_cache_bundles())
        while limit and len(_BUNDLE_STAGE_CACHE) > limit:
            _BUNDLE_STAGE_CACHE.popitem(last=False)
            _STAGE_STATS["evictions"] += 1
    return staged


def prefetch_native_moe_stage(layer: int, expert_ids: list[int],
                              hidden: int, inter: int, group: int, bits: int,
                              num_experts: int) -> bool:
    """提前把预测专家拷入 MLX-managed staging cache，供后续 fused native op 复用。"""
    if not config.native_moe():
        return False
    if not config.native_moe_stage_prefetch():
        return False
    if config.native_moe_synthetic():
        return False
    compute_dir = _compute_dir()
    if not compute_dir or not _has_matching_compute_buffers(
        compute_dir, layer, hidden, inter, group, bits, num_experts):
        return False
    try:
        uniq = list(dict.fromkeys(int(e) for e in expert_ids))
        if not uniq:
            return False
        for proj, out_dim, in_dim in (
            ("gate_proj", inter, hidden),
            ("up_proj", inter, hidden),
            ("down_proj", hidden, inter),
        ):
            for e in uniq:
                _stage_one_projection(
                    compute_dir, int(layer), proj, e, out_dim, in_dim, group, bits, num_experts)
        return True
    except Exception:
        if config.native_moe_raise():
            raise
        return False


def stage_cache_stats() -> dict:
    slot_stats = {
        "pools": len(_SLOT_POOLS),
        "entries": sum(len(pool.slot_of) for pool in _SLOT_POOLS.values()),
        "hits": sum(pool.hits for pool in _SLOT_POOLS.values()),
        "misses": sum(pool.misses for pool in _SLOT_POOLS.values()),
        "evictions": sum(pool.evictions for pool in _SLOT_POOLS.values()),
        "rebuilds": sum(pool.rebuilds for pool in _SLOT_POOLS.values()),
    }
    return {
        **_STAGE_STATS,
        "route_sync_s": round(_STAGE_STATS["route_sync_s"], 6),
        "stage_s": round(_STAGE_STATS["stage_s"], 6),
        "enqueue_s": round(_STAGE_STATS["enqueue_s"], 6),
        "expert_entries": len(_EXPERT_STAGE_CACHE),
        "bundle_entries": len(_BUNDLE_STAGE_CACHE),
        "slot_pool": slot_stats,
    }


def note_route_sync(seconds: float) -> None:
    _STAGE_STATS["route_sync_s"] += float(seconds)


def can_native_moe(layer: int, hidden: int, inter: int, group: int, bits: int,
                   num_experts: int) -> bool:
    """只做缓存化的轻量可用性判断；失败时避免上层触发 route host 同步。"""
    _STAGE_STATS["can_checks"] += 1
    if not config.native_moe():
        _STAGE_STATS["can_false"] += 1
        return False
    if not config.native_moe_mlx_op():
        _STAGE_STATS["can_false"] += 1
        return False
    if config.native_moe_synthetic():
        ok = _load_ext() is not None
        _STAGE_STATS["can_true" if ok else "can_false"] += 1
        return ok
    if (hidden * bits) % 32 != 0 or (inter * bits) % 32 != 0:
        _STAGE_STATS["can_false"] += 1
        return False
    compute_dir = _compute_dir()
    if not compute_dir:
        _STAGE_STATS["can_false"] += 1
        return False
    ok = (
        _load_ext() is not None
        and _has_matching_compute_buffers(
            compute_dir, int(layer), hidden, inter, group, bits, num_experts)
    )
    _STAGE_STATS["can_true" if ok else "can_false"] += 1
    return ok


def try_native_moe(layer: int, expert_ids: list[int], x: mx.array, scores: mx.array,
                   hidden: int, inter: int, group: int, bits: int,
                   num_experts: int) -> "mx.array | None":
    """尝试 native fused MoE，失败返回 None 让调用方回退 MLX 路径。"""
    if not config.native_moe():
        return None
    if not config.native_moe_mlx_op():
        return None
    if (hidden * bits) % 32 != 0 or (inter * bits) % 32 != 0:
        return None
    if not expert_ids:
        return None
    if not can_native_moe(layer, hidden, inter, group, bits, num_experts):
        return None
    compute_dir = _compute_dir()
    ext = _load_ext()
    if ext is None:
        return None
    synthetic = config.native_moe_synthetic()
    if (not synthetic and not _has_matching_compute_buffers(
            compute_dir, layer, hidden, inter, group, bits, num_experts)):
        return None
    try:
        expert_arr = mx.array([int(e) for e in expert_ids], dtype=mx.uint32)
        if synthetic:
            t_enqueue = time.perf_counter()
            y = ext.fused_moe(
                x.astype(mx.float32),
                expert_arr,
                scores.astype(mx.float32),
                compute_dir,
                int(layer),
                hidden,
                inter,
                group,
                bits,
                num_experts,
                True,
            )
            _STAGE_STATS["enqueue_s"] += time.perf_counter() - t_enqueue
        else:
            t_stage = time.perf_counter()
            if config.native_moe_slot_pool():
                pool = _slot_pool(
                    compute_dir, int(layer), hidden, inter, group, bits, num_experts)
                local_slots, pool_arrays = pool.acquire(expert_ids)
                local_arr = mx.array(local_slots, dtype=mx.uint32)
                (
                    gate_w, gate_s, gate_b,
                    up_w, up_s, up_b,
                    down_w, down_s, down_b,
                ) = pool_arrays
            else:
                local_arr = None
                (
                    gate_w, gate_s, gate_b,
                    up_w, up_s, up_b,
                    down_w, down_s, down_b,
                ) = _stage_bundle(
                    compute_dir, int(layer), expert_ids,
                    hidden, inter, group, bits, num_experts)
            _STAGE_STATS["stage_s"] += time.perf_counter() - t_stage
            t_enqueue = time.perf_counter()
            if local_arr is not None:
                y = ext.fused_moe_slots(
                    x.astype(mx.float32),
                    local_arr,
                    scores.astype(mx.float32),
                    gate_w, gate_s, gate_b,
                    up_w, up_s, up_b,
                    down_w, down_s, down_b,
                    hidden, inter, group, bits,
                )
            else:
                y = ext.fused_moe_staged(
                    x.astype(mx.float32),
                    scores.astype(mx.float32),
                    gate_w, gate_s, gate_b,
                    up_w, up_s, up_b,
                    down_w, down_s, down_b,
                    hidden, inter, group, bits,
                )
            _STAGE_STATS["enqueue_s"] += time.perf_counter() - t_enqueue
        _STAGE_STATS["calls"] += 1
        return y
    except Exception:
        if config.native_moe_raise():
            raise
        return None
