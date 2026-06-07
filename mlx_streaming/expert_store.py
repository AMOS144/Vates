"""路线 B 专家后端：每层独立 LRU 缓存 + 按需切片，统计命中率。

与早期版本的两处提速改动：
- **每层独立 LRU**：每个 MoE 层各自一个 LRU，容量 = 每层槽数。解码自回归时相邻
  token 在同一层常路由到重叠专家，per-layer 隔离能显著抬高命中率（不再被别层挤掉）。
- **去掉每次驱逐的 clear_cache 抖动**：驱逐只丢引用，靠 MLX 自带缓冲复用 + 外部
  set_cache_limit 控制上限，避免低命中率下疯狂 malloc/free。

stacked[layer] = {"weight": (E,O,I) [, "scales", "biases"]}（惰性，未 eval）。
fetch(layer, expert_ids) 只取这几个专家、堆叠成 (k, ...) 返回。
"""
import os
from collections import OrderedDict, Counter
from typing import Dict, List

import mlx.core as mx

from mlx_streaming.mem import clear_cache


class _PerLayerLru:
    """每层一个 OrderedDict 的 LRU 集合，容量为「每层槽数」。"""

    def __init__(self, capacity: int, clear_on_evict: bool = False):
        self.capacity = capacity
        self.clear_on_evict = clear_on_evict
        self._caches: "Dict[int, OrderedDict[int, Dict[str, mx.array]]]" = {}
        self.hits = 0
        self.misses = 0

    def _layer_cache(self, layer: int) -> "OrderedDict[int, Dict[str, mx.array]]":
        c = self._caches.get(layer)
        if c is None:
            c = OrderedDict()
            self._caches[layer] = c
        return c

    def resident_count(self) -> int:
        return sum(len(c) for c in self._caches.values())

    def get_or_load(self, layer: int, e: int, loader) -> Dict[str, mx.array]:
        cache = self._layer_cache(layer)
        if e in cache:
            self.hits += 1
            cache.move_to_end(e)
        else:
            self.misses += 1
            cache[e] = loader(layer, e)
            cache.move_to_end(e)
            while len(cache) > self.capacity:
                cache.popitem(last=False)   # 仅丢引用；不在热路径 clear_cache
                if self.clear_on_evict:
                    clear_cache()
        return cache[e]

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0


def _stack_picked(picked: List[Dict[str, mx.array]]) -> Dict[str, mx.array]:
    keys = picked[0].keys()
    return {k: mx.stack([p[k] for p in picked]) for k in keys}


def _stack_cache_size(size: int | None) -> int:
    if size is not None:
        return max(0, int(size))
    return max(0, int(os.environ.get("EXPERT_STACK_CACHE", "0")))


def _stack_cache_get(cache, size: int, layer: int, key: tuple[int, ...]):
    if size <= 0:
        return None
    layer_cache = cache.get(layer)
    if layer_cache is None or key not in layer_cache:
        return None
    layer_cache.move_to_end(key)
    return layer_cache[key]


def _stack_cache_put(cache, size: int, layer: int, key: tuple[int, ...],
                     value: Dict[str, mx.array]) -> None:
    if size <= 0:
        return
    layer_cache = cache.get(layer)
    if layer_cache is None:
        layer_cache = OrderedDict()
        cache[layer] = layer_cache
    layer_cache[key] = value
    layer_cache.move_to_end(key)
    while len(layer_cache) > size:
        layer_cache.popitem(last=False)


class ResidentExpertPool:
    """每层一个连续常驻池：(capacity,*shape) 张量 + slot LRU。

    命中只返回槽位、不写池；miss 只把单个专家原地写进它的槽位（_write_slot）。
    loader(layer, e) -> Dict[str, mx.array]，单个专家的参数（未堆叠）。
    """

    def __init__(self, capacity: int, loader, layer_caps: "Dict[int, int] | None" = None):
        self.capacity = capacity
        self.loader = loader
        # 每层独立上限(profile 驱动)：uniform 预留 capacity 槽里大量层实际工作集远小于
        # capacity，预留即浪费。按各层真实高水位分配 → 无损省内存(命中率/输出/吞吐不变)。
        # 缺省层用全局 capacity。一次性分配(不动态增长，避免 concat 复制整池的稳态开销)。
        self.layer_caps: "Dict[int, int]" = dict(layer_caps or {})
        self._pools: "Dict[int, Dict[str, mx.array]]" = {}
        self._slot_of: "Dict[int, OrderedDict[int, int]]" = {}
        self._free: "Dict[int, list]" = {}
        self.hits = 0
        self.misses = 0

    def cap_for(self, layer: int) -> int:
        """该层池容量：profile 指定则用之(上限 capacity)，否则用全局 capacity。"""
        return min(self.layer_caps.get(layer, self.capacity), self.capacity)

    def _ensure_layer(self, layer: int):
        if layer not in self._slot_of:
            self._slot_of[layer] = OrderedDict()
            self._free[layer] = list(range(self.cap_for(layer)))

    def _alloc_pool(self, layer: int, sample: Dict[str, mx.array]):
        n = self.cap_for(layer)
        self._pools[layer] = {
            k: mx.zeros((n,) + v.shape, dtype=v.dtype)
            for k, v in sample.items()
        }

    def allocated_slots(self, layer: int) -> int:
        """该层池张量物理行数(= cap_for，已分配时)，用于内存核算/测试。"""
        return self.cap_for(layer) if layer in self._pools else 0

    def _write_slot(self, layer: int, slot: int, expert: Dict[str, mx.array]):
        pool = self._pools[layer]
        for k, v in expert.items():
            pool[k][slot] = v        # de-risk 选定：原地写，单槽、不拷贝整池

    def resident_count(self, layer: int) -> int:
        return len(self._slot_of.get(layer, ()))

    def acquire(self, layer: int, expert_ids: List[int]):
        # 唯一专家集合(保序去重)：池只需同时容纳本次请求的唯一专家
        uniq = list(dict.fromkeys(int(e) for e in expert_ids))
        cap = self.cap_for(layer)
        if len(uniq) > cap:
            raise ValueError(
                f"本次请求 {len(uniq)} 个唯一专家 > 该层池容量 {cap}")
        self._ensure_layer(layer)
        slot_of, free = self._slot_of[layer], self._free[layer]
        for e in uniq:
            if e in slot_of:
                self.hits += 1
                slot_of.move_to_end(e)        # 触摸为最近使用，避免本次内被驱逐
                continue
            self.misses += 1
            expert = self.loader(layer, e)
            if layer not in self._pools:
                self._alloc_pool(layer, expert)
            slot = free.pop(0) if free else slot_of.popitem(last=False)[1]
            self._write_slot(layer, slot, expert)
            slot_of[e] = slot
            slot_of.move_to_end(e)
        # slots 与原始 expert_ids 一一对应(含重复)，便于直接 reshape 成 routing 索引
        slots = [slot_of[int(e)] for e in expert_ids]
        return self._pools[layer], slots

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0


class LruExpertStore:
    """内存堆叠后端：从常驻（惰性）堆叠张量按专家切片。容量为「每层槽数」。"""

    def __init__(self, stacked: Dict[int, Dict[str, mx.array]], capacity: int,
                 clear_on_evict: bool = False, stack_cache_size: int | None = None):
        self._stacked = stacked
        self.capacity = capacity
        self._lru = _PerLayerLru(capacity, clear_on_evict)
        self.stack_cache_size = _stack_cache_size(stack_cache_size)
        self._stack_cache: "Dict[int, OrderedDict[tuple[int, ...], Dict[str, mx.array]]]" = {}

    @property
    def hits(self) -> int:
        return self._lru.hits

    @property
    def misses(self) -> int:
        return self._lru.misses

    def resident_count(self) -> int:
        return self._lru.resident_count()

    def cap_for(self, layer: int) -> int:
        return self.capacity

    def _load_one(self, layer: int, e: int) -> Dict[str, mx.array]:
        src = self._stacked[layer]
        out = {}
        for k, arr in src.items():
            sub = arr[e]            # 切第 e 个专家
            mx.eval(sub)            # 显式物化这一个专家
            out[k] = sub
        return out

    def fetch(self, layer: int, expert_ids: List[int]) -> Dict[str, mx.array]:
        key = tuple(int(e) for e in expert_ids)
        cached = _stack_cache_get(self._stack_cache, self.stack_cache_size, layer, key)
        if cached is not None:
            for e in key:
                self._lru.get_or_load(layer, e, self._load_one)
            return cached
        picked = [self._lru.get_or_load(layer, int(e), self._load_one) for e in expert_ids]
        stacked = _stack_picked(picked)
        _stack_cache_put(self._stack_cache, self.stack_cache_size, layer, key, stacked)
        return stacked

    def hit_rate(self) -> float:
        return self._lru.hit_rate()


class FileExpertStore:
    """文件后端专家缓存：从离线拆分的 per-expert safetensors 按需加载，每层独立 LRU。

    文件命名：{root}/layer{layer:02d}_expert{e:03d}.safetensors，
    内容为扁平 dict，键形如 "gate_proj.weight"/"gate_proj.scales"/"up_proj.weight"...
    capacity 语义为「每层槽数」（worst-case 常驻 ≈ capacity × MoE 层数）。
    """

    def __init__(self, root: str, capacity: int, clear_on_evict: bool = False,
                 record: bool = False, stack_cache_size: int | None = None,
                 layer_caps: "Dict[int, int] | None" = None):
        self.root = root
        self.capacity = capacity
        self._lru = _PerLayerLru(capacity, clear_on_evict)
        self.stack_cache_size = _stack_cache_size(stack_cache_size)
        self._stack_cache: "Dict[int, OrderedDict[tuple[int, ...], Dict[str, mx.array]]]" = {}
        # ③ 热专家常驻：钉住的专家永不进 LRU、永不驱逐
        self._pinned: "Dict[int, Dict[int, Dict[str, mx.array]]]" = {}
        self.pinned_hits = 0
        # 连续常驻池后端（acquire 路径用），与 LRU stack 路径共享 _load_one
        # layer_caps：每层独立池容量(profile 驱动的无损省内存)，缺省层用全局 capacity
        self._resident = ResidentExpertPool(capacity, loader=self._load_one,
                                            layer_caps=layer_caps)
        # 激活频率统计（校准阶段用）
        self.record = record
        self._counts: "Dict[int, Counter]" = {}

    def cap_for(self, layer: int) -> int:
        """该层常驻池容量(profile 指定则用之，否则全局 capacity)。"""
        return self._resident.cap_for(layer)

    @property
    def hits(self) -> int:
        return self._lru.hits + self.pinned_hits + self._resident.hits

    @property
    def misses(self) -> int:
        return self._lru.misses + self._resident.misses

    def resident_count(self) -> int:
        pinned = sum(len(d) for d in self._pinned.values())
        return self._lru.resident_count() + pinned

    def pinned_count(self) -> int:
        return sum(len(d) for d in self._pinned.values())

    def path(self, layer: int, e: int) -> str:
        import os
        return os.path.join(self.root, f"layer{layer:02d}_expert{e:03d}.safetensors")

    def _load_one(self, layer: int, e: int) -> Dict[str, mx.array]:
        w = mx.load(self.path(layer, e))
        mx.eval(w)             # 只物化这一个专家
        return w

    def note(self, layer: int, expert_ids: List[int]) -> None:
        """记录一次激活（校准阶段调用），用于挑热专家。"""
        c = self._counts.get(layer)
        if c is None:
            c = Counter()
            self._counts[layer] = c
        for e in expert_ids:
            c[int(e)] += 1

    def recorded_layers(self) -> List[int]:
        return list(self._counts.keys())

    def hot(self, layer: int, h: int) -> List[int]:
        """返回该层激活最频繁的 h 个专家 id。"""
        c = self._counts.get(layer)
        if not c:
            return []
        return [e for e, _ in c.most_common(h)]

    def pin(self, layer: int, expert_ids: List[int]) -> None:
        """把这些专家加载并钉为常驻（不计入 LRU、不可驱逐）。"""
        d = self._pinned.setdefault(layer, {})
        for e in expert_ids:
            e = int(e)
            if e not in d:
                d[e] = self._load_one(layer, e)

    def reset_stats(self) -> None:
        self._lru.hits = 0
        self._lru.misses = 0
        self.pinned_hits = 0
        self._resident.hits = 0
        self._resident.misses = 0

    def acquire(self, layer: int, expert_ids: List[int]):
        """连续常驻池取专家：返回 (pool_arrays, slots)。命中零拷贝，miss 单槽写。"""
        if self.record:
            self.note(layer, expert_ids)
        return self._resident.acquire(layer, expert_ids)

    def fetch(self, layer: int, expert_ids: List[int]) -> Dict[str, mx.array]:
        if self.record:
            self.note(layer, expert_ids)
        key = tuple(int(e) for e in expert_ids)
        cached = _stack_cache_get(self._stack_cache, self.stack_cache_size, layer, key)
        if cached is not None:
            pinned = self._pinned.get(layer)
            for e in key:
                if pinned is not None and e in pinned:
                    self.pinned_hits += 1
                else:
                    self._lru.get_or_load(layer, e, self._load_one)
            return cached
        pinned = self._pinned.get(layer)
        picked = []
        for e in expert_ids:
            e = int(e)
            if pinned is not None and e in pinned:
                self.pinned_hits += 1
                picked.append(pinned[e])
            else:
                picked.append(self._lru.get_or_load(layer, e, self._load_one))
        stacked = _stack_picked(picked)
        _stack_cache_put(self._stack_cache, self.stack_cache_size, layer, key, stacked)
        return stacked

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
