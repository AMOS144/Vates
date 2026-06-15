"""常驻专家池：每层一块连续 GPU 张量 + slot LRU，支持按需增长、pin、GPU 侧重映射。

设计要点：
- 命中只返回槽位、不写池；miss 只把单个专家原地写进它的槽位（`_write_slot`），避免拷贝整池。
- 池按需增长（grow-on-demand）：起步 `_POOL_INIT_SLOTS` 行，工作集扩大时 ~1.5× 增至天花板
  `cap_for(layer)` 才开始 LRU 淘汰。默认无 profile 即自动右尺寸，容量内永不超预算。
- `acquire_gpu`：decode 热路径用 GPU 查找表做纯 GPU slot 重映射，命中层零 host 往返，
  仅一次 miss 标志同步；真 miss 才回退 host 读盘路径。
"""
from collections import OrderedDict, Counter
from typing import Dict, List

import mlx.core as mx

from mlx_streaming import config

# 池按需增长(grow-on-demand)的初始物理行数。
# 起步小,工作集扩大时按 ~1.5× 增长,封顶 cap_for(默认=全局 capacity)。
# 好处:无需 profile 也能自动右尺寸(内存≈实际工作集),且对任何 prompt 自适应、容量内永不超预算。
_POOL_INIT_SLOTS = 16


class ResidentExpertPool:
    """每层一个连续常驻池：(capacity,*shape) 张量 + slot LRU。

    命中只返回槽位、不写池；miss 只把单个专家原地写进它的槽位（_write_slot）。
    loader(layer, e) -> Dict[str, mx.array]，单个专家的参数（未堆叠）。
    """

    def __init__(self, capacity: int, loader, layer_caps: "Dict[int, int] | None" = None):
        self.capacity = capacity
        self.loader = loader
        # cap_for(layer) 是该层物理行数的「天花板」：profile 指定则用之，否则=全局 capacity。
        # 池按需增长(grow-on-demand)：起步小、随工作集扩大增至天花板才开始 LRU 淘汰。
        # 因此默认(无 profile)即自动右尺寸——内存≈实际工作集,对任意 prompt 自适应,
        # 且天花板内永不超预算(超预算只是更慢,不崩)。增长偶发(预热期),稳态零拷贝。
        self.layer_caps: "Dict[int, int]" = dict(layer_caps or {})
        self._pools: "Dict[int, Dict[str, mx.array]]" = {}
        self._slot_of: "Dict[int, OrderedDict[int, int]]" = {}
        self._free: "Dict[int, list]" = {}
        # 每层当前物理已分配行数(按需增长,≤ cap_for)。grow-on-demand 的核心状态。
        self._alloc: "Dict[int, int]" = {}
        # 每层 GPU 查找表(全局专家 id → slot,-1=不在池),供 acquire_gpu 在命中层
        # 做纯 GPU 重映射(消除每层 .tolist 同步)。与 _slot_of 同步维护(仅在表已建时)。
        self._slot_table: "Dict[int, mx.array]" = {}
        # 每层 pinned 专家集合：预取进 resident pool 后不参与 LRU 驱逐。
        # 用于 K>2 / 小槽位时保住热专家工作集，降低真实 miss。
        self._pinned: "Dict[int, set[int]]" = {}
        # 可选替换策略：默认 lru；EVICT_POLICY=lfu 时用短窗口频率 + LRU tie-break。
        self.eviction_policy = config.evict_policy().lower()
        self.lfu_decay_interval = config.lfu_decay_interval()
        self._freq: "Dict[int, Counter[int]]" = {}
        self._access_count: "Dict[int, int]" = {}
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0
        self.prefetch_loads = 0
        # GPU remap 路径取证:命中层走纯 GPU 快路径次数 vs 有 miss 回退 host 次数
        self.gpu_fastpath = 0
        self.gpu_fallback = 0

    def cap_for(self, layer: int) -> int:
        """该层池容量：profile 指定则用之(上限 capacity)，否则用全局 capacity。"""
        return min(self.layer_caps.get(layer, self.capacity), self.capacity)

    def _ensure_layer(self, layer: int):
        if layer not in self._slot_of:
            self._slot_of[layer] = OrderedDict()
            # 空闲槽列表懒填:首次 miss 分配池时再灌入,之后按需增长时追加。
            # 注意:始终原地 mutate 这个 list,绝不重新绑定,确保 acquire 里持有的本地引用同步可见。
            self._free[layer] = []
            self._pinned[layer] = set()
            self._freq[layer] = Counter()
            self._access_count[layer] = 0

    def _alloc_pool(self, layer: int, sample: Dict[str, mx.array]):
        """首次为某层分配池张量,起步 _POOL_INIT_SLOTS 行(不超过该层天花板)。"""
        n = min(_POOL_INIT_SLOTS, self.cap_for(layer))
        self._pools[layer] = {
            k: mx.zeros((n,) + v.shape, dtype=v.dtype)
            for k, v in sample.items()
        }
        self._alloc[layer] = n
        self._free[layer].extend(range(n))        # 原地 mutate,不重新绑定

    def _grow_pool(self, layer: int, new_n: int):
        """把某层池物理行数扩到 new_n(封顶 cap_for),拼接保留已驻行的数据与 slot 索引。"""
        old_n = self._alloc[layer]
        new_n = min(new_n, self.cap_for(layer))
        if new_n <= old_n:
            return
        old = self._pools[layer]
        self._pools[layer] = {
            k: mx.concatenate(
                [v, mx.zeros((new_n - old_n,) + v.shape[1:], dtype=v.dtype)],
                axis=0)
            for k, v in old.items()
        }
        self._free[layer].extend(range(old_n, new_n))   # 新行追加为空闲,原地 mutate
        self._alloc[layer] = new_n

    def allocated_slots(self, layer: int) -> int:
        """该层池张量当前物理行数(按需增长,≤ cap_for),用于内存核算/测试。"""
        return self._alloc.get(layer, 0)

    def _write_slot(self, layer: int, slot: int, expert: Dict[str, mx.array]):
        pool = self._pools[layer]
        for k, v in expert.items():
            pool[k][slot] = v        # de-risk 选定：原地写，单槽、不拷贝整池

    def resident_count(self, layer: int) -> int:
        return len(self._slot_of.get(layer, ()))

    def resident_experts(self, layer: int) -> set[int]:
        """返回该层 resident pool 里当前已有的专家 id 集合（trace/probe 用）。"""
        return set(self._slot_of.get(layer, {}).keys())

    def resident_lru_scores(self, layer: int) -> dict[int, float]:
        """返回 resident 专家的 LRU 新近度分数：0=最久未用，1=最近使用。"""
        keys = list(self._slot_of.get(layer, {}).keys())
        if not keys:
            return {}
        denom = max(1, len(keys) - 1)
        return {int(e): i / denom for i, e in enumerate(keys)}

    def _note_access(self, layer: int, expert_ids: List[int]) -> None:
        if self.eviction_policy != "lfu":
            return
        self._ensure_layer(layer)
        freq = self._freq[layer]
        for e in expert_ids:
            freq[int(e)] += 1
        self._access_count[layer] += len(expert_ids)
        if self._access_count[layer] >= self.lfu_decay_interval:
            for e in list(freq):
                freq[e] //= 2
                if freq[e] <= 0:
                    del freq[e]
            self._access_count[layer] = 0

    def _choose_victim(self, layer: int, current: set[int]) -> int:
        slot_of = self._slot_of[layer]
        pinned = self._pinned.get(layer, set())
        # 绝不驱逐当前请求(current)专家：否则 acquire 末尾按 slot_of 取槽会 KeyError。
        # 只在「非 current 且非 pinned」中选受害者；选不出来说明本次唯一专家+不可驱逐者已超容量，
        # 由调用方(acquire 的 len(uniq)>cap 守卫)负责拦截，这里直接报清晰错误。
        candidates = [e for e in slot_of if e not in pinned and e not in current]
        if not candidates:
            raise ValueError(
                f"layer {layer} resident pool has no evictable non-current slot "
                f"(capacity={self.cap_for(layer)}, pinned={len(pinned)}, current={len(current)})")
        if self.eviction_policy != "lfu":
            return candidates[0]
        freq = self._freq.get(layer, Counter())
        return min(enumerate(candidates), key=lambda x: (freq.get(x[1], 0), x[0]))[1]

    def _place_expert(self, layer: int, e: int, expert: Dict[str, mx.array],
                      current: "set[int] | None" = None) -> int:
        """把专家写进 resident pool，必要时增长或驱逐非 pinned LRU，返回 slot。"""
        current = current or {e}
        cap = self.cap_for(layer)
        slot_of, free = self._slot_of[layer], self._free[layer]
        if layer not in self._pools:
            self._alloc_pool(layer, expert)
        if not free and self._alloc[layer] < cap:
            cur = self._alloc[layer]
            self._grow_pool(layer, cur + max(1, cur // 2))
        if free:
            slot = free.pop(0)
        else:
            evicted_e = self._choose_victim(layer, current)
            slot = slot_of.pop(evicted_e)
            self._clear_table(layer, evicted_e)
        self._write_slot(layer, slot, expert)
        slot_of[e] = slot
        slot_of.move_to_end(e)
        self._set_table(layer, e, slot)
        return slot

    def pin(self, layer: int, expert_ids: List[int]) -> None:
        """预取并钉住专家到 resident pool；pinned 专家不被 LRU 驱逐。

        pin 是显式预取，不计入 hit/miss 统计；后续 acquire/acquire_gpu 命中才计 hit。
        """
        loaded = {int(e): self.loader(layer, int(e)) for e in expert_ids}
        self.pin_loaded(layer, loaded)

    def pin_loaded(self, layer: int, experts: "Dict[int, Dict[str, mx.array]]") -> None:
        """把已加载专家钉进 resident pool，避免 FileExpertStore.pin 重复读/持有两份。"""
        uniq = list(dict.fromkeys(int(e) for e in experts))
        cap = self.cap_for(layer)
        self._ensure_layer(layer)
        pinned = self._pinned[layer]
        if len(pinned | set(uniq)) > cap:
            raise ValueError(
                f"pin 后 pinned 数量 {len(pinned | set(uniq))} > 该层池容量 {cap}")
        slot_of = self._slot_of[layer]
        for e in uniq:
            if e in slot_of:
                pinned.add(e)
                slot_of.move_to_end(e)
                continue
            self._place_expert(layer, e, experts[e], current={e})
            pinned.add(e)

    def prefetch(self, layer: int, expert_ids: List[int]) -> None:
        """把专家预取进 resident pool，但不计入 hit/miss，且可被后续 LRU 驱逐。"""
        uniq = list(dict.fromkeys(int(e) for e in expert_ids))
        cap = self.cap_for(layer)
        if len(uniq) > cap:
            uniq = uniq[:cap]
        self._ensure_layer(layer)
        slot_of = self._slot_of[layer]
        current = set(uniq)
        for e in uniq:
            if e in slot_of:
                slot_of.move_to_end(e)
                self.prefetch_hits += 1
                continue
            expert = self.loader(layer, e)
            self._place_expert(layer, e, expert, current=current)
            self.prefetch_loads += 1

    def acquire(self, layer: int, expert_ids: List[int]):
        # 唯一专家集合(保序去重)：池只需同时容纳本次请求的唯一专家
        uniq = list(dict.fromkeys(int(e) for e in expert_ids))
        cap = self.cap_for(layer)
        if len(uniq) > cap:
            raise ValueError(
                f"本次请求 {len(uniq)} 个唯一专家 > 该层池容量 {cap}")
        self._ensure_layer(layer)
        self._note_access(layer, uniq)
        uniq_set = set(uniq)
        slot_of, free = self._slot_of[layer], self._free[layer]
        for e in uniq:
            if e in slot_of:
                self.hits += 1
                slot_of.move_to_end(e)        # 触摸为最近使用，避免本次内被驱逐
                continue
            self.misses += 1
            expert = self.loader(layer, e)
            self._place_expert(layer, e, expert, current=uniq_set)
        # slots 与原始 expert_ids 一一对应(含重复)，便于直接 reshape 成 routing 索引
        slots = [slot_of[int(e)] for e in expert_ids]
        return self._pools[layer], slots

    def _set_table(self, layer: int, e: int, slot: int):
        t = self._slot_table.get(layer)
        if t is not None:
            t[int(e)] = slot       # (num_experts,) 小张量原地改,代价可忽略

    def _clear_table(self, layer: int, e: int):
        t = self._slot_table.get(layer)
        if t is not None:
            t[int(e)] = -1

    def _ensure_table(self, layer: int, num_experts: int) -> mx.array:
        t = self._slot_table.get(layer)
        if t is None or int(t.shape[0]) != num_experts:
            self._ensure_layer(layer)
            tab = [-1] * num_experts
            for e, slot in self._slot_of[layer].items():
                tab[int(e)] = int(slot)
            t = mx.array(tab, dtype=mx.int32)
            self._slot_table[layer] = t
        return t

    def acquire_gpu(self, layer: int, inds: mx.array, num_experts: int):
        """GPU 侧 slot 重映射:命中层零 host 往返(仅一次 miss 标志同步)。

        返回 (pool_arrays, local)。inds 为路由结果(decode 时形如 (1,1,k))。
        全命中 → local = 表[inds] 纯 GPU;有 miss → 回退既有 host acquire 读盘并维护表后重算 local。
        """
        table = self._ensure_table(layer, num_experts)
        local = mx.take(table, inds)
        n_miss = int(mx.sum((local < 0).astype(mx.int32)))   # 唯一一次 GPU→CPU 同步
        if n_miss == 0:
            self.gpu_fastpath += 1
            self.hits += int(inds.size)     # decode top-k 各专家互异 → size 即唯一命中数
            return self._pools[layer], local
        # 有 miss:回退 host 路径(读盘、写槽、维护表),再用更新后的表重算 local
        self.gpu_fallback += 1
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        pool_arrays, _ = self.acquire(layer, flat)
        local = mx.take(self._slot_table[layer], inds)
        return pool_arrays, local

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
