"""常驻专家池：每层一块连续 GPU 张量 + slot LRU，支持按需增长、pin、GPU 侧重映射。

设计要点：
- 命中只返回槽位、不写池；miss 只把单个专家原地写进它的槽位（`_write_slot`），避免拷贝整池。
- 池按需增长（grow-on-demand）：起步 `_POOL_INIT_SLOTS` 行，工作集扩大时 ~1.5× 增至天花板
  `cap_for(layer)` 才开始 LRU 淘汰。默认无 profile 即自动右尺寸，容量内永不超预算。
- `acquire_gpu`：decode 热路径用 GPU 查找表做纯 GPU slot 重映射，命中层零 host 往返，
  仅一次 miss 标志同步；真 miss 才回退 host 读盘路径。
"""
import os
from collections import OrderedDict, Counter
from typing import Dict, List

import mlx.core as mx

# 诊断门控：staging/acquire_gpu 路径消费侧字节级真值校验（混合 ahead 损坏取证用，默认关）。
# 在 acquire_gpu 全命中快路径返回前，对本层真实路由命中的每个专家，把池槽字节与磁盘真值
# 逐 key 比对；不一致即「池槽装错字节」铁证。受 STG_VERIFY=1 控制，对主路径零影响（默认 off）。
_STG_VERIFY = os.environ.get("STG_VERIFY") == "1"
_stg_verify_state = {"ok": 0, "bad": 0, "printed": 0, "calls": 0, "first_bad_call": None}

from mlx_streaming import config

# Route 3 Phase 1 底座：spec/dual 模式下池 buffer 改由 C++ 拥有(mx::allocator + no-op deleter)，
# 地址进程内恒定、永不被 MLX donation/迁移，供侧区/demand 后台 pread 安全直写（替代消费侧 MLX scatter）。
# POOL_OWNED=0 可强制回退 mx.zeros(仅供 A/B 对照)。
_POOL_OWNED = os.environ.get("POOL_OWNED", "1") == "1"
# mx.Dtype -> C++ pool_owned_zeros 接受的 dtype 名。
_DTYPE_NAME = {
    mx.uint32: "uint32", mx.uint16: "uint16", mx.uint8: "uint8",
    mx.int32: "int32", mx.int16: "int16",
    mx.bfloat16: "bfloat16", mx.float16: "float16", mx.float32: "float32",
}


def _owned_pool(sample: "Dict[str, mx.array]", n: int) -> "Dict[str, mx.array]":
    """用 C++-owned buffer 建 (n,*shape) 的 per-key 池数组（地址恒定，供 C++ 直写）。"""
    import mlx_streaming.native_moe_ext as _N
    out = {}
    for k, v in sample.items():
        name = _DTYPE_NAME.get(v.dtype)
        if name is None:
            raise RuntimeError(f"pool_owned_zeros 不支持 dtype {v.dtype} (key={k})")
        out[k] = _N.pool_owned_zeros([int(n)] + [int(d) for d in v.shape], name)
    return out


# 池按需增长(grow-on-demand)的初始物理行数。
# 起步小,工作集扩大时按 ~1.5× 增长,封顶 cap_for(默认=全局 capacity)。
# 好处:无需 profile 也能自动右尺寸(内存≈实际工作集),且对任何 prompt 自适应、容量内永不超预算。
_POOL_INIT_SLOTS = 16


class ResidentExpertPool:
    """每层一个连续常驻池：(capacity,*shape) 张量 + slot LRU。

    命中只返回槽位、不写池；miss 只把单个专家原地写进它的槽位（_write_slot）。
    loader(layer, e) -> Dict[str, mx.array]，单个专家的参数（未堆叠）。
    """

    def __init__(self, capacity: int, loader, layer_caps: "Dict[int, int] | None" = None,
                 spec_slots: int = 0, batch_loader=None, stacked_batch_loader=None,
                 spec_gens: int = 1):
        self.capacity = capacity
        self.loader = loader
        # 可选批量加载器 batch_loader(layer, [ids]) -> {e: expert_dict}：acquire 用它把本层所有
        # miss 一次并行读（8-worker pread），取代逐专家串行 loader。None 时退回串行 loader。
        self.batch_loader = batch_loader
        # 可选「批量+预堆叠」加载器 stacked_batch_loader(layer, [ids]) -> {k:(N,*shape)}：
        # 在 batch_loader 基础上再把 6N 次碎 mx.array 物化折成每段一次，写池时直接整批 scatter，
        # 既省 frombuffer/mx.array 构造又省 _write_slots_batch 里的 mx.stack。优先级最高。
        self.stacked_batch_loader = stacked_batch_loader
        # The former side-slot count is now a speculative-admission quota.
        # Physical rows belong to the single merged main pool in every mode.
        self.spec_slots = int(spec_slots)
        self.spec_gens = max(1, int(spec_gens))
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
        # 可选替换策略：默认 lfu（短窗口频率 + LRU tie-break，实测比 lru 命中更高）；EVICT_POLICY=lru 回退纯 LRU。
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
        # dual 路径真实区槽状态由 C++ demand_dual 唯一权威(无 opt-out)。spec 模式 + native 已编译即启用;
        # 非 spec(spec_slots==0)或 native 缺失时保持 Python 权威路径(仅 prefill/host/非双源用)。
        # 启用后 _slot_of/_free/_freq 在 dual 路径不再维护,resident_experts/_count 改查 C++ g_real(预取过滤要用)。
        self._native_demand = False
        if int(spec_slots) > 0:
            try:
                import mlx_streaming.native_moe_ext as _N
                self._native_demand = hasattr(_N, "demand_dual")
            except Exception:
                self._native_demand = False
        if self._native_demand:
            # 复刻基线 8-worker 并行读：demand miss 的 pread 派给 BgReader 并行执行（高优队列）。
            import mlx_streaming.native_moe_ext as _N
            try:
                _N.bg_reader_start(
                    int(os.environ.get("DEMAND_WORKERS", "8")),
                    int(os.environ.get("PREFETCH_LOW_WORKERS", "0")),
                )
            except Exception:
                pass
        if self._native_demand and os.environ.get("DEMAND_TIMING") == "1":
            import atexit
            import mlx_streaming.native_moe_ext as _N
            _N.demand_timing_enable(True)
            atexit.register(lambda: print(
                "[DEMAND_TIMING ms] inds/pool/side_snap/real_lock/core/build =",
                [round(x / 1e3, 1) for x in _N.demand_timings()], flush=True))

    def cap_for(self, layer: int) -> int:
        """该层池容量：profile 指定则用之(上限 capacity)，否则用全局 capacity。"""
        return min(self.layer_caps.get(layer, self.capacity), self.capacity)

    def native_real_cap_for(self, layer: int) -> int:
        """Return the demand-owned prefix of the physical allocation."""
        cap = self.cap_for(layer)
        if config.prefetch_isolated_side_for(layer) and int(layer) > 0:
            return max(1, cap - self.spec_gens * self.spec_slots)
        return cap

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
        if self.spec_slots > 0:
            n = self.cap_for(layer)
            # Direct prefetch reserves rows inside this same allocation.
            if _POOL_OWNED:
                self._pools[layer] = _owned_pool(sample, n)
            else:
                self._pools[layer] = {
                    k: mx.zeros((n,) + v.shape, dtype=v.dtype) for k, v in sample.items()}
        else:
            n = min(_POOL_INIT_SLOTS, self.cap_for(layer))
            self._pools[layer] = {
                k: mx.zeros((n,) + v.shape, dtype=v.dtype)
                for k, v in sample.items()
            }
        self._alloc[layer] = n
        real = self.cap_for(layer) if self.spec_slots > 0 else n
        self._free[layer].extend(range(real))

    def _grow_pool(self, layer: int, new_n: int):
        """把某层池物理行数扩到 new_n(封顶 cap_for),拼接保留已驻行的数据与 slot 索引。"""
        if self.spec_slots > 0:
            return   # 侧区模式：预分配满，永不 grow（保 C++ 写指针稳定）
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

    def preallocate(self, layer: int, sample: "Dict[str, mx.array]", cap: int):
        """满 cap 预分配 typed 池并 eval 固定 data 指针；幂等。

        供"C++ 直写槽"机制使用：调用后该层不再 grow/不再走 _place_expert(MLX scatter)，
        槽位由调用方直接 mutate _slot_of + _set_table 管理、字节由 C++ pread 写入。
        """
        self._ensure_layer(layer)
        if layer in self._pools:
            return
        if _POOL_OWNED:
            self._pools[layer] = _owned_pool(sample, cap)   # C++ 拥有，地址恒定供直写
        else:
            self._pools[layer] = {
                k: mx.zeros((cap,) + v.shape, dtype=v.dtype) for k, v in sample.items()
            }
        mx.eval(list(self._pools[layer].values()))   # 固定 data 指针
        self._alloc[layer] = cap
        # 登记所有物理行为空闲,供共用分配器(_alloc_slot)按 free 优先分配；
        # 已被 slot_of 占用的行不在此列(暖池复用时 slot_of 已满 → free 为空 → 走驱逐)。
        occupied = set(self._slot_of[layer].values())
        self._free[layer].extend(r for r in range(cap) if r not in occupied)

    def allocated_slots(self, layer: int) -> int:
        """该层池张量当前物理行数(按需增长,≤ cap_for),用于内存核算/测试。"""
        return self._alloc.get(layer, 0)

    def _pool_owned(self, layer: int) -> bool:
        """该层池是否为 C++-owned buffer（spec/dual 模式 + POOL_OWNED）。
        owned 池禁止任何 MLX scatter（会重分配 buffer、孤立侧区直写），落池一律走 C++ 直写。"""
        return _POOL_OWNED and self.spec_slots > 0

    def _write_slot(self, layer: int, slot: int, expert: Dict[str, mx.array]):
        if self._pool_owned(layer):
            self._write_slots_batch(layer, [slot], [expert])   # 单槽也走 C++ 直写，避免 scatter
            return
        pool = self._pools[layer]
        for k, v in expert.items():
            pool[k][slot] = v        # de-risk 选定：原地写，单槽、不拷贝整池

    def _write_slots_batch(self, layer: int, slots: List[int],
                           experts: "List[Dict[str, mx.array]]") -> None:
        """把多个专家一次性写进各自槽位。

        owned 池：C++ memcpy 直写各行（无 MLX scatter，保 buffer 地址恒定 → 侧区直写不被孤立）。
        非 owned：每个 key 一次 stack + fancy-index scatter（原碎 kernel 最少化路径）。
        """
        if not slots:
            return
        pool = self._pools[layer]
        if self._pool_owned(layer):
            import mlx_streaming.native_moe_ext as _N
            keys = list(pool.keys())
            pool_list = [pool[k] for k in keys]
            srcs_flat = [e[k] for k in keys for e in experts]   # key-major：与 C++ pool_write_rows 约定一致
            mx.eval(srcs_flat)                                   # 物化源，固定 data 指针
            _N.pool_write_rows(pool_list, srcs_flat, [int(s) for s in slots])
            return
        idx = mx.array(slots, dtype=mx.int32)
        for k in pool:
            pool[k][idx] = mx.stack([e[k] for e in experts], axis=0)

    def resident_count(self, layer: int) -> int:
        if self._native_demand:                      # 方案B：C++ g_real 为真实区权威
            import mlx_streaming.native_moe_ext as N
            return int(N.real_region_count(int(layer)))
        return len(self._slot_of.get(layer, ()))

    def resident_experts(self, layer: int) -> set[int]:
        """返回该层 resident pool 里当前已有的专家 id 集合（trace/probe + 侧区预取过滤用）。"""
        if self._native_demand:                      # 方案B：查 C++ g_real，保预取过滤一致
            import mlx_streaming.native_moe_ext as N
            flat = (
                N.real_verified_contents(int(layer))
                if config.prefetch_direct_slots()
                else N.real_region_contents(int(layer))
            )
            return {int(flat[i]) for i in range(0, len(flat), 2)}
        return set(self._slot_of.get(layer, {}).keys())

    def _bootstrap_dual_pool(self, layer: int) -> None:
        """首次为该层只预分配统一主池 cap 行，并初始化 C++ 权威映射。幂等。

        池结构(per-key typed 数组)由 Python 从一个样本专家的 shape 建；字节后续由 C++ demand pread 写入。
        Python 侧 _slot_of/_free 在方案B dual 路径不再使用（真实区状态归 C++ g_real）。
        """
        self._ensure_layer(layer)
        if layer not in self._pools:
            sample = self.loader(layer, 0)           # 仅取 shape/dtype，不写入池
            cap = self.cap_for(layer)
            n = cap
            if _POOL_OWNED:
                self._pools[layer] = _owned_pool(sample, n)   # C++ 拥有，地址恒定供直写
            else:
                self._pools[layer] = {
                    k: mx.zeros((n,) + v.shape, dtype=v.dtype) for k, v in sample.items()
                }
            mx.eval(list(self._pools[layer].values()))   # 固定 data 指针，供 C++ memcpy
            self._alloc[layer] = n
            import mlx_streaming.native_moe_ext as N
            N.real_init(int(layer), int(self.native_real_cap_for(layer)))

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
        if self.lfu_decay_interval > 0 and self._access_count[layer] >= self.lfu_decay_interval:
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

    def _alloc_slot(self, layer: int, e: int, current: "set[int]") -> "tuple[int, bool]":
        """为 e 分配/复用槽（free 优先，否则驱逐非 pinned/非 current 的 LRU 最旧），
        更新 slot_of + table，但不写数据。返回 (slot, is_new)。

        供 C++ 直写预取(prefetch_cpp)与 demand 写入(_place_expert)**共用**：保证两条路径
        从同一套 _free/_slot_of/_slot_table 分配，绝不抢占同一物理槽。
        """
        slot_of, free = self._slot_of[layer], self._free[layer]
        if e in slot_of:
            slot_of.move_to_end(e)
            return slot_of[e], False
        cap = self.cap_for(layer)
        if not free and self._alloc.get(layer, 0) < cap:
            cur = self._alloc[layer]
            self._grow_pool(layer, cur + max(1, cur // 2))
        if free:
            slot = free.pop(0)
        else:
            evicted_e = self._choose_victim(layer, current)
            slot = slot_of.pop(evicted_e)
            self._clear_table(layer, evicted_e)
        slot_of[e] = slot
        slot_of.move_to_end(e)
        self._set_table(layer, e, slot)
        return slot, True

    def _place_expert(self, layer: int, e: int, expert: Dict[str, mx.array],
                      current: "set[int] | None" = None) -> int:
        """把专家写进 resident pool，必要时增长或驱逐非 pinned LRU，返回 slot。"""
        current = current or {e}
        if layer not in self._pools:
            self._alloc_pool(layer, expert)
        slot, _ = self._alloc_slot(layer, e, current)
        self._write_slot(layer, slot, expert)
        return slot

    def _place_experts(self, layer: int, ids: List[int],
                       experts: "Dict[int, Dict[str, mx.array]]",
                       current: "set[int]") -> List[int]:
        """批量把一组 miss 专家写进 resident pool：共用槽分配器 + 单次批量 scatter。

        与逐个 _place_expert 语义等价（同样的 grow/驱逐/槽分配顺序），
        只把数据写入合并成 _write_slots_batch，去掉每 token 上百次碎 scatter。
        """
        if not ids:
            return []
        if layer not in self._pools:
            self._alloc_pool(layer, experts[ids[0]])
        slots = [self._alloc_slot(layer, e, current)[0] for e in ids]
        self._write_slots_batch(layer, slots, [experts[e] for e in ids])
        return slots

    def _place_experts_stacked(self, layer: int, ids: List[int],
                               stacked: "Dict[str, mx.array]",
                               current: "set[int]") -> List[int]:
        """批量写一组 miss：stacked 为按 ids 顺序预堆叠的 {k:(N,*shape)}，
        每 key 直接一次 fancy-index scatter（连 _write_slots_batch 的 mx.stack 都省了）。
        """
        if not ids:
            return []
        if layer not in self._pools:
            self._alloc_pool(layer, {k: v[0] for k, v in stacked.items()})
        slots = [self._alloc_slot(layer, e, current)[0] for e in ids]
        pool = self._pools[layer]
        if self._pool_owned(layer):
            import mlx_streaming.native_moe_ext as _N
            keys = list(pool.keys())
            pool_list = [pool[k] for k in keys]
            stacked_list = [stacked[k] for k in keys]
            mx.eval(stacked_list)                          # 物化预堆叠源
            _N.pool_write_stacked(pool_list, stacked_list, [int(s) for s in slots])
            return slots
        idx = mx.array(slots, dtype=mx.int32)
        for k in pool:
            pool[k][idx] = stacked[k]
        return slots

    def prefetch_cpp(self, layer: int, expert_ids, submit_fn) -> list:
        """用 C++ 直写把预测专家投机预取进池（与 demand fallback 共用槽分配器）。

        submit_fn(slot, expert)：提交一次异步 C++ 后台读到该物理槽（调用方负责后续 wait）。
        已驻 → 触摸 LRU 跳过；未驻 → 分配槽并 submit。返回新提交的 expert 列表。
        投机性质：不计 hit/miss，分得的槽可被后续 LRU 驱逐；无可驱逐槽时跳过该专家
        （交给 demand fallback），绝不抛错打断热路径。
        """
        self._ensure_layer(layer)
        if layer not in self._pools:
            return []                                  # 池未建（首 token 预热）→ 全交 fallback
        current = {int(e) for e in expert_ids}
        slot_of = self._slot_of[layer]
        submitted = []
        for e in expert_ids:
            e = int(e)
            if e in slot_of:
                slot_of.move_to_end(e)
                continue
            try:
                slot, _ = self._alloc_slot(layer, e, current)
            except ValueError:
                continue                               # 无可驱逐槽 → 跳过，留给 demand fallback
            submit_fn(slot, e)
            submitted.append(e)
        return submitted

    def pin(self, layer: int, expert_ids: List[int]) -> None:
        """预取并钉住专家到 resident pool；pinned 专家不被 LRU 驱逐。

        pin 是显式预取，不计入 hit/miss 统计；后续 acquire/acquire_gpu 命中才计 hit。
        """
        uniq = list(dict.fromkeys(int(e) for e in expert_ids))
        if not uniq:
            return
        loaded = {e: self.loader(layer, e) for e in uniq}
        if self._native_demand:
            # dual 热路径的槽账本归 C++ g_real 权威：由它分配并登记不可驱逐
            # pin，再复用 owned-pool 批量写把完整专家字节同步落入对应真实区槽。
            self._bootstrap_dual_pool(layer)
            import mlx_streaming.native_moe_ext as N
            slots = list(N.real_pin(
                int(layer), uniq, int(self.native_real_cap_for(layer)),
            ))
            if len(slots) != len(uniq):
                raise RuntimeError("native real_pin 返回槽数与专家数不一致")
            self._write_slots_batch(
                layer, slots, [loaded[expert] for expert in uniq],
            )
            return
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

    def acquire(self, layer: int, expert_ids: List[int], protect: "set[int] | None" = None):
        """protect：额外的「不可驱逐」专家集（除本批 miss 外）。dual 路径传入本前向全部 inds，
        确保落 miss 腾槽时不驱逐本前向仍需的命中专家（否则其 slot 被复用 → 消费侧 gather 错字节）。
        仅影响驱逐候选，不改 hit/miss 记账口径。"""
        # 唯一专家集合(保序去重)：池只需同时容纳本次请求的唯一专家
        uniq = list(dict.fromkeys(int(e) for e in expert_ids))
        cap = self.cap_for(layer)
        if len(uniq) > cap:
            raise ValueError(
                f"本次请求 {len(uniq)} 个唯一专家 > 该层池容量 {cap}")
        self._ensure_layer(layer)
        self._note_access(layer, uniq)
        uniq_set = set(uniq)
        # 驱逐保护集 = 本批 miss ∪ 调用方额外指定(如本前向全部 inds 命中专家)。
        current = uniq_set if protect is None else (uniq_set | protect)
        slot_of, free = self._slot_of[layer], self._free[layer]
        # 先处理命中（触摸 LRU），再把所有 miss 收成一批
        misses = []
        for e in uniq:
            if e in slot_of:
                self.hits += 1
                slot_of.move_to_end(e)        # 触摸为最近使用，避免本次内被驱逐
            else:
                misses.append(e)
        if misses:
            self.misses += len(misses)
            if self.stacked_batch_loader is not None:
                # 批量读 + 批量物化：每段一次 mx.array，写池每 key 一次 scatter（碎 kernel 最少）
                stacked = self.stacked_batch_loader(layer, misses)
                self._place_experts_stacked(layer, misses, stacked, current=current)
            else:
                if self.batch_loader is not None:
                    # 一次并行读本层所有 miss（8-worker pread）；保序写池，LRU/驱逐语义不变
                    loaded = self.batch_loader(layer, misses)
                else:
                    loaded = {e: self.loader(layer, e) for e in misses}
                # 批量写：每 key 仅一次 stacked scatter，取代逐专家逐段碎 scatter
                self._place_experts(layer, misses, loaded, current=current)
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
            if self.eviction_policy == "lfu":   # 全命中层也计频:piggyback 上面 n_miss 的 drain,inds 已物化
                self._note_access(layer, [int(i) for i in inds.reshape(-1).tolist()])
            return self._pools[layer], local
        # 有 miss:回退 host 路径(读盘、写槽、维护表),再用更新后的表重算 local
        self.gpu_fallback += 1
        flat = [int(i) for i in inds.reshape(-1).tolist()]
        pool_arrays, _ = self.acquire(layer, flat)
        local = mx.take(self._slot_table[layer], inds)
        return pool_arrays, local

    def verify_acquire_bytes(self, layer, inds, stg=None):
        """诊断(STG_VERIFY)：acquire 后把本层真实路由命中专家的池槽字节与磁盘真值逐 key 比对。

        发现不一致即打印 (call, layer, expert, slot, gen, key)：池槽装错字节的铁证，
        与 timing/gen 竞态对应。call 是全局校验调用序号(≈token 前向序，便于和首分歧 token 关联)。
        stg：可选 NativeStagingManager，用于查该专家落池所用 gen。
        """
        st = _stg_verify_state
        st["calls"] += 1
        call = st["calls"]
        pool = self._pools.get(layer)
        slot_of = self._slot_of.get(layer)
        if pool is None or slot_of is None:
            return
        flat = {int(i) for i in inds.reshape(-1).tolist()}
        for e in flat:
            slot = slot_of.get(e)
            if slot is None:
                continue
            try:
                truth = self.loader(layer, e)
            except Exception:
                continue
            bad_key = None
            for k in pool:
                if k not in truth:
                    continue
                a = pool[k][slot]
                b = truth[k]
                if a.shape != b.shape or not bool(mx.all(a == b)):
                    bad_key = k
                    break
            if bad_key is None:
                st["ok"] += 1
            else:
                st["bad"] += 1
                if st["first_bad_call"] is None:
                    st["first_bad_call"] = call
                gen = None
                if stg is not None:
                    gen = stg.placed_gen.get((layer, e))
                if st["printed"] < 24:
                    st["printed"] += 1
                    print(f"[STG_VERIFY] BAD call={call} layer={layer} expert={e} "
                          f"slot={slot} gen={gen} key={bad_key} "
                          f"(ok={st['ok']} bad={st['bad']})", flush=True)

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0
