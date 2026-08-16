"""路线 B 专家后端：每层独立 LRU 缓存 + 按需切片，统计命中率。

与早期版本的两处提速改动：
- **每层独立 LRU**：每个 MoE 层各自一个 LRU，容量 = 每层槽数。解码自回归时相邻
  token 在同一层常路由到重叠专家，per-layer 隔离能显著抬高命中率（不再被别层挤掉）。
- **去掉每次驱逐的 clear_cache 抖动**：驱逐只丢引用，靠 MLX 自带缓冲复用 + 外部
  set_cache_limit 控制上限，避免低命中率下疯狂 malloc/free。

stacked[layer] = {"weight": (E,O,I) [, "scales", "biases"]}（惰性，未 eval）。
fetch(layer, expert_ids) 只取这几个专家、堆叠成 (k, ...) 返回。
"""
import json
import os
import struct
from collections import OrderedDict, Counter
from typing import Dict, List

import mlx.core as mx
import numpy as np

from mlx_streaming import config
from mlx_streaming.core.mem import clear_cache
from mlx_streaming.core.cache.resident_pool import ResidentExpertPool, _POOL_INIT_SLOTS  # noqa: F401

# macOS fcntl：F_NOCACHE 提示内核不要把读过的页留在 page cache。
_F_NOCACHE = 48
# safetensors dtype -> (numpy 读入类型, 需 view 的 mlx 类型)。BF16 numpy 无原生类型,
# 先读 uint16 原始位再 .view(bfloat16)。其余类型按需补。
_ST_DTYPE = {
    "U32": (np.uint32, None),
    "U16": (np.uint16, None),
    "I32": (np.int32, None),
    "F32": (np.float32, None),
    "F16": (np.float16, None),
    "BF16": (np.uint16, mx.bfloat16),
}


def _pread_all(fd: int, size: int) -> bytes:
    """把整文件读出(os.pread 可能短读,循环补齐)。"""
    chunks = []
    off = 0
    while off < size:
        b = os.pread(fd, size - off, off)
        if not b:
            break
        chunks.append(b)
        off += len(b)
    return b"".join(chunks)


def load_safetensors_nocache(path: str) -> "Dict[str, mx.array]":
    """用 F_NOCACHE 读 safetensors,返回与 mx.load 等价的 {name: mx.array}。

    绕过 OS page cache:每次都真实读 NVMe,基准结果可复现(不被页缓存冷热污染)。
    与 mx.load 的数值/dtype 完全一致(U32 权重、BF16 scales/biases)。
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            import fcntl
            fcntl.fcntl(fd, _F_NOCACHE, 1)
        except (OSError, ValueError, ImportError):
            pass
        size = os.fstat(fd).st_size
        data = _pread_all(fd, size)
    finally:
        os.close(fd)
    n = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + n])
    base = 8 + n
    out: "Dict[str, mx.array]" = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        np_dt, view_dt = _ST_DTYPE[info["dtype"]]
        s, e = info["data_offsets"]
        arr = mx.array(np.frombuffer(data[base + s:base + e], dtype=np_dt).reshape(info["shape"]))
        if view_dt is not None:
            arr = arr.view(view_dt)
        out[name] = arr
    return out


class _PerLayerLru:
    """每层一个 OrderedDict 的 LRU 集合，容量为「每层槽数」。"""

    def __init__(self, capacity: int, clear_on_evict: bool = False):
        self.capacity = capacity
        self.clear_on_evict = clear_on_evict
        self._caches: "Dict[int, OrderedDict[int, Dict[str, mx.array]]]" = {}
        self.hits = 0
        self.misses = 0
        self.prefetch_hits = 0
        self.prefetch_loads = 0

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
    return max(0, config.expert_stack_cache())


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
        # 可选 per-layer bundle：把同层所有 per-expert safetensors 合并成一个文件，
        # 运行时每层只 mx.load 一次，减少大量小文件 open/header parse 开销。
        self.use_bundles = config.expert_bundle()
        self.bundle_dir = config.expert_bundle_dir(os.path.join(root, "layer_bundles"))
        self.bundle_cache_size = config.expert_bundle_cache()
        self._bundle_cache: "OrderedDict[int, Dict[str, mx.array]]" = OrderedDict()
        self._bundle_key_cache: "Dict[int, Dict[int, list[tuple[str, str]]]]" = {}
        self.bundle_loads = 0
        # 连续常驻池后端（acquire 路径用），与 LRU stack 路径共享 _load_one
        # layer_caps：每层独立池容量(profile 驱动的无损省内存)，缺省层用全局 capacity
        # demand miss 批量加载接线(三选一,acquire 优先 stacked > batch > 逐专家):
        # - native(默认,且非 async_prefetch):C++ blob_load 直进 MLX 数组、绕 GIL、惰性,返回
        #   逐专家 {e:{k}} → 走 _place_experts。native 不消费 async prefetch buffer,故 gate 在
        #   not async_prefetch(async 路径改用 _batch_load_resident,它会先吃 buffer)。
        # - batch_miss_read(默认):numpy 批量堆叠,每段一次 mx.array(取代逐专家 6N 次构造)。
        # - 都关:逐专家 loader。
        if config.native_demand_loader() and not config.async_prefetch():
            _batch_loader = self._batch_load_native
            _stacked_loader = None
        elif config.batch_miss_read():
            _batch_loader = self._batch_load_resident
            _stacked_loader = (self._batch_load_stacked
                               if not config.async_prefetch() else None)
        else:
            _batch_loader = None
            _stacked_loader = None
        self._resident = ResidentExpertPool(
            capacity, loader=self._load_one_resident, layer_caps=layer_caps,
            batch_loader=_batch_loader, stacked_batch_loader=_stacked_loader)
        # 激活频率统计（校准阶段用）
        self.record = record
        self._counts: "Dict[int, Counter]" = {}
        self.async_prefetch = config.async_prefetch()
        self.prefetch_buffer_size = config.prefetch_buffer_experts()
        self._prefetch_buffer: "OrderedDict[tuple[int, int], Dict[str, mx.array]]" = OrderedDict()
        self.prefetch_buffer_hits = 0
        self.prefetch_submitted = 0
        self.prefetch_dropped = 0
        # dual-source：decode 走 staging 双源命中计数（不写池的零拷贝命中）
        self.staging_hits = 0
        # 可选 blob miss-loader（STREAM_BLOB_LOADER=1，由 model_builder 注入）：
        # miss 时用每专家连续 blob 的并行 pread 取代 per-expert safetensors 的 mx.load，
        # 复用常驻池 GPU-remap 快路径（命中零编排），小 EXPERT_SLOTS 即低内存。
        self._blob_loader = None
        # 可选后台预取器（STREAM_BLOB_BG=1，由 model_builder 注入）：在独立 stream 上
        # 提前物化预测专家，promote_prefetched 把它们写进常驻池槽（主线程、便宜 scatter）。
        self._bg = None

    def cap_for(self, layer: int) -> int:
        """该层常驻池容量(profile 指定则用之，否则全局 capacity)。"""
        return self._resident.cap_for(layer)

    def resident_experts(self, layer: int) -> set[int]:
        return self._resident.resident_experts(layer)

    def resident_lru_scores(self, layer: int) -> dict[int, float]:
        return self._resident.resident_lru_scores(layer)

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
        if self._resident._native_demand:
            import mlx_streaming.native_moe_ext as N
            return sum(
                len(N.real_pinned_contents(int(layer)))
                for layer in self._resident._pools
            )
        return sum(len(d) for d in self._pinned.values())

    def path(self, layer: int, e: int) -> str:
        import os
        return os.path.join(self.root, f"layer{layer:02d}_expert{e:03d}.safetensors")

    def bundle_path(self, layer: int) -> str:
        return os.path.join(self.bundle_dir, f"layer{layer:02d}.safetensors")

    def _load_layer_bundle(self, layer: int) -> Dict[str, mx.array]:
        cached = self._bundle_cache.get(layer)
        if cached is not None:
            self._bundle_cache.move_to_end(layer)
            return cached
        bundle = mx.load(self.bundle_path(layer))
        self.bundle_loads += 1
        self._bundle_cache[layer] = bundle
        self._bundle_cache.move_to_end(layer)
        key_map: "Dict[int, list[tuple[str, str]]]" = {}
        for full_key in bundle:
            if not full_key.startswith("expert"):
                continue
            head, rest = full_key.split(".", 1)
            e = int(head[len("expert"):])
            key_map.setdefault(e, []).append((rest, full_key))
        self._bundle_key_cache[layer] = key_map
        while len(self._bundle_cache) > self.bundle_cache_size:
            old_layer, _ = self._bundle_cache.popitem(last=False)
            self._bundle_key_cache.pop(old_layer, None)
        return bundle

    def _raw_load_one(self, layer: int, e: int) -> Dict[str, mx.array]:
        if self._blob_loader is not None:
            # blob 单专家：1 次连续 pread + 物化，结构与 mx.load 的 per-expert dict 一致。
            return self._blob_loader.load_experts(int(layer), [int(e)])[int(e)]
        if self.use_bundles and os.path.exists(self.bundle_path(layer)):
            bundle = self._load_layer_bundle(layer)
            keys = self._bundle_key_cache.get(layer, {}).get(int(e), [])
            return {short: bundle[full] for short, full in keys}
        # 默认走 F_NOCACHE 读取(绕过 OS page cache):基准每次真实读盘、可复现。
        # mx.load 用 mmap 会经页缓存,重复跑被缓存命中污染→速度飘。EXPERT_NOCACHE=0 切回 mmap。
        if config.expert_nocache():
            w = load_safetensors_nocache(self.path(layer, e))
        else:
            w = mx.load(self.path(layer, e))
        # 默认惰性:不在此 eval,让读盘并入 MLX 异步图、与计算重叠,token 末统一 eval。
        # 旧版每专家强制 mx.eval(w) 会每 token 触发 ~42 次同步、打碎流水线(实测 ~2.4× 慢)。
        # 数值完全等价(惰性求值仍精确)。EAGER_EXPERT_LOAD=1 可恢复旧的逐专家强制物化。
        if config.eager_expert_load():
            mx.eval(w)
        return w

    def _load_one(self, layer: int, e: int) -> Dict[str, mx.array]:
        return self._raw_load_one(layer, e)

    def _load_one_resident(self, layer: int, e: int) -> Dict[str, mx.array]:
        """resident demand loader：优先消费 async prefetch buffer/inflight。"""
        key = (int(layer), int(e))
        if self.async_prefetch:
            cached = self._prefetch_buffer.pop(key, None)
            if cached is not None:
                self.prefetch_buffer_hits += 1
                self._resident.misses = max(0, self._resident.misses - 1)
                return cached
        return self._raw_load_one(layer, e)

    def _batch_load_resident(self, layer: int, ids: List[int]) -> Dict[int, Dict[str, mx.array]]:
        """批量 resident demand loader：先消费 async prefetch buffer，剩余 miss 一次并行读。

        关键：blob 路径用一次 load_experts(layer, rest)（8-worker 并行 pread），取代逐专家
        串行 pread（实测串行 6.4GB/s → 并行 8 路 22GB/s）。结果与逐专家加载等价。
        """
        ids = [int(e) for e in ids]
        out: "Dict[int, Dict[str, mx.array]]" = {}
        rest: List[int] = []
        for e in ids:
            if self.async_prefetch:
                cached = self._prefetch_buffer.pop((int(layer), e), None)
                if cached is not None:
                    self.prefetch_buffer_hits += 1
                    self._resident.misses = max(0, self._resident.misses - 1)
                    out[e] = cached
                    continue
            rest.append(e)
        if rest:
            if self._blob_loader is not None:
                out.update(self._blob_loader.load_experts(int(layer), rest))
            else:
                for e in rest:
                    out[e] = self._raw_load_one(layer, e)
        return out

    def _batch_load_stacked(self, layer: int, ids: List[int]) -> Dict[str, mx.array]:
        """批量+预堆叠 demand loader：blob 路径一次 load_experts_stacked（并行 pread + 每段一次
        物化），返回 {k:(N,*shape)} 按 ids 顺序。无 blob loader 时退化为逐专家物化后堆叠。

        仅在 async_prefetch 关闭时接入（不消费 prefetch buffer），保证不漏 buffer 命中。
        """
        ids = [int(e) for e in ids]
        if self._blob_loader is not None:
            return self._blob_loader.load_experts_stacked(int(layer), ids)
        per = {e: self._raw_load_one(layer, e) for e in ids}
        keys = list(per[ids[0]].keys())
        return {k: mx.stack([per[e][k] for e in ids], axis=0) for k in keys}

    def _batch_load_native(self, layer: int, ids: List[int]) -> "Dict[int, Dict[str, mx.array]]":
        """批量 native demand loader:blob 路径用 C++ blob_load 把整批字节 pread 直进 MLX 数组
        (eval 时在 C++ 跑、绕 GIL、惰性),返回逐专家 {e:{proj.tensor:arr}}(供 _place_experts)。

        相比 numpy 路径(frombuffer+stack+mx.array 同步拷贝),把字节物化从 acquire 挪到 token 末
        eval、并绕开 GIL。无 blob loader 时退回逐专家 raw 物化(行为等价)。view_bf16=True 以
        兼容 affine 量化(uint16 scales/biases→bf16);mxfp4 的 uint8 scales 忽略此参数。
        """
        ids = [int(e) for e in ids]
        if self._blob_loader is not None:
            return self._blob_loader.load_experts_native(int(layer), ids, view_bf16=True)
        return {e: self._raw_load_one(layer, e) for e in ids}

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
        if self._resident._native_demand:
            # dual 模式只保留 owned resident pool 中的一份字节；避免旧
            # FileExpertStore._pinned 再持有一整份重复专家张量。
            self._resident.pin(layer, expert_ids)
            return
        d = self._pinned.setdefault(layer, {})
        for e in expert_ids:
            e = int(e)
            if e not in d:
                d[e] = self._load_one(layer, e)
        # 连续 resident pool 是 decode/MTP 热路径；同步预取进去并标记不可驱逐，
        # 否则旧 _pinned 只会影响 fetch/stack 路径，无法降低 acquire_gpu 的 miss。
        self._resident.pin_loaded(layer, {int(e): d[int(e)] for e in expert_ids})

    def prefetch(self, layer: int, expert_ids: List[int]) -> None:
        """预取到连续 resident pool，不计入 demand miss/hit。"""
        if not self.async_prefetch:
            self._resident.prefetch(layer, expert_ids)
            return
        resident = self._resident.resident_experts(layer)
        for e in dict.fromkeys(int(x) for x in expert_ids):
            key = (int(layer), int(e))
            if e in resident:
                self._resident.prefetch_hits += 1
                continue
            if key in self._prefetch_buffer:
                self._resident.prefetch_hits += 1
                continue
            self._prefetch_buffer[key] = self._raw_load_one(int(layer), int(e))
            self._prefetch_buffer.move_to_end(key)
            self.prefetch_submitted += 1
            self._resident.prefetch_loads += 1
            while len(self._prefetch_buffer) > self.prefetch_buffer_size:
                self._prefetch_buffer.popitem(last=False)
                self.prefetch_dropped += 1

    def promote_prefetched(self, layer: int) -> int:
        """把后台预取器中该层所有「已物化」专家写进常驻池槽（主线程、默认 stream）。

        同层(AHEAD=0)预取的就是本层将用到的专家，应真正放进池槽以转 miss 为 hit。
        互保护：置入时把「本批就绪的全部专家」当作 current，使它们彼此不被驱逐，
        只挤掉池里陈旧的非本批专家；池满且无可驱逐者时跳过剩余，留给 demand 回退。
        返回写入的专家数。
        """
        if self._bg is None:
            return 0
        ready = self._bg.take_ready_layer(int(layer))
        self._bg.note_promote(int(layer), len(ready))
        if not ready:
            return 0
        self._resident._ensure_layer(int(layer))
        slot_of = self._resident._slot_of[int(layer)]
        protect = {int(e) for e in ready}
        placed = 0
        for e, d in ready.items():
            e = int(e)
            if e in slot_of:
                continue
            try:
                self._resident._place_expert(int(layer), e, d, current=protect)
            except ValueError:
                # 池满且无非本批可驱逐 → 剩余专家留给 demand 路径
                break
            placed += 1
        return placed

    def wait_prefetch(self) -> None:
        """测试/诊断用：等待当前所有 async prefetch 完成。"""
        return

    def release_fetch_cache(self) -> int:
        """释放 fetch/stack 路径（prefill 用）的常驻缓存，返回释放的专家数。

        decode 单 token 走 acquire_gpu/_resident 池，永不再碰 _lru/_stack_cache；
        prefill 结束后这套缓存（每层 ≤cap 个专家 + 堆叠张量）是纯死重，约占
        cap×MoE层数 个专家的内存（实测 cap=16 时 ~9.2GB）。在 prefill→decode
        边界调用一次即可把这块预算还给系统/池。MTP 多 token 会按需重新填充。
        """
        freed = self._lru.resident_count()
        self._lru._caches.clear()
        self._stack_cache.clear()
        clear_cache()
        return freed

    def reset_stats(self) -> None:
        self._lru.hits = 0
        self._lru.misses = 0
        self.pinned_hits = 0
        self._resident.hits = 0
        self._resident.misses = 0
        self._resident.gpu_fastpath = 0
        self._resident.gpu_fallback = 0
        self._resident.prefetch_hits = 0
        self._resident.prefetch_loads = 0
        if hasattr(self._resident, "_demand_layer_stats"):
            self._resident._demand_layer_stats.clear()
        self.prefetch_buffer_hits = 0
        self.prefetch_submitted = 0
        self.prefetch_dropped = 0
        self.staging_hits = 0
        staging = getattr(self, "_staging", None)
        if staging is not None and hasattr(staging, "reset_stats"):
            staging.reset_stats()

    def acquire(self, layer: int, expert_ids: List[int]):
        """连续常驻池取专家：返回 (pool_arrays, slots)。命中零拷贝，miss 单槽写。"""
        if self.record:
            self.note(layer, expert_ids)
        return self._resident.acquire(layer, expert_ids)

    def acquire_gpu(self, layer: int, inds: mx.array, num_experts: int):
        """GPU 侧 slot 重映射(decode 热路径):命中零 host 往返。record 模式才回 CPU 记账。"""
        if self.record:
            self.note(layer, [int(i) for i in inds.reshape(-1).tolist()])
        return self._resident.acquire_gpu(layer, inds, num_experts)

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
