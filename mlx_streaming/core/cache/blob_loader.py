"""全流式 blob 专家源：并行 pread + F_NOCACHE + 即时物化为 MLX 量化数组。

字节布局由 prep/blob_layout.py 统一描述（单一真相源），支持两种格式：
- v1 affine（expert_blob_v1）：每 proj 段序 [weight, scales, biases]；weight uint32，
  scales/biases 存 uint16 原始位、用时 .view(bfloat16)。
- v2 mxfp4（expert_blob_v2_mxfp4）：每 proj 段序 [weight, scales]（无 biases）；
  weight uint32，scales 为 uint8 原始位、绝不 view(bf16)。
计算复用 MLX quantized_matmul / gather_qmm（不碰已 NO-GO 的 fused kernel）。
"""
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from mlx_streaming import config

# macOS fcntl：F_NOCACHE 提示内核不要把读过的页留在 page cache（支撑低内存目标）。
_F_NOCACHE = 48


class BlobExpertSource:
    def __init__(self, blob_dir: str, hidden: int, inter: int, group: int, bits: int,
                 num_experts: int, workers: int = 8, nocache: bool = True,
                 quant_mode: str = "affine", blob_format: "str | None" = None):
        self.dir = blob_dir
        self.h, self.i, self.g, self.b, self.ne = hidden, inter, group, bits, num_experts
        self.workers = workers
        self.nocache = nocache
        # blob 格式：mxfp4 默认 v2（无 biases、scales uint8），其余默认 v1 affine。
        self.quant_mode = quant_mode
        self.blob_format = blob_format or (
            "expert_blob_v2_mxfp4" if quant_mode == "mxfp4" else "expert_blob_v1")
        self._segs, self.stride = self._layout()
        self._fds: "dict[int, int]" = {}
        # 后台预取：只预读原始字节进 _pf_cache（不建 mx.array，可后台线程跑）；
        # 滚动窗口只保留最近 window 层的预取字节，支撑低内存。
        self.window = config.stream_blob_window()
        self._pf_pool = ThreadPoolExecutor(max_workers=workers)
        self._pf_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
        self._pf_futures: "dict[tuple, object]" = {}
        self._pf_layers: "list[int]" = []
        self._lock = threading.Lock()
        self.preads = 0
        self.prefetch_hits = 0

    def _layout(self):
        # 段表统一由 blob_layout.layout_for 给出（dtype 为字符串 "uint32"/"uint16"/"uint8"）。
        from mlx_streaming.prep.blob_layout import layout_for
        return layout_for(self.blob_format, self.h, self.i, self.b, self.g)

    def _fd(self, layer: int) -> int:
        fd = self._fds.get(layer)
        if fd is None:
            fd = os.open(os.path.join(self.dir, f"layer{layer:02d}.blob"), os.O_RDONLY)
            if self.nocache:
                try:
                    import fcntl
                    fcntl.fcntl(fd, _F_NOCACHE, 1)
                except (OSError, ValueError):
                    pass
            self._fds[layer] = fd
        return fd

    def _materialize(self, raw: bytes, view_bf16: bool = True) -> dict:
        # view_bf16=False：scales/biases 保留 uint16（.view(bfloat16) 在后台线程会报
        # no Stream(gpu,0)，故后台物化时关掉，由主线程消费时再 view）。
        # 段表 dtype 为字符串：仅 v1 affine 的 uint16 scales/biases 是 bf16 位重解释；
        # mxfp4 的 uint8 scales 保持原样，绝不 view(bf16)。
        out = {}
        off = 0
        for proj, tensor, dt, shape, nb in self._segs:
            np_dt = np.dtype(dt)
            view = np.frombuffer(raw, dtype=np_dt, count=nb // np_dt.itemsize, offset=off).reshape(shape)
            arr = mx.array(view)
            if view_bf16 and dt == "uint16" and tensor in ("scales", "biases"):
                arr = arr.view(mx.bfloat16)
            out[f"{proj}.{tensor}"] = arr
            off += nb
        return out

    def _pread(self, layer: int, e: int) -> bytes:
        return os.pread(self._fd(layer), self.stride, e * self.stride)

    def prefetch_async(self, layer: int, expert_ids) -> None:
        """后台并行预读预测专家的字节（仅 IO，不建 mx.array）。供跨层预测调用。"""
        for e in (int(x) for x in expert_ids):
            key = (layer, e)
            with self._lock:
                if key in self._pf_cache or key in self._pf_futures:
                    continue
                self._pf_futures[key] = self._pf_pool.submit(self._pf_job, layer, e)
        self._evict_window(layer)

    def _pf_job(self, layer: int, e: int) -> bytes:
        raw = self._pread(layer, e)
        with self._lock:
            self._pf_cache[(layer, e)] = raw
            self._pf_futures.pop((layer, e), None)
        return raw

    def _evict_window(self, layer: int) -> None:
        with self._lock:
            if layer not in self._pf_layers:
                self._pf_layers.append(layer)
            while len(self._pf_layers) > self.window:
                old = self._pf_layers.pop(0)
                for key in [k for k in self._pf_cache if k[0] == old]:
                    del self._pf_cache[key]

    def wait_prefetch(self) -> None:
        with self._lock:
            futs = list(self._pf_futures.values())
        for f in futs:
            f.result()

    def release_prefetched(self, layer: int, expert_ids) -> None:
        """Release consumed raw-byte lookahead without disturbing other layers."""
        with self._lock:
            for expert in expert_ids:
                self._pf_cache.pop((int(layer), int(expert)), None)

    def read_raw(self, layer: int, expert_ids) -> "list[bytes]":
        """取原始字节：优先用预取缓存/在途结果，未命中的并行 pread（仅 IO）。"""
        ids = [int(e) for e in expert_ids]
        results: "dict[int, bytes]" = {}
        misses = []
        for e in ids:
            key = (layer, e)
            with self._lock:
                if key in self._pf_cache:
                    results[e] = self._pf_cache[key]
                    self.prefetch_hits += 1
                    continue
                fut = self._pf_futures.get(key)
            if fut is not None:
                results[e] = fut.result()
                self.prefetch_hits += 1
            else:
                misses.append(e)

        def rd(e):
            return e, self._pread(layer, e)

        if misses:
            self.preads += len(misses)
            if self.workers <= 1 or len(misses) <= 1:
                for e in misses:
                    results[e] = self._pread(layer, e)
            else:
                # 复用持久线程池（之前每次调用新建/销毁 ThreadPoolExecutor → 大量线程创建 + GIL 争用）。
                for e, raw in self._pf_pool.map(rd, misses):
                    results[e] = raw
        return [results[e] for e in ids]

    def load_experts(self, layer: int, expert_ids, view_bf16: bool = True) -> dict:
        """返回 {expert_id: {proj.tensor: mx.array}}。并行读字节，lazy 物化 mx.array。

        view_bf16=False：scales/biases 留 uint16（供后台线程用，避免 .view 报 no Stream）。
        注：零拷贝(preadv 进 MLX buffer)实测在真实 lazy-eval 流里更慢——为拿可写 buffer
        需 per-expert mx.eval 同步，代价高于省掉的拷贝(见报告)。故保留 lazy frombuffer 路径。
        """
        ids = [int(e) for e in expert_ids]
        raws = self.read_raw(layer, ids)
        return {e: self._materialize(raw, view_bf16=view_bf16) for e, raw in zip(ids, raws)}

    def load_experts_stacked(self, layer: int, expert_ids, view_bf16: bool = True) -> dict:
        """批量物化：每段只建一个 (N,*shape) mx.array（np.stack 后单次构造），
        取代逐专家 6N 次 frombuffer+mx.array。返回 {proj.tensor: (N,*shape)}，按 ids 顺序。

        与 load_experts 同源（同段表、同 view_bf16 语义），只是把"逐专家逐段"折成
        "逐段一次堆叠"——demand 多 miss 消费的碎 mx.array 构造从 6N 降到 6。
        """
        ids = [int(e) for e in expert_ids]
        raws = self.read_raw(layer, ids)
        out = {}
        off = 0
        for proj, tensor, dt, shape, nb in self._segs:
            np_dt = np.dtype(dt)
            cnt = nb // np_dt.itemsize
            views = [np.frombuffer(raw, dtype=np_dt, count=cnt, offset=off).reshape(shape)
                     for raw in raws]
            arr = mx.array(np.stack(views, axis=0))      # (N, *shape) 单次构造
            if view_bf16 and dt == "uint16" and tensor in ("scales", "biases"):
                arr = arr.view(mx.bfloat16)
            out[f"{proj}.{tensor}"] = arr
            off += nb
        if config.expert_major_double_buffer():
            self.release_prefetched(layer, ids)
        return out

    def load_experts_native(self, layer: int, expert_ids, view_bf16: bool = False) -> dict:
        """native 物化：用 C++ blob_load 把字节 pread 进 MLX 数组（eval 时在 C++ 跑、绕 GIL），
        Python 侧只建惰性切片/view 图（无数据拷贝）。目的是把"拷贝"挪出 GIL，减少后台线程
        对主线程 per-layer 派发的争用。
        """
        import mlx_streaming.native_moe_ext as _N
        ids = [int(e) for e in expert_ids]
        path = os.path.join(self.dir, f"layer{layer:02d}.blob")
        raw = _N.blob_load(path, mx.array(ids, dtype=mx.uint32), self.stride)  # [n, stride] uint8（lazy）
        out = {}
        for i, e in enumerate(ids):
            row = raw[i]
            d = {}
            off = 0
            for proj, tensor, dt, shape, nb in self._segs:
                seg = row[off:off + nb]
                # 按段 dtype 重解释：weight→uint32；v1 affine 的 uint16 scales/biases
                # 可再 view(bf16)；mxfp4 的 uint8 scales 保持 uint8。
                if dt == "uint32":
                    arr = seg.view(mx.uint32).reshape(shape)
                elif dt == "uint16":
                    arr = seg.view(mx.uint16).reshape(shape)
                    if view_bf16:
                        arr = arr.view(mx.bfloat16)
                else:  # uint8（mxfp4 scales）
                    arr = seg.reshape(shape)
                d[f"{proj}.{tensor}"] = arr
                off += nb
            out[e] = d
        return out

    def keys(self) -> "list[str]":
        return [f"{proj}.{tensor}" for proj, tensor, _, _, _ in self._segs]

    def acquire(self, layer: int, expert_ids):
        """对齐 ResidentExpertPool.acquire：返回 (pool_arrays, slots)。

        pool_arrays: {proj.tensor: (n_uniq,...) 堆叠数组}，喂给 _sub.forward。
        slots: 每个 expert_id 在 unique 列表中的下标（= local，reshape 前）。
        """
        ids = [int(e) for e in expert_ids]
        uniq = list(dict.fromkeys(ids))
        experts = self.load_experts(layer, uniq)
        pool = {k: mx.stack([experts[e][k] for e in uniq], axis=0) for k in self.keys()}
        pos = {e: i for i, e in enumerate(uniq)}
        slots = [pos[e] for e in ids]
        return pool, slots

    def close(self):
        self._pf_pool.shutdown(wait=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
