"""后台专家预取器：在独立 MLX stream 上物化预测专家（私有 array），交接给主线程。

经 gating 测试验证可行（probe_multistream_gate / probe_multistream_handoff）：
- 后台线程 `with mx.stream(s2):` 物化 + mx.eval，可与主线程计算重叠、不崩。
- 只物化「私有」array，绝不写主线程共享池（跨 stream 写共享张量会报 no Stream 错）。
- 主线程只消费「已 eval」的交接 array → 跨 stream 读安全。
"""
import queue
import threading
from collections import OrderedDict

import mlx.core as mx

from mlx_streaming import config


class BackgroundExpertPrefetcher:
    def __init__(self, blob_source, window: int = 3, native: bool = False):
        self._src = blob_source
        self._stream = mx.new_stream(mx.default_device())
        self._q: "queue.Queue" = queue.Queue()
        self._ready: "OrderedDict[tuple, dict]" = OrderedDict()
        self._ready_layers: "list[int]" = []
        self._window = window
        self._native = native
        self._lock = threading.Lock()
        self._stop = False
        self.submitted = 0
        self.materialized = 0
        self.taken = 0
        # 就绪率诊断：promote 时已物化好的(ready_on_time) vs 提交了但还在飞行中(not_ready)。
        # 用于量化 attention/GDN 窗口是否够长隐藏 I/O。
        self.ready_on_time = 0
        self.not_ready = 0
        self.materialize_s = 0.0  # bg 线程物化(load+eval)累计墙钟,验证"串行加性"假设
        self._native_mat = config.native_materialize()
        self._inflight: "dict[int, set]" = {}
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def submit(self, layer: int, expert_ids) -> None:
        ids = [int(e) for e in expert_ids]
        if not ids:
            return
        with self._lock:
            self.submitted += len(ids)
            self._inflight.setdefault(int(layer), set()).update(ids)
        self._q.put((int(layer), ids))

    def _loop(self) -> None:
        while not self._stop:
            try:
                layer, ids = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                import time as _t
                _t0 = _t.perf_counter()
                with mx.stream(self._stream):
                    # view_bf16=False：后台不做 .view(bfloat16)（会报 no Stream(gpu,0)）；
                    # scales/biases 留 uint16，主线程 take 时再 view。
                    # NATIVE_MATERIALIZE=1：用 C++ blob_load 把拷贝挪出 GIL（减少对主线程争用）。
                    if self._native_mat:
                        experts = self._src.load_experts_native(layer, ids, view_bf16=False)
                    else:
                        experts = self._src.load_experts(layer, ids, view_bf16=False)
                    mx.eval([v for d in experts.values() for v in d.values()])
                self.materialize_s += _t.perf_counter() - _t0
                with self._lock:
                    pend = self._inflight.get(layer)
                    for e, d in experts.items():
                        self._ready[(layer, int(e))] = d
                        self.materialized += 1
                        if pend is not None:
                            pend.discard(int(e))
                    if layer not in self._ready_layers:
                        self._ready_layers.append(layer)
                    while len(self._ready_layers) > self._window:
                        old = self._ready_layers.pop(0)
                        for k in [k for k in self._ready if k[0] == old]:
                            del self._ready[k]
            except Exception:
                # 后台失败不影响主线程：主线程会优雅回退到同步 demand 路径。
                pass

    @staticmethod
    def _view_bf16(d: dict) -> dict:
        """主线程消费时把 scales/biases 从 uint16 位重解释回 bfloat16。"""
        out = {}
        for k, v in d.items():
            out[k] = v.view(mx.bfloat16) if (k.endswith(".scales") or k.endswith(".biases")) else v
        return out

    def take_ready(self, layer: int, e: int) -> "dict | None":
        with self._lock:
            d = self._ready.pop((int(layer), int(e)), None)
            if d is None:
                return None
            self.taken += 1
        return self._view_bf16(d)

    def take_ready_layer(self, layer: int) -> "dict[int, dict]":
        """取走某层所有就绪专家：{expert_id: {proj.tensor: mx.array}}（主线程调用）。"""
        raw = {}
        with self._lock:
            for key in [k for k in self._ready if k[0] == int(layer)]:
                raw[key[1]] = self._ready.pop(key)
                self.taken += 1
        return {e: self._view_bf16(d) for e, d in raw.items()}

    def ready_count(self, layer: int) -> int:
        with self._lock:
            return sum(1 for k in self._ready if k[0] == int(layer))

    def note_promote(self, layer: int, ready_now: int) -> None:
        """promote_prefetched 调用：记录本次 promote 时已就绪数 vs 仍在飞行中(窗口没盖住)。"""
        with self._lock:
            self.ready_on_time += int(ready_now)
            pend = self._inflight.get(int(layer))
            if pend:
                self.not_ready += len(pend)
                pend.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"submitted": self.submitted, "materialized": self.materialized,
                    "taken": self.taken, "ready": len(self._ready),
                    "ready_on_time": self.ready_on_time, "not_ready": self.not_ready,
                    "materialize_s": round(self.materialize_s, 3)}

    def close(self) -> None:
        self._stop = True
        self._t.join(timeout=2)
