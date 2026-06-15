"""native-fused-prefetch 的 miss→hit 落地:per-layer staging buffer + 池 promote。

机制(零主线程 host 同步):
- submit(layer, inds_lazy):用下层 gate 预测的 lazy inds 调 prefetch_into_staging →
  在 GPU 完成回调里(C++)把专家字节 pread 进该层 staging buffer,并记录 (expert→row)。
- promote(layer, store):主线程读 C++ 记录(纯锁,无 GPU 同步)→ 把 staging 行切片(惰性 view)
  → 复用 ResidentExpertPool._place_expert 写进池槽 + 更新 slot 表 → acquire_gpu 命中。
预读(IO)在回调里、主线程零 pread/零 .tolist/零 np→mx.array。
"""
import os

import mlx.core as mx


class NativeStagingManager:
    """环形 staging：每层 N 块 buffer 轮转，避免"惰性 scatter 还没 eval、下个 forward 已覆盖"竞态。
    submit 写 ring[layer][rr] 并记 last[layer]=rr；promote 读 ring[layer][last[layer]]。
    只要 scatter 在 N 个 submit 内 eval，被消费的 buffer 就不会被覆盖。
    """
    def __init__(self, blob_source, budget: int, ring: int = 4):
        self.src = blob_source            # BlobExpertSource:有 dir / stride / _segs
        self.budget = int(budget)
        self.stride = int(blob_source.stride)
        self.ring = max(2, int(ring))
        self._ring: "dict[int, list]" = {}     # layer -> [mx.array]*ring
        self._rr: "dict[int, int]" = {}        # layer -> 下一个要写的环索引
        self._gen = 1                          # 全局单调 generation
        self._gen_buf: "dict[int, mx.array]" = {}  # gen -> 该次 submit 写的 buffer（promote 按 gen 严格匹配）
        self.submitted = 0
        self.promoted = 0

    def _bufs(self, layer: int) -> list:
        bl = self._ring.get(layer)
        if bl is None:
            bl = []
            for _ in range(self.ring):
                a = mx.zeros((self.budget, self.stride), dtype=mx.uint8)
                mx.eval(a)
                bl.append(a)
            self._ring[layer] = bl
            self._rr[layer] = 0
        return bl

    def submit(self, layer: int, inds_lazy: mx.array):
        """inds_lazy:lazy uint32 [k]（k≤budget）目标层预测专家。返回 dummy（需折进图触发回调）。
        关键：buffer 与 gen 绑定（self._gen_buf[gen]=buf），handler 在 C++ 里原子记 (gen,映射)。
        promote 按 gen 取回**同一块** buffer，杜绝"submit 时 Python 记 buffer、handler 时 C++ 记映射"的解耦。
        """
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        bufs = self._bufs(layer)
        rr = self._rr[layer]
        buf = bufs[rr]
        self._rr[layer] = (rr + 1) % self.ring
        gen = self._gen
        self._gen += 1
        self._gen_buf[gen] = buf
        # 限制 _gen_buf 大小（每层最多 ring 个在飞 gen；保守留 ring*层数 上限不必，定期裁剪）。
        if len(self._gen_buf) > self.ring * 64:
            for g in sorted(self._gen_buf)[:len(self._gen_buf) - self.ring * 64]:
                self._gen_buf.pop(g, None)
        path = os.path.join(self.src.dir, f"layer{layer:02d}.blob")
        self.submitted += 1
        return _N.prefetch_into_staging(buf, inds_lazy, layer, gen, path, self.stride)

    def promote(self, layer: int, store) -> int:
        """把该层 staging 里已就绪专家写进常驻池（主线程、acquire 前）。"""
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        flat = _N.prefetch_staging_take(layer)
        if not flat:
            return 0
        gen = int(flat[0])
        stg = self._gen_buf.pop(gen, None)     # 按 handler 记的 gen 取回**正是它写过**的那块 buffer
        if stg is None:
            return 0                            # buffer 已被回收/不匹配 → 跳过（不会错配）
        store._resident._ensure_layer(layer)    # promote 在 acquire 前，池可能还没建
        resident = (store.resident_experts(layer)
                    if hasattr(store, "resident_experts") else set())
        pairs = [(int(flat[i]), int(flat[i + 1])) for i in range(1, len(flat), 2)]
        protect = {e for e, _ in pairs}
        # 惰性切片直接放入池：正确性由 gen-匹配（切到 handler 真正写的那块 buffer）+ 环形 buffer
        # （该 buffer 在 ring 次 submit 内不被覆盖，而 pool scatter 会在本前向读池时 eval）保证。
        # 不在此 mx.eval（那是每层 host 同步，在 verify 大流水线上会 2× 拖慢热路径）。
        pend = [(e, self._slice(stg[row])) for e, row in pairs if e not in resident]
        placed = 0
        for e, d in pend:
            try:
                store._resident._place_expert(layer, e, d, current=protect)
            except ValueError:
                break
            placed += 1
        self.promoted += placed
        return placed

    def _slice(self, row: mx.array) -> dict:
        out = {}
        off = 0
        for proj, tensor, dt, shape, nb in self.src._segs:
            seg = row[off:off + nb]
            if tensor == "weight":
                arr = seg.view(mx.uint32).reshape(shape)
            else:
                arr = seg.view(mx.uint16).reshape(shape).view(mx.bfloat16)
            out[f"{proj}.{tensor}"] = arr
            off += nb
        return out
