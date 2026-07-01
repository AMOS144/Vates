"""native-fused-prefetch 的 miss→hit 落地：per-layer 环形 staging buffer + 池 promote。

机制（预读 IO 全在 GPU 完成回调里、主线程零 pread/零 .tolist/零 np→mx.array）：
- submit(layer, inds_lazy)：取该层环里的一块 buffer，调 prefetch_into_staging →
  GPU 完成回调（C++）把专家字节 pread 进这块 buffer，并按 gen 原子记录 (expert→row)。
- promote(layer, store)：读 C++ 记录（纯锁）→ 按 gen 取回 handler 真正写过的那块 buffer →
  惰性切片直接 _place_expert 写进池槽（不 eval，避免每层 host 同步拖慢 verify 流水线）。

为何必须 per-layer（而非全局共享小环）：预读在 GPU 完成回调里异步发生，且 batched verify
是「整块前向、最后一次 eval」——同一前向里全部 48 层的 submit 都在飞行，其 handler 可能要到
接近末尾才触发。若用全局小环，buffer 会在它的 handler 落地/promote 消费**之前**就被后续层的
submit 复用、被 handler 覆盖成别层数据（gen 匹配只保证 buffer 对象、保证不了内容新鲜）→ 非确定性
串扰损坏（实测全局环 n_mismatch=60~85）。所以每层各留独立 buffer，环大小只用于跨 token 复用。

内存 = 层数 × ring × budget × stride。ring 默认从 4 降到 2（STAGING_RING），在保持逐位精确的前提下
把 staging 占用减半（48×2×budget×stride）。
"""
import os
import time

import mlx.core as mx

from mlx_streaming import config
from mlx_streaming.core.profiling import note_tprof, TPROF_ON

# 诊断门控（STG_VERIFY）：开启时 promote 记录每个 (layer, expert) 落池所用 gen，供消费侧字节校验
# 定位损坏来自哪一代 staging。默认关，对主路径零影响（关时连 dict 写入都不发生）。
_STG_VERIFY = os.environ.get("STG_VERIFY") == "1"


def route_used_subset(cand: "list[int]", route_inds: mx.array, num_experts: int) -> "set[int]":
    """返回 cand 中“被 route_inds 真实路由到”的子集（GPU membership，drain ≤len(cand)）。

    cand：预读候选专家 id（host 已知，通常 ≤budget）。
    route_inds：本层真实路由 id（GPU lazy，任意 shape，内部 reshape(-1)）。
    num_experts：专家总数（建 mask 用）。
    cand 为空 → 返回空集（不触发任何 GPU op）。

    仅对预读候选做成员测试：不物化全量路由，只把 ≤len(cand) 个布尔拉回 host。
    """
    if not cand:
        return set()
    route_mask = mx.zeros((num_experts,), dtype=mx.uint8)
    route_mask[route_inds.reshape(-1)] = 1                       # GPU scatter（幂等）
    hit = route_mask[mx.array(cand, dtype=mx.uint32)]            # GPU gather，≤len(cand)
    hit_list = hit.tolist()                                      # 唯一一次极小 drain
    return {int(cand[i]) for i in range(len(cand)) if hit_list[i]}


class NativeStagingManager:
    """per-layer 环形 staging：每层 ring 块 buffer 轮转，按 gen 严格匹配回收。

    submit 写 ring[layer][rr]、记 (gen→buffer)；C++ handler 按 gen 原子记 (expert→row)；
    promote 按 gen 取回**正是 handler 写过**的那块 buffer。只要某层的 buffer 在该层下一次（ring 个
    submit 之后）被复用前，其 handler 已落地且 promote 的惰性切片已随本前向 eval 被消费，即不串扰。
    """
    def __init__(self, blob_source, budget: int, ring: int | None = None):
        self.src = blob_source            # BlobExpertSource：有 dir / stride / _segs
        self.budget = int(budget)
        self.stride = int(blob_source.stride)
        # 安全下限=2：MTP 每步 verify+replay 会对同层各 submit 一次（2 个在飞），ring=1 实测串扰损坏。
        self.ring = max(2, int(ring) if ring is not None else config.staging_ring())
        self._ring: "dict[int, list]" = {}     # layer -> [mx.array]*ring
        self._rr: "dict[int, int]" = {}        # layer -> 下一个要写的环索引
        self._gen = 1                          # 全局单调 generation
        self._gen_buf: "dict[int, mx.array]" = {}  # gen -> 该次 submit 写的 buffer
        self.submitted = 0
        self.promoted = 0
        # 诊断用:每层最近一次 promote take 到的"已就绪(C++ pread 完成)专家集"。
        # miss_attrib 据此把 miss_A 拆成 时序(不在就绪集) / 驱逐(在就绪集但 acquire 前已不在池)。
        self.last_ready: "dict[int, set]" = {}
        # 诊断用(STG_VERIFY)：记录每个 (layer, expert) 最近一次被 promote 写池时所用的 gen，
        # 供消费侧字节校验定位「损坏来自哪一代 staging」。默认不影响主路径(只是写 dict)。
        self.placed_gen: "dict[tuple, int]" = {}

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

    def submit(self, layer: int, inds_lazy: mx.array, resident: "list[int] | None" = None):
        """方案B：inds_lazy 为"预测宽集合"(lazy uint32 [N]，按门控分降序，N 可远大于 budget)。

        resident：目标层提交时刻的常驻专家快照（host 侧）。C++ 回调按 resident 过滤、再按降序
        取前 budget 个"缺口"专家 pread 进 buffer（budget=buffer 行数）。返回 dummy（折进图触发回调）。
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
        if len(self._gen_buf) > self.ring * 64:
            for g in sorted(self._gen_buf)[:len(self._gen_buf) - self.ring * 64]:
                self._gen_buf.pop(g, None)
        path = f"{self.src.dir}/layer{layer:02d}.blob"
        self.submitted += 1
        res = [int(e) for e in (resident or [])]
        # cap=self.budget：回调最多往 buffer 写 budget 行（覆盖缺口分布，p99≈15 → budget=16 够）。
        return _N.prefetch_into_staging(
            buf, inds_lazy, layer, gen, path, self.stride, res, self.budget,
            config.staging_pread_parallel())

    def promote(self, layer: int, store, used: "set[int] | None" = None,
                route_inds: "mx.array | None" = None,
                num_experts: "int | None" = None) -> int:
        """把该层 staging 里已就绪专家写进常驻池（主线程、acquire 前）。

        used：本层真实路由专家集合（host 已知时传入）。只 promote "预读好 ∩ used" 的专家。
        route_inds/num_experts：GPU 重映射路径专用——host 没有现成 used 时，
            用 route_used_subset 仅对预读候选做 GPU membership 现算 used（drain ≤budget）。
        三者皆缺 → 整批写入（兜底，行为同改动前）。
        """
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        _pt0 = time.perf_counter() if TPROF_ON else 0.0
        flat = _N.prefetch_staging_take(layer)
        if TPROF_ON:
            note_tprof("take_s", time.perf_counter() - _pt0)
        if not flat:
            self.last_ready[layer] = set()    # 本层无就绪:miss_A 全归"时序(pread 未完成)"
            if TPROF_ON:
                note_tprof("promote_s", time.perf_counter() - _pt0, count_key="promote_n")
            return 0
        gen = int(flat[0])
        # 记录本层"已就绪(pread 完成)"专家集:无论 buffer 是否匹配，pread 都已完成。
        self.last_ready[layer] = {int(flat[i]) for i in range(1, len(flat), 2)}
        stg = self._gen_buf.pop(gen, None)     # 按 handler 记的 gen 取回**正是它写过**的那块 buffer
        if stg is None:
            if TPROF_ON:
                note_tprof("promote_s", time.perf_counter() - _pt0, count_key="promote_n")
            return 0                            # buffer 已被回收/不匹配 → 跳过（不会错配）
        store._resident._ensure_layer(layer)    # promote 在 acquire 前，池可能还没建
        resident = (store.resident_experts(layer)
                    if hasattr(store, "resident_experts") else set())
        pairs = [(int(flat[i]), int(flat[i + 1])) for i in range(1, len(flat), 2)]
        # GPU 重映射路径无现成 used：仅对预读候选做 GPU membership 现算（不物化全量路由）。
        if used is None and route_inds is not None and num_experts is not None:
            _tr0 = time.perf_counter() if TPROF_ON else 0.0
            used = route_used_subset([e for e, _ in pairs], route_inds, int(num_experts))
            if TPROF_ON:
                note_tprof("route_s", time.perf_counter() - _tr0)
        # 受保护（不被驱逐）= 本层真实路由（若已知），否则退化为全部预读专家。
        protect = set(used) if used is not None else {e for e, _ in pairs}
        # 惰性切片直接放入池：正确性由 gen-匹配（切到 handler 真正写的那块 buffer）+ per-layer 环
        # （该 buffer 在该层 ring 次 submit 内不被覆盖，而 pool scatter 会在本前向读池时 eval）保证。
        # 不在此 mx.eval（那是每层 host 同步，在 verify 大流水线上会显著拖慢热路径）。
        # used 已知时只保留命中真实路由的专家：假阳性不写池（省 scatter、不挤掉有用专家）。
        _tpl0 = time.perf_counter() if TPROF_ON else 0.0
        pend = [(e, self._slice(stg[row])) for e, row in pairs
                if e not in resident and (used is None or e in used)]
        placed = 0
        for e, d in pend:
            try:
                store._resident._place_expert(layer, e, d, current=protect)
            except ValueError:
                break
            if _STG_VERIFY:
                self.placed_gen[(layer, e)] = gen   # 诊断（默认关）：记录该专家这次落池用的 gen
            placed += 1
        if TPROF_ON:
            # place_s 含惰性切片构建 + 池 scatter 入图(不含其 GPU 执行);place_experts=实际写入专家数。
            note_tprof("place_s", time.perf_counter() - _tpl0,
                       count_key="place_experts", count=placed)
            note_tprof("promote_s", time.perf_counter() - _pt0, count_key="promote_n")
        self.promoted += placed
        return placed

    # ---- zero-copy 双源解码：池侧区直接散写（C++ 维护侧区 e→物理行） ----
    def submit_pool_sideregion(self, layer, pred, resident, pool_list, base_row, gen=0):
        """zero-copy 预取：触发 C++ 把预读专家段散写进池侧区行（填代 gen）。pool_list 为 _segs 顺序的 per-key 池数组。
        seg_nbytes 取自 blob 段表（与 pool key 顺序一致）。spec_slots = 侧区行数（=self.budget）。"""
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        seg_nbytes = [int(nb) for *_, nb in self.src._segs]      # 段字节（与 pool key 顺序一致）
        path = f"{self.src.dir}/layer{layer:02d}.blob"
        self.submitted += 1
        res = [int(e) for e in (resident or [])]
        return _N.prefetch_pool_sideregion(
            pool_list, seg_nbytes, pred, layer, path, self.stride, res,
            int(self.budget), int(base_row), gen=int(gen))

    def sideregion_contents(self, layer, gen=0):
        """读该层某代 C++ 侧区缓存当前内容 {expert: 物理侧区行}（纯锁，不消费）。"""
        import mlx_streaming.native_moe_ext as _N
        flat = _N.sideregion_contents(int(layer), int(gen))
        return {int(flat[i]): int(flat[i + 1]) for i in range(0, len(flat), 2)}

    def sideregion_kv(self, layer, gen=0):
        """读该层某代侧区为 (keys uint32, vals int32) 两个 device mx.array（C++ 直接建，供快路径合并）。"""
        import mlx_streaming.native_moe_ext as _N
        return _N.sideregion_kv(int(layer), int(gen))

    def sideregion_reset(self):
        """清空 C++ 侧区缓存（换 prompt/重置统计时调用）。"""
        import mlx_streaming.native_moe_ext as _N
        _N.sideregion_reset()

    def _slice(self, row: mx.array) -> dict:
        # 按段 dtype 通用分派（支持 v1 affine 与 v2 mxfp4）：
        # - uint32：weight，原样 reshape；
        # - uint16：affine 的 scales/biases，view(bfloat16)；
        # - uint8 ：mxfp4 的 scales，保持原样，绝不 view bf16。
        out = {}
        off = 0
        for proj, tensor, dt, shape, nb in self.src._segs:
            seg = row[off:off + nb]
            if dt == "uint32":
                arr = seg.view(mx.uint32).reshape(shape)
            elif dt == "uint16":
                arr = seg.view(mx.uint16).reshape(shape).view(mx.bfloat16)
            else:  # uint8（mxfp4 scales）
                arr = seg.reshape(shape)
            out[f"{proj}.{tensor}"] = arr
            off += nb
        return out


class _StagingSide:
    """把 NativeStagingManager.sideregion_contents 适配成 acquire_gpu_dual 期望的 .contents 接口。

    供 block.py 的双源取用路径使用。gen=读代：双缓冲下消费上一前向填好的那一代。
    """
    def __init__(self, stg, gen: int = 0):
        self._stg = stg
        self._gen = int(gen)

    def contents(self, layer):
        return self._stg.sideregion_contents(layer, self._gen)

    def kv(self, layer):
        # C++ 直接吐 (keys uint32, vals int32) device 数组，供 acquire_gpu_dual 快路径零 host 胶水合并。
        return self._stg.sideregion_kv(layer, self._gen)
