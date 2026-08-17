"""native-fused-prefetch 的 miss→hit 落地：全局多-bank staging。

机制（预读 IO 全在 GPU 完成回调里、主线程零 pread/零 .tolist/零 np→mx.array）：
- submit(layer, inds_lazy)：取该层环里的一块 buffer，调 prefetch_into_staging →
  GPU 完成回调（C++）把专家字节 pread 进这块 buffer，并按 gen 原子记录 (expert→row)。
- promote(layer, store)：读 C++ 记录（纯锁）→ 按 gen 取回 handler 真正写过的那块 buffer →
  惰性切片直接 _place_expert 写进池槽（不 eval，避免每层 host 同步拖慢 verify 流水线）。

每个bank从submit开始由(layer,generation)独占，直到目标demand晋升或迟到任务被回收；
资源不足时安全跳过预取，绝不覆盖在途字节。内存与层数无关。
"""
import os
import time

import mlx.core as mx

from mlx_streaming import config
from mlx_streaming.core.profiling import note_tprof, TPROF_ON

# 诊断门控（STG_VERIFY）：开启时 promote 记录每个 (layer, expert) 落池所用 gen，供消费侧字节校验
# 定位损坏来自哪一代 staging。默认关，对主路径零影响（关时连 dict 写入都不发生）。
_STG_VERIFY = os.environ.get("STG_VERIFY") == "1"


def _typed_seg(seg, dt, shape):
    """按段 dtype 重解释：uint32 原样 / uint16→bfloat16 / uint8(mxfp4 scales) 原样。"""
    if dt == "uint32":
        return seg.view(mx.uint32).reshape(shape)
    if dt == "uint16":
        return seg.view(mx.uint16).reshape(shape).view(mx.bfloat16)
    return seg.reshape(shape)


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
    """全局多-bank staging，支持同一目标的 early + refinement 两次提交。"""
    def __init__(self, blob_source, budget: int, ring: int | None = None):
        self.src = blob_source            # BlobExpertSource：有 dir / stride / _segs
        self.budget = int(budget)
        self.read_budget = max(
            1, min(
                self.budget,
                int(os.environ.get("GLOBAL_STAGING_READ_SLOTS", self.budget)),
            ),
        )
        self.stride = int(blob_source.stride)
        self.ring = max(
            2, int(ring) if ring is not None else config.global_staging_banks(),
        )
        self._banks: "list[mx.array] | None" = None
        self._bank_owner: "list[tuple[int, int, str, int] | None]" = [
            None for _ in range(self.ring)
        ]
        self._gen_bank: "dict[int, int]" = {}
        self._layer_gens: "dict[int, list[int]]" = {}
        self._gen = 1                          # 全局单调 generation
        self._gen_buf: "dict[int, mx.array]" = {}  # gen -> 该次 submit 写的 buffer
        self._late_promoter = None
        self.submitted = 0
        self.host_ready_ids = 0
        self.skipped_no_bank = 0
        self.max_busy_banks = 0
        self.promoted = 0
        # 诊断用:每层最近一次 promote take 到的"已就绪(C++ pread 完成)专家集"。
        # miss_attrib 据此把 miss_A 拆成 时序(不在就绪集) / 驱逐(在就绪集但 acquire 前已不在池)。
        self.last_ready: "dict[int, set]" = {}
        # 诊断用(STG_VERIFY)：记录每个 (layer, expert) 最近一次被 promote 写池时所用的 gen，
        # 供消费侧字节校验定位「损坏来自哪一代 staging」。默认不影响主路径(只是写 dict)。
        self.placed_gen: "dict[tuple, int]" = {}

    def reset_stats(self) -> None:
        self.submitted = 0
        self.host_ready_ids = 0
        self.promoted = 0
        self.skipped_no_bank = 0
        self.max_busy_banks = sum(
            owner is not None for owner in self._bank_owner
        )

    def _bufs(self, layer: int) -> list:
        del layer
        if self._banks is None:
            self._banks = []
            for _ in range(self.ring):
                a = mx.zeros((self.read_budget, self.stride), dtype=mx.uint8)
                mx.eval(a)
                self._banks.append(a)
        return self._banks

    def register_late_promoter(self, promoter) -> None:
        self._late_promoter = promoter

    def _release_bank(self, bank: int) -> None:
        owner = self._bank_owner[int(bank)]
        if owner is None:
            return
        layer, gen, _state, _forward_id = owner
        try:
            import mlx_streaming.native_moe_ext as native
            native.prefetch_staging_forget(int(layer), int(gen))
        except (ImportError, AttributeError):
            # Unit-test fakes and non-native inspection paths need no cleanup.
            pass
        self._gen_bank.pop(gen, None)
        self._gen_buf.pop(gen, None)
        gens = self._layer_gens.get(layer, [])
        if gen in gens:
            gens.remove(gen)
        if not gens:
            self._layer_gens.pop(layer, None)
        self._bank_owner[int(bank)] = None

    def _reap_missed(self) -> None:
        import mlx_streaming.native_moe_ext as native
        for bank, owner in enumerate(tuple(self._bank_owner)):
            if owner is None or owner[2] not in ("missed", "attached"):
                continue
            layer, gen, _state, _forward_id = owner
            if _state == "attached":
                if (native.prefetch_staging_consumed(layer, gen)
                        and native.prefetch_staging_finished(layer, gen)):
                    self._release_bank(bank)
                continue
            flat = native.prefetch_staging_take(layer, gen)
            finished = native.prefetch_staging_finished(layer, gen)
            staging = self._gen_buf.get(gen)
            if (
                config.prefetch_staging_late_promote()
                and flat and len(flat) > 1
                and self._late_promoter is not None and staging is not None
            ):
                self._late_promoter(
                    int(layer), staging, [int(value) for value in flat[1:]],
                )
            if finished:
                self._release_bank(bank)

    def submit(
        self, layer: int, inds_lazy: mx.array,
        resident: "list[int] | None" = None, *, source_layer=-1,
        forward_id=-1, stream=None,
        priority=0,
    ):
        """方案B：inds_lazy 为"预测宽集合"(lazy uint32 [N]，按门控分降序，N 可远大于 budget)。

        resident：目标层提交时刻的常驻专家快照（host 侧）。C++ 回调按 resident 过滤、再按降序
        取前 budget 个"缺口"专家 pread 进 buffer（budget=buffer 行数）。返回 dummy（折进图触发回调）。
        """
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        self._reap_missed()
        bufs = self._bufs(layer)
        busy = sum(owner is not None for owner in self._bank_owner)
        self.max_busy_banks = max(self.max_busy_banks, busy)
        bank = next(
            (index for index, owner in enumerate(self._bank_owner)
             if owner is None),
            None,
        )
        if bank is None:
            self.skipped_no_bank += 1
            return None
        buf = bufs[bank]
        gen = self._gen
        self._gen += 1
        self._bank_owner[bank] = (
            layer, gen, "reserved", int(forward_id),
        )
        self._gen_bank[gen] = bank
        self._layer_gens.setdefault(layer, []).append(gen)
        self._gen_buf[gen] = buf
        path = (
            self.src.native_blob_path(layer)
            if hasattr(self.src, "native_blob_path")
            else f"{self.src.dir}/layer{layer:02d}.blob"
        )
        self.submitted += 1
        res = [int(e) for e in (resident or [])]
        # Physical bank rows are independent from the target pool's bounded
        # speculative-admission allowance.
        kwargs = {"stream": stream} if stream is not None else {}
        return _N.prefetch_into_staging(
            buf, inds_lazy, layer, gen, path, self.stride, res, self.read_budget,
            config.staging_pread_parallel(), int(source_layer), int(forward_id),
            int(priority),
            **kwargs)

    def submit_ready_ids(
        self, layer: int, expert_ids, resident=None, *, source_layer=-1,
        forward_id=-1,
    ) -> bool:
        """Start a staging read for IDs already materialized by source demand."""
        values = [int(value) for value in expert_ids]
        if not values:
            return False
        import mlx_streaming.native_moe_ext as native

        layer = int(layer)
        self._reap_missed()
        bufs = self._bufs(layer)
        busy = sum(owner is not None for owner in self._bank_owner)
        self.max_busy_banks = max(self.max_busy_banks, busy)
        bank = next(
            (index for index, owner in enumerate(self._bank_owner)
             if owner is None),
            None,
        )
        if bank is None:
            self.skipped_no_bank += 1
            return False
        gen = self._gen
        self._gen += 1
        buf = bufs[bank]
        self._bank_owner[bank] = (
            layer, gen, "reserved", int(forward_id),
        )
        self._gen_bank[gen] = bank
        self._layer_gens.setdefault(layer, []).append(gen)
        self._gen_buf[gen] = buf
        path = (
            self.src.native_blob_path(layer)
            if hasattr(self.src, "native_blob_path")
            else f"{self.src.dir}/layer{layer:02d}.blob"
        )
        self.submitted += 1
        native.prefetch_staging_ready_ids(
            buf, values, layer, gen, path, self.stride,
            [int(value) for value in (resident or ())], self.read_budget,
            config.staging_pread_parallel(), int(source_layer),
            int(forward_id), 0,
        )
        self.host_ready_ids += len(values)
        return True

    def wait_for_demand(self, forward_id: int, layer: int, expert_ids) -> None:
        """Join only real-route experts already pending in early/refinement."""
        import mlx_streaming.native_moe_ext as native

        native.prefetch_staging_wait_experts(
            int(forward_id), int(layer), expert_ids,
        )

    @staticmethod
    def note_prejoin(forward_id: int, layer: int, expert_ids) -> None:
        import mlx_streaming.native_moe_ext as native

        native.prefetch_staging_note_prejoin(
            int(forward_id), int(layer), expert_ids,
        )

    @staticmethod
    def finish_demand(forward_id: int, layer: int) -> None:
        import mlx_streaming.native_moe_ext as native

        native.prefetch_staging_finish_demand(int(forward_id), int(layer))

    def take_ready(self, layer: int):
        """Return ready bank leases after the target route synchronization."""
        import mlx_streaming.native_moe_ext as native
        layer = int(layer)
        leases = []
        for gen in tuple(self._layer_gens.get(layer, ())):
            bank = self._gen_bank.get(gen)
            if bank is None:
                continue
            flat = native.prefetch_staging_take(layer, gen)
            if not flat:
                self._bank_owner[bank] = (
                    layer, gen, "missed", int(self._bank_owner[bank][3]),
                )
                continue
            pairs = [int(value) for value in flat[1:]]
            self.last_ready.setdefault(layer, set()).update(pairs[0::2])
            finished = native.prefetch_staging_finished(layer, gen)
            if not finished:
                self._bank_owner[bank] = (
                    layer, gen, "missed", int(self._bank_owner[bank][3]),
                )
            leases.append((bank, self._gen_buf[gen], pairs, finished))
        return leases

    def attach_for_demand(self, layer: int, forward_id: int):
        """Attach target banks without inspecting readiness during graph build."""
        layer = int(layer)
        forward_id = int(forward_id)
        leases = []
        for gen in tuple(self._layer_gens.get(layer, ())):
            bank = self._gen_bank.get(gen)
            if bank is None:
                continue
            owner = self._bank_owner[bank]
            if (owner is None or owner[2] != "reserved"
                    or int(owner[3]) != forward_id):
                continue
            self._bank_owner[bank] = (
                layer, gen, "attached", forward_id,
            )
            leases.append((bank, self._gen_buf[gen], int(gen)))
        return leases

    def release(self, bank: int) -> None:
        self._release_bank(int(bank))

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
    def submit_pool_sideregion(
        self, layer, pred, resident, pool_list, base_row, gen=0, *,
        source_layer=-1, forward_id=-1, priority=0, stream=None,
    ):
        """zero-copy 预取：触发 C++ 把预读专家段散写进池侧区行（填代 gen）。pool_list 为 _segs 顺序的 per-key 池数组。
        seg_nbytes 取自 blob 段表（与 pool key 顺序一致）。spec_slots = 侧区行数（=self.budget）。"""
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        seg_nbytes = [int(nb) for *_, nb in self.src._segs]      # 段字节（与 pool key 顺序一致）
        path = f"{self.src.dir}/layer{layer:02d}.blob"
        self.submitted += 1
        res = [int(e) for e in (resident or [])]
        kwargs = {}
        if stream is not None:
            kwargs["stream"] = stream
        return _N.prefetch_pool_sideregion(
            pool_list, seg_nbytes, pred, layer, path, self.stride, res,
            int(self.budget), int(base_row), gen=int(gen),
            source_layer=int(source_layer), forward_id=int(forward_id),
            priority=int(priority), **kwargs)

    def submit_unified_ready(
        self, layer, expert_ids, resident, pool_list, *, source_layer,
        forward_id, real_cap,
    ):
        """Submit already-materialized IDs without an MLX callback primitive."""
        import mlx_streaming.native_moe_ext as _N
        layer = int(layer)
        seg_nbytes = [int(nb) for *_, nb in self.src._segs]
        path = f"{self.src.dir}/layer{layer:02d}.blob"
        self.submitted += 1
        self.host_ready_ids = getattr(self, "host_ready_ids", 0) + len(expert_ids)
        _N.prefetch_unified_ready_ids(
            pool_list, seg_nbytes, [int(value) for value in expert_ids],
            layer, path, int(self.stride),
            [int(value) for value in (resident or ())], int(self.budget),
            int(real_cap), int(source_layer), int(forward_id),
        )

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
            out[f"{proj}.{tensor}"] = _typed_seg(seg, dt, shape)
            off += nb
        return out
