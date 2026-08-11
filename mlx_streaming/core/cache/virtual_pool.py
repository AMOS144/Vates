"""VirtualPool：预取调度器 + 全局 staging 晋升协调器（统一收口）。

一个对象承担两职，因为 block.py 用同一个 `self._vpool` 属性：

1. ahead 调度（两种模式都用）：`ahead_for` / `targets_for` 决定「第 L 层应预读哪些层」——
   目标早层用小 ahead 保召回、cutoff 后用大 ahead 抢时序。切换处允许一个源层服务两个
   目标，避免旧 source-based 切换漏掉目标层。`_native_fused_prefetch` 靠它选目标层。

2. unified-pool 协调（ZEROCOPY_DUAL_SOURCE 模式）：默认 direct-slot 在单一主池
   中预留最终行，SSD 直写后才原子发布同一张 expert→slot 表；兼容模式
   才先写少量全局 staging bank 并在 demand 时晋升。

构造两种签名并存（互斥使用）：
- 调度器：VirtualPool(num_layers=.., cutoff=.., ahead_lo=.., ahead_hi=..)
- 协调器：VirtualPool(resident, staging, spec_slots)，dual-source 下再补调度参数即可两职合一。
"""
import mlx.core as mx

from mlx_streaming import config

# 方案B STG_VERIFY 校验累计态（诊断用，默认路径不触及）。
_stg_verify_state = {"ok": 0, "bad": 0, "printed": 0, "calls": 0}


class VirtualPool:
    def __init__(self, resident=None, staging=None, spec_slots=None, *,
                 num_layers=None, cutoff=None, ahead_lo=None, ahead_hi=None,
                 ahead_profile=None):
        # --- 双源协调（resident/staging 存在时启用）---
        self._rp = resident
        self._stg = staging
        self._spec = int(spec_slots) if spec_slots is not None else 0
        self._direct_slots = bool(config.prefetch_direct_slots())
        self._gen = 0
        self._last_layer = -1        # 前向边界检测：层号回绕(<= 上次) 即新前向
        # (logical forward, target layer) -> frozen source-time candidate
        # state.  VirtualPool 不是 nn.Module，存 lazy arrays 不会进入参数树。
        self._progressive = {}
        self._progressive_last_width = {}
        self._progressive_waitable = {}
        self._progressive_ready_dummy = {}
        self._progressive_acceptance = {}
        self._progressive_stream = None
        # --- ahead 调度 ---
        self._num_layers = int(num_layers) if num_layers is not None else 0
        self._cutoff = int(cutoff) if cutoff is not None else 0
        self._a_lo = max(1, int(ahead_lo)) if ahead_lo is not None else 1
        self._a_hi = max(1, int(ahead_hi)) if ahead_hi is not None else 1
        self._ahead_profile = {
            int(target): int(ahead)
            for target, ahead in (ahead_profile or {}).items()
        }
        if (
            self._rp is not None
            and self._stg is not None
            and self._spec > 0
            and not self._direct_slots
            and hasattr(self._stg, "register_late_promoter")
        ):
            self._stg.register_late_promoter(self._late_promote_staged)

    # ---- ahead 调度 ----
    def ahead_for(self, target_layer: int) -> int:
        """返回某个目标层应保留的 ahead。

        ``cutoff=6, lo=1, hi=3`` 的语义是目标 1..6 用 T-1，目标
        7..last 用 T-3。必须按 target 判断；旧实现按 source 判断会漏掉
        target 7、8。
        """
        target = int(target_layer)
        if target in self._ahead_profile:
            return self._ahead_profile[target]
        return self._a_lo if target <= self._cutoff else self._a_hi

    def targets_for(self, src_layer: int) -> "tuple[int, ...]":
        """返回源层要提交的全部目标层，每个合法目标恰好映射一次。"""
        source = int(src_layer)
        if source < 0 or source >= self._num_layers - 1:
            return ()
        return tuple(
            target
            for target in range(source + 1, self._num_layers)
            if target - self.ahead_for(target) == source
        )

    def target_for(self, src_layer: int) -> int:
        """兼容旧的单目标调用者；多目标调度必须使用 :meth:`targets_for`。"""
        targets = self.targets_for(src_layer)
        return targets[0] if targets else 0

    # ---- 双源双缓冲协调 ----
    def begin_forward(self, layer_idx: int):
        """每个 MoE 块 __call__ 开头调；层号回绕(<= 上次) 判为新前向 → 代 +1。
        稳健：不依赖首个 MoE 层是 layer 0、也不要求 MoE 层连续。"""
        if layer_idx <= self._last_layer:
            self._gen += 1
            # 上一个 forward 到此已经越过所有 target demand；未被消费的项只
            # 可能来自首次空 cache/prefill，不能泄漏到下一次 refinement。
            self._progressive = {
                key: value
                for key, value in self._progressive.items()
                if key[0] >= self._gen
            }
            self._progressive_last_width = {
                key: value
                for key, value in self._progressive_last_width.items()
                if key[0] >= self._gen
            }
            self._progressive_waitable = {
                key: value
                for key, value in self._progressive_waitable.items()
                if key[0] >= self._gen
            }
            self._progressive_ready_dummy = {
                key: value
                for key, value in self._progressive_ready_dummy.items()
                if key[0] >= self._gen
            }
            # Acceptance samples are intentionally retained until the final
            # benchmark/report barrier.  Evaluating them here would restore a
            # per-layer ids.eval() sync; pruning would silently erase them.
        self._last_layer = layer_idx

    def _gens(self) -> int:
        # 代数取自常驻池;单代(=1)时读=填=0(持久 LFU 单区),双代(=2)时交替(%2==&1)。
        g = getattr(self._rp, "spec_gens", 2) if self._rp is not None else 2
        return max(1, int(g))

    def read_gen(self) -> int:
        return (self._gen - 1) % self._gens()   # 读上一前向填好的代;单代恒 0

    def fill_gen(self) -> int:
        return self._gen % self._gens()         # fill 写本代;单代恒 0

    def acquire(self, layer, inds, num_experts, *, seq_len=None, layer_cap=None):
        """统一取用入口（GPU-remap 路径）：对外呈现「所有专家都在」的视角。

        返回 (pool_arrays, local, n_experts)，计算侧零分支：
        - direct-slot：真实行与预取最终行都可被 GPU remap 直接寻址；
        - compatibility staging：已完成 staging 先晋升真实行，再 native demand；
        - 非 dual GPU-remap：acquire_gpu；n_experts = layer_cap。
        （host/fetch 路径见 acquire_host，其输入是 host 侧已 .tolist 的 flat，语义不同故分开。）
        """
        cap = int(layer_cap) if layer_cap is not None else self._rp.cap_for(layer)
        if self._stg is not None and self._spec > 0:
            side_gen = self.read_gen()
            # C++ demand_dual 是双源 decode 真实区的唯一权威（每层 1 次 inds 同步、零主线程落池/记账）。
            # 无 Python 退路：native 未编译 demand_dual → 明确报错（decode 依赖 native 全套能力）。
            if not getattr(self._rp, "_native_demand", False):
                raise RuntimeError(
                    "双源 decode 需要 native demand_dual（已成唯一权威），但未检出。请编译 native_moe_ext。")
            if self._direct_slots:
                pool, local, n_exp = self._acquire_direct(
                    layer, inds, side_gen, cap, seq_len=seq_len,
                )
            else:
                pool, local, n_exp = self._acquire_native(
                    layer, inds, side_gen, cap, seq_len=seq_len,
                )
            return pool, local, n_exp
        pool, local = self._rp.acquire_gpu(layer, inds, num_experts)
        return pool, local, cap

    def _native_meta(self, layer):
        """缓存每层 demand_dual 的不变入参（pool_list/seg_nbytes/path），避免逐层重建 host 胶水。"""
        cache = getattr(self, "_nd_meta", None)
        if cache is None:
            cache = self._nd_meta = {}
        m = cache.get(layer)
        if m is None:
            stg = self._stg
            segs = stg.src._segs                             # (proj, tensor, dt, shape, nb)，与池 key 同序
            pool_list = [self._rp._pools[layer][f"{p}.{t}"] for p, t, *_ in segs]
            path = (stg.src.native_blob_path(layer)
                    if hasattr(stg.src, "native_blob_path")
                    else f"{stg.src.dir}/layer{int(layer):02d}.blob")
            m = (pool_list, [int(nb) for *_, nb in segs], path, int(stg.stride))
            cache[layer] = m
        return m

    def _late_promote_staged(self, layer, staging, staging_map, actual_ids=None):
        """把已完成的 staging 候选按 speculative 语义并入统一主池。"""
        import mlx_streaming.native_moe_ext as native

        layer = int(layer)
        self._rp._bootstrap_dual_pool(layer)
        pool_list, seg_nbytes, _path, _stride = self._native_meta(layer)
        args = (
            pool_list, seg_nbytes, layer, int(self._rp.cap_for(layer)),
            int(self._spec), staging, staging_map,
        )
        if actual_ids is not None:
            return native.demand_promote_staged(*args, actual_ids)
        return native.late_promote_staged(*args)

    def _acquire_native(self, layer, inds, side_gen, cap, *, seq_len=None):
        """方案B 取用：委派 C++ demand_dual（每层 1 次 inds 同步 + 并行 worker pread 落池），
        更新 rp 统计计数（供报告口径一致）。"""
        import mlx_streaming.native_moe_ext as N
        rp = self._rp
        rp._bootstrap_dual_pool(layer)                       # 首次建池 + real_init（幂等）
        pool_list, seg_nbytes, path, stride = self._native_meta(layer)
        progressive_key = (int(self._gen), int(layer))
        progressive_demand = progressive_key in self._progressive_last_width
        if progressive_demand:
            if self._progressive_waitable.get(progressive_key, False):
                # The route is now known. Join only its experts that early/tail
                # already reserved; unpredicted experts remain fallback.
                self._wait_progressive_at_demand(layer, inds, side_gen)
            else:
                self._stg.note_prejoin(int(self._gen), int(layer), inds)
        # 与源分支一致：一个 native 状态机内只同步一次 inds，同时完成所有
        # ready bank 的路由命中晋升、speculative 准入、miss pread 与 remap。
        leases = self._stg.take_ready(layer)
        banks = [lease[0] for lease in leases]
        staging_list = [lease[1] for lease in leases]
        staging_maps = [lease[2] for lease in leases]
        finished = [lease[3] for lease in leases]
        try:
            local = N.demand_staged_multi(
                inds, pool_list, seg_nbytes, int(layer), path, stride, int(cap),
                rp.eviction_policy == "lfu", int(rp.lfu_decay_interval),
                int(self._spec), staging_list, staging_maps,
                forward_id=int(self._gen),
                sequence_length=(-1 if seq_len is None else int(seq_len)),
            )
        finally:
            for bank, is_finished in zip(banks, finished):
                if is_finished:
                    self._stg.release(bank)
            if progressive_demand:
                self._stg.finish_demand(int(self._gen), int(layer))
                self._progressive_last_width.pop(progressive_key, None)
                self._progressive_waitable.pop(progressive_key, None)
        self._record_progressive_acceptance(layer, inds)
        st = N.demand_last_stats()                           # [hitpos, misspos, loads, fallback012]
        if st[3] == 2:
            # C++ 已在真实区发生任何分配/驱逐前判定：本次非 side 唯一专家 + 无关
            # pinned 无法同时容纳。临时堆叠完整真实路由，保证小 per-layer cap 只影响
            # 偶发 overflow 的性能，不影响模型数值或下一次 resident 状态。
            pool, local, n_exp = self._temporary_fetch(layer, inds)
            rp.misses += n_exp
            rp.gpu_fallback += 1
            return pool, local, n_exp
        rp.hits += st[0]
        rp.misses += st[2]
        if st[3] == 0:
            rp.gpu_fastpath += 1
        else:
            rp.gpu_fallback += 1
        from mlx_streaming import config
        if config.stg_verify():                              # 诊断:方案B 池字节逐 key 真值校验(默认关)
            self._verify_native_bytes(layer, inds, local)
        return (
            rp._pools[layer], local,
            int(cap),
        )

    def _acquire_direct(self, layer, inds, side_gen, cap, *, seq_len=None):
        """Consume prefetch rows in-place; no staging-to-main promotion."""
        import mlx_streaming.native_moe_ext as N

        rp = self._rp
        rp._bootstrap_dual_pool(layer)
        # Layer 0 has no legal cross-layer producer. Its direct prefetch rows
        # are therefore ordinary final pool rows owned by the real table,
        # rather than an unreachable reserved range.
        demand_cap = int(rp.native_real_cap_for(layer))
        pool_list, seg_nbytes, path, stride = self._native_meta(layer)
        progressive_key = (int(self._gen), int(layer))
        ready_dummy = self._progressive_ready_dummy.pop(progressive_key, None)
        if ready_dummy is not None:
            inds = inds + (
                ready_dummy.reshape(()).astype(inds.dtype) * 0
            )
        if config.prefetch_pinned_gpu_demand():
            local = N.demand_gpu_remap_only(
                inds, int(layer), int(side_gen), demand_cap, False,
            )
            return (
                rp._pools[layer], local,
                int(rp.allocated_slots(layer)),
            )
        if ready_dummy is not None and config.prefetch_exact_gpu_demand():
            local = N.demand_gpu_remap_only(
                inds, int(layer), int(side_gen), demand_cap, False,
            )
            self._progressive_last_width.pop(progressive_key, None)
            self._progressive_waitable.pop(progressive_key, None)
            self._defer_progressive_acceptance(layer, inds)
            return (
                rp._pools[layer], local,
                int(rp.allocated_slots(layer)),
            )
        progressive_demand = progressive_key in self._progressive_last_width
        pre_wait_deadline = False
        if (
            progressive_demand
            and self._progressive_waitable.get(progressive_key, False)
            and config.prefetch_progressive_demand_wait()
            and not config.demand_async()
        ):
            if config.prefetch_deadline_prof():
                # User-facing hit rate is the state at target entry, before
                # this optional compatibility wait.  Snapshot it now so rows
                # completed while waiting cannot masquerade as prefetch hits.
                N.demand_deadline_snapshot(
                    inds, int(layer), int(side_gen), True,
                )
                pre_wait_deadline = True
            N.sideregion_wait_experts(
                int(self._gen), int(layer), int(side_gen), inds,
            )
        if config.demand_async():
            if pre_wait_deadline:
                raise RuntimeError(
                    "DEMAND_ASYNC 与 PREFETCH_PROGRESSIVE_DEMAND_WAIT 不兼容",
                )
            demand_args = (
                inds, pool_list, seg_nbytes, int(layer), int(side_gen), path,
                stride, demand_cap, rp.eviction_policy == "lfu",
                int(rp.lfu_decay_interval),
            )
            demand_kwargs = dict(
                forward_id=int(self._gen),
                sequence_length=(-1 if seq_len is None else int(seq_len)),
                use_side=False,
                wait_for_pending=(
                    config.prefetch_wait_predicted_pending()
                    or (
                    progressive_demand
                    and self._progressive_waitable.get(progressive_key, False)
                    and config.prefetch_progressive_demand_wait()
                    )
                ),
                wait_for_refinement=(
                    progressive_demand
                    and self._progressive_waitable.get(progressive_key, False)
                    and config.prefetch_progressive_demand_wait()
                ),
                evaluator_submit=config.demand_async_python_submit(),
            )
            if config.demand_async_python_submit():
                entry_local, final_local = N.demand_dual_split_async(
                    *demand_args, **demand_kwargs,
                )
                # Non-blocking evaluator-owned submission: the remap command
                # buffer can complete and start true-miss I/O before the final
                # SharedEvent wait is encoded. No ids.eval() and no native
                # max_ops no-op padding are needed.
                mx.async_eval(entry_local)
                local = final_local
            else:
                local = N.demand_dual_async(
                    *demand_args, **demand_kwargs,
                )
        else:
            local = N.demand_dual(
                inds, pool_list, seg_nbytes, int(layer), int(side_gen), path,
                stride, demand_cap, rp.eviction_policy == "lfu",
                int(rp.lfu_decay_interval), forward_id=int(self._gen),
                sequence_length=(-1 if seq_len is None else int(seq_len)),
                use_side=False,
                record_deadline=not pre_wait_deadline,
            )
        if config.demand_async():
            # Completion-handler statistics are not ready while the lazy graph
            # is being assembled. Native cumulative counters are collected at
            # the final graph synchronization instead of forcing a host wait.
            self._defer_progressive_acceptance(layer, inds)
            return (
                rp._pools[layer], local,
                int(rp.allocated_slots(layer)),
            )
        st = N.demand_last_stats()
        rp.hits += st[0]
        rp.misses += st[2]
        if st[3] == 0:
            rp.gpu_fastpath += 1
        else:
            rp.gpu_fallback += 1
        if progressive_demand:
            self._progressive_last_width.pop(progressive_key, None)
            self._progressive_waitable.pop(progressive_key, None)
        self._record_progressive_acceptance(layer, inds)
        return (
            rp._pools[layer], local,
            int(rp.allocated_slots(layer)),
        )

    def _wait_progressive_at_demand(self, layer, inds, side_gen) -> bool:
        """Join only real-route rows already pending in multi-step staging."""
        del side_gen  # physical side generations no longer exist
        key = (int(self._gen), int(layer))
        if key not in self._progressive_last_width:
            return False
        self._stg.wait_for_demand(int(self._gen), int(layer), inds)
        return True

    def _record_progressive_acceptance(self, layer, actual_ids) -> bool:
        key = (int(self._gen), int(layer))
        if not self._defer_progressive_acceptance(layer, actual_ids):
            return False
        state = self._progressive_acceptance.pop(key)
        self._materialize_progressive_acceptance(key, state)
        return True

    def _defer_progressive_acceptance(self, layer, actual_ids) -> bool:
        """Attach truth without synchronizing the per-layer lazy GPU route."""
        key = (int(self._gen), int(layer))
        state = self._progressive_acceptance.get(key)
        if state is None:
            return False
        state["actual_ids"] = actual_ids
        return True

    @staticmethod
    def _materialize_progressive_acceptance(key, state) -> None:
        from mlx_streaming.core.prefetch import progressive_acceptance

        progressive_acceptance.record(
            int(key[1]),
            candidate_ids=state["candidate_ids"],
            selected_ids=state["selected_ids"],
            online_width=state["online_width"],
            actual_ids=state["actual_ids"],
            resident=state["resident"],
        )

    def flush_progressive_acceptance(self) -> int:
        """Materialize completed samples once, after the generation barrier."""
        completed = [
            (key, state)
            for key, state in self._progressive_acceptance.items()
            if "actual_ids" in state
        ]
        for key, _state in completed:
            self._progressive_acceptance.pop(key, None)
        for key, state in completed:
            self._materialize_progressive_acceptance(key, state)
        return len(completed)

    def reset_progressive_acceptance_pending(self) -> None:
        self._progressive_acceptance.clear()

    def record_rerank_acceptance(
        self, target_layer, *, candidate_ids, selected_ids, online_width,
        resident,
    ) -> None:
        """Retain an early-only rerank sample until target truth is available.

        All arrays stay lazy here.  Target demand attaches ``actual_ids`` and
        the benchmark's final barrier materializes the sample once, so audit
        mode does not restore a source- or target-layer ``ids.eval()``.
        """
        key = (int(self._gen), int(target_layer))
        if key in self._progressive_acceptance:
            raise RuntimeError(
                f"duplicate rerank acceptance state for forward/target={key}",
            )
        self._progressive_acceptance[key] = {
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
            "online_width": online_width,
            "resident": tuple(int(value) for value in (resident or ())),
        }

    def _temporary_fetch(self, layer, inds):
        """为真实区 over-cap demand 构造不写 resident 的临时连续专家批。"""
        import mlx.core as mx

        flat = [int(value) for value in inds.reshape(-1).tolist()]
        unique = list(dict.fromkeys(flat))
        remap = {expert: row for row, expert in enumerate(unique)}
        rp = self._rp
        if rp.stacked_batch_loader is not None:
            stacked = rp.stacked_batch_loader(layer, unique)
        else:
            if rp.batch_loader is not None:
                loaded = rp.batch_loader(layer, unique)
            else:
                loaded = {expert: rp.loader(layer, expert) for expert in unique}
            first = loaded[unique[0]]
            stacked = {
                key: mx.stack([loaded[expert][key] for expert in unique])
                for key in first
            }
        local = mx.array(
            [remap[expert] for expert in flat], dtype=inds.dtype,
        ).reshape(inds.shape)
        return stacked, local, len(unique)

    def _verify_native_bytes(self, layer, inds, local):
        """诊断(STG_VERIFY，方案B)：校验「字节落池不变量」——真实区每个占用槽的池字节 == 该槽当前
        C++ 属主专家(g_real)的 blob 真值。这是 C++ 接管落池的字节等价铁证；发现不一致即池装错字节。

        注：不以 local→expert 为判据（local 可能因跨调用/多模型共享 g_real 而滞后于 g_real，属路由级
        问题、非落池字节问题；逐位权威信号是 e2e n_mismatch）。
        """
        st = _stg_verify_state
        st["calls"] += 1
        pool = self._rp._pools.get(layer)
        if pool is None:
            return
        import mlx_streaming.native_moe_ext as N
        stg = self._stg
        path = f"{stg.src.dir}/layer{int(layer):02d}.blob"
        segs = stg.src._segs
        flat = N.real_region_contents(int(layer))               # [expert0,slot0,expert1,slot1,...]
        for j in range(0, len(flat), 2):
            e, slot = flat[j], flat[j + 1]
            raw = N.blob_load(path, mx.array([e], dtype=mx.uint32), int(stg.stride))[0]
            bad, off = None, 0
            for p, t, dt, shape, nb in segs:
                k = f"{p}.{t}"
                pv = pool[k][slot].reshape(-1).view(mx.uint8)
                if not bool(mx.all(pv == raw[off:off + nb])):
                    bad = k
                    break
                off += nb
            if bad is None:
                st["ok"] += 1
            else:
                st["bad"] += 1
                if st["printed"] < 12:
                    st["printed"] += 1
                    print(f"[STG_VERIFY-DUAL] BAD 落池字节错 call={st['calls']} layer={layer} "
                          f"expert={e} slot={slot} key={bad} (ok={st['ok']} bad={st['bad']})", flush=True)

    def acquire_host(self, layer, flat, inds_shape, inds_dtype, layer_cap):
        """host/fetch 路径收口（prefill/大 seq 或关 GPU-remap）：flat 为 host 侧路由 id 列表。

        返回 (pool_arrays, local, n_experts)，与 block.py 原 host/fetch 分支逐元素等价：
        - uniq <= cap：acquire(flat)，local 为槽位；n_experts = layer_cap。
        - uniq > cap：fetch(uniq_sorted)，local 为 remap 到 [0,uniq) 连续索引；n_experts = uniq 数。
        """
        import mlx.core as mx
        cap = int(layer_cap)
        uniq_set = set(flat)
        if len(uniq_set) <= cap:
            pool, slots = self._rp.acquire(layer, flat)
            local = mx.array(slots, dtype=inds_dtype).reshape(inds_shape)
            return pool, local, cap
        uniq_sorted = sorted(uniq_set)
        remap = {g: i for i, g in enumerate(uniq_sorted)}
        local = mx.array([remap[i] for i in flat], dtype=inds_dtype).reshape(inds_shape)
        fetched = self._rp.fetch(layer, uniq_sorted)
        return fetched, local, len(uniq_sorted)

    def prefetch(self, layer, pred, resident, pool_list, *, source_layer=None):
        """Submit either into final direct rows or the global staging ring."""
        if self._direct_slots:
            gen = self.fill_gen()
            # Negative base selects unified-main ownership in native code;
            # its magnitude is the full real capacity.  No side rows/table.
            base = -int(self._rp.cap_for(layer))
            return self._stg.submit_pool_sideregion(
                layer, pred, resident, pool_list, base, gen=gen,
                source_layer=(-1 if source_layer is None else int(source_layer)),
                forward_id=int(self._gen),
            )
        del pool_list
        return self._stg.submit(
            layer, pred, resident,
            source_layer=(-1 if source_layer is None else int(source_layer)),
            forward_id=int(self._gen),
        )

    # ---- progressive early-core + exact late-fill coordination ----
    def record_progressive(
        self,
        target_layer,
        *,
        candidate_logits,
        early_ids,
        resident,
        pool_list,
        top_k,
        candidate_width=64,
        early_dummy=None,
    ):
        """Freeze the original source candidate state for same-forward T-1."""
        key = (int(self._gen), int(target_layer))
        if key in self._progressive:
            raise RuntimeError(
                f"duplicate progressive source state for forward/target={key}",
            )
        self._progressive[key] = {
            "candidate_logits": candidate_logits,
            "early_ids": early_ids,
            "resident": tuple(int(value) for value in (resident or ())),
            "pool_list": list(pool_list),
            "top_k": int(top_k),
            "candidate_width": int(candidate_width),
            "early_dummy": early_dummy,
        }

    def progressive_stream(self):
        """Dedicated device stream for non-blocking exact tail refinement."""
        if self._progressive_stream is None:
            self._progressive_stream = mx.new_stream(mx.default_device())
        return self._progressive_stream

    def has_progressive(self, target_layer) -> bool:
        return (int(self._gen), int(target_layer)) in self._progressive

    def refine_progressive(
        self, target_layer, exact_logits, *, source_layer, exact_route=False,
    ):
        """Submit the full core-preserving final set from exact T-1 logits."""
        key = (int(self._gen), int(target_layer))
        state = self._progressive.pop(key, None)
        if state is None:
            return None
        from mlx_streaming.core.prefetch.progressive import (
            exact_route_union_ids,
            refined_ids,
        )

        if exact_route and config.prefetch_exact_gpu_demand():
            # The exact adjacent decoder computation is moved (and later
            # reused), so its real route can close the last few top64 misses
            # before a handler-free GPU-only demand.
            ids, width = exact_route_union_ids(
                exact_logits,
                top_k=state["top_k"],
                side_capacity=self._spec,
                resident=state["resident"],
            )
        elif exact_route:
            # Adjacent attention provides a better ranking signal, not a new
            # candidate universe.  Keep the same frozen raw-top64 contract so
            # acceptance and production are measuring the same rerank.
            ids, width = refined_ids(
                exact_logits,
                state["candidate_logits"],
                state["early_ids"],
                top_k=state["top_k"],
                # Exact ranking does not relax the production output budget.
                # The merged pool may retain many more rows, but one K3
                # occurrence must not silently expand to the cache size.
                side_capacity=min(
                    self._spec, config.prefetch_progressive_max_width(),
                ),
                resident=state["resident"],
                candidate_width=state["candidate_width"],
                union_margin=config.prefetch_progressive_union_margin_for(
                    int(target_layer),
                ),
            )
        else:
            ids, width = refined_ids(
                exact_logits,
                state["candidate_logits"],
                state["early_ids"],
                top_k=state["top_k"],
                # Physical side rows are a persistent cache and may be larger
                # than one occurrence's legal rerank set.  Never let cache sizing
                # silently widen the logical prediction output.
                side_capacity=min(
                    self._spec, config.prefetch_progressive_max_width(),
                ),
                resident=state["resident"],
                candidate_width=state["candidate_width"],
                union_margin=config.prefetch_progressive_union_margin_for(
                    int(target_layer),
                ),
            )
        # Preserve callback order across streams without a host wait: the
        # final full-union submission cannot become ready before early submit.
        early_dummy = state.get("early_dummy")
        if early_dummy is not None:
            dependency = mx.sum(early_dummy.astype(mx.uint32)) * 0
            ids = ids + dependency.astype(ids.dtype)
        self._progressive_last_width[key] = width
        if config.prefetch_audit_prof():
            candidate_logits = state["candidate_logits"]
            num_experts = int(candidate_logits.shape[-1])
            candidate_width = min(
                num_experts, int(state["candidate_width"]),
            )
            candidate_ids = mx.argpartition(
                candidate_logits.reshape(-1, num_experts),
                kth=-candidate_width,
                axis=-1,
            )[:, -candidate_width:]
            self._progressive_acceptance[key] = {
                "candidate_ids": candidate_ids,
                "selected_ids": ids,
                "online_width": width,
                "resident": state["resident"],
            }
        # refined_ids 为了让 route-critical 读取优先，布局为
        # [selected exact tail, early core, unselected padding]，并不是旧的
        # [early core, tail]。兼容 staging 路径必须从前缀取得 tail；按
        # core_width 切片会恰好丢掉排序最靠前的精确专家。direct-slot 路径
        # 仍提交完整合法并集，以便 LFU 在一次 victim 选择中保护 early+tail。
        core_width = int(state["early_ids"].size)
        late_capacity = int(ids.size) - core_width
        late_ids = ids[:late_capacity]
        if int(late_ids.size) > 0:
            late_count = mx.maximum(0, width - core_width)
            late_ids = mx.where(
                mx.arange(int(late_ids.size)) < late_count,
                late_ids,
                # Fixed-shape padding must stay inside the selected set.  In
                # particular K3 can legally need zero tail experts; padding
                # with an unselected tail would violate the 150% union cap.
                mx.broadcast_to(state["early_ids"][:1], late_ids.shape),
            )
            if self._direct_slots:
                # Side LFU protects only experts present in this submission's
                # P-set while choosing victims.  Passing just the tail lets it
                # evict early-core rows moments before target demand.  Resubmit
                # the full legal union: already-published core rows are native
                # hits (no reread), while the complete P-set remains protected.
                gen = self.fill_gen()
                base = -int(self._rp.cap_for(target_layer))
                dummy = self._stg.submit_pool_sideregion(
                    int(target_layer), ids, state["resident"],
                    state["pool_list"], base, gen=gen,
                    source_layer=int(source_layer), forward_id=int(self._gen),
                    stream=self.progressive_stream(),
                    priority=2 if exact_route else 1,
                )
            else:
                dummy = self._stg.submit(
                    int(target_layer), late_ids, state["resident"],
                    source_layer=int(source_layer), forward_id=int(self._gen),
                    stream=self.progressive_stream(),
                    priority=1,
                )
        else:
            dummy = None
        # A full bank ring is a legal prefetch drop, not a reason to wait for
        # an event that can never be published. Demand will use ready early
        # rows and ordinary fallback for the missing tail.
        self._progressive_waitable[key] = dummy is not None
        if exact_route and dummy is not None:
            self._progressive_ready_dummy[key] = dummy
        num_experts = int(exact_logits.shape[-1])
        route_ids = mx.contiguous(mx.argpartition(
            exact_logits.reshape(-1, num_experts),
            kth=-state["top_k"],
            axis=-1,
        )[:, -state["top_k"]:].reshape(-1).astype(mx.uint32))
        return dummy, route_ids

    def wait_progressive(self, target_layer, route_ids):
        """兼容诊断开关：全局 staging 不再等待旧侧区 pending map。"""
        del target_layer
        mx.eval(route_ids)
