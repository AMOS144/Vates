"""VirtualPool：预取调度器 + 双源双缓冲协调器（统一收口）。

一个对象承担两职，因为 block.py 用同一个 `self._vpool` 属性：

1. ahead 调度（两种模式都用）：`ahead_for` / `target_for` 决定「第 L 层应预读哪层」——
   早层用小 ahead 保召回、cutoff 起用大 ahead 抢时序。`_native_fused_prefetch` 靠它选目标层。

2. 双源双缓冲协调（仅 ZEROCOPY_DUAL_SOURCE 模式）：`begin_forward` / `read_gen` /
   `fill_gen` / `acquire` / `prefetch`。侧区分两「代(gen)」物理行，每前向读上一代、fill 写
   另一代，根除「本前向消费 gather 读的物理行在本前向 eval 期被 fill 覆盖」的竞态
   （见 spec 2026-06-25-qwen-virtual-pool-double-buffer）。对外只暴露「专家→物理行」单次
   gather 接口，消费者每层零 host 同步。

构造两种签名并存（互斥使用）：
- 调度器：VirtualPool(num_layers=.., cutoff=.., ahead_lo=.., ahead_hi=..)
- 协调器：VirtualPool(resident, staging, spec_slots)，dual-source 下再补调度参数即可两职合一。
"""
import mlx.core as mx

from mlx_streaming.core.prefetch.native_staging import _StagingSide

# 方案B STG_VERIFY 校验累计态（诊断用，默认路径不触及）。
_stg_verify_state = {"ok": 0, "bad": 0, "printed": 0, "calls": 0}


class VirtualPool:
    def __init__(self, resident=None, staging=None, spec_slots=None, *,
                 num_layers=None, cutoff=None, ahead_lo=None, ahead_hi=None):
        # --- 双源协调（resident/staging 存在时启用）---
        self._rp = resident
        self._stg = staging
        self._spec = int(spec_slots) if spec_slots is not None else 0
        self._gen = 0
        self._last_layer = -1        # 前向边界检测：层号回绕(<= 上次) 即新前向
        # --- ahead 调度 ---
        self._num_layers = int(num_layers) if num_layers is not None else 0
        self._cutoff = int(cutoff) if cutoff is not None else 0
        self._a_lo = max(1, int(ahead_lo)) if ahead_lo is not None else 1
        self._a_hi = max(1, int(ahead_hi)) if ahead_hi is not None else 1

    # ---- ahead 调度 ----
    def ahead_for(self, src_layer: int) -> int:
        # cutoff：早层用 lo（保召回），cutoff 起用 hi（保时序）。
        return self._a_lo if int(src_layer) < self._cutoff else self._a_hi

    def target_for(self, src_layer: int) -> int:
        # 目标层 = src+ahead。末层无可预读 → 返回 0（跳过）。
        # 不再 clamp 到末层：clamp 会让多个源层同时预读同一末层，对该层每前向 submit 次数 >
        # staging ring，其环形 buffer 在惰性切片被本前向 eval 消费前就被后续 submit 完成回调覆盖成
        # 别的专家字节 → 池槽装错字节。越界即跳过（交给 demand 读真值），保证每层每前向至多 1 次 submit。
        L = int(src_layer)
        if L >= self._num_layers - 1:
            return 0
        tgt = L + self.ahead_for(L)
        if tgt > self._num_layers - 1:          # 越界（本会触发 clamp 堆叠）→ 跳过
            return 0
        return tgt

    # ---- 双源双缓冲协调 ----
    def begin_forward(self, layer_idx: int):
        """每个 MoE 块 __call__ 开头调；层号回绕(<= 上次) 判为新前向 → 代 +1。
        稳健：不依赖首个 MoE 层是 layer 0、也不要求 MoE 层连续。"""
        if layer_idx <= self._last_layer:
            self._gen += 1
            # 新前向开头排空上一前向提交的侧区 fill：C++-owned 池 buffer 由后台异步直写，
            # 若消费前 fill 未写完，GPU gather 会读到半写侧区行（DUAL_VERIFY BAD）。
            # drain 阻塞到在途 fill 全部写完 → 本前向要消费的侧区行字节必已就绪。
            if self._spec > 0:
                import mlx_streaming.native_moe_ext as _N
                _N.sideregion_drain()
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
        - dual（有侧区 staging 且 spec>0）：真实区表 ∪ 侧区(读代) → acquire_gpu_dual；
          n_experts = layer_cap + spec_gens*spec_slots。
        - 非 dual GPU-remap：acquire_gpu；n_experts = layer_cap。
        （host/fetch 路径见 acquire_host，其输入是 host 侧已 .tolist 的 flat，语义不同故分开。）
        """
        cap = int(layer_cap) if layer_cap is not None else self._rp.cap_for(layer)
        if self._stg is not None and self._spec > 0:
            side_gen = self.read_gen()
            # 方案B：C++ demand_dual 全接管真实区（每层 1 次 inds 同步、零主线程落池/记账）。
            # pinned 非空的层不支持（方案B 假设 PIN_HOT=0），退回 Python 权威路径。
            if getattr(self._rp, "_native_demand", False) and not self._rp._pinned.get(layer):
                pool, local = self._acquire_native(layer, inds, side_gen, cap)
            else:
                side = _StagingSide(self._stg, side_gen)
                pool, local = self._rp.acquire_gpu_dual(layer, inds, num_experts, side)
            n_exp = cap + self._rp.spec_gens * self._rp.spec_slots
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
            m = (pool_list, [int(nb) for *_, nb in segs],
                 f"{stg.src.dir}/layer{int(layer):02d}.blob", int(stg.stride))
            cache[layer] = m
        return m

    def _acquire_native(self, layer, inds, side_gen, cap):
        """方案B 取用：委派 C++ demand_dual（每层 1 次 inds 同步 + 并行 worker pread 落池），
        更新 rp 统计计数（供报告口径一致）。"""
        import mlx_streaming.native_moe_ext as N
        rp = self._rp
        # 方案B 容量前提：真实区仅 cap 槽，本次唯一路由专家数 ≤ inds.size。若 inds.size > cap，
        # 会超容量（多余 miss 落槽0/脏字节）→ 逐位不再正确。用 shape 判定(无同步)，一次性告警。
        if int(inds.size) > int(cap) and not getattr(self, "_overcap_warned", False):
            self._overcap_warned = True
            import sys
            print(f"[NATIVE_DEMAND_DUAL] 警告：inds.size={int(inds.size)} > cap={int(cap)}，"
                  f"真实区超容量，逐位将不正确。请将 EXPERT_SLOTS 提到 ≥ seq·top_k。", file=sys.stderr, flush=True)
        rp._bootstrap_dual_pool(layer)                       # 首次建池 + real_init（幂等）
        pool_list, seg_nbytes, path, stride = self._native_meta(layer)
        local = N.demand_dual(inds, pool_list, seg_nbytes, int(layer), int(side_gen), path,
                              stride, int(cap),
                              rp.eviction_policy == "lfu", int(rp.lfu_decay_interval))
        st = N.demand_last_stats()                           # [hitpos, misspos, loads, fallback01]
        rp.hits += st[0]
        rp.misses += st[2]
        if st[3] == 0:
            rp.gpu_fastpath += 1
        else:
            rp.gpu_fallback += 1
        from mlx_streaming import config
        if config.stg_verify():                              # 诊断:方案B 池字节逐 key 真值校验(默认关)
            self._verify_native_bytes(layer, inds, local)
        return rp._pools[layer], local

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

    def prefetch(self, layer, pred, resident, pool_list):
        """向 fill 代 submit 预读：base_row = cap_for(layer) + fill_gen*spec。"""
        g = self.fill_gen()
        base = self._rp.cap_for(layer) + g * self._spec
        return self._stg.submit_pool_sideregion(layer, pred, resident, pool_list, base, gen=g)
