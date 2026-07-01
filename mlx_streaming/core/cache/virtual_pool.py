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
from mlx_streaming.core.prefetch.native_staging import _StagingSide


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
        self._last_layer = layer_idx

    def _gens(self) -> int:
        # 代数取自常驻池;单代(=1)时读=填=0(持久 LFU 单区),双代(=2)时交替(%2==&1)。
        g = getattr(self._rp, "spec_gens", 2) if self._rp is not None else 2
        return max(1, int(g))

    def read_gen(self) -> int:
        return (self._gen - 1) % self._gens()   # 读上一前向填好的代;单代恒 0

    def fill_gen(self) -> int:
        return self._gen % self._gens()         # fill 写本代;单代恒 0

    def acquire(self, layer, inds, num_experts):
        """真实区表 ∪ 侧区(读代)e2r → 单次 gather。返回 (pool_arrays, local)。"""
        side = _StagingSide(self._stg, self.read_gen())
        return self._rp.acquire_gpu_dual(layer, inds, num_experts, side)

    def prefetch(self, layer, pred, resident, pool_list):
        """向 fill 代 submit 预读：base_row = cap_for(layer) + fill_gen*spec。"""
        g = self.fill_gen()
        base = self._rp.cap_for(layer) + g * self._spec
        return self._stg.submit_pool_sideregion(layer, pred, resident, pool_list, base, gen=g)
