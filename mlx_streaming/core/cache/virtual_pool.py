"""VirtualPool：per-layer 自适应 ahead 调度器（cutoff）。

只负责「第 L 层应预测多远（ahead）/ 目标层是哪层」这一调度决策：
早层用小 ahead 保召回、晚层用大 ahead 抢时序（Phase 0 实测）。
不持有 staging/池、不碰 submit/promote/acquire —— 字节搬运与图内 scatter
仍由 block.py 既有路径完成（保留 promote→miss_attrib→acquire 顺序，正确性已验证）。
"""


class VirtualPool:
    def __init__(self, num_layers, cutoff, ahead_lo, ahead_hi):
        self._num_layers = int(num_layers)
        self._cutoff = int(cutoff)
        self._a_lo = max(1, int(ahead_lo))
        self._a_hi = max(1, int(ahead_hi))

    def ahead_for(self, src_layer: int) -> int:
        # cutoff：早层用 lo（保召回），cutoff 起用 hi（保时序）。
        return self._a_lo if int(src_layer) < self._cutoff else self._a_hi

    def target_for(self, src_layer: int) -> int:
        # 目标层 = src+ahead。末层无可预读 → 返回 0（跳过）。
        # 不再 clamp 到末层：clamp 会让多个源层（如 ahead=3 时 44/45/46）同时预读同一末层，
        # 对该层每前向 submit 次数 > staging ring（=2），其环形 buffer 在 promote 的惰性切片被
        # 本前向 eval 消费前就被后续 submit 的完成回调覆盖成别的专家字节 → 池槽装错字节（确定性
        # 损坏，混合 ahead 下首现于第 ~50 个 token）。越界即跳过（交给 demand 读真值），保证每层
        # 每前向至多 1 次 submit，ring=2 充足、无覆盖竞态。末层仍由恰好命中它的源层（N-1-ahead）预读。
        L = int(src_layer)
        if L >= self._num_layers - 1:
            return 0
        tgt = L + self.ahead_for(L)
        if tgt > self._num_layers - 1:          # 越界（本会触发 clamp 堆叠）→ 跳过
            return 0
        return tgt
