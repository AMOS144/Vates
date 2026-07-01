"""真实 MTP drafter：把 Qwen3NextMTP 包成 mtp_generate 需要的 draft/sync 接口。"""
import mlx.core as mx

from mlx_streaming.mtp.qwen3_next_mtp import mtp_step
from mlx_streaming.mtp.kv_cache import _snapshot, _restore


class MTPDrafter:
    """把 Qwen3NextMTP 包成 mtp_generate 需要的 drafter 接口。"""

    def __init__(self, mtp, lm_head):
        self.mtp = mtp
        self.lm_head = lm_head
        self.embed_tokens = mtp.embed_tokens

    def make_cache(self):
        from mlx_lm.models import cache as kc
        return [kc.KVCache()]            # MTP 单层全注意力

    def draft(self, H_last, x_ids, mtp_cache, K, topk: int = 0):
        # 注意:每草稿一次 int(argmax) host 同步反而最快 —— 它让 MLX 把每步 draft 的
        # 图保持很小、及时释放;试过"全程 GPU argmax、末尾一次同步"攒大惰性图,draft
        # 慢 5×、端到端 24.9→16.3(A/B 证伪,见报告)。保持逐步同步。
        # topk>0(探针用):额外返回每个位置 MTP 的 top-k 候选 id(降序),量树形展开的救回上界。
        drafts, cands = [], []
        h, cur = H_last, x_ids
        for _ in range(K):
            logits, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
            lg = logits[0].reshape(-1)
            if topk > 0:
                order = [int(i) for i in mx.argsort(lg)[-topk:].tolist()][::-1]  # 降序 top-k
                d = order[0]
                cands.append(order)
            else:
                d = int(mx.argmax(lg))
            drafts.append(d)
            h, cur = mh, mx.array([[d]])
        if topk > 0:
            return drafts, cands
        return drafts

    def draft_tree(self, H_last, x_ids, mtp_cache, K):
        """最小树:位置1 展开 top-2,返回两条链 (chainA, chainB),各长 K。

        chainA = [d1a(top-1), d2a, d3a]（同 draft）；chainB = [d1b(top-2), d2b, d3b],
        d2b/d3b 从 d1b 续抽。两链共享 pos1 之后的 MTP 隐状态 mh1,但各自从 d1a/d1b 分叉;
        用 mtp_cache 快照在续抽 A 后回到 pos1 态再续抽 B,保证 B 不带 A 的递归污染。
        """
        logits1, mh1 = mtp_step(self.mtp, H_last, x_ids, self.lm_head, mtp_cache[0])
        lg = logits1[0].reshape(-1)
        top2 = [int(i) for i in mx.argsort(lg)[-2:].tolist()][::-1]   # [d1a, d1b] 降序
        d1a, d1b = top2[0], top2[1]
        snap_pos1 = _snapshot(mtp_cache)          # x 处理后、分叉前的 MTP 递归态

        def _continue(first):
            chain = [first]
            h, cur = mh1, mx.array([[first]])
            for _ in range(K - 1):
                lo, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
                d = int(mx.argmax(lo[0]))
                chain.append(d)
                h, cur = mh, mx.array([[d]])
            return chain

        chainA = _continue(d1a)
        _restore(mtp_cache, snap_pos1)            # 回到 pos1 态,B 链从同一起点分叉
        chainB = _continue(d1b)
        return chainA, chainB

    def draft_paths(self, H_last, x_ids, mtp_cache, K, P):
        """完整树 batch-of-paths:位置1 展开 top-P,返回 P 条链(各长 K)。

        位置1 取 MTP top-P 候选 [d1_0..d1_{P-1}];每个候选从 pos1 的共享 MTP 递归态 mh1 分叉,
        贪婪续抽 K-1 个 token 成一条链。用 mtp_cache 快照保证各链从同一起点分叉、互不污染。
        P=1 退化为普通链;P=2 等价 draft_tree。
        """
        logits1, mh1 = mtp_step(self.mtp, H_last, x_ids, self.lm_head, mtp_cache[0])
        lg = logits1[0].reshape(-1)
        firsts = [int(i) for i in mx.argsort(lg)[-P:].tolist()][::-1]   # top-P 降序
        snap_pos1 = _snapshot(mtp_cache)

        def _continue(first):
            chain = [first]
            h, cur = mh1, mx.array([[first]])
            for _ in range(K - 1):
                lo, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
                d = int(mx.argmax(lo[0]))
                chain.append(d)
                h, cur = mh, mx.array([[d]])
            return chain

        paths = []
        for j, f in enumerate(firsts):
            if j > 0:
                _restore(mtp_cache, snap_pos1)         # 回到 pos1 态,从同一起点分叉
            paths.append(_continue(f))
        return paths

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        """用已接受 token 的真实主模型 hidden 推进 MTP KV cache。

        MTP 在位置 i 消费 (H_i, t_{i+1}) 预测 t_{i+2};因此提交 accepted prefix
        `[t_{i+1}, ..., t_{i+n}]` 时,hidden 序列应为 `[H_i, ..., H_{i+n-1}]`。
        """
        from mlx_streaming.mtp.qwen3_next_mtp import mtp_advance
        h_seq = mx.concatenate([prev_H, rH[:, :-1, :]], axis=1)
        H = mtp_advance(self.mtp, h_seq, replay_in, mtp_cache[0])
        mx.eval(H)
