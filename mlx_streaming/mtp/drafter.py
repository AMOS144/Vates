"""真实 MTP drafter：把 Qwen3NextMTP 包成 mtp_generate 需要的 draft/sync 接口。"""
import mlx.core as mx

from mlx_streaming.mtp.qwen3_next_mtp import mtp_step


class MTPDrafter:
    """把 Qwen3NextMTP 包成 mtp_generate 需要的 drafter 接口。"""

    def __init__(self, mtp, lm_head):
        self.mtp = mtp
        self.lm_head = lm_head
        self.embed_tokens = mtp.embed_tokens

    def make_cache(self):
        from mlx_lm.models import cache as kc
        return [kc.KVCache()]            # MTP 单层全注意力

    def draft(self, H_last, x_ids, mtp_cache, K):
        # 注意:每草稿一次 int(argmax) host 同步反而最快 —— 它让 MLX 把每步 draft 的
        # 图保持很小、及时释放;试过"全程 GPU argmax、末尾一次同步"攒大惰性图,draft
        # 慢 5×、端到端 24.9→16.3(A/B 证伪,见报告)。保持逐步同步。
        drafts = []
        h, cur = H_last, x_ids
        for _ in range(K):
            logits, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
            d = int(mx.argmax(logits[0]))
            drafts.append(d)
            h, cur = mh, mx.array([[d]])
        return drafts

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        """用已接受 token 的真实主模型 hidden 推进 MTP KV cache。

        MTP 在位置 i 消费 (H_i, t_{i+1}) 预测 t_{i+2};因此提交 accepted prefix
        `[t_{i+1}, ..., t_{i+n}]` 时,hidden 序列应为 `[H_i, ..., H_{i+n-1}]`。
        """
        from mlx_streaming.mtp.qwen3_next_mtp import mtp_advance
        h_seq = mx.concatenate([prev_H, rH[:, :-1, :]], axis=1)
        H = mtp_advance(self.mtp, h_seq, replay_in, mtp_cache[0])
        mx.eval(H)
