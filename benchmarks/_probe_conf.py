"""置信度↔接受率相关性探针:判断"动态自适应深度(P-MTP 置信门控)"在本系统有没有信号。

原理:动态深度要成立,MTP 在第 i 位的 top-1 softmax 置信度 p_i 必须能预测该位草稿是否被接受。
若高置信步接受率显著高于低置信步,则可按累计置信度门控 n_max(高→抽深、低→抽浅省专家加载)。

实现无需改热路径:包一层 drafter,draft() 复制原抽样循环但额外记录每位 top-1 概率;sync() 收到的
replay_in 长度 = 本步接受长度(min(matched+1,K)),据此还原逐位接受。走 plain 单链(TREE_TOP2=0)。

口径(K=3):acc_len>=2 ⇔ 第0位草稿被接受;acc_len>=3 ⇔ 第1位也被接受(第2位与 matched==K 混叠,不测)。
"""
import json
import os

import mlx.core as mx
from mlx_lm.models.qwen3_next import ModelArgs

from mlx_streaming.mtp.drafter import MTPDrafter
from mlx_streaming.mtp.qwen3_next_mtp import load_mtp, mtp_step
from mlx_streaming.mtp.generate import mtp_generate
from mlx_streaming.model_builder import build_streaming_model
from mlx_streaming import config as _cfg

os.environ["TREE_TOP2"] = "0"          # 走 plain 单链
K = int(os.environ.get("K", "3"))
MAXTOK = int(os.environ.get("MAXTOK", "128"))

PROMPTS = [
    "用三句话解释什么是混合专家模型。",
    "写一段 Python 代码，演示如何用 LRU 缓存函数结果。",
    "为什么模型量化会影响困惑度和生成质量？",
    "用英文写一段关于 speculative decoding 的技术摘要。",
    "请写一个短故事，主题是工程师在午夜调试模型推理性能。",
    "请给出一个使用 Python 解析 JSONL 文件并统计字段频率的例子。",
]


class ConfProbeDrafter:
    """包装真实 drafter:记录每步逐位 top-1 概率,并在 sync 时配对该步接受长度。"""

    def __init__(self, mtp, lm_head):
        self.inner = MTPDrafter(mtp, lm_head)
        self.mtp = mtp
        self.lm_head = lm_head
        self.records = []          # [(probs[list], acc_len)]
        self._last_probs = None

    def make_cache(self):
        return self.inner.make_cache()

    def draft(self, H_last, x_ids, mtp_cache, K, topk=0):
        drafts, probs = [], []
        h, cur = H_last, x_ids
        for _ in range(K):
            logits, mh = mtp_step(self.mtp, h, cur, self.lm_head, mtp_cache[0])
            lg = logits[0].reshape(-1)
            p = mx.softmax(lg)
            d = int(mx.argmax(lg))
            probs.append(float(p[d]))
            drafts.append(d)
            h, cur = mh, mx.array([[d]])
        self._last_probs = probs
        return drafts

    def sync(self, prev_H, rH, replay_in, mtp_cache):
        if self._last_probs is not None:
            self.records.append((self._last_probs, int(replay_in.shape[1])))
        self.inner.sync(prev_H, rH, replay_in, mtp_cache)


def _bin_stats(pairs):
    """pairs: [(p, accepted_bool)]。按 p 分 5 桶,返回每桶 (lo, hi, n, accept_rate)。"""
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [a for p, a in pairs if lo <= p < hi]
        n = len(sub)
        rate = sum(sub) / n if n else float("nan")
        out.append((lo, hi, n, rate))
    return out


def main():
    model, tok, store = build_streaming_model()
    with open(_cfg.qn_config()) as f:
        args = ModelArgs.from_dict(json.load(f))
    mtp = load_mtp(args, _cfg.mtp_out(), quantize=True)
    mtp.embed_tokens = model.model.embed_tokens
    drafter = ConfProbeDrafter(mtp, model.lm_head)

    for p in PROMPTS:
        enc = mx.array([tok.encode(p)])
        mtp_generate(model, drafter, tok, enc, MAXTOK, K=K, ids_mode=True)

    recs = drafter.records
    # 第0位:p0 → 是否接受(acc_len>=2)
    pos0 = [(pr[0], acc >= 2) for pr, acc in recs]
    # 第1位:仅在第0位已接受(acc>=2)的步;p1 → 是否再接受(acc_len>=3)
    pos1 = [(pr[1], acc >= 3) for pr, acc in recs if acc >= 2]

    print(f"\n=== 置信度↔接受率(6 prompt,共 {len(recs)} 步)===")
    for name, pairs in (("第0位 p0→accept", pos0), ("第1位 p1→accept(限第0位已中)", pos1)):
        print(f"\n[{name}]  样本={len(pairs)}")
        print(f"{'区间':>12} {'n':>5} {'接受率':>8}")
        for lo, hi, n, rate in _bin_stats(pairs):
            r = f"{rate:.3f}" if n else "--"
            print(f"  [{lo:.1f},{hi:.1f}) {n:>5} {r:>8}")

    # 整体相关性:低置信 vs 高置信步的接受率差(以 0.5 为界)
    def _split(pairs):
        lo = [a for p, a in pairs if p < 0.5]
        hi = [a for p, a in pairs if p >= 0.5]
        rl = sum(lo) / len(lo) if lo else float("nan")
        rh = sum(hi) / len(hi) if hi else float("nan")
        return len(lo), rl, len(hi), rh

    print("\n=== 低/高置信分界(p<0.5 vs p>=0.5)===")
    for name, pairs in (("第0位", pos0), ("第1位", pos1)):
        nl, rl, nh, rh = _split(pairs)
        print(f"{name}: 低置信 n={nl} 接受率={rl:.3f} | 高置信 n={nh} 接受率={rh:.3f} | 差={rh - rl:+.3f}")


if __name__ == "__main__":
    main()
