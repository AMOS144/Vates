"""离线分析：从 route_trace 事件流计算「跨-occurrence 同层」大窗口信号对 routed/miss 的 recall。

事件流按执行顺序排列。对层 L 的第 i 次出现，用其前 history_n 次出现的 routed 并集作为预测集
pred（= 上一个/前几个 token 同层路由，窗口可达整 token，最大）。命门指标是 recall_miss：
pred 能否覆盖「本次 routed 但当前不常驻」的 miss——即真正需要预取的那部分专家。
"""
from collections import defaultdict, deque


def crosstoken_recall(events, history_n: int = 1) -> dict:
    hist = defaultdict(lambda: deque(maxlen=history_n))
    hit_full = tot_routed = 0
    hit_miss = tot_miss = 0
    pred_size_sum = miss_sum = 0
    n_scored = 0
    for ev in events:
        layer = int(ev["layer"])
        routed = {int(e) for e in ev.get("experts", [])}
        miss = {int(e) for e in ev.get("miss", [])}
        h = hist[layer]
        if h:  # 有历史才计分（首次出现无预测来源）
            pred = set().union(*h)
            hit_full += len(pred & routed)
            tot_routed += len(routed)
            hit_miss += len(pred & miss)
            tot_miss += len(miss)
            pred_size_sum += len(pred)
            miss_sum += len(miss)
            n_scored += 1
        h.append(routed)
    return {
        "history_n": history_n,
        "n_scored": n_scored,
        "recall_full": round(hit_full / max(1, tot_routed), 4),
        "recall_miss": round(hit_miss / max(1, tot_miss), 4),
        "tot_miss": tot_miss,
        "avg_pred_size": round(pred_size_sum / max(1, n_scored), 2),
        "avg_miss": round(miss_sum / max(1, n_scored), 2),
    }
