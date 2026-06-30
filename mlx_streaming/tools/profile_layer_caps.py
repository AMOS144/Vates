"""按层 cap 分配:在同样总预算下,把槽位从"用不满"的层挪给"命中低"的层。

做法:对每层算 LFU hit(c) 曲线(c=0..工作集),用多选背包 DP 在 Σcap ≤ 预算下最大化
Σhit(精确、容忍非凹),产出 {layer: cap}(写入 pool_profile.json,被 model_builder 加载)。
离线证据:cap=16 完美分配约 +1pp(0.690→0.700),免费、零运行时成本、可叠 LFU。
"""


def allocate_caps(htab: "dict[int, list[float]]", budget: int, floor: int = 0) -> "dict[int, int]":
    """多选背包 DP:每层在 [floor, cmax_li] 选一个 cap,Σcap ≤ budget,最大化 Σhit。

    htab[layer] = [hit@0, hit@1, ..., hit@cmax]。返回 {layer: cap}。
    LFU hit(c) 近似凹但不保证,故用精确 DP 而非贪心注水。
    """
    layers = list(htab)
    NEG = float("-inf")
    dp = [NEG] * (budget + 1)
    dp[0] = 0.0
    choice = [[0] * (budget + 1) for _ in layers]
    for i, li in enumerate(layers):
        H = htab[li]
        cmax_li = len(H) - 1
        lo = min(floor, cmax_li)
        ndp = [NEG] * (budget + 1)
        for b in range(budget + 1):
            if dp[b] == NEG:
                continue
            base = dp[b]
            hi = min(cmax_li, budget - b)
            for c in range(lo, hi + 1):
                v = base + H[c]
                nb = b + c
                if v > ndp[nb]:
                    ndp[nb] = v
                    choice[i][nb] = c
        dp = ndp
    best_b = max(range(budget + 1), key=lambda b: dp[b])
    caps: "dict[int, int]" = {}
    b = best_b
    for i in range(len(layers) - 1, -1, -1):
        c = choice[i][b]
        caps[layers[i]] = c
        b -= c
    return caps
