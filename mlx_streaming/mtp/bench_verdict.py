"""最小树 A/B 评测的纯裁决逻辑:中位数 + go/no-go 判定(无副作用,可单测)。"""


def median(xs):
    """样本中位数(偶数个取中间两数均值)。xs 非空。"""
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def verdict_from_delta(delta, exact_all, margin=0.05):
    """按跨 prompt 中位相对提速 delta 与 lossless 门给出裁决。

    - exact_all=False → "bug"(lossless 硬门一票否决)
    - delta >  margin → "go"
    - delta < -margin → "no-go"
    - 其余(含边界) → "even"
    """
    if not exact_all:
        return "bug"
    if delta > margin:
        return "go"
    if delta < -margin:
        return "no-go"
    return "even"
