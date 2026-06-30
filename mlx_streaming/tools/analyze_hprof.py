"""分析 staging 完成回调触发时刻日志,判定"回调扎堆在 eval 尾"还是"逐层铺开"。

输入:STAGING_HPROF 产出的 jsonl,每行 [gen, layer, t_fire_seconds]。
方法:按时间间隔把回调流切成若干"前向簇"(簇间是计算大间隔),对每簇统计回调的
时间跨度与簇内相邻间隔中位数。若簇内跨度 << 前向墙钟(~0.65s)、相邻间隔≈0,
则回调扎堆在一瞬(支持"扎堆 eval 尾");若簇内跨度≈前向墙钟、相邻间隔≈每层算时,
则逐层铺开(证伪扎堆)。
"""
import json
import statistics
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ab/hprof.jsonl"
    # 簇切分阈值:相邻回调间隔 > gap_split 秒 → 视为跨前向边界。
    gap_split = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    rows = load(path)
    if not rows:
        print("no records")
        return
    rows.sort(key=lambda r: r[2])  # 按触发时刻排序
    ts = [r[2] for r in rows]
    print(f"总回调数 = {len(rows)}, 总时间跨度 = {ts[-1]-ts[0]:.3f}s")

    # 切簇
    clusters = []
    cur = [rows[0]]
    for prev, r in zip(rows, rows[1:]):
        if r[2] - prev[2] > gap_split:
            clusters.append(cur)
            cur = [r]
        else:
            cur.append(r)
    clusters.append(cur)

    # 簇间间隔(≈前向墙钟,即一次 eval+其后计算的时长)
    cluster_starts = [c[0][2] for c in clusters]
    inter_cluster = [b - a for a, b in zip(cluster_starts, cluster_starts[1:])]

    spans = []          # 每簇内回调时间跨度(max-min)
    intra_gaps = []     # 每簇内相邻回调间隔
    sizes = []
    for c in clusters:
        cts = [r[2] for r in c]
        spans.append(cts[-1] - cts[0])
        sizes.append(len(c))
        for a, b in zip(cts, cts[1:]):
            intra_gaps.append(b - a)

    def med(x):
        return statistics.median(x) if x else float("nan")

    print(f"\n切出 {len(clusters)} 个前向簇(间隔阈值 {gap_split*1000:.0f}ms)")
    print(f"  每簇回调数      中位 = {med(sizes):.0f}   (范围 {min(sizes)}~{max(sizes)})")
    print(f"  簇间间隔(≈前向墙钟) 中位 = {med(inter_cluster)*1000:.1f}ms" if inter_cluster else "  簇间间隔: n/a")
    print(f"  簇内回调跨度    中位 = {med(spans)*1000:.2f}ms   ← 关键:48+ 个回调铺开多久")
    print(f"  簇内相邻间隔    中位 = {med(intra_gaps)*1000:.3f}ms   ← 关键:≈0=扎堆一瞬, ≈13ms=逐层")

    # 判定
    fwd = med(inter_cluster) if inter_cluster else float("nan")
    span = med(spans)
    if inter_cluster and fwd > 0:
        ratio = span / fwd
        print(f"\n  簇内跨度 / 前向墙钟 = {ratio:.3f}")
        if ratio < 0.15:
            print("  → 判定:回调【扎堆】在前向内极短一段(支持'扎堆 eval 尾'假设)。")
        elif ratio > 0.6:
            print("  → 判定:回调【逐层铺开】在整段前向(证伪'扎堆 eval 尾')。")
        else:
            print("  → 判定:介于两者之间,部分铺开。")


if __name__ == "__main__":
    main()
