"""离线路由 trace 的专家缓存替换策略模拟。

输入为 `probe_route_trace` 产出的 JSONL：每行包含 layer 和 experts。模拟只关注
“该层这次需要哪些专家”，不触碰真实权重，用来估计 LRU / 近似 Belady 的 miss 上界。
"""
import argparse
import json
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path


def _next_use(events: list[list[int]], start: int, expert: int, window: int | None) -> int | None:
    """返回 expert 下一次出现的相对距离；None 表示窗口内不再出现。"""
    end = len(events) if window is None else min(len(events), start + 1 + window)
    for j in range(start + 1, end):
        if expert in events[j]:
            return j - start
    return None


def _choose_victim(
    resident: "OrderedDict[int, None]",
    current: set[int],
    pinned: set[int],
    avoid: set[int],
    events: list[list[int]],
    idx: int,
    policy: str,
    window: int,
) -> int:
    candidates = [e for e in resident if e not in pinned and e not in current]
    if not candidates:
        candidates = [e for e in resident if e not in pinned]
    if not candidates:
        raise ValueError("没有可驱逐专家：pinned 数量已占满 capacity")
    if policy == "lru":
        return candidates[0]
    if policy == "avoid":
        cold = [e for e in candidates if e not in avoid]
        return cold[0] if cold else candidates[0]
    if policy == "window":
        future = set()
        for ev in events[idx + 1: idx + 1 + window]:
            future.update(ev)
        cold = [e for e in candidates if e not in future]
        return cold[0] if cold else candidates[0]
    if policy == "belady":
        best_e = candidates[0]
        best_dist = -1
        for e in candidates:
            dist = _next_use(events, idx, e, None)
            if dist is None:
                return e
            if dist > best_dist:
                best_dist = dist
                best_e = e
        return best_e
    raise ValueError(f"未知 policy: {policy}")


def simulate_layer_events(
    events: list[list[int]],
    capacity: int,
    policy: str = "lru",
    window: int = 4,
    pinned: set[int] | None = None,
    protected_ratio: float = 0.25,
    avoid_events: "list[set[int]] | None" = None,
) -> dict:
    """模拟单层专家请求序列，返回 hit/miss 统计。"""
    pinned = set(pinned or ())
    if len(pinned) > capacity:
        raise ValueError("pinned 数量不能超过 capacity")
    if policy == "2q":
        return _simulate_layer_events_2q(events, capacity, pinned, protected_ratio)
    resident: "OrderedDict[int, None]" = OrderedDict((e, None) for e in pinned)
    hits = misses = 0
    avoid_events = avoid_events or [set() for _ in events]
    for idx, raw in enumerate(events):
        uniq = list(dict.fromkeys(int(e) for e in raw))
        current = set(uniq)
        if len(current) > capacity:
            raise ValueError(f"单次请求专家数 {len(current)} > capacity {capacity}")
        for e in uniq:
            if e in resident:
                hits += 1
                resident.move_to_end(e)
                continue
            misses += 1
            while len(resident) >= capacity:
                victim = _choose_victim(
                    resident, current, pinned, avoid_events[idx], events, idx, policy, window)
                del resident[victim]
            resident[e] = None
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / max(1, hits + misses), 4),
        "events": len(events),
    }


def _simulate_layer_events_2q(
    events: list[list[int]],
    capacity: int,
    pinned: set[int],
    protected_ratio: float,
) -> dict:
    """Segmented LRU / 2Q: 新专家进 probation，二次命中后进 protected。"""
    protected_cap = max(1, int(capacity * protected_ratio))
    protected_cap = min(protected_cap, max(1, capacity - len(pinned)))
    probation: "OrderedDict[int, None]" = OrderedDict()
    protected: "OrderedDict[int, None]" = OrderedDict((e, None) for e in pinned)
    resident = set(pinned)
    hits = misses = 0

    def trim_protected():
        while len([e for e in protected if e not in pinned]) + len(pinned) > protected_cap:
            for cand in list(protected):
                if cand not in pinned:
                    del protected[cand]
                    probation[cand] = None
                    break
            else:
                break

    def evict(current: set[int]):
        candidates = [e for e in probation if e not in pinned and e not in current]
        if not candidates:
            candidates = [e for e in protected if e not in pinned and e not in current]
        if not candidates:
            candidates = [e for e in probation if e not in pinned]
        if not candidates:
            candidates = [e for e in protected if e not in pinned]
        if not candidates:
            raise ValueError("没有可驱逐专家：pinned 数量已占满 capacity")
        victim = candidates[0]
        probation.pop(victim, None)
        protected.pop(victim, None)
        resident.remove(victim)

    for raw in events:
        uniq = list(dict.fromkeys(int(e) for e in raw))
        current = set(uniq)
        if len(current) > capacity:
            raise ValueError(f"单次请求专家数 {len(current)} > capacity {capacity}")
        for e in uniq:
            if e in resident:
                hits += 1
                if e in protected:
                    protected.move_to_end(e)
                else:
                    probation.pop(e, None)
                    protected[e] = None
                    trim_protected()
                continue
            misses += 1
            while len(resident) >= capacity:
                evict(current)
            probation[e] = None
            resident.add(e)
    return {
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / max(1, hits + misses), 4),
        "events": len(events),
    }


def _load_trace(path: str) -> tuple[dict[int, list[list[int]]], dict[int, list[set[int]]]]:
    by_layer: dict[int, list[list[int]]] = defaultdict(list)
    avoid_by_layer: dict[int, list[set[int]]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        layer = int(rec["layer"])
        by_layer[layer].append([int(e) for e in rec["experts"]])
        avoid_by_layer[layer].append({int(e) for e in rec.get("avoid", [])})
    return by_layer, avoid_by_layer


def _hot_pins(events: list[list[int]], n: int) -> set[int]:
    c: Counter[int] = Counter()
    for ev in events:
        c.update(set(ev))
    return {e for e, _ in c.most_common(n)}


def simulate_trace(path: str, capacity: int, pin_hot: int = 0, window: int = 4) -> dict:
    by_layer, avoid_by_layer = _load_trace(path)
    has_avoid = any(any(ev for ev in events) for events in avoid_by_layer.values())
    policies = ["lru", "2q"]
    if has_avoid:
        policies.append("avoid")
    policies.extend(["window", "belady"])
    out = {"capacity": capacity, "pin_hot": pin_hot, "window": window, "policies": {}}
    for policy in policies:
        total_hits = total_misses = total_events = 0
        for _layer, events in by_layer.items():
            pins = _hot_pins(events, pin_hot) if pin_hot else set()
            r = simulate_layer_events(
                events, capacity, policy=policy, window=window, pinned=pins,
                avoid_events=avoid_by_layer.get(_layer))
            total_hits += r["hits"]
            total_misses += r["misses"]
            total_events += r["events"]
        out["policies"][policy] = {
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": round(total_hits / max(1, total_hits + total_misses), 4),
            "events": total_events,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, nargs="+", default=[160, 192, 208, 224, 256])
    ap.add_argument("--pin-hot", type=int, default=0)
    ap.add_argument("--window", type=int, default=4)
    args = ap.parse_args()
    rows = [simulate_trace(args.trace, s, pin_hot=args.pin_hot, window=args.window)
            for s in args.slots]
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
