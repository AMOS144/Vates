"""MoE 路由 trace 采集工具（仅 probe 使用）。

热路径默认完全关闭；开启后会把每次 MoE block 的 layer 和专家集合记录下来，
用于离线替换策略模拟。注：开启会触发路由 ids 的 CPU 同步，不用于正式性能测试。
"""
import json


_enabled = False
_events: list[dict] = []


def enable() -> None:
    global _enabled, _events
    _enabled = True
    _events = []


def disable() -> None:
    global _enabled
    _enabled = False


def record(layer: int, experts, miss=None, resident=None, resident_rank=None) -> None:
    if not _enabled:
        return
    vals = [int(e) for e in experts]
    rec = {"layer": int(layer), "experts": sorted(set(vals))}
    if miss is not None:
        rec["miss"] = sorted({int(e) for e in miss})
    if resident is not None:
        rec["resident"] = sorted({int(e) for e in resident})
    if resident_rank is not None:
        rec["resident_rank"] = [[int(e), float(v)] for e, v in resident_rank.items()]
    _events.append(rec)


def dump_jsonl(path: str) -> int:
    with open(path, "w") as f:
        for rec in _events:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(_events)


def size() -> int:
    return len(_events)


def events() -> list[dict]:
    return list(_events)
