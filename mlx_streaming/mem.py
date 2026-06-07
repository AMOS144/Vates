"""内存度量统一口径。macOS 上 ru_maxrss 单位是字节。"""
import resource
from dataclasses import dataclass

import mlx.core as mx


def rss_bytes() -> int:
    # macOS: ru_maxrss 已是字节；Linux 是 KB（本项目目标是 macOS）
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _call(name: str) -> int:
    fn = getattr(mx, name, None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", object()), name, None)
    try:
        return int(fn()) if fn else 0
    except Exception:
        return 0


@dataclass
class MemSnapshot:
    rss_bytes: int
    mlx_active_bytes: int
    mlx_peak_bytes: int


def snapshot() -> MemSnapshot:
    return MemSnapshot(
        rss_bytes=rss_bytes(),
        mlx_active_bytes=_call("get_active_memory"),
        mlx_peak_bytes=_call("get_peak_memory"),
    )


def clear_cache() -> None:
    fn = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", object()), "clear_cache", None)
    if fn:
        fn()


def reset_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", object()), "reset_peak_memory", None)
    if fn:
        fn()
