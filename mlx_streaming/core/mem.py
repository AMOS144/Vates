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


def _set_limit(name: str, nbytes: int) -> bool:
    """调用 mx.set_*_limit(字节);存在且成功返回 True。"""
    fn = getattr(mx, name, None) or getattr(getattr(mx, "metal", object()), name, None)
    if fn is None:
        return False
    try:
        fn(int(nbytes))
        return True
    except Exception:
        return False


def setup_memory_hygiene(cache_gb: float = 2.0, wired_gb: float = 0.0) -> dict:
    """长期运行的内存防御:bound MLX 缓冲缓存 + 可选 wire 工作集防 macOS 压缩器。

    - cache_gb>0:set_cache_limit,封顶 MLX 可回收缓冲,防长会话里缓存膨胀把常驻推过墙。
    - wired_gb>0:set_wired_limit,把这么多 GB 的 GPU 缓冲钉为常驻(wired),macOS 不再
      压缩/换出这些页 → 长跑延迟稳定。务必 < 系统建议工作集(本机 26.8GB),否则饿死系统。
    返回实际生效项,便于启动日志记录。
    """
    applied = {}
    if cache_gb and cache_gb > 0:
        applied["cache_limit_gb"] = cache_gb if _set_limit("set_cache_limit", cache_gb * 1e9) else None
    if wired_gb and wired_gb > 0:
        applied["wired_limit_gb"] = wired_gb if _set_limit("set_wired_limit", wired_gb * 1e9) else None
    return applied
