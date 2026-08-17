"""Identify prompt ingestion without guessing from the sequence length.

Decode and MTP verification share the same model forward function as prefill.
An explicit context keeps their token-major hot path intact while making every
public prefill call use the memory-bounded Expert-major implementation.
"""

from contextlib import contextmanager
from contextvars import ContextVar


_EXPERT_MAJOR_PREFILL_ACTIVE = ContextVar(
    "expert_major_prefill_active", default=False,
)


def expert_major_prefill_active() -> bool:
    return bool(_EXPERT_MAJOR_PREFILL_ACTIVE.get())


@contextmanager
def expert_major_prefill_scope():
    token = _EXPERT_MAJOR_PREFILL_ACTIVE.set(True)
    try:
        yield
    finally:
        _EXPERT_MAJOR_PREFILL_ACTIVE.reset(token)
