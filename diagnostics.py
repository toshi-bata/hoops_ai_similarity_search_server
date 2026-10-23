"""Per-request phase timing, independent of search and embedding.

``_PhaseTimer`` accumulates labelled phase durations (milliseconds) for a
single request and emits them as INFO log lines. It is a plain stopwatch: it
belongs to neither the search path nor the embedding path, so it lives in its
own leaf module that both can share.

This module depends on nothing beyond the standard library. It must never
import ``config``, ``core`` or ``embeddings`` so it stays a true leaf: a tool
that only measures elapsed time must not be able to pull the rest of the
server in behind it.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class _PhaseTimer:
    """Accumulate labelled phase durations (milliseconds) for one request.

    Only ever constructed when timing is enabled
    (``config._search_timing_enabled``). ``lap(label)`` records the time since
    the previous lap (or construction); ``emit(tag, context)`` logs one INFO
    line per phase plus a TOTAL, prefixed with the request context.
    """

    __slots__ = ("_start", "_last", "_marks")

    def __init__(self) -> None:
        now = time.perf_counter()
        self._start = now
        self._last = now
        self._marks: list[tuple[str, float]] = []

    def lap(self, label: str) -> None:
        now = time.perf_counter()
        self._marks.append((label, (now - self._last) * 1000.0))
        self._last = now

    def emit(self, tag: str, context: dict[str, Any]) -> None:
        total = (time.perf_counter() - self._start) * 1000.0
        ctx = " ".join(f"{k}={v}" for k, v in context.items())
        for label, ms in self._marks:
            logger.info("[%s] %s | %s = %.1f ms", tag, ctx, label, ms)
        logger.info("[%s] %s | TOTAL = %.1f ms", tag, ctx, total)
