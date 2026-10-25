"""API-key authentication, identifier validation and a simple in-memory rate limiter."""
from __future__ import annotations

import hmac
import re
import time
from collections import defaultdict, deque
from typing import Deque, Dict

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def valid_identifier(value: str) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


def check_api_key(provided: str | None, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` calls per ``window`` seconds per client."""

    def __init__(self, limit: int, window: float = 60.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = self.clock()
        q = self._hits[client]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True
