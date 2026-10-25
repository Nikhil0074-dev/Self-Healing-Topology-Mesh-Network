"""Records the previous route of each flow so a change can be undone."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class RollbackManager:
    def __init__(self, depth: int = 10):
        self.depth = depth
        self._stack: Dict[str, List[Tuple[int, Optional[List[str]]]]] = defaultdict(list)

    def record(self, flow_id: str, path: Optional[List[str]], step: int) -> None:
        s = self._stack[flow_id]
        s.append((step, list(path) if path else None))
        del s[:-self.depth]

    def last(self, flow_id: str) -> Optional[Tuple[int, Optional[List[str]]]]:
        s = self._stack.get(flow_id)
        return s[-1] if s else None

    def pop(self, flow_id: str) -> Optional[Tuple[int, Optional[List[str]]]]:
        s = self._stack.get(flow_id)
        return s.pop() if s else None
