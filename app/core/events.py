"""Event record shared by the detector, controller, storage and dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Event:
    step: int
    kind: str
    entity: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "kind": self.kind, "entity": self.entity, "detail": dict(self.detail)}
