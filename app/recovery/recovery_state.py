"""Controller state machine and recovery-plan records."""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.events import Event


class ControllerState(str, enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILURE_SUSPECTED = "FAILURE_SUSPECTED"
    RECOVERY_PLANNING = "RECOVERY_PLANNING"
    RECOVERY_APPLYING = "RECOVERY_APPLYING"
    RECOVERY_VERIFYING = "RECOVERY_VERIFYING"
    PARTITIONED = "PARTITIONED"
    REINTEGRATING = "REINTEGRATING"
    SAFE_MODE = "SAFE_MODE"


class StateMachine:
    def __init__(self) -> None:
        self.state = ControllerState.HEALTHY
        self.transitions: List[Event] = []

    def transition(self, new: ControllerState, step: int, reason: str = "") -> Optional[Event]:
        if new == self.state:
            return None
        ev = Event(step, "state_change", new.value, {"from": self.state.value, "reason": reason})
        self.state = new
        self.transitions.append(ev)
        return ev


@dataclass
class RecoveryPlan:
    plan_id: str
    flow_id: str
    source: str
    destination: str
    candidate_path: Optional[List[str]]
    predicted_cost: Optional[float]
    previous_path: Optional[List[str]]
    failed_elements: List[str] = field(default_factory=list)
    status: str = "proposed"           # proposed | applied | rejected | install_failed | verify_failed | no_feasible_route
    validation_status: str = "pending"  # pending | valid | invalid
    decision_reason: str = ""
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        return self.validation_status == "valid"
