"""Detector configuration and node heartbeat tracking (with hysteresis)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.events import Event
from app.monitoring.telemetry_collector import Observation
from app.topology.graph_state import BROKEN, Health, TopologyState


@dataclass
class DetectorConfig:
    suspect_after: int = 1          # consecutive misses -> SUSPECTED
    down_after: int = 3             # consecutive misses -> DOWN
    probation_successes: int = 5    # consecutive successes needed before a link is trusted again
    flap_window: int = 60
    flap_threshold: int = 3         # >= this many DOWN events in flap_window -> hold-down
    flap_multiplier: int = 3
    loss_enter: float = 0.12        # separate enter/exit thresholds = hysteresis
    loss_exit: float = 0.06
    latency_enter_ratio: float = 2.0
    latency_exit_ratio: float = 1.5
    ewma_alpha: float = 0.4
    max_stale_steps: int = 10
    node_down_after: int = 3
    node_up_after: int = 3


class HeartbeatTracker:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg

    def update(self, state: TopologyState, obs: Observation) -> List[Event]:
        events: List[Event] = []
        t, cfg = obs.step, self.cfg
        for n, beat in obs.node_beats.items():
            ns = state.nodes[n]
            if beat.ok:
                ns.miss_streak, ns.ok_streak = 0, ns.ok_streak + 1
                ns.last_seen_step = t
                ns.cpu, ns.mem, ns.queue, ns.load = beat.cpu, beat.mem, beat.queue, beat.load
            else:
                ns.miss_streak, ns.ok_streak = ns.miss_streak + 1, 0
            st = ns.status
            if st == Health.DOWN:
                if beat.ok:
                    ns.status, ns.probation_required = Health.PROBATION, cfg.node_up_after
                    events.append(Event(t, "node_probation", n))
            elif st == Health.PROBATION:
                if not beat.ok:
                    ns.status = Health.DOWN
                elif ns.ok_streak >= ns.probation_required:
                    ns.status = Health.HEALTHY
                    events.append(Event(t, "node_up", n))
            else:
                if ns.miss_streak >= cfg.node_down_after:
                    ns.status = Health.DOWN
                    ns.down_events.append(t)
                    events.append(Event(t, "node_down", n))
                elif ns.miss_streak >= cfg.suspect_after:
                    if st != Health.SUSPECTED:
                        events.append(Event(t, "node_suspected", n))
                    ns.status = Health.SUSPECTED
                elif beat.ok and st == Health.SUSPECTED:
                    ns.status = Health.HEALTHY
        return events
