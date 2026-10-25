"""Link failure/degradation detector with hysteresis, hold-down and reintegration probation."""
from __future__ import annotations

from typing import List, Optional

from app.core.events import Event
from app.detection.heartbeat import DetectorConfig, HeartbeatTracker
from app.monitoring.telemetry_collector import Observation
from app.topology.graph_state import BROKEN, USABLE_LINK, Health, LinkState, TopologyState


class FailureDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.heartbeats = HeartbeatTracker(self.cfg)

    def _quality_status(self, ls: LinkState) -> Health:
        cfg = self.cfg
        ratio = ls.ewma_latency / max(ls.base_latency, 1e-6)
        if ls.status == Health.DEGRADED:
            return Health.DEGRADED if (ls.ewma_loss > cfg.loss_exit or ratio > cfg.latency_exit_ratio) else Health.HEALTHY
        return Health.DEGRADED if (ls.ewma_loss > cfg.loss_enter or ratio > cfg.latency_enter_ratio) else Health.HEALTHY

    def update(self, state: TopologyState, obs: Observation) -> List[Event]:
        t, cfg = obs.step, self.cfg
        events = self.heartbeats.update(state, obs)
        for key, ls in state.links.items():
            probe = obs.link_probes.get(key)
            ok = probe is not None and probe.ok
            if probe is None:
                if ls.status in USABLE_LINK and ls.age(t) > cfg.max_stale_steps:
                    ls.status = Health.UNKNOWN
                    events.append(Event(t, "link_unknown", f"{key[0]}|{key[1]}"))
                continue
            name = f"{key[0]}|{key[1]}"
            if ok:
                ls.miss_streak, ls.ok_streak = 0, ls.ok_streak + 1
                ls.last_obs_step = t
                ls.prev_ewma_latency = ls.ewma_latency
                if ls.status in BROKEN:
                    ls.ewma_latency, ls.ewma_loss = probe.latency_ms, probe.loss
                else:
                    a = cfg.ewma_alpha
                    ls.ewma_latency = a * probe.latency_ms + (1 - a) * ls.ewma_latency
                    ls.ewma_loss = a * probe.loss + (1 - a) * ls.ewma_loss
                ls.last_loss, ls.util, ls.jitter = probe.loss, probe.util, probe.jitter_ms
            else:
                ls.miss_streak, ls.ok_streak = ls.miss_streak + 1, 0

            if ls.status == Health.DOWN:
                if ok:
                    flapping = ls.recent_failures(t, cfg.flap_window) >= cfg.flap_threshold
                    ls.probation_required = cfg.probation_successes * (cfg.flap_multiplier if flapping else 1)
                    ls.status = Health.PROBATION
                    events.append(Event(t, "link_probation", name, {"required": ls.probation_required}))
            elif ls.status == Health.PROBATION:
                if not ok:
                    ls.status = Health.DOWN
                    ls.down_events.append(t)   # a failed probation counts toward flap detection
                    events.append(Event(t, "link_probation_failed", name))
                elif ls.ok_streak >= ls.probation_required:
                    ls.status = self._quality_status(ls)
                    events.append(Event(t, "link_up", name))
            else:
                if ls.miss_streak >= cfg.down_after:
                    ls.status = Health.DOWN
                    ls.down_events.append(t)
                    events.append(Event(t, "link_down", name))
                elif ls.miss_streak >= cfg.suspect_after:
                    if ls.status != Health.SUSPECTED:
                        events.append(Event(t, "link_suspected", name))
                    ls.status = Health.SUSPECTED
                else:
                    new = self._quality_status(ls)
                    if new != ls.status:
                        kind = "link_degraded" if new == Health.DEGRADED else "link_recovered"
                        events.append(Event(t, kind, name, {"loss": round(ls.ewma_loss, 3),
                                                            "latency": round(ls.ewma_latency, 2)}))
                    ls.status = new
        return events
