"""In-memory registry of live simulated experiments used by the API."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.monitoring.metrics import MetricsCollector
from simulator.experiment_runner import RunConfig, World, build_world, step_world


@dataclass
class LiveExperiment:
    experiment_id: str
    world: World
    metrics: MetricsCollector
    t: int = 0
    false_down: int = 0
    total_down: int = 0
    pending_restores: List[tuple] = field(default_factory=list)  # (step, kind, target)


class ExperimentManager:
    def __init__(self, predictor=None, repo=None):
        self.predictor = predictor
        self.repo = repo
        self.experiments: Dict[str, LiveExperiment] = {}
        self.latest: Optional[str] = None
        self._ids = itertools.count(1)

    def create(self, cfg: RunConfig, warmup: int = 5) -> LiveExperiment:
        eid = f"experiment_{next(self._ids):03d}"
        cfg.scenario = cfg.scenario or "hazard"
        world = build_world(cfg, self.predictor if cfg.strategy.startswith("gnn") else None)
        exp = LiveExperiment(eid, world, MetricsCollector(world.flows))
        self.experiments[eid] = exp
        self.latest = eid
        if self.repo:
            self.repo.save_experiment(eid, cfg.__dict__, cfg.seed, "running")
        self.advance(eid, warmup)
        return exp

    def get(self, eid: Optional[str] = None) -> LiveExperiment:
        eid = eid or self.latest
        if eid is None or eid not in self.experiments:
            raise KeyError(eid or "no experiment")
        return self.experiments[eid]

    def advance(self, eid: str, n: int) -> LiveExperiment:
        exp = self.get(eid)
        for _ in range(n):
            self._apply_restores(exp)
            r = step_world(exp.world, exp.t, exp.metrics)
            exp.total_down += r["down"]
            exp.false_down += r["false_down"]
            exp.t += 1
        return exp

    def _apply_restores(self, exp: LiveExperiment) -> None:
        due = [p for p in exp.pending_restores if p[0] <= exp.t]
        exp.pending_restores = [p for p in exp.pending_restores if p[0] > exp.t]
        for _, kind, target in due:
            exp.world.sim.apply_event(kind, target)

    def inject(self, eid: Optional[str], kind: str, target, duration: Optional[int]) -> None:
        exp = self.get(eid)
        exp.world.sim.apply_event(kind, target)
        # Durations apply only to outage events; explicit *_up events are final.
        if duration and kind in ("link_down", "node_down"):
            restore = {"link_down": "link_up", "node_down": "node_up"}[kind]
            exp.pending_restores.append((exp.t + int(duration), restore, target))

    def finish_metrics(self, eid: Optional[str] = None) -> dict:
        exp = self.get(eid)
        m = exp.metrics.finalize(exp.world.controller, exp.false_down, exp.total_down, exp.world.predictor)
        m["steps"] = exp.t
        return m
