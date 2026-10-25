"""Run one controller/strategy against one scenario, or a whole experiment suite."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from app.monitoring.metrics import MetricsCollector
from app.monitoring.telemetry_collector import TelemetryCollector
from app.recovery.healing_controller import ChangePolicy, HealingController
from app.routing.baseline_router import (ReactiveRouter, RoutingStrategy, ScoringRouter, ShortestPathRouter,
                                         StaticRouter)
from app.routing.gnn_router import GNNRouter
from simulator.failure_injector import SCENARIO_NAMES, Scenario, build_scenario
from simulator.network_state import (CONGEST, DEGRADE, LINK_DOWN, LINK_UP, MODEL, NODE_DOWN, NODE_UP,
                                     MeshSimulator, SimConfig, edge_key)
from simulator.topology_generator import generate_topology
from simulator.traffic_generator import generate_flows

STRATEGIES = ("static", "shortest", "reactive", "heuristic", "gnn", "gnn_latency", "gnn_risk")


@dataclass
class RunConfig:
    scenario: str = "single_link_failure"
    strategy: str = "heuristic"
    seed: int = 0
    topo_kind: str = "geometric"
    n_nodes: int = 24
    n_flows: int = 6
    n_steps: int = 200
    demand_range: tuple = (3.0, 12.0)
    use_hysteresis: bool = True
    use_min_improvement: bool = True
    install_fail_prob: float = 0.0
    label: str = ""

    def name(self) -> str:
        return self.label or self.strategy


@dataclass
class World:
    cfg: RunConfig
    graph: object
    flows: list
    scenario: Scenario
    sim: MeshSimulator
    controller: HealingController
    collector: TelemetryCollector
    predictor: object = None


@dataclass
class RunResult:
    config: RunConfig
    summary: dict
    events: List[dict]
    frames: List[dict] = field(default_factory=list)
    world: Optional[World] = None


def make_strategy(name: str, predictor=None) -> RoutingStrategy:
    if name == "static":
        return StaticRouter()
    if name == "shortest":
        return ShortestPathRouter()
    if name == "reactive":
        return ReactiveRouter()
    if name == "heuristic":
        return ScoringRouter()
    if name == "gnn":
        return GNNRouter(predictor, "both")
    if name == "gnn_latency":
        return GNNRouter(predictor, "latency")
    if name == "gnn_risk":
        return GNNRouter(predictor, "risk")
    raise ValueError(f"unknown strategy {name!r}; choose from {STRATEGIES}")


def build_world(cfg: RunConfig, predictor=None) -> World:
    graph = generate_topology(cfg.topo_kind, cfg.n_nodes, cfg.seed)
    flows = generate_flows(graph, cfg.n_flows, cfg.seed, cfg.demand_range)
    scenario = build_scenario(cfg.scenario, graph, flows, cfg.seed, cfg.n_steps)
    sim = MeshSimulator(graph, flows, SimConfig(hazard_scale=scenario.hazard_scale,
                                                install_fail_prob=cfg.install_fail_prob), seed=cfg.seed)
    strategy = make_strategy(cfg.strategy, predictor)
    policy = ChangePolicy(use_hysteresis=cfg.use_hysteresis, use_min_improvement=cfg.use_min_improvement)
    controller = HealingController(graph, flows, strategy, sim, change_policy=policy,
                                   experiment_id=f"{cfg.scenario}-{cfg.name()}-{cfg.seed}")
    controller.bootstrap()
    collector = TelemetryCollector(sim.nodes, sim.edges)
    if predictor is not None:
        predictor.override = None
        predictor.reset_stats()
    return World(cfg, graph, flows, scenario, sim, controller, collector, predictor)


def apply_scenario_event(world: World, ev) -> None:
    if ev.kind == MODEL:
        if world.predictor is not None:
            world.predictor.override = {0: None, 1: "unavailable", 2: "invalid"}[int(ev.value)]
        return
    world.sim.apply_event(ev.kind, ev.target, ev.value)


def step_world(world: World, t: int, metrics: Optional[MetricsCollector] = None) -> dict:
    """Advance one time step; returns counters of newly declared link-down events."""
    sim, ctl = world.sim, world.controller
    for ev in world.scenario.events_at(t):
        apply_scenario_event(world, ev)
    sim.step(t)
    flow_steps = sim.send_traffic(t)
    obs = world.collector.from_raw(sim.observe(t), t)
    n0 = len(ctl.events)
    ctl.step(obs)
    new_down = [e for e in ctl.events[n0:] if e.kind == "link_down"]
    false_down = 0
    for e in new_down:
        a, b = e.entity.split("|")
        if sim.link_up(edge_key(a, b)):
            false_down += 1
    if metrics is not None:
        metrics.record(t, flow_steps, sim)
    return {"down": len(new_down), "false_down": false_down}


def run_experiment(cfg: RunConfig, predictor=None, record_frames: bool = False, frame_every: int = 1,
                   on_step: Optional[Callable[[int, World], None]] = None, keep_world: bool = False) -> RunResult:
    world = build_world(cfg, predictor if cfg.strategy.startswith("gnn") else None)
    metrics = MetricsCollector(world.flows)
    total_down = false_down = 0
    frames: List[dict] = []
    for t in range(cfg.n_steps):
        r = step_world(world, t, metrics)
        total_down += r["down"]
        false_down += r["false_down"]
        if on_step is not None:
            on_step(t, world)
        if record_frames and t % frame_every == 0:
            frames.append({"t": t, "links": world.controller.state.status_map(),
                           "routes": {k: list(v) if v else None for k, v in world.controller.routes.items()},
                           "state": world.controller.sm.state.value,
                           "physical_down": [e for e in world.sim.edges if not world.sim.link_up(e)]})
    pred = world.predictor
    summary = metrics.finalize(world.controller, false_down, total_down, pred)
    events = [e.to_dict() for e in world.controller.events]
    return RunResult(cfg, summary, events, frames, world if keep_world else None)


# ---------------------------------------------------------------------- suites
_WORKER_PRED = None


def _init_worker(model_path: Optional[str]) -> None:
    global _WORKER_PRED
    if model_path and os.path.exists(model_path):
        from gnn.inference import GNNPredictor
        _WORKER_PRED = GNNPredictor.load(model_path)


def _run_one(cfg: RunConfig) -> dict:
    res = run_experiment(cfg, _WORKER_PRED)
    row = {k: (str(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()}
    row["method"] = cfg.name()
    row.update(res.summary)
    return row


def run_suite(configs: List[RunConfig], model_path: Optional[str] = None, workers: int = 1) -> pd.DataFrame:
    if workers <= 1:
        _init_worker(model_path)
        rows = [_run_one(c) for c in configs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(model_path,)) as ex:
            rows = list(ex.map(_run_one, configs, chunksize=4))
    return pd.DataFrame(rows)


def default_variants() -> List[dict]:
    """Strategy variants incl. ablations (labels are used as the 'method' column)."""
    v = [{"strategy": s} for s in STRATEGIES]
    v += [{"strategy": "heuristic", "use_hysteresis": False, "label": "heuristic_no_hysteresis"},
          {"strategy": "heuristic", "use_min_improvement": False, "label": "heuristic_no_min_improvement"},
          {"strategy": "gnn", "use_hysteresis": False, "label": "gnn_no_hysteresis"},
          {"strategy": "gnn", "use_min_improvement": False, "label": "gnn_no_min_improvement"}]
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Run routing experiments")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIO_NAMES))
    ap.add_argument("--strategies", nargs="*", default=None, help="subset of method labels")
    ap.add_argument("--topo", default="geometric")
    ap.add_argument("--nodes", type=int, default=24)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--model", default="models/link_gnn_v1.pt")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="experiments/results/results.csv")
    a = ap.parse_args()
    variants = default_variants()
    if a.strategies:
        variants = [v for v in variants if v.get("label", v["strategy"]) in a.strategies]
    model = a.model if os.path.exists(a.model) else None
    if model is None and any(v["strategy"].startswith("gnn") for v in variants):
        print("note: no model file found; GNN methods will run in fallback (heuristic) mode")
    cfgs = [RunConfig(scenario=sc, seed=a.seed0 + s, topo_kind=a.topo, n_nodes=a.nodes, n_steps=a.steps, **v)
            for sc, s, v in itertools.product(a.scenarios, range(a.seeds), variants)]
    t0 = time.time()
    df = run_suite(cfgs, model, a.workers)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs in {time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
