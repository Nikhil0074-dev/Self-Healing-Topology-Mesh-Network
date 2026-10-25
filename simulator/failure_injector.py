"""Scripted failure scenarios and random training-event generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from simulator.network_state import (CONGEST, DEGRADE, LINK_DOWN, LINK_UP, MODEL, NODE_DOWN,
                                     NODE_UP, edge_key)
from simulator.traffic_generator import Flow

SCENARIO_NAMES = (
    "single_link_failure", "node_failure", "gradual_degradation", "congestion",
    "partition", "partition_recovery", "link_flapping", "model_failure", "hazard",
)


@dataclass(frozen=True)
class ScenarioEvent:
    step: int
    kind: str
    target: object = None
    value: float = 0.0


@dataclass
class Scenario:
    name: str
    events: List[ScenarioEvent]
    hazard_scale: float
    description: str
    _by_step: Dict[int, List[ScenarioEvent]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for ev in self.events:
            self._by_step.setdefault(ev.step, []).append(ev)

    def events_at(self, t: int) -> List[ScenarioEvent]:
        return self._by_step.get(t, [])


def _initial_route(graph: nx.Graph, flow: Flow) -> List[str]:
    return nx.shortest_path(graph, flow.src, flow.dst, weight="base_latency")


def _keeps_flows_connected(graph: nx.Graph, flows: Sequence[Flow], remove_edges=(), remove_nodes=()) -> bool:
    g = graph.copy()
    g.remove_edges_from(remove_edges)
    g.remove_nodes_from(remove_nodes)
    for f in flows:
        if f.src not in g or f.dst not in g or not nx.has_path(g, f.src, f.dst):
            return False
    return True


def _pick_route_link(graph: nx.Graph, flows: Sequence[Flow], rng: np.random.Generator, avoid=()) -> Tuple[str, str]:
    route = _initial_route(graph, flows[0])
    links = [edge_key(u, v) for u, v in zip(route, route[1:]) if edge_key(u, v) not in avoid]
    order = list(rng.permutation(len(links)))
    for i in order:
        if _keeps_flows_connected(graph, flows, remove_edges=[links[i]]):
            return links[i]
    for i in order:
        if _keeps_flows_connected(graph, flows[:1], remove_edges=[links[i]]):
            return links[i]
    return links[order[0]]


def _pick_route_node(graph: nx.Graph, flows: Sequence[Flow], rng: np.random.Generator) -> str:
    endpoints = {f.src for f in flows} | {f.dst for f in flows}
    route = _initial_route(graph, flows[0])
    inner = [n for n in route[1:-1]] or [n for n in graph.nodes if n not in endpoints]
    order = list(rng.permutation(len(inner)))
    for i in order:
        if inner[i] not in endpoints and _keeps_flows_connected(graph, flows, remove_nodes=[inner[i]]):
            return inner[i]
    for i in order:
        if _keeps_flows_connected(graph, flows[:1], remove_nodes=[inner[i]]):
            return inner[i]
    return inner[order[0]]


def build_scenario(name: str, graph: nx.Graph, flows: Sequence[Flow], seed: int, n_steps: int = 200) -> Scenario:
    """Create a scripted scenario. Times assume n_steps >= 180."""
    rng = np.random.default_rng([int(seed), 31])
    ev: List[ScenarioEvent] = []
    hz = 0.3
    if name == "single_link_failure":
        e = _pick_route_link(graph, flows, rng)
        ev += [ScenarioEvent(60, LINK_DOWN, e), ScenarioEvent(140, LINK_UP, e)]
        desc = f"Link {e} fails at t=60 and returns at t=140."
    elif name == "node_failure":
        n = _pick_route_node(graph, flows, rng)
        ev += [ScenarioEvent(60, NODE_DOWN, n), ScenarioEvent(150, NODE_UP, n)]
        desc = f"Intermediate node {n} fails at t=60 and returns at t=150."
    elif name == "gradual_degradation":
        e = _pick_route_link(graph, flows, rng)
        for k in range(61):
            ev.append(ScenarioEvent(50 + k, DEGRADE, e, k / 60.0))
        ev += [ScenarioEvent(140, LINK_DOWN, e), ScenarioEvent(170, LINK_UP, e), ScenarioEvent(170, DEGRADE, e, 0.0)]
        desc = f"Link {e} degrades t=50..110, holds, fails at t=140."
    elif name == "congestion":
        route = _initial_route(graph, flows[0])
        links = [edge_key(u, v) for u, v in zip(route, route[1:])]
        picks = links[: max(1, min(2, len(links)))]
        for e in picks:
            bw = graph.edges[e]["bandwidth"]
            ev += [ScenarioEvent(50, CONGEST, e, 0.92 * bw), ScenarioEvent(150, CONGEST, e, 0.0)]
        desc = f"Background load saturates links {picks} during t=50..150."
    elif name in ("partition", "partition_recovery"):
        s, d = flows[0].src, flows[0].dst
        cut = sorted(edge_key(*c) for c in nx.minimum_edge_cut(graph, s, d))
        for e in cut:
            ev.append(ScenarioEvent(60, LINK_DOWN, e))
        if name == "partition_recovery":
            first = cut[0]
            ev += [ScenarioEvent(130, LINK_UP, first), ScenarioEvent(134, LINK_DOWN, first),
                   ScenarioEvent(138, LINK_UP, first), ScenarioEvent(142, LINK_DOWN, first)]
            for e in cut:
                ev.append(ScenarioEvent(150, LINK_UP, e))
            desc = f"Cut {cut} at t=60; link {first} flaps t=130..142; all restored at t=150."
        else:
            desc = f"Min-cut {cut} fails at t=60 and is never restored."
    elif name == "link_flapping":
        e = _pick_route_link(graph, flows, rng)
        t, down = 50, True
        while t < 150:
            ev.append(ScenarioEvent(t, LINK_DOWN if down else LINK_UP, e))
            down = not down
            t += int(rng.integers(4, 8))
        ev.append(ScenarioEvent(150, LINK_UP, e))
        desc = f"Link {e} flaps between t=50 and t=150."
    elif name == "model_failure":
        e1 = _pick_route_link(graph, flows, rng)
        e2 = _pick_route_link(graph, flows, rng, avoid=(e1,))
        ev += [ScenarioEvent(40, MODEL, None, 1.0), ScenarioEvent(60, LINK_DOWN, e1),
               ScenarioEvent(110, MODEL, None, 2.0), ScenarioEvent(120, LINK_DOWN, e2),
               ScenarioEvent(150, MODEL, None, 0.0), ScenarioEvent(150, LINK_UP, e1),
               ScenarioEvent(160, LINK_UP, e2)]
        desc = "GNN unavailable t=40..110, returns invalid output t=110..150; two link failures."
    elif name == "hazard":
        hz = 1.5
        desc = "No scripted events; stress-driven stochastic link/node failures."
    elif name == "random_training":
        hz = 1.5
        ev = random_training_events(graph, n_steps, rng)
        desc = "Random degradations, congestion and node outages for dataset generation."
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return Scenario(name, sorted(ev, key=lambda x: x.step), hz, desc)


def random_training_events(graph: nx.Graph, n_steps: int, rng: np.random.Generator) -> List[ScenarioEvent]:
    edges = sorted(edge_key(u, v) for u, v in graph.edges)
    nodes = sorted(graph.nodes)
    ev: List[ScenarioEvent] = []
    for _ in range(int(rng.integers(4, 9))):
        kind = rng.choice(["degrade", "congest", "node"], p=[0.5, 0.35, 0.15])
        start = int(rng.integers(10, max(11, n_steps - 80)))
        if kind == "degrade":
            e = edges[int(rng.integers(len(edges)))]
            ramp = int(rng.integers(15, 50))
            peak = float(rng.uniform(0.4, 1.0))
            hold = int(rng.integers(5, 30))
            for k in range(ramp + 1):
                ev.append(ScenarioEvent(start + k, DEGRADE, e, peak * k / ramp))
            ev.append(ScenarioEvent(start + ramp + hold, DEGRADE, e, 0.0))
        elif kind == "congest":
            e = edges[int(rng.integers(len(edges)))]
            bw = graph.edges[e]["bandwidth"]
            dur = int(rng.integers(20, 60))
            ev += [ScenarioEvent(start, CONGEST, e, float(rng.uniform(0.6, 1.1)) * bw),
                   ScenarioEvent(start + dur, CONGEST, e, 0.0)]
        else:
            n = nodes[int(rng.integers(len(nodes)))]
            dur = int(rng.integers(10, 40))
            ev += [ScenarioEvent(start, NODE_DOWN, n), ScenarioEvent(start + dur, NODE_UP, n)]
    return [e for e in ev if e.step < n_steps]
