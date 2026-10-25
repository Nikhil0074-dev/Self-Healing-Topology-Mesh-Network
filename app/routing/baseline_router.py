"""Routing strategies: static, shortest-path, reactive, heuristic-aware (and GNN-assisted)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from app.routing.candidate_paths import generate_candidate_paths
from app.routing.cost_function import (CostWeights, EdgeEstimate, heuristic_p_fail, link_weight,
                                       route_cost)
from app.routing.route_validator import ValidationPolicy, validate_route
from app.topology.graph_state import Edge, Health, TopologyState, edge_key
from simulator.traffic_generator import Flow


@dataclass
class Proposal:
    path: List[str]
    cost: float
    current_cost: Optional[float]
    reason: str
    ranked: List[Tuple[List[str], float]] = field(default_factory=list)


class RoutingStrategy:
    name = "base"
    proactive = False         # may the strategy ask for a change while the route still works?
    fallback_active = False   # True when an ML component is bypassed this step

    def initial_route(self, flow: Flow, graph: nx.Graph) -> Optional[List[str]]:
        try:
            return list(nx.shortest_path(graph, flow.src, flow.dst, weight="base_latency"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def prepare(self, state: TopologyState, step: int) -> None:
        self.fallback_active = False

    def propose(self, flow: Flow, current: Optional[List[str]], state: TopologyState, step: int,
                policy: ValidationPolicy, broken: bool, graph: nx.Graph) -> Optional[Proposal]:
        raise NotImplementedError


class StaticRouter(RoutingStrategy):
    name = "static"

    def propose(self, *args, **kwargs) -> Optional[Proposal]:
        return None


class _DijkstraRouter(RoutingStrategy):
    """Re-routes only after the current route is confirmed broken."""
    k = 4

    def weight(self, u: str, v: str, d: dict) -> float:
        raise NotImplementedError

    def propose(self, flow, current, state, step, policy, broken, graph) -> Optional[Proposal]:
        if not broken:
            return None
        paths = generate_candidate_paths(graph, flow.src, flow.dst, self.k, self.weight)
        for p in paths:
            if validate_route(p, flow, state, step, policy, current=current, relax_capacity=True).ok:
                return Proposal(p, 0.0, None, f"{self.name}: route broken; best remaining path", [(p, 0.0)])
        return None


class ShortestPathRouter(_DijkstraRouter):
    name = "shortest"

    def weight(self, u, v, d):
        return d["base_latency"]


class ReactiveRouter(_DijkstraRouter):
    name = "reactive"

    def weight(self, u, v, d):
        return d["latency"] / (1.0 - min(d["loss"], 0.9))


class ScoringRouter(RoutingStrategy):
    """Heuristic-aware router: scores candidates with the multi-objective cost."""
    name = "heuristic"
    proactive = True

    def __init__(self, weights: Optional[CostWeights] = None, k: int = 5):
        self.weights = weights or CostWeights()
        self.k = k
        self._est: Dict[Edge, EdgeEstimate] = {}

    def heuristic_estimates(self, state: TopologyState, step: int) -> Dict[Edge, EdgeEstimate]:
        return {k: EdgeEstimate(ls.ewma_latency, min(ls.ewma_loss, 0.99), ls.util, heuristic_p_fail(ls, step))
                for k, ls in state.links.items()}

    def prepare(self, state: TopologyState, step: int) -> None:
        self.fallback_active = False
        self._est = self.heuristic_estimates(state, step)

    def propose(self, flow, current, state, step, policy, broken, graph) -> Optional[Proposal]:
        est, w = self._est, self.weights

        def wf(u: str, v: str, d: dict) -> float:
            e = est.get(edge_key(u, v))
            return link_weight(e, w) if e else 1e3

        ranked: List[Tuple[List[str], float]] = []
        for p in generate_candidate_paths(graph, flow.src, flow.dst, self.k, wf):
            c, _ = route_cost(p, est, w)
            if validate_route(p, flow, state, step, policy, current=current,
                              relax_capacity=broken, cost=c).ok:
                ranked.append((p, c))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[1])
        cur_cost = None
        if current and not broken:
            c, _ = route_cost(current, est, w)
            cur_cost = c if math.isfinite(c) else None
        best, best_cost = ranked[0]
        reason = (f"{self.name}: lowest predicted cost {best_cost:.3f}"
                  + (f" vs current {cur_cost:.3f}" if cur_cost is not None else " (current route unusable)"))
        return Proposal(best, best_cost, cur_cost, reason, ranked)
