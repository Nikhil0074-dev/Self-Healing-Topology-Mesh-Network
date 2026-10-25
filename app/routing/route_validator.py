"""Hard safety checks that every route must pass before installation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Sequence, Set, Tuple

from app.topology.graph_state import Edge, Health, TopologyState, USABLE_LINK, edge_key
from simulator.traffic_generator import Flow


@dataclass
class ValidationPolicy:
    max_hops: int = 12
    max_link_util: float = 0.95
    allow_degraded: bool = True
    max_staleness: int = 5
    max_cost: float = 50.0
    forbidden_nodes: Set[str] = field(default_factory=set)
    forbidden_links: Set[Edge] = field(default_factory=set)


@dataclass
class ValidationResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)


def validate_route(path: Optional[Sequence[str]], flow: Flow, state: TopologyState, step: int,
                   policy: ValidationPolicy, current: Optional[Sequence[str]] = None,
                   relax_capacity: bool = False, cost: Optional[float] = None) -> ValidationResult:
    reasons: List[str] = []
    if not path or len(path) < 2:
        return ValidationResult(False, ["empty path"])
    if path[0] != flow.src or path[-1] != flow.dst:
        reasons.append("endpoints do not match flow")
    if len(set(path)) != len(path):
        reasons.append("forwarding loop")
    if len(path) - 1 > policy.max_hops:
        reasons.append("too many hops")
    cur_edges = {edge_key(a, b) for a, b in zip(current, current[1:])} if current else set()
    for n in path:
        ns = state.nodes.get(n)
        if ns is None:
            reasons.append(f"unknown node {n}")
        elif ns.status != Health.HEALTHY:
            reasons.append(f"node {n} is {ns.status.value}")
        if n in policy.forbidden_nodes:
            reasons.append(f"node {n} forbidden by policy")
    for a, b in zip(path, path[1:]):
        k = edge_key(a, b)
        ls = state.links.get(k)
        if ls is None:
            reasons.append(f"no link {a}-{b}")
            continue
        allowed = USABLE_LINK if policy.allow_degraded else {Health.HEALTHY}
        if ls.status not in allowed:
            reasons.append(f"link {a}-{b} is {ls.status.value}")
        if ls.age(step) > policy.max_staleness:
            reasons.append(f"link {a}-{b} measurement stale")
        if k in policy.forbidden_links:
            reasons.append(f"link {a}-{b} forbidden by policy")
        if not relax_capacity:
            extra = 0.0 if k in cur_edges else flow.demand_mbps / ls.bandwidth
            if ls.util + extra > policy.max_link_util:
                reasons.append(f"link {a}-{b} over capacity")
    if cost is not None and (not math.isfinite(cost) or cost > policy.max_cost):
        reasons.append("cost not finite or above limit")
    return ValidationResult(not reasons, reasons)
