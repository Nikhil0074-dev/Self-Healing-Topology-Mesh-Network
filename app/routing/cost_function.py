"""Route cost  J(P) = wL*L + wR*R + wC*C + wH*H  and per-link estimates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from app.topology.graph_state import Edge, Health, LinkState, edge_key

RISK_FLOOR = 0.02  # baseline failure probability that is not worth routing around


@dataclass(frozen=True)
class CostWeights:
    w_latency: float = 1.0
    w_reliability: float = 3.0
    w_congestion: float = 0.3
    w_hops: float = 0.1
    latency_ref_ms: float = 100.0


@dataclass(frozen=True)
class EdgeEstimate:
    latency_ms: float
    loss: float
    util: float
    p_fail: float   # probability of failure within the prediction horizon


def heuristic_p_fail(ls: LinkState, step: int) -> float:
    """Non-ML failure-risk estimate from history and current loss."""
    p = 0.04 * ls.recent_failures(step, 100)
    if ls.status == Health.SUSPECTED:
        p += 0.3
    p += 0.5 * max(0.0, ls.ewma_loss - 0.08)
    return min(0.9, p)


def route_cost(path: Sequence[str], est: Dict[Edge, EdgeEstimate], w: CostWeights) -> Tuple[float, Dict[str, float]]:
    lat, survive, max_util = 0.0, 1.0, 0.0
    for u, v in zip(path, path[1:]):
        e = est.get(edge_key(u, v))
        if e is None:
            return math.inf, {}
        lat += e.latency_ms
        survive *= (1.0 - min(e.loss, 0.99)) * (1.0 - max(0.0, e.p_fail - RISK_FLOOR))
        max_util = max(max_util, e.util)
    L = lat / w.latency_ref_ms
    R = 1.0 - survive
    C = max(0.0, (max_util - 0.6) / 0.4)
    H = (len(path) - 1) / 10.0
    cost = w.w_latency * L + w.w_reliability * R + w.w_congestion * C + w.w_hops * H
    return cost, {"latency": L, "reliability": R, "congestion": C, "hops": H}


def link_weight(e: EdgeEstimate, w: CostWeights) -> float:
    """Additive per-link surrogate used only to *generate* candidate paths."""
    return (w.w_latency * e.latency_ms / w.latency_ref_ms
            + w.w_reliability * (e.loss + max(0.0, e.p_fail - RISK_FLOOR))
            + w.w_congestion * max(0.0, (e.util - 0.6) / 0.4) * 0.3
            + w.w_hops * 0.1 + 1e-6)
