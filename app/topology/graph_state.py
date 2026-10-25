"""Controller-side belief about the network (built only from telemetry)."""
from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

import networkx as nx

Edge = Tuple[str, str]


def edge_key(u: str, v: str) -> Edge:
    return (u, v) if u <= v else (v, u)


class Health(str, enum.Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    SUSPECTED = "SUSPECTED"
    DOWN = "DOWN"
    PROBATION = "PROBATION"   # reachable again, not yet trusted for production traffic
    UNKNOWN = "UNKNOWN"       # no recent measurement


USABLE_LINK = frozenset({Health.HEALTHY, Health.DEGRADED})
BROKEN = frozenset({Health.DOWN, Health.PROBATION})


@dataclass
class LinkState:
    key: Edge
    base_latency: float
    bandwidth: float
    status: Health = Health.HEALTHY
    ewma_latency: float = 0.0
    prev_ewma_latency: float = 0.0
    ewma_loss: float = 0.01
    last_loss: float = 0.01
    util: float = 0.0
    jitter: float = 0.0
    last_obs_step: int = 0
    miss_streak: int = 0
    ok_streak: int = 0
    probation_required: int = 0
    down_events: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.ewma_latency <= 0:
            self.ewma_latency = self.base_latency
        if self.prev_ewma_latency <= 0:
            self.prev_ewma_latency = self.ewma_latency

    def recent_failures(self, step: int, window: int = 100) -> int:
        return sum(1 for s in self.down_events if step - s <= window)

    def age(self, step: int) -> int:
        return max(0, step - self.last_obs_step)


@dataclass
class NodeState:
    key: str
    status: Health = Health.HEALTHY
    miss_streak: int = 0
    ok_streak: int = 0
    probation_required: int = 0
    cpu: float = 0.05
    mem: float = 0.3
    queue: float = 0.0
    load: float = 0.0
    last_seen_step: int = 0
    down_events: List[int] = field(default_factory=list)

    def recent_failures(self, step: int, window: int = 100) -> int:
        return sum(1 for s in self.down_events if step - s <= window)


class TopologyState:
    def __init__(self, graph: nx.Graph, history_len: int = 500):
        self.nodes: Dict[str, NodeState] = {n: NodeState(n) for n in sorted(graph.nodes)}
        self.links: Dict[Edge, LinkState] = {}
        for u, v, d in graph.edges(data=True):
            k = edge_key(u, v)
            self.links[k] = LinkState(k, d["base_latency"], d["bandwidth"])
        self.node_list: List[str] = sorted(self.nodes)
        self.link_list: List[Edge] = sorted(self.links)
        self.history: Deque[dict] = deque(maxlen=history_len)

    def has_link(self, u: str, v: str) -> bool:
        return edge_key(u, v) in self.links

    def build_graph(self, strict: bool = True) -> nx.Graph:
        """strict=True: only trusted elements (for choosing routes).
        strict=False: everything not confirmed down (for connectivity/partition analysis)."""
        g = nx.Graph()
        for n, ns in self.nodes.items():
            ok = ns.status == Health.HEALTHY if strict else ns.status not in BROKEN
            if ok:
                g.add_node(n)
        for k, ls in self.links.items():
            ok = ls.status in USABLE_LINK if strict else ls.status not in BROKEN
            if ok and k[0] in g and k[1] in g:
                g.add_edge(k[0], k[1], latency=ls.ewma_latency, loss=ls.ewma_loss, util=ls.util,
                           bandwidth=ls.bandwidth, base_latency=ls.base_latency, jitter=ls.jitter,
                           status=ls.status.value)
        return g

    def snapshot(self, step: int) -> dict:
        snap = {
            "step": step,
            "down_links": [k for k, l in self.links.items() if l.status == Health.DOWN],
            "probation_links": [k for k, l in self.links.items() if l.status == Health.PROBATION],
            "degraded_links": [k for k, l in self.links.items() if l.status == Health.DEGRADED],
            "suspected_links": [k for k, l in self.links.items() if l.status == Health.SUSPECTED],
            "down_nodes": [n for n, s in self.nodes.items() if s.status in BROKEN],
        }
        self.history.append(snap)
        return snap

    def status_map(self) -> Dict[Edge, str]:
        return {k: l.status.value for k, l in self.links.items()}
