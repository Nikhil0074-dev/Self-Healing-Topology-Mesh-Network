"""Traffic-flow generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class Flow:
    flow_id: str
    src: str
    dst: str
    demand_mbps: float
    # "telemetry" traffic is buffered during a partition; "control" traffic expires.
    traffic_class: str = "telemetry"


def generate_flows(
    graph: nx.Graph,
    n_flows: int,
    seed: int,
    demand_range: Tuple[float, float] = (3.0, 12.0),
    min_hops: int = 3,
) -> List[Flow]:
    rng = np.random.default_rng([int(seed), 11])
    nodes = sorted(graph.nodes)
    hops = dict(nx.all_pairs_shortest_path_length(graph))
    pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:] if hops[u].get(v, 0) >= min_hops]
    if len(pairs) < n_flows:
        pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:] if hops[u].get(v, 0) >= 1]
    n = min(n_flows, len(pairs))
    idx = rng.choice(len(pairs), size=n, replace=False)
    flows: List[Flow] = []
    for k, i in enumerate(sorted(int(x) for x in idx)):
        u, v = pairs[i]
        if rng.random() < 0.5:
            u, v = v, u
        demand = float(rng.uniform(*demand_range))
        cls = "telemetry" if k % 2 == 0 else "control"
        flows.append(Flow(f"flow_{k}", u, v, demand, cls))
    return flows
