"""Candidate path generation (k shortest simple paths)."""
from __future__ import annotations

from itertools import islice
from typing import Callable, List, Union

import networkx as nx


def generate_candidate_paths(graph: nx.Graph, src: str, dst: str, k: int,
                             weight: Union[str, Callable] = "latency") -> List[List[str]]:
    if src == dst or src not in graph or dst not in graph:
        return []
    try:
        return [list(p) for p in islice(nx.shortest_simple_paths(graph, src, dst, weight=weight), k)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
