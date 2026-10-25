"""Connected-component and partition analysis."""
from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

import networkx as nx


def connected_components(g: nx.Graph) -> List[Set[str]]:
    comps = [set(c) for c in nx.connected_components(g)]
    comps.sort(key=lambda c: (-len(c), min(c)))
    return comps


def component_map(g: nx.Graph) -> Dict[str, int]:
    return {n: i for i, c in enumerate(connected_components(g)) for n in c}


def is_reachable(g: nx.Graph, src: str, dst: str) -> bool:
    return src in g and dst in g and nx.has_path(g, src, dst)


def isolated_nodes(g: nx.Graph) -> List[str]:
    return sorted(n for n in g.nodes if g.degree(n) == 0)


def unreachable_pairs(g: nx.Graph, pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(s, d) for s, d in pairs if not is_reachable(g, s, d)]


def partition_summary(g: nx.Graph) -> dict:
    comps = connected_components(g)
    return {"n_components": len(comps), "sizes": [len(c) for c in comps], "isolated": isolated_nodes(g)}
