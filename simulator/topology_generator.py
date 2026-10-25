"""Random mesh-topology generation with link/node attributes."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import networkx as nx
import numpy as np

TOPOLOGY_KINDS = ("geometric", "small_world", "grid", "scale_free")
TRAIN_KINDS = ("geometric", "small_world")
UNSEEN_KINDS = ("grid", "scale_free")


def node_name(i: int) -> str:
    return f"node_{i:02d}"


def _geometric(n: int, rng: np.random.Generator) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    pos = {i: (float(rng.random()), float(rng.random())) for i in range(n)}
    radius = 0.22 * math.sqrt(20.0 / n)
    while True:
        g = nx.random_geometric_graph(n, radius, pos=pos)
        if nx.is_connected(g):
            return g, pos
        radius += 0.02


def _grid(n: int, rng: np.random.Generator) -> Tuple[nx.Graph, Dict[int, Tuple[float, float]]]:
    rows = max(2, int(round(math.sqrt(n))))
    cols = max(2, int(math.ceil(n / rows)))
    g2 = nx.grid_2d_graph(rows, cols)
    for r in range(rows - 1):
        for c in range(cols - 1):
            if rng.random() < 0.15:
                g2.add_edge((r, c), (r + 1, c + 1))
    g = nx.convert_node_labels_to_integers(g2, ordering="sorted")
    # Keep the first n row-major cells. This induced prefix remains connected
    # while making the requested node count exact for non-square sizes.
    g.remove_nodes_from([i for i in g.nodes if i >= n])
    pos = {r * cols + c: (c / max(cols - 1, 1), r / max(rows - 1, 1))
           for r in range(rows) for c in range(cols) if r * cols + c < n}
    return g, pos


def generate_topology(kind: str = "geometric", n_nodes: int = 20, seed: int = 0) -> nx.Graph:
    """Return a connected undirected graph with node and link attributes.

    Link attributes: base_latency (ms), bandwidth (Mbps), base_loss (fraction).
    Node attributes: pos (x, y), cpu_capacity (Mbps of forwarding capacity).
    """
    if kind not in TOPOLOGY_KINDS:
        raise ValueError(f"unknown topology kind {kind!r}; choose from {TOPOLOGY_KINDS}")
    if n_nodes < 6:
        raise ValueError("n_nodes must be at least 6")
    rng = np.random.default_rng([int(seed), 7])
    pos = None
    if kind == "geometric":
        g, pos = _geometric(n_nodes, rng)
    elif kind == "small_world":
        g = nx.connected_watts_strogatz_graph(n_nodes, 4, 0.25, tries=200, seed=int(seed))
    elif kind == "grid":
        g, pos = _grid(n_nodes, rng)
    else:
        g = nx.barabasi_albert_graph(n_nodes, 2, seed=int(seed))
    if pos is None:
        pos = nx.spring_layout(g, seed=int(seed))
    n = g.number_of_nodes()
    out = nx.Graph()
    for i in sorted(g.nodes):
        out.add_node(node_name(i), pos=(float(pos[i][0]), float(pos[i][1])), cpu_capacity=150.0)
    for u, v in sorted((min(a, b), max(a, b)) for a, b in g.edges):
        if kind == "geometric":
            d = math.dist(pos[u], pos[v])
            lat = d * 80.0 + 2.0
        else:
            lat = float(rng.uniform(3.0, 22.0))
        out.add_edge(
            node_name(u),
            node_name(v),
            base_latency=float(lat),
            bandwidth=float(rng.choice([50.0, 100.0, 100.0, 200.0])),
            base_loss=float(rng.uniform(0.001, 0.01)),
        )
    assert out.number_of_nodes() == n and nx.is_connected(out)
    return out
