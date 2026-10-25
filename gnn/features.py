"""Feature extraction from the controller's *observed* state (never ground truth).

Scaling constants are fixed (documented below) rather than fitted, so no statistics from
validation/test data can leak into preprocessing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from app.topology.graph_state import Health, TopologyState, USABLE_LINK

NODE_FEATURES = ["alive", "cpu", "queue", "mem", "degree", "load", "failure_count", "last_seen_age"]
EDGE_FEATURES = ["latency", "latency_ratio", "loss", "bandwidth", "util", "jitter", "available",
                 "failure_freq", "meas_age", "lat_trend", "loss_trend", "degraded", "suspected"]
LAT_INDEX = EDGE_FEATURES.index("latency")
LAT_SCALE = 100.0   # latency feature = ms / 100


def schema_hash() -> str:
    return hashlib.sha256(("|".join(NODE_FEATURES) + "#" + "|".join(EDGE_FEATURES)).encode()).hexdigest()[:12]


@dataclass
class GraphTensors:
    node_x: torch.Tensor          # [N, Fn]
    edge_list: torch.Tensor       # [E, 2] long (undirected, each link once)
    edge_x: torch.Tensor          # [E, Fe]
    usable: torch.Tensor          # [E] bool - link currently believed usable
    nodes: List[str]
    edges: List[Tuple[str, str]]


def build_graph_tensors(state: TopologyState, step: int) -> GraphTensors:
    nodes = state.node_list
    edges = state.link_list
    idx = {n: i for i, n in enumerate(nodes)}
    deg = {n: 0 for n in nodes}
    for k in edges:
        if state.links[k].status in USABLE_LINK:
            deg[k[0]] += 1
            deg[k[1]] += 1
    nx_rows = []
    for n in nodes:
        ns = state.nodes[n]
        nx_rows.append([
            1.0 if ns.status == Health.HEALTHY else 0.0,
            ns.cpu,
            min(ns.queue / 200.0, 2.0),
            ns.mem,
            min(deg[n] / 8.0, 2.0),
            min(ns.load / 150.0, 2.0),
            min(ns.recent_failures(step, 100) / 5.0, 1.0),
            min(max(0, step - ns.last_seen_step) / 20.0, 1.0),
        ])
    ex_rows, usable = [], []
    for k in edges:
        ls = state.links[k]
        is_use = ls.status in USABLE_LINK
        usable.append(is_use)
        ex_rows.append([
            ls.ewma_latency / LAT_SCALE,
            min(ls.ewma_latency / max(ls.base_latency, 1e-6), 5.0) / 5.0,
            min(max(ls.ewma_loss, 0.0), 1.0),
            ls.bandwidth / 200.0,
            min(max(ls.util, 0.0), 1.5),
            min(ls.jitter / 20.0, 2.0),
            1.0 if is_use else 0.0,
            min(ls.recent_failures(step, 100) / 5.0, 1.0),
            min(ls.age(step) / 10.0, 1.0),
            float(np.clip((ls.ewma_latency - ls.prev_ewma_latency) / max(ls.base_latency, 1.0), -1.0, 1.0)),
            float(np.clip(ls.last_loss - ls.ewma_loss, -1.0, 1.0)),
            1.0 if ls.status == Health.DEGRADED else 0.0,
            1.0 if ls.status == Health.SUSPECTED else 0.0,
        ])
    edge_list = torch.tensor([[idx[a], idx[b]] for a, b in edges], dtype=torch.long)
    return GraphTensors(torch.tensor(nx_rows, dtype=torch.float32), edge_list,
                        torch.tensor(ex_rows, dtype=torch.float32), torch.tensor(usable, dtype=torch.bool),
                        list(nodes), list(edges))
