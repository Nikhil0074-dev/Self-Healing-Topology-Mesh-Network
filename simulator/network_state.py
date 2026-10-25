"""Discrete-time mesh-network simulator (ground truth + data plane).

The simulator owns the *physical truth* (which links are really up, true latency, ...).
The controller never reads that truth: it only sees noisy probe observations, and it
changes forwarding through ``install_route`` / ``probe_path``.

Random streams are split so the environment (failures, stress) is identical across
routing strategies for the same seed:
  rng_env : stress / hazard / outage durations     (constant draws per step)
  rng_pkt : packet sampling and telemetry probes   (constant draws per step)
  rng_ctl : controller-initiated route verification
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from simulator.traffic_generator import Flow

Edge = Tuple[str, str]

LINK_DOWN = "link_down"
LINK_UP = "link_up"
NODE_DOWN = "node_down"
NODE_UP = "node_up"
DEGRADE = "degrade"
CONGEST = "congest"
MODEL = "model"


def edge_key(u: str, v: str) -> Edge:
    return (u, v) if u <= v else (v, u)


@dataclass
class SimConfig:
    hazard_scale: float = 1.0
    node_hazard_scale: float = 1.0
    base_hazard: float = 0.0004
    max_hazard: float = 0.06
    cascade_hazard: float = 0.003
    burst_prob: float = 0.005
    node_base_hazard: float = 0.0002
    down_min: int = 8
    down_max: int = 30
    node_down_min: int = 10
    node_down_max: int = 40
    packets_per_flow: int = 5
    probes_per_link: int = 10
    probe_blip_prob: float = 0.01
    install_fail_prob: float = 0.0
    buffer_limit: int = 200
    buffer_ttl: int = 60
    verify_packets: int = 5


@dataclass
class LinkTruth:
    latency: float
    loss: float
    up: bool = True
    down_until: int = -1
    forced_down: bool = False
    stress: float = 0.15
    severity: float = 0.0
    bg_load: float = 0.0
    util: float = 0.0
    jitter: float = 0.0
    s_eff: float = 0.0
    fail_count: int = 0


@dataclass
class NodeTruth:
    alive: bool = True
    down_until: int = -1
    forced_down: bool = False
    cpu: float = 0.05
    mem: float = 0.3
    queue: float = 0.0
    load: float = 0.0
    fail_count: int = 0


@dataclass
class FlowStep:
    flow_id: str
    sent: int
    delivered: int
    latencies: List[float]
    route_valid: bool
    buffered: int = 0
    expired: int = 0
    late_delivered: int = 0


class MeshSimulator:
    def __init__(self, graph: nx.Graph, flows: Sequence[Flow], config: Optional[SimConfig] = None, seed: int = 0):
        self.graph = graph
        self.flows: Dict[str, Flow] = {f.flow_id: f for f in flows}
        self.cfg = config or SimConfig()
        self.seed = int(seed)
        self.rng_env = np.random.default_rng([self.seed, 101])
        self.rng_pkt = np.random.default_rng([self.seed, 202])
        self.rng_ctl = np.random.default_rng([self.seed, 303])
        self.nodes: List[str] = sorted(graph.nodes)
        self.edges: List[Edge] = sorted(edge_key(u, v) for u, v in graph.edges)
        self.link: Dict[Edge, LinkTruth] = {}
        for e in self.edges:
            d = graph.edges[e]
            self.link[e] = LinkTruth(latency=d["base_latency"], loss=d["base_loss"])
        self.node: Dict[str, NodeTruth] = {n: NodeTruth() for n in self.nodes}
        self.node_links: Dict[str, List[Edge]] = {n: [] for n in self.nodes}
        for e in self.edges:
            self.node_links[e[0]].append(e)
            self.node_links[e[1]].append(e)
        self.adj_links: Dict[Edge, List[Edge]] = {
            e: [x for x in self.node_links[e[0]] + self.node_links[e[1]] if x != e] for e in self.edges
        }
        self.routes: Dict[str, Optional[List[str]]] = {fid: None for fid in self.flows}
        self.buffers: Dict[str, Deque[int]] = {fid: deque() for fid in self.flows}
        self._late: Dict[str, int] = {fid: 0 for fid in self.flows}
        self.t = 0

    # ------------------------------------------------------------------ truth helpers
    def link_up(self, e: Edge) -> bool:
        lt = self.link[e]
        return lt.up and self.node[e[0]].alive and self.node[e[1]].alive

    def path_valid(self, path: Sequence[str]) -> bool:
        if not path or any(n not in self.node or not self.node[n].alive for n in path):
            return False
        for u, v in zip(path, path[1:]):
            e = edge_key(u, v)
            if e not in self.link or not self.link_up(e):
                return False
        return True

    def truth_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(n for n in self.nodes if self.node[n].alive)
        for e in self.edges:
            if self.link_up(e):
                g.add_edge(*e, latency=self.link[e].latency)
        return g

    def reference_latency(self, src: str, dst: str) -> Optional[float]:
        g = self.truth_graph()
        if src not in g or dst not in g:
            return None
        try:
            return float(nx.shortest_path_length(g, src, dst, weight="latency"))
        except nx.NetworkXNoPath:
            return None

    def route_latency_now(self, path: Sequence[str]) -> Optional[float]:
        if not self.path_valid(path):
            return None
        return float(sum(self.link[edge_key(u, v)].latency for u, v in zip(path, path[1:])))

    def physical_components(self) -> List[set]:
        return [set(c) for c in nx.connected_components(self.truth_graph())]

    # ------------------------------------------------------------------ dynamics
    def step(self, t: int) -> None:
        self.t = t
        cfg = self.cfg
        E, N = len(self.edges), len(self.nodes)
        rng = self.rng_env
        noise = rng.normal(size=(E, 3))
        u_burst = rng.random(E)
        u_haz = rng.random(E)
        burst_mag = rng.uniform(0.3, 0.6, size=E)
        dur_l = rng.integers(cfg.down_min, cfg.down_max + 1, size=E)
        n_noise = rng.normal(size=(N, 3))
        u_nhaz = rng.random(N)
        n_dur = rng.integers(cfg.node_down_min, cfg.node_down_max + 1, size=N)

        for e in self.edges:
            lt = self.link[e]
            if not lt.up and not lt.forced_down and t >= lt.down_until:
                lt.up = True
        for n in self.nodes:
            nt = self.node[n]
            if not nt.alive and not nt.forced_down and t >= nt.down_until:
                nt.alive = True

        link_load = {e: self.link[e].bg_load for e in self.edges}
        node_load = {n: 0.0 for n in self.nodes}
        for fid, path in self.routes.items():
            if path and self.path_valid(path):
                d = self.flows[fid].demand_mbps
                for n in path:
                    node_load[n] += d
                for u, v in zip(path, path[1:]):
                    link_load[edge_key(u, v)] += d

        for i, n in enumerate(self.nodes):
            nt = self.node[n]
            cap = self.graph.nodes[n]["cpu_capacity"]
            nt.load = node_load[n]
            nt.cpu = float(np.clip(node_load[n] / cap + 0.05 + 0.02 * n_noise[i, 0], 0.0, 1.0))
            nt.mem = float(np.clip(0.3 + 0.4 * nt.cpu + 0.02 * n_noise[i, 1], 0.0, 1.0))
            nt.queue = float(max(0.0, (nt.cpu - 0.7) * 300.0 + 2.0 * n_noise[i, 2]))
            if nt.alive and not nt.forced_down:
                p = (cfg.node_base_hazard + 0.01 * max(0.0, nt.cpu - 0.85)) * cfg.node_hazard_scale
                if u_nhaz[i] < p:
                    nt.alive = False
                    nt.down_until = t + int(n_dur[i])
                    nt.fail_count += 1

        old_stress = {e: self.link[e].stress for e in self.edges}
        for i, e in enumerate(self.edges):
            lt = self.link[e]
            d = self.graph.edges[e]
            nbrs = self.adj_links[e]
            nbr_mean = float(np.mean([old_stress[x] for x in nbrs])) if nbrs else lt.stress
            lt.stress += 0.08 * (0.15 - lt.stress) + 0.06 * (nbr_mean - lt.stress) + 0.04 * noise[i, 0]
            if u_burst[i] < cfg.burst_prob:
                lt.stress += burst_mag[i]
            lt.stress = float(np.clip(lt.stress, 0.0, 1.0))
            util = link_load[e] / d["bandwidth"]
            lt.util = float(util)
            sev = lt.severity
            s_eff = min(1.5, lt.stress + 0.8 * sev + 0.5 * max(0.0, util - 0.7))
            lt.s_eff = float(s_eff)
            base = d["base_latency"]
            lat = base * (1.0 + 0.9 * s_eff + 1.5 * max(0.0, util - 0.6) ** 2) * (1.0 + 2.0 * sev)
            lat += 0.02 * base * noise[i, 1]
            lt.latency = float(max(0.1 * base, lat))
            lt.loss = float(np.clip(d["base_loss"] + 0.3 * s_eff ** 2 + 0.35 * sev ** 2 + 0.2 * max(0.0, util - 0.9), 0.0, 0.9))
            lt.jitter = float(abs(0.05 * lt.latency * (1.0 + 2.0 * s_eff) + 0.3 * noise[i, 2]))
            if lt.up and not lt.forced_down:
                p = cfg.base_hazard + cfg.max_hazard * min(s_eff, 1.0) ** 3
                if any(not self.link[x].up for x in nbrs):
                    p += cfg.cascade_hazard
                p *= cfg.hazard_scale
                if u_haz[i] < p:
                    lt.up = False
                    lt.down_until = t + int(dur_l[i])
                    lt.fail_count += 1

    # ------------------------------------------------------------------ scripted events
    def apply_event(self, kind: str, target, value: float = 0.0) -> None:
        if kind == LINK_DOWN:
            lt = self.link[edge_key(*target)]
            if lt.up:
                lt.fail_count += 1
            lt.up, lt.forced_down = False, True
        elif kind == LINK_UP:
            lt = self.link[edge_key(*target)]
            lt.up, lt.forced_down, lt.down_until = True, False, -1
        elif kind == NODE_DOWN:
            nt = self.node[target]
            if nt.alive:
                nt.fail_count += 1
            nt.alive, nt.forced_down = False, True
        elif kind == NODE_UP:
            nt = self.node[target]
            nt.alive, nt.forced_down, nt.down_until = True, False, -1
        elif kind == DEGRADE:
            self.link[edge_key(*target)].severity = float(np.clip(value, 0.0, 1.0))
        elif kind == CONGEST:
            self.link[edge_key(*target)].bg_load = float(max(0.0, value))
        else:
            raise ValueError(f"unknown event kind {kind!r}")

    # ------------------------------------------------------------------ data plane
    def send_traffic(self, t: int) -> Dict[str, FlowStep]:
        cfg = self.cfg
        P = cfg.packets_per_flow
        out: Dict[str, FlowStep] = {}
        for fid, flow in self.flows.items():
            u = self.rng_pkt.random(P)
            jit = self.rng_pkt.normal(size=P)
            path = self.routes[fid]
            late = self._late[fid]
            self._late[fid] = 0
            buf = self.buffers[fid]
            expired = 0
            while buf and t - buf[0] > cfg.buffer_ttl:
                buf.popleft()
                expired += 1
            delivered, lats, valid, buffered = 0, [], False, 0
            if path is None:
                if flow.traffic_class == "telemetry":
                    for _ in range(P):
                        if len(buf) < cfg.buffer_limit:
                            buf.append(t)
                            buffered += 1
            else:
                valid = self.path_valid(path)
                if valid:
                    edges = [edge_key(a, b) for a, b in zip(path, path[1:])]
                    psucc = float(np.prod([1.0 - self.link[e].loss for e in edges]))
                    base = sum(self.link[e].latency for e in edges) + sum(0.02 * self.node[n].queue for n in path)
                    spread = float(np.sqrt(sum(self.link[e].jitter ** 2 for e in edges)))
                    for k in range(P):
                        if u[k] < psucc:
                            delivered += 1
                            lats.append(max(0.1, base + spread * jit[k]))
            out[fid] = FlowStep(fid, P, delivered, lats, valid, buffered, expired, late)
        return out

    def install_route(self, flow_id: str, path: Optional[Sequence[str]]) -> bool:
        if flow_id not in self.flows:
            raise KeyError(flow_id)
        if self.rng_ctl.random() < self.cfg.install_fail_prob:
            return False
        if path is None:
            self.routes[flow_id] = None
            return True
        self.routes[flow_id] = list(path)
        buf = self.buffers[flow_id]
        self._late[flow_id] += len(buf)
        buf.clear()
        return True

    def current_route(self, flow_id: str) -> Optional[List[str]]:
        r = self.routes[flow_id]
        return list(r) if r else None

    def probe_path(self, path: Sequence[str]) -> Tuple[bool, Optional[float]]:
        """Active end-to-end probe along ``path`` (data-plane verification)."""
        n = self.cfg.verify_packets
        u = self.rng_ctl.random(n)
        jit = self.rng_ctl.normal(size=n)
        if not self.path_valid(path):
            return False, None
        edges = [edge_key(a, b) for a, b in zip(path, path[1:])]
        psucc = float(np.prod([1.0 - self.link[e].loss for e in edges]))
        base = sum(self.link[e].latency for e in edges)
        lats = [max(0.1, base + 0.03 * base * jit[k]) for k in range(n) if u[k] < psucc]
        if len(lats) * 2 > n:  # majority of probes must arrive
            return True, float(np.mean(lats))
        return False, None

    # ------------------------------------------------------------------ telemetry
    def observe(self, t: int) -> dict:
        """Raw (noisy) probe results; converted to typed records by TelemetryCollector."""
        cfg = self.cfg
        E, N, K = len(self.edges), len(self.nodes), cfg.probes_per_link
        recv_draw = self.rng_pkt.random((E, K))
        blips = self.rng_pkt.random(E)
        noise = self.rng_pkt.normal(size=(E, 3))
        nblip = self.rng_pkt.random(N)
        nnoise = self.rng_pkt.normal(size=(N, 3))
        links = {}
        for i, e in enumerate(self.edges):
            lt = self.link[e]
            received = int((recv_draw[i] >= lt.loss).sum()) if self.link_up(e) else 0
            if blips[i] < cfg.probe_blip_prob:
                received = 0
            if received > 0:
                lat = max(0.1, lt.latency + (0.03 * lt.latency + 0.2) * noise[i, 0])
                util = float(np.clip(lt.util + 0.02 * noise[i, 1], 0.0, 2.0))
                jitter = float(max(0.0, lt.jitter + 0.1 * noise[i, 2]))
            else:
                lat, util, jitter = None, 0.0, 0.0
            links[e] = (K, received, lat, 1.0 - received / K, util, jitter)
        nodes = {}
        for i, n in enumerate(self.nodes):
            nt = self.node[n]
            ok = nt.alive and nblip[i] >= cfg.probe_blip_prob
            if ok:
                nodes[n] = (True, float(np.clip(nt.cpu + 0.02 * nnoise[i, 0], 0, 1)),
                            float(np.clip(nt.mem + 0.02 * nnoise[i, 1], 0, 1)),
                            float(max(0.0, nt.queue + nnoise[i, 2])), nt.load)
            else:
                nodes[n] = (False, 0.0, 0.0, 0.0, 0.0)
        return {"step": t, "links": links, "nodes": nodes}
