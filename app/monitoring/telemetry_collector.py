"""Typed telemetry records with validation, staleness and duplicate rejection."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

Edge = Tuple[str, str]


@dataclass(frozen=True)
class LinkProbe:
    link: Edge
    step: int
    sent: int
    received: int
    latency_ms: Optional[float]
    loss: float
    util: float
    jitter_ms: float
    window: int = 1
    source: str = "probe-agent"
    seq: int = 0

    @property
    def ok(self) -> bool:
        return self.received > 0


@dataclass(frozen=True)
class NodeBeat:
    node: str
    step: int
    ok: bool
    cpu: float
    mem: float
    queue: float
    load: float
    source: str = "heartbeat-agent"
    seq: int = 0


@dataclass
class Observation:
    step: int
    link_probes: Dict[Edge, LinkProbe] = field(default_factory=dict)
    node_beats: Dict[str, NodeBeat] = field(default_factory=dict)


class TelemetryCollector:
    def __init__(self, known_nodes: Iterable[str], known_links: Iterable[Edge], max_lag: int = 5):
        self.known_nodes = set(known_nodes)
        self.known_links = set(known_links)
        self.max_lag = max_lag
        self._last_seq: Dict[Tuple[str, str, object], int] = {}
        self.rejected = {"unknown": 0, "stale": 0, "invalid": 0}

    def _fresh(self, source: str, kind: str, ident: object, seq: int) -> bool:
        key = (source, kind, ident)
        if seq <= self._last_seq.get(key, -1):
            self.rejected["stale"] += 1
            return False
        self._last_seq[key] = seq
        return True

    @staticmethod
    def _finite(*vals: Optional[float]) -> bool:
        return all(v is None or (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0) for v in vals)

    def accept_link(self, p: LinkProbe, now: int) -> bool:
        if p.link not in self.known_links:
            self.rejected["unknown"] += 1
            return False
        if now - p.step > self.max_lag:
            self.rejected["stale"] += 1
            return False
        if not self._finite(p.latency_ms, p.loss, p.util, p.jitter_ms) \
                or p.sent <= 0 or not 0 <= p.received <= p.sent or p.loss > 1.0:
            self.rejected["invalid"] += 1
            return False
        return self._fresh(p.source, "link", p.link, p.seq)

    def accept_node(self, b: NodeBeat, now: int) -> bool:
        if b.node not in self.known_nodes:
            self.rejected["unknown"] += 1
            return False
        if now - b.step > self.max_lag:
            self.rejected["stale"] += 1
            return False
        if not self._finite(b.cpu, b.mem, b.queue, b.load):
            self.rejected["invalid"] += 1
            return False
        return self._fresh(b.source, "node", b.node, b.seq)

    def from_raw(self, raw: dict, step: int) -> Observation:
        obs = Observation(step)
        for e, (sent, recv, lat, loss, util, jit) in raw["links"].items():
            p = LinkProbe(e, raw["step"], sent, recv, lat, loss, util, jit, seq=raw["step"])
            if self.accept_link(p, step):
                obs.link_probes[e] = p
        for n, (ok, cpu, mem, queue, load) in raw["nodes"].items():
            b = NodeBeat(n, raw["step"], ok, cpu, mem, queue, load, seq=raw["step"])
            if self.accept_node(b, step):
                obs.node_beats[n] = b
        return obs
