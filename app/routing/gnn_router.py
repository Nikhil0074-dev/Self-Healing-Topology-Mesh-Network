"""GNN-assisted router: same scoring/validation pipeline as the heuristic router, but link
estimates blend GNN predictions. Falls back to heuristic estimates if the model is unusable."""
from __future__ import annotations

from typing import Dict, Optional

from app.routing.baseline_router import ScoringRouter
from app.routing.cost_function import CostWeights, EdgeEstimate
from app.topology.graph_state import Edge, TopologyState


class GNNRouter(ScoringRouter):
    """mode: "both" (latency + risk), "latency" (latency only), "risk" (failure risk only)."""

    def __init__(self, predictor, mode: str = "both", blend: float = 0.6,
                 weights: Optional[CostWeights] = None, k: int = 5):
        super().__init__(weights, k)
        if mode not in ("both", "latency", "risk"):
            raise ValueError("mode must be 'both', 'latency' or 'risk'")
        self.predictor, self.mode, self.blend = predictor, mode, blend
        self.name = "gnn" if mode == "both" else f"gnn_{mode}"

    def prepare(self, state: TopologyState, step: int) -> None:
        base = self.heuristic_estimates(state, step)
        pred = None
        if self.predictor is not None:
            pred = self.predictor.predict(state, step)
        if pred is None:
            self.fallback_active = True
            self._est = base
            return
        self.fallback_active = False
        est: Dict[Edge, EdgeEstimate] = {}
        for k, b in base.items():
            p = pred.get(k)
            if p is None:
                est[k] = b
                continue
            lat = b.latency_ms
            pf = b.p_fail
            if self.mode in ("both", "latency"):
                lat = self.blend * p.latency_ms + (1.0 - self.blend) * b.latency_ms
            if self.mode in ("both", "risk"):
                pf = p.p_fail
            est[k] = EdgeEstimate(lat, b.loss, b.util, pf)
        self._est = est
