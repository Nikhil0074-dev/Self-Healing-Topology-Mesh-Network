"""Frozen-model inference with validity checks (fall back to conventional routing if invalid)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from app.topology.graph_state import TopologyState
from gnn.features import EDGE_FEATURES, LAT_SCALE, build_graph_tensors, schema_hash

AGE_INDEX = EDGE_FEATURES.index("meas_age")
from gnn.model import LinkQualityGNN

Edge = Tuple[str, str]


@dataclass(frozen=True)
class LinkPrediction:
    p_fail: float
    latency_ms: float


class GNNPredictor:
    """override: None | "unavailable" | "invalid"  (used to test the fallback path)."""

    def __init__(self, model: LinkQualityGNN, platt: Tuple[float, float] = (1.0, 0.0), meta: Optional[dict] = None):
        self.model = model.eval()
        self.platt = platt
        self.meta = meta or {}
        self.override: Optional[str] = None
        self.last_error: Optional[str] = None
        self.calls = 0
        self.total_ms = 0.0

    @staticmethod
    def load(path: str) -> "GNNPredictor":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck["schema_hash"] != schema_hash():
            raise ValueError("model feature schema does not match the current feature definition")
        model = LinkQualityGNN(**ck["model_cfg"])
        model.load_state_dict(ck["state_dict"])
        return GNNPredictor(model, tuple(ck["platt"]), {k: ck[k] for k in ("version", "horizon", "metrics") if k in ck})

    def reset_stats(self) -> None:
        self.calls, self.total_ms = 0, 0.0

    def predict(self, state: TopologyState, step: int) -> Optional[Dict[Edge, LinkPrediction]]:
        """Return a prediction per link, or None if the model cannot be trusted this step."""
        self.last_error = None
        if self.override == "unavailable":
            self.last_error = "model service unavailable"
            return None
        t0 = time.perf_counter()
        try:
            g = build_graph_tensors(state, step)
            if not (torch.isfinite(g.node_x).all() and torch.isfinite(g.edge_x).all()):
                self.last_error = "non-finite input features"
                return None
            if g.node_x.abs().max() > 10 or g.edge_x.abs().max() > 10:
                self.last_error = "input features out of distribution"
                return None
            use = g.edge_x[g.usable]
            if use.numel() and use[:, AGE_INDEX].mean() > 0.8:  # mean measurement age of usable links
                self.last_error = "telemetry too stale"
                return None
            with torch.no_grad():
                out = self.model(g.node_x, g.edge_list, g.edge_x)
            if self.override == "invalid":
                out = torch.full_like(out, float("nan"))
            logits, lat = out[:, 0], out[:, 1]
            if not (torch.isfinite(logits).all() and torch.isfinite(lat).all()):
                self.last_error = "non-finite model output"
                return None
            probs = torch.sigmoid(self.platt[0] * logits + self.platt[1])
            lat_ms = lat * LAT_SCALE
            if (lat_ms < -1.0).any() or (lat_ms > 5000.0).any():
                self.last_error = "latency prediction out of bounds"
                return None
            res = {e: LinkPrediction(float(probs[i]), float(max(0.1, lat_ms[i]))) for i, e in enumerate(g.edges)}
            return res
        finally:
            self.calls += 1
            self.total_ms += (time.perf_counter() - t0) * 1000.0
