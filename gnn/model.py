"""GraphSAGE-style edge-aware GNN written in pure PyTorch (no extra graph library needed).

Each layer:  m_ij = ReLU(W_m [h_j || e_ij]);  h_i' = LayerNorm(h_i + ReLU(W_s h_i + W_a mean_j m_ij))
Link head (symmetric in the two endpoints) predicts, per undirected link:
    out[:,0] = failure-within-horizon logit
    out[:,1] = change in latency (units of 100 ms) relative to the current smoothed latency
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.features import EDGE_FEATURES, LAT_INDEX, NODE_FEATURES


class SageEdgeLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Linear(hidden + edge_dim, hidden)
        self.self_lin = nn.Linear(hidden, hidden)
        self.agg_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        m = F.relu(self.msg(torch.cat([h[src], edge_attr], dim=-1)))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        deg = torch.zeros(h.size(0), 1, dtype=h.dtype, device=h.device).index_add_(
            0, dst, torch.ones(m.size(0), 1, dtype=h.dtype, device=h.device)).clamp(min=1.0)
        out = F.relu(self.self_lin(h) + self.agg_lin(agg / deg))
        return self.norm(h + out)


class LinkQualityGNN(nn.Module):
    def __init__(self, node_dim: int = len(NODE_FEATURES), edge_dim: int = len(EDGE_FEATURES),
                 hidden: int = 48, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cfg = {"node_dim": node_dim, "edge_dim": edge_dim, "hidden": hidden, "layers": layers,
                    "dropout": dropout}
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.layers = nn.ModuleList([SageEdgeLayer(hidden, edge_dim) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 2))

    def forward(self, node_x: torch.Tensor, edge_list: torch.Tensor, edge_x: torch.Tensor) -> torch.Tensor:
        ei = torch.cat([edge_list.t(), edge_list.flip(1).t()], dim=1)
        ea = torch.cat([edge_x, edge_x], dim=0)
        h = self.node_enc(node_x)
        for layer in self.layers:
            h = layer(h, ei, ea)
        hu, hv = h[edge_list[:, 0]], h[edge_list[:, 1]]
        out = self.head(torch.cat([hu + hv, (hu - hv).abs(), edge_x], dim=-1))
        lat = edge_x[:, LAT_INDEX] + out[:, 1]   # residual on current latency
        return torch.stack([out[:, 0], lat], dim=-1)


def collate(samples: Sequence[dict]) -> dict:
    """Merge several graphs into one disjoint batch."""
    nx_, el, ex, yf, yl, mf, ml = [], [], [], [], [], [], []
    off = 0
    for s in samples:
        nx_.append(s["node_x"])
        el.append(s["edge_list"] + off)
        ex.append(s["edge_x"])
        yf.append(s["y_fail"])
        yl.append(s["y_lat"])
        mf.append(s["m_fail"])
        ml.append(s["m_lat"])
        off += s["node_x"].size(0)
    return {"node_x": torch.cat(nx_), "edge_list": torch.cat(el), "edge_x": torch.cat(ex),
            "y_fail": torch.cat(yf), "y_lat": torch.cat(yl), "m_fail": torch.cat(mf), "m_lat": torch.cat(ml)}
