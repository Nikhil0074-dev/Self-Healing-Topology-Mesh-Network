"""Evaluation metrics for the link-quality GNN (numpy only, no sklearn)."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch

from gnn.features import EDGE_FEATURES, LAT_INDEX, LAT_SCALE
from gnn.model import LinkQualityGNN, collate


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ss):           # average ranks over ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / (np.arange(len(yy)) + 1)
    return float((prec * yy).sum() / yy.sum())


def prf(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = p >= thr
    y = y.astype(bool)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.999, 60))):
        f = prf(y, p, float(t))["f1"]
        if f > best:
            best, thr = f, float(t)
    return thr


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Dict[str, float]:
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"brier": brier, "ece": float(ece)}


@torch.no_grad()
def predict_samples(model: LinkQualityGNN, samples: Sequence[dict], batch: int = 32):
    model.eval()
    logits, lat, batches = [], [], []
    for i in range(0, len(samples), batch):
        b = collate(samples[i:i + batch])
        out = model(b["node_x"], b["edge_list"], b["edge_x"])
        logits.append(out[:, 0])
        lat.append(out[:, 1])
        batches.append(b)
    cat = lambda k: torch.cat([b[k] for b in batches])
    return (torch.cat(logits), torch.cat(lat), cat("y_fail"), cat("y_lat"), cat("m_fail"), cat("m_lat"), cat("edge_x"))


def evaluate_model(model: LinkQualityGNN, samples: Sequence[dict], platt=(1.0, 0.0), thr: float = None) -> dict:
    logits, lat, yf, yl, mf, ml, ex = predict_samples(model, samples)
    p = torch.sigmoid(platt[0] * logits + platt[1])[mf].numpy()
    y = yf[mf].numpy()
    res: Dict[str, float] = {"n_links": int(mf.sum()), "positive_rate": float(y.mean()) if len(y) else float("nan")}
    res["auc"] = roc_auc(y, p)
    res["avg_precision"] = average_precision(y, p)
    res.update(calibration(y, p))
    thr = best_f1_threshold(y, p) if thr is None else thr
    res["threshold"] = thr
    res.update(prf(y, p, thr))
    # non-learned baseline: rank links by current loss + failure history + utilisation
    f = {n: i for i, n in enumerate(EDGE_FEATURES)}
    base = (ex[:, f["loss"]] + 0.2 * ex[:, f["failure_freq"]] + 0.1 * ex[:, f["util"]])[mf].numpy()
    res["baseline_auc"] = roc_auc(y, base)
    res["baseline_avg_precision"] = average_precision(y, base)
    pred_ms = (lat[ml] * LAT_SCALE).numpy()
    true_ms = (yl[ml] * LAT_SCALE).numpy()
    cur_ms = (ex[ml, LAT_INDEX] * LAT_SCALE).numpy()
    res["lat_mae"] = float(np.mean(np.abs(pred_ms - true_ms))) if len(true_ms) else float("nan")
    res["lat_rmse"] = float(np.sqrt(np.mean((pred_ms - true_ms) ** 2))) if len(true_ms) else float("nan")
    res["persistence_lat_mae"] = float(np.mean(np.abs(cur_ms - true_ms))) if len(true_ms) else float("nan")
    res["persistence_lat_rmse"] = float(np.sqrt(np.mean((cur_ms - true_ms) ** 2))) if len(true_ms) else float("nan")
    return res
