"""Train the link-quality GNN on episode-split data and save a versioned checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from gnn.dataset import HORIZON, build_splits, class_balance
from gnn.evaluate import evaluate_model, predict_samples
from gnn.features import EDGE_FEATURES, NODE_FEATURES, schema_hash
from gnn.model import LinkQualityGNN, collate

VERSION = "link_gnn_v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_loss(model: LinkQualityGNN, b: dict, pos_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(b["node_x"], b["edge_list"], b["edge_x"])
    mf, ml = b["m_fail"], b["m_lat"]
    lf = F.binary_cross_entropy_with_logits(out[mf, 0], b["y_fail"][mf], pos_weight=pos_weight) if mf.any() \
        else out.sum() * 0.0
    ll = F.smooth_l1_loss(out[ml, 1], b["y_lat"][ml], beta=0.05) if ml.any() else out.sum() * 0.0
    return lf + 5.0 * ll, lf, ll


@torch.no_grad()
def dataset_loss(model: LinkQualityGNN, samples: Sequence[dict], pos_weight: torch.Tensor) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(samples), 64):
        b = collate(samples[i:i + 64])
        loss, _, _ = batch_loss(model, b, pos_weight)
        tot += loss.item() * len(samples[i:i + 64])
        n += len(samples[i:i + 64])
    return tot / max(n, 1)


def fit_platt(logits: np.ndarray, y: np.ndarray, iters: int = 200) -> Tuple[float, float]:
    """Calibrate probabilities on validation data:  p = sigmoid(a*logit + b)."""
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lg, yy = torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(a * lg + b, yy)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a), float(b)


def train(splits: Dict[str, List[dict]], epochs: int = 30, lr: float = 2e-3, hidden: int = 48, layers: int = 2,
          batch_size: int = 16, patience: int = 6, seed: int = 0, out: str = "models/link_gnn_v1.pt",
          verbose: bool = True) -> dict:
    set_seed(seed)
    tr, va = splits["train"], splits["val"]
    pos = class_balance(tr)
    pw = torch.tensor(min(max((1 - pos) / max(pos, 1e-6), 1.0), 20.0))
    model = LinkQualityGNN(hidden=hidden, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        run = 0.0
        for i in range(0, len(order), batch_size):
            b = collate([tr[j] for j in order[i:i + batch_size]])
            opt.zero_grad()
            loss, _, _ = batch_loss(model, b, pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * len(order[i:i + batch_size])
        vl = dataset_loss(model, va, pw)
        if verbose:
            print(f"epoch {ep + 1:02d} train={run / len(tr):.4f} val={vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    logits, _, yf, _, mf, _, _ = predict_samples(model, va)
    platt = fit_platt(logits[mf].numpy(), yf[mf].numpy())
    val_metrics = evaluate_model(model, va, platt)
    thr = val_metrics["threshold"]          # threshold chosen on validation only
    metrics = {"val": val_metrics, "test": evaluate_model(model, splits["test"], platt, thr),
               "test_unseen": evaluate_model(model, splits["test_unseen"], platt, thr)}
    ck = {"version": VERSION, "state_dict": model.state_dict(), "model_cfg": model.cfg, "platt": list(platt),
          "schema_hash": schema_hash(), "node_features": NODE_FEATURES, "edge_features": EDGE_FEATURES,
          "horizon": HORIZON, "metrics": metrics, "train_args": {"epochs": epochs, "lr": lr, "seed": seed,
                                                                 "pos_weight": float(pw), "batch_size": batch_size}}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(ck, out)
    with open(os.path.splitext(out)[0] + "_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="experiments/results/dataset.pt")
    ap.add_argument("--out", default="models/link_gnn_v1.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    splits = torch.load(a.data, weights_only=False)
    m = train(splits, epochs=a.epochs, seed=a.seed, out=a.out)
    for split in ("val", "test", "test_unseen"):
        r = m[split]
        print(f"{split:12s} AUC={r['auc']:.3f} (baseline {r['baseline_auc']:.3f}) AP={r['avg_precision']:.3f} "
              f"(baseline {r['baseline_avg_precision']:.3f}) F1={r['f1']:.3f} brier={r['brier']:.4f} "
              f"latMAE={r['lat_mae']:.2f} (persistence {r['persistence_lat_mae']:.2f})")
