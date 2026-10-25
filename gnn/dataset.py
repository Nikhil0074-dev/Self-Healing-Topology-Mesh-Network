"""Dataset generation: run the full controller pipeline on random episodes and label each
snapshot with what *actually happened* in the following ``horizon`` steps.

Labels (per usable link at snapshot time t, link physically up at t):
    y_fail : 1 if the link is physically down at any step in (t, t+horizon]
    y_lat  : mean true latency over the steps in the window where the link is up (units of 100 ms)
Features come only from controller-observed state at t (see gnn/features.py), so no label leakage.
Splitting is by *episode* (and by topology family for the unseen-topology test), never by snapshot.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from gnn.features import LAT_SCALE, build_graph_tensors
from simulator.experiment_runner import RunConfig, World, run_experiment
from simulator.topology_generator import TRAIN_KINDS, UNSEEN_KINDS

HORIZON = 10


def generate_episode(cfg: RunConfig, horizon: int = HORIZON, sample_every: int = 4, warmup: int = 10) -> List[dict]:
    snaps: List[dict] = []
    up_log: List[np.ndarray] = []
    lat_log: List[np.ndarray] = []

    def hook(t: int, w: World) -> None:
        sim = w.sim
        up_log.append(np.array([sim.link_up(e) for e in sim.edges], dtype=bool))
        lat_log.append(np.array([sim.link[e].latency for e in sim.edges], dtype=np.float32))
        if t >= warmup and t % sample_every == 0 and t + horizon < cfg.n_steps:
            g = build_graph_tensors(w.controller.state, t)
            assert g.edges == sim.edges, "edge order mismatch between controller and simulator"
            snaps.append({"t": t, "g": g, "truth_up_now": up_log[-1].copy()})

    run_experiment(cfg, None, on_step=hook)
    up = np.stack(up_log)
    lat = np.stack(lat_log)
    out = []
    for s in snaps:
        t, g = s["t"], s["g"]
        win_up = up[t + 1: t + 1 + horizon]
        win_lat = lat[t + 1: t + 1 + horizon]
        y_fail = (~win_up).any(axis=0)
        n_up = win_up.sum(axis=0)
        mean_lat = np.where(n_up > 0, (win_lat * win_up).sum(axis=0) / np.maximum(n_up, 1), 0.0)
        valid = g.usable.numpy() & s["truth_up_now"]
        m_lat = valid & (n_up > 0)
        out.append({
            "node_x": g.node_x, "edge_list": g.edge_list, "edge_x": g.edge_x,
            "y_fail": torch.tensor(y_fail, dtype=torch.float32),
            "y_lat": torch.tensor(mean_lat / LAT_SCALE, dtype=torch.float32),
            "m_fail": torch.tensor(valid, dtype=torch.bool),
            "m_lat": torch.tensor(m_lat, dtype=torch.bool),
            "t": t,
        })
    return out


def generate_dataset(n_episodes: int, seed0: int, kinds: Sequence[str], n_steps: int = 300,
                     sizes: Sequence[int] = (16, 20, 24, 28), horizon: int = HORIZON, tag: str = "train",
                     verbose: bool = True) -> List[dict]:
    data: List[dict] = []
    for i in range(n_episodes):
        seed = seed0 + i
        kind = kinds[i % len(kinds)]
        n = sizes[(i // len(kinds)) % len(sizes)]
        cfg = RunConfig(scenario="random_training", strategy="heuristic", seed=seed, topo_kind=kind, n_nodes=n,
                        n_flows=8, n_steps=n_steps, demand_range=(4.0, 18.0))
        for s in generate_episode(cfg, horizon):
            s["episode"], s["topo_kind"], s["seed"], s["n_nodes"] = f"{tag}_{seed}", kind, seed, n
            data.append(s)
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{tag}] {i + 1}/{n_episodes} episodes, {len(data)} snapshots", flush=True)
    return data


def build_splits(n_train: int = 60, n_val: int = 12, n_test: int = 12, n_unseen: int = 12, n_steps: int = 300,
                 verbose: bool = True) -> Dict[str, List[dict]]:
    """Disjoint seeds for every split; the unseen split uses topology families never trained on."""
    return {
        "train": generate_dataset(n_train, 10_000, TRAIN_KINDS, n_steps, tag="train", verbose=verbose),
        "val": generate_dataset(n_val, 20_000, TRAIN_KINDS, n_steps, tag="val", verbose=verbose),
        "test": generate_dataset(n_test, 30_000, TRAIN_KINDS, n_steps, tag="test", verbose=verbose),
        "test_unseen": generate_dataset(n_unseen, 40_000, UNSEEN_KINDS, n_steps, sizes=(20, 25, 30),
                                        tag="unseen", verbose=verbose),
    }


def class_balance(samples: Sequence[dict]) -> float:
    pos = sum(float((s["y_fail"] * s["m_fail"]).sum()) for s in samples)
    tot = sum(float(s["m_fail"].sum()) for s in samples)
    return pos / tot if tot else 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/dataset.pt")
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--val", type=int, default=12)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--unseen", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    sp = build_splits(a.train, a.val, a.test, a.unseen, a.steps)
    for k, v in sp.items():
        print(f"{k}: {len(v)} snapshots, positive rate {class_balance(v):.3%}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(sp, a.out)
