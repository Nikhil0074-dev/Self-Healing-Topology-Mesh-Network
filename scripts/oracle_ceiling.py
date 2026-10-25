"""Upper bound on failure-risk ranking quality in this simulator.

Scores each link by its *true* current hazard (uses hidden simulator state that no controller can see)
and measures AUC against the real failures in the next ``horizon`` steps. A learned model cannot be
expected to beat this number, so it is the right yardstick for the GNN's failure-risk AUC.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from gnn.evaluate import roc_auc
from simulator.experiment_runner import RunConfig, run_experiment

H = 10
ys, ss = [], []
for seed in range(8):
    up, hz = [], []

    def hook(t, w):
        s, cfg = w.sim, w.sim.cfg
        up.append(np.array([s.link_up(e) for e in s.edges]))
        row = []
        for e in s.edges:
            p = cfg.base_hazard + cfg.max_hazard * min(s.link[e].s_eff, 1.0) ** 3
            if any(not s.link[x].up for x in s.adj_links[e]):
                p += cfg.cascade_hazard
            row.append(p * cfg.hazard_scale)
        hz.append(row)

    run_experiment(RunConfig(scenario="random_training", strategy="heuristic", seed=seed, n_nodes=20, n_flows=8,
                             n_steps=300, demand_range=(4, 18)), None, on_step=hook)
    up, hz = np.array(up), np.array(hz)
    for t in range(10, 290, 4):
        y = (~up[t + 1:t + 1 + H]).any(axis=0)
        ys.append(y[up[t]])
        ss.append(hz[t][up[t]])
y, s = np.concatenate(ys), np.concatenate(ss)
print(f"oracle-hazard AUC = {roc_auc(y, s):.3f}   (positive rate {y.mean():.3f})")
