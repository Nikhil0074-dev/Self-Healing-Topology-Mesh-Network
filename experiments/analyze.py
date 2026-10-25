"""Aggregate experiment results: mean +/- 95% CI per method and paired bootstrap tests.

Usage: python -m experiments.analyze experiments/results/results.csv --out experiments/results/report.md
"""
from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd

KEY_METRICS = ["avg_latency_ms", "p95_latency_ms", "pdr", "availability", "recovery_time_detection",
               "route_churn", "control_msgs", "route_stretch"]
PRIMARY = ["p95_latency_ms", "recovery_time_detection"]   # fixed before running experiments
LOWER_IS_BETTER = {"avg_latency_ms", "p95_latency_ms", "recovery_time_detection", "recovery_time_occurrence",
                   "route_churn", "control_msgs", "route_stretch"}


def ci95(x: pd.Series) -> float:
    x = x.dropna()
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def summarize(df: pd.DataFrame, metrics: List[str] = KEY_METRICS) -> pd.DataFrame:
    g = df.groupby(["scenario", "method"])
    mean, ci = g[metrics].mean(), g[metrics].agg(ci95)
    out = mean.round(3).astype(str) + " ± " + ci.round(3).astype(str)
    return out.reset_index()


def paired_bootstrap(df: pd.DataFrame, a: str, b: str, metric: str, scenario: str | None = None,
                     n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean difference (a - b) over matched (scenario, seed) pairs with a bootstrap 95% CI."""
    d = df if scenario is None else df[df.scenario == scenario]
    pa = d[d.method == a].set_index(["scenario", "seed"])[metric]
    pb = d[d.method == b].set_index(["scenario", "seed"])[metric]
    diff = (pa - pb).dropna().to_numpy()
    if len(diff) < 3:
        return {"n": len(diff), "mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan")}
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min(((boots <= 0).sum() + 1) / (n_boot + 1), ((boots >= 0).sum() + 1) / (n_boot + 1))
    return {"n": len(diff), "mean_diff": float(diff.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0))}


def build_report(df: pd.DataFrame) -> str:
    lines = ["# Experiment report", "",
             f"Runs: {len(df)} | scenarios: {df.scenario.nunique()} | methods: {df.method.nunique()} | "
             f"seeds per cell: {df.groupby(['scenario', 'method']).size().min()}", "",
             "Values are mean ± 95% CI (normal approximation over seeds). Lower is better except pdr/availability.", ""]
    for sc, sub in summarize(df).groupby("scenario"):
        lines += [f"## {sc}", "", sub.drop(columns="scenario").to_markdown(index=False), ""]
    lines += ["## Paired comparisons (GNN-assisted vs heuristic-aware, matched seeds)", "",
              "Negative difference = GNN lower. Primary metrics were fixed in advance; others are secondary.", "",
              "| scope | metric | n | mean diff | 95% CI | p (bootstrap) |", "|---|---|---|---|---|---|"]
    for metric in PRIMARY + ["availability", "route_churn", "pdr"]:
        for sc in [None] + sorted(df.scenario.unique()):
            r = paired_bootstrap(df, "gnn", "heuristic", metric, sc)
            if r["n"] < 3:
                continue
            tag = "PRIMARY " if metric in PRIMARY else ""
            lines.append(f"| {sc or 'all scenarios'} | {tag}{metric} | {r['n']} | {r['mean_diff']:+.3f} | "
                         f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.3f} |")
    lines += ["", "Note: many comparisons are reported; p-values are not corrected for multiplicity. "
              "Treat only the pre-declared primary metrics as confirmatory."]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="experiments/results/report.md")
    a = ap.parse_args()
    report = build_report(pd.read_csv(a.csv))
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report)
    print(report[:3000])
