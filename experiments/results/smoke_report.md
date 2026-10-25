# Experiment report

Runs: 1 | scenarios: 1 | methods: 1 | seeds per cell: 1

Values are mean ± 95% CI (normal approximation over seeds). Lower is better except pdr/availability.

## hazard

| method   | avg_latency_ms   | p95_latency_ms   | pdr         | availability   | recovery_time_detection   | route_churn   | control_msgs   | route_stretch   |
|:---------|:-----------------|:-----------------|:------------|:---------------|:--------------------------|:--------------|:---------------|:----------------|
| reactive | 162.432 ± nan    | 196.876 ± nan    | 0.887 ± nan | 0.967 ± nan    | 1.0 ± nan                 | 1.0 ± nan     | 25.0 ± nan     | 1.018 ± nan     |

## Paired comparisons (GNN-assisted vs heuristic-aware, matched seeds)

Negative difference = GNN lower. Primary metrics were fixed in advance; others are secondary.

| scope | metric | n | mean diff | 95% CI | p (bootstrap) |
|---|---|---|---|---|---|

Note: many comparisons are reported; p-values are not corrected for multiplicity. Treat only the pre-declared primary metrics as confirmatory.