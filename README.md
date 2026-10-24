# Self-Healing Topology Mesh Network

GNN-assisted self-healing routing for a dynamic mesh, with a controlled detect → plan → validate → apply → verify → roll-back loop, partition handling, a simulator, a FastAPI service, a Streamlit dashboard and a reproducible evaluation. Everything runs on CPU; the network is **simulated** (NetworkX-based, discrete time), not packet-level.

## Quick start
```bash
pip install -r requirements.txt
python -m pytest tests -q                       # 51 tests
bash scripts/run_pipeline.sh                    # tests -> dataset -> train -> experiments -> report
# Windows PowerShell
.\scripts\run_pipeline.ps1
# or step by step
python -m gnn.dataset                           # generate labelled snapshots (episode-split)
python -m gnn.train                             # writes models/link_gnn_v1.pt (+ _metrics.json)
python -m simulator.experiment_runner --seeds 6 --out experiments/results/main.csv
python -m experiments.analyze experiments/results/main.csv
uvicorn app.main:app --reload                   # API   (header X-API-Key, default "dev-key")
streamlit run dashboard/dashboard.py            # dashboard
```
Without a trained model the GNN strategies run in **SAFE_MODE** (heuristic fallback) – nothing breaks.

## Architecture (matches the six layers of the proposal)
| Layer | Code |
|---|---|
| 1 Dashboard / API | `dashboard/dashboard.py`, `app/main.py` |
| 2 Telemetry | `app/monitoring/telemetry_collector.py` (typed records, seq/staleness/duplicate rejection) |
| 3 Topology & state | `app/topology/*`, `app/detection/*` (hysteresis, probation, flap hold-down) |
| 4 Routing engine | `app/routing/*`, `gnn/*` (GNN scores, k-shortest candidates, cost J(P)) |
| 5 Safe actuation | `app/recovery/*` (state machine, validation, hysteresis, cooldown, verify, rollback) |
| 6 Data plane | `simulator/network_state.py` (independent of the GNN) |

Key design points: the controller never reads simulator ground truth (only noisy probes); plans are computed
server-side and re-validated at apply time; failed install/verification triggers rollback or the next candidate;
partitioned flows are marked `UNREACHABLE` (never a fake route) and restored only after link probation succeeds;
telemetry traffic is buffered during a partition while control traffic expires.

## GNN
Pure-PyTorch GraphSAGE-style edge-aware network (no PyG needed, ~21k parameters). Predicts per link (a) failure
within 10 steps and (b) future latency. Features come only from controller-observed state; scaling constants are
fixed (nothing fitted on val/test); splits are by **episode**, with a test set of **unseen topology families**
(grid, scale-free). Probabilities are Platt-calibrated on validation data; the F1 threshold is chosen on validation.
The checkpoint stores a feature-schema hash and loading refuses a mismatch. Inference output is checked
(finite, bounded, fresh telemetry) and any failure falls back to conventional routing.

## Results (honest summary — see `experiments/results/report.md`)
**Model quality** (`models/link_gnn_v1_metrics.json`, ~4 % positive rate)

| split | AUC | baseline AUC (loss/history/util) | avg-precision | Brier | ECE |
|---|---|---|---|---|---|
| test (seen families) | 0.618 | 0.538 | 0.070 | 0.038 | 0.007 |
| test (unseen topologies) | ≈0.60 | ≈0.54 | ≈0.07 | – | – |

* `python scripts/oracle_ceiling.py`: a predictor that knows the simulator's **hidden** stress reaches only
  AUC ≈ 0.63. The GNN (0.62) is therefore essentially at the ceiling: failures in this simulator are mostly
  stochastic. Three hyper-parameter variants all plateaued at the same value, so it is not under-training.
* Latency prediction is **not better than persistence** (MAE 1.03 vs 1.03 ms): link latency barely changes
  over a 10-step horizon.

**System level** (6 seeds × 9 scenarios + unseen grid / scale-free topologies; paired bootstrap, matched seeds)
* Adaptive routing (reactive / shortest / heuristic / GNN) raises availability from ≈0.5 (static) to ≈0.9+.
* **GNN-assisted vs heuristic-aware: no evidence of improvement.** Primary metrics: p95 latency differs by
  +0.8 ms overall (GNN slightly *worse*, CI [+0.2, +1.5]); detection→recovery time difference −1.0 steps,
  CI [−2.6, +0.4] (not significant). H0 is **not rejected**; GNN is better in a few cells (e.g. scale-free
  p95 −1.9 ms) and worse in others – consistent with noise plus the weak predictive signal above.
* Proactive routers (heuristic/GNN) change routes ~2.5× more than reactive ones for no availability gain
  in this simulator. Their value would depend on failures being more predictable than here.
* **Ablation:** hysteresis + minimum-improvement threshold are what matter: removing hysteresis raises route
  churn from ≈22 to ≈124 (heuristic) and ≈20 to ≈58 (GNN) and control messages ~6×.
* Fallback test (`model_failure`): GNN unavailable/invalid for 110 steps → SAFE_MODE, 0 controller errors,
  availability equal to the non-ML baselines.
* GNN inference ≈ 1.7 ms per step on one CPU core.

## Limitations
Simulated physics, synthetic data, undirected links, single-controller view, recovery-time metric includes
flows whose endpoint node itself was down (they recover only when it returns), results from 6–8 seeds,
p-values not multiplicity-corrected. The GNN claim is therefore *"matches the achievable failure-ranking
ceiling but did not improve routing outcomes here"*, which is a legitimate negative result.

## Layout
`app/` controller, API · `gnn/` features/model/train/eval/inference · `simulator/` topology, traffic, failures,
data plane, experiment runner · `experiments/` analysis + results · `dashboard/` · `tests/` · `scripts/`
