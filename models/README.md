Trained checkpoints are written here by `python -m gnn.train` (e.g. `link_gnn_v1.pt`, plus `link_gnn_v1_metrics.json`).
A checkpoint stores weights, model config, Platt-calibration parameters, the feature-schema hash and evaluation metrics.
`GNNPredictor.load` refuses a checkpoint whose feature-schema hash differs from `gnn/features.py`.
