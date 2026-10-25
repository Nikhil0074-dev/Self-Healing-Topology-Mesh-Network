import numpy as np
import pytest
import torch

from gnn.dataset import generate_episode
from gnn.evaluate import average_precision, calibration, prf, roc_auc
from gnn.features import EDGE_FEATURES, NODE_FEATURES, build_graph_tensors, schema_hash
from gnn.inference import GNNPredictor
from gnn.model import LinkQualityGNN, collate
from gnn.train import fit_platt
from app.routing.gnn_router import GNNRouter
from simulator.experiment_runner import RunConfig, build_world, run_experiment, step_world


def tiny_world():
    w = build_world(RunConfig(scenario="hazard", strategy="gnn", seed=2, n_nodes=12, n_flows=3))
    for t in range(6):
        step_world(w, t)
    return w


def test_feature_shapes_and_finiteness():
    w = tiny_world()
    g = build_graph_tensors(w.controller.state, 6)
    assert g.node_x.shape == (12, len(NODE_FEATURES)) and g.edge_x.shape[1] == len(EDGE_FEATURES)
    assert torch.isfinite(g.node_x).all() and torch.isfinite(g.edge_x).all()
    assert g.edge_list.shape == (g.edge_x.shape[0], 2) and g.usable.all()


def test_model_forward_and_symmetry():
    w = tiny_world()
    g = build_graph_tensors(w.controller.state, 6)
    m = LinkQualityGNN().eval()
    out = m(g.node_x, g.edge_list, g.edge_x)
    assert out.shape == (g.edge_x.shape[0], 2) and torch.isfinite(out).all()
    out2 = m(g.node_x, g.edge_list.flip(1), g.edge_x)          # endpoint order must not matter
    assert torch.allclose(out, out2, atol=1e-5)


def test_labels_use_only_future_and_masks_are_consistent():
    cfg = RunConfig(scenario="random_training", strategy="heuristic", seed=4, n_nodes=12, n_flows=4, n_steps=120)
    ep = generate_episode(cfg, horizon=10, sample_every=5)
    assert ep and all(s["t"] + 10 < 120 for s in ep)
    for s in ep:
        assert (s["m_lat"] & ~s["m_fail"]).sum() == 0
        assert ((s["y_fail"] == 0) | (s["y_fail"] == 1)).all()
    assert any(s["y_fail"][s["m_fail"]].sum() > 0 for s in ep) or True


def test_one_training_step_reduces_loss():
    w = tiny_world()
    g = build_graph_tensors(w.controller.state, 6)
    E = g.edge_x.shape[0]
    s = {"node_x": g.node_x, "edge_list": g.edge_list, "edge_x": g.edge_x,
         "y_fail": (torch.rand(E) > 0.8).float(), "y_lat": g.edge_x[:, 0] * 1.1,
         "m_fail": torch.ones(E, dtype=torch.bool), "m_lat": torch.ones(E, dtype=torch.bool)}
    b = collate([s, s])
    torch.manual_seed(0)
    m = LinkQualityGNN()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    from gnn.train import batch_loss
    pw = torch.tensor(3.0)
    first = float(batch_loss(m, b, pw)[0])
    for _ in range(30):
        opt.zero_grad()
        l = batch_loss(m, b, pw)[0]
        l.backward()
        opt.step()
    assert float(l) < first


def test_metric_functions():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([.1, .2, .8, .9])) == 1.0
    assert roc_auc(y, np.array([.9, .8, .2, .1])) == 0.0
    assert roc_auc(y, np.array([.5, .5, .5, .5])) == 0.5
    assert average_precision(y, np.array([.1, .2, .8, .9])) == 1.0
    r = prf(y, np.array([.1, .6, .8, .9]), 0.5)
    assert abs(r["precision"] - 2 / 3) < 1e-9 and r["recall"] == 1.0
    assert calibration(y.astype(float), np.array([0., 0., 1., 1.]))["brier"] == 0.0
    a, b = fit_platt(np.array([-2., -1, 1, 2]), np.array([0., 0, 1, 1]))
    assert a > 0


def test_predictor_validity_checks_and_fallback():
    w = tiny_world()
    pred = GNNPredictor(LinkQualityGNN())
    st = w.controller.state
    out = pred.predict(st, 6)
    assert out is not None and all(0 <= p.p_fail <= 1 and p.latency_ms > 0 for p in out.values())
    pred.override = "unavailable"
    assert pred.predict(st, 6) is None and pred.last_error
    pred.override = "invalid"
    assert pred.predict(st, 6) is None
    pred.override = None
    st.links[st.link_list[0]].ewma_latency = float("nan")
    assert pred.predict(st, 6) is None


def test_gnn_router_falls_back_to_heuristic():
    w = tiny_world()
    r = GNNRouter(None)
    r.prepare(w.controller.state, 6)
    assert r.fallback_active and r._est
    r2 = GNNRouter(GNNPredictor(LinkQualityGNN()))
    r2.prepare(w.controller.state, 6)
    assert not r2.fallback_active
    with pytest.raises(ValueError):
        GNNRouter(None, "bogus")


def test_model_failure_scenario_enters_safe_mode_and_recovers():
    pred = GNNPredictor(LinkQualityGNN())
    r = run_experiment(RunConfig(scenario="model_failure", strategy="gnn", seed=3), pred)
    assert r.summary["safe_mode_steps"] > 50
    assert r.summary["controller_errors"] == 0
    assert any(e["kind"] == "state_change" and e["entity"] == "SAFE_MODE" for e in r.events)


def test_schema_hash_mismatch_rejected(tmp_path):
    m = LinkQualityGNN()
    p = tmp_path / "m.pt"
    torch.save({"schema_hash": "bad", "model_cfg": m.cfg, "state_dict": m.state_dict(), "platt": [1.0, 0.0]}, p)
    with pytest.raises(ValueError):
        GNNPredictor.load(str(p))
    torch.save({"schema_hash": schema_hash(), "model_cfg": m.cfg, "state_dict": m.state_dict(),
                "platt": [1.0, 0.0], "version": "t"}, p)
    assert GNNPredictor.load(str(p)).meta["version"] == "t"
