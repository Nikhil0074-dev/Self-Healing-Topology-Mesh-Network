import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.storage.repository import Repository

H = {"X-API-Key": "secret"}


@pytest.fixture()
def client():
    app = create_app(Settings(api_key="secret", db_path=":memory:", model_path="none.pt", rate_limit_per_min=1000))
    return TestClient(app)


def create(client, **kw):
    body = {"n_nodes": 14, "n_flows": 3, "seed": 2, "strategy": "heuristic", "scenario": "hazard", **kw}
    r = client.post("/api/v1/experiments", json=body, headers=H)
    assert r.status_code == 201, r.text
    return r.json()["experiment_id"]


def test_health_and_auth(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert client.post("/api/v1/experiments", json={}).status_code == 401
    assert client.post("/api/v1/experiments", json={}, headers={"X-API-Key": "bad"}).status_code == 401


def test_read_endpoints_before_experiment_404(client):
    assert client.get("/api/v1/topology").status_code == 404


def test_full_flow(client):
    eid = create(client)
    topo = client.get("/api/v1/topology").json()
    assert len(topo["nodes"]) == 14 and topo["components"]
    assert client.get("/api/v1/nodes").status_code == 200 and client.get("/api/v1/links").status_code == 200
    link = topo["links"][0]
    r = client.post("/api/v1/failures", headers=H, json={"experiment_id": eid, "failure_type": "link_down",
                    "source_node": link["source"], "target_node": link["target"], "duration_seconds": 5})
    assert r.status_code == 201
    adv = client.post(f"/api/v1/experiments/{eid}/advance", headers=H, json={"steps": 15})
    assert adv.status_code == 200 and adv.json()["step"] >= 20
    assert client.get(f"/api/v1/experiments/{eid}").json()["controller"]["routes"]
    assert any(e["kind"] == "link_down" for e in client.get("/api/v1/events?limit=500").json())
    m = client.get("/api/v1/metrics").json()
    assert "pdr" in m and "route_churn" in m
    flows = list(client.get(f"/api/v1/experiments/{eid}").json()["controller"]["routes"])
    p = client.post("/api/v1/healing/plan", headers=H, json={"experiment_id": eid, "flow_id": flows[0]})
    assert p.status_code == 201 and p.json()["status"] in ("proposed", "no_feasible_route")
    if p.json()["status"] == "proposed":
        a = client.post("/api/v1/healing/apply", headers=H, json={"experiment_id": eid, "plan_id": p.json()["plan_id"]})
        assert a.status_code == 200 and a.json()["status"] in ("applied", "rejected", "verify_failed", "install_failed")
    ev = client.post("/api/v1/routes/evaluate", json={"experiment_id": eid, "source": topo["nodes"][0]["id"],
                                                      "destination": topo["nodes"][-1]["id"], "k": 3})
    assert ev.status_code == 200 and ev.json()["status"] in ("ok", "no_feasible_route")


def test_validation_errors(client):
    eid = create(client)
    bad = client.post("/api/v1/failures", headers=H, json={"failure_type": "link_down", "source_node": "x y",
                                                           "target_node": "node_01"})
    assert bad.status_code == 422
    assert client.post("/api/v1/failures", headers=H, json={"failure_type": "explode"}).status_code == 422
    assert client.post("/api/v1/failures", headers=H, json={"failure_type": "link_down", "source_node": "node_00",
                                                            "target_node": "nope"}).status_code == 422
    assert client.post("/api/v1/experiments", headers=H, json={"strategy": "magic"}).status_code == 422
    assert client.post("/api/v1/healing/apply", headers=H, json={"plan_id": "plan_99999"}).status_code == 404
    assert client.post("/api/v1/healing/rollback", headers=H, json={"flow_id": "flow_0"}).status_code == 409


def test_timed_explicit_up_event_does_not_crash(client):
    eid = create(client, n_nodes=10, n_flows=1)
    topo = client.get("/api/v1/topology", params={"experiment_id": eid}).json()
    link = topo["links"][0]
    r = client.post("/api/v1/failures", headers=H, json={"experiment_id": eid, "failure_type": "link_up",
                    "source_node": link["source"], "target_node": link["target"], "duration_seconds": 5})
    assert r.status_code == 201, r.text


def test_settings_reject_non_positive_rate_limit():
    with pytest.raises(ValueError):
        Settings(rate_limit_per_min=0)
    with pytest.raises(ValueError):
        Settings(rate_limit_per_min=-1)
def test_audit_log_written(client):
    create(client)
    rows = client.app.state.repo.get_audit()
    assert rows and rows[0]["action"] == "create_experiment"


def test_repository_roundtrip():
    repo = Repository()
    repo.save_experiment("e1", {"a": 1}, 3)
    repo.save_metrics("e1", {"pdr": 0.9, "label": "x"})
    repo.save_events("e1", [{"step": 1, "kind": "k", "entity": "x", "detail": {"a": 1}}])
    repo.register_model("v1", "p", "h")
    assert repo.get_experiment("e1")["config"] == {"a": 1}
    assert repo.get_metrics("e1") == {"pdr": 0.9}
    assert repo.get_events("e1")[0]["detail"] == {"a": 1}
    assert repo.list_models()[0]["version"] == "v1"
    repo.save_plan("e1", {"plan_id": "p1", "flow_id": "f1", "source": "a", "destination": "b",
                         "candidate_path": ["a", "b"], "status": "proposed", "validation_status": "valid",
                         "decision_reason": "initial", "step": 1})
    repo.save_plan("e1", {"plan_id": "p1", "flow_id": "f1", "source": "a", "destination": "b",
                         "candidate_path": ["a", "c", "b"], "status": "applied", "validation_status": "valid",
                         "decision_reason": "updated", "step": 2})
    assert len(repo._all("SELECT * FROM route_plans WHERE plan_id=?", ("p1",))) == 1
    repo.set_status("e1", "done")
    assert repo.get_experiment("e1")["status"] == "done"
