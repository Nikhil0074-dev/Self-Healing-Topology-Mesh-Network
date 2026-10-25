import math

import pytest

from app.recovery.recovery_state import ControllerState
from simulator.experiment_runner import RunConfig, build_world, run_experiment, step_world
from simulator.network_state import LINK_DOWN, LINK_UP, NODE_DOWN, edge_key


def quiet_world(strategy="reactive", scenario="single_link_failure", seed=3, **kw):
    w = build_world(RunConfig(scenario=scenario, strategy=strategy, seed=seed, **kw))
    w.sim.cfg.hazard_scale = 0.0          # deterministic: only scripted faults
    w.sim.cfg.node_hazard_scale = 0.0
    w.sim.cfg.probe_blip_prob = 0.0
    w.scenario.events.clear()
    w.scenario._by_step.clear()
    return w


def route_links(path):
    return {edge_key(a, b) for a, b in zip(path, path[1:])}


def test_link_failure_is_healed_and_verified():
    w = quiet_world()
    for t in range(10):
        step_world(w, t)
    fid = "flow_0"
    old = list(w.controller.routes[fid])
    link = edge_key(old[1], old[2])
    w.sim.apply_event(LINK_DOWN, link)
    for t in range(10, 20):
        step_world(w, t)
    new = w.controller.routes[fid]
    assert new is not None and link not in route_links(new)
    assert w.sim.path_valid(new)
    assert any(e.kind == "route_changed" for e in w.controller.events)


def test_link_returns_only_after_probation():
    w = quiet_world()
    for t in range(5):
        step_world(w, t)
    link = edge_key(*w.controller.routes["flow_0"][:2])
    w.sim.apply_event(LINK_DOWN, link)
    for t in range(5, 12):
        step_world(w, t)
    assert w.controller.state.links[link].status.value == "DOWN"
    w.sim.apply_event(LINK_UP, link)
    step_world(w, 12)
    assert w.controller.state.links[link].status.value == "PROBATION"
    assert w.controller.sm.state == ControllerState.REINTEGRATING
    for t in range(13, 20):
        step_world(w, t)
    assert w.controller.state.links[link].status.value in ("HEALTHY", "DEGRADED")


def test_partition_reports_no_route_and_never_fakes_recovery():
    w = build_world(RunConfig(scenario="partition", strategy="reactive", seed=3))
    w.sim.cfg.hazard_scale = 0.0
    for t in range(100):
        step_world(w, t)
    ctl = w.controller
    f0 = w.flows[0]
    assert not any(f0.dst in c and f0.src in c for c in w.sim.physical_components())
    assert ctl.flow_status["flow_0"] == "UNREACHABLE"
    assert ctl.routes["flow_0"] is None
    assert ctl.sm.state == ControllerState.PARTITIONED
    assert any(e.kind == "partition_detected" for e in ctl.events)


def test_partition_recovery_restores_traffic():
    w = build_world(RunConfig(scenario="partition_recovery", strategy="reactive", seed=3, n_steps=200))
    w.sim.cfg.hazard_scale = 0.0
    for t in range(200):
        step_world(w, t)
    ctl = w.controller
    assert ctl.flow_status["flow_0"] == "OK" and ctl.routes["flow_0"] is not None
    assert w.sim.path_valid(ctl.routes["flow_0"])


def test_no_forwarding_loops_ever():
    for scenario in ("node_failure", "link_flapping", "hazard"):
        w = build_world(RunConfig(scenario=scenario, strategy="heuristic", seed=5, n_steps=200))
        for t in range(200):
            step_world(w, t)
            for p in w.controller.routes.values():
                assert p is None or len(set(p)) == len(p)


def test_install_failure_does_not_corrupt_state():
    w = quiet_world(install_fail_prob=1.0)
    w.sim.cfg.install_fail_prob = 0.0
    for t in range(5):
        step_world(w, t)
    old = list(w.controller.routes["flow_0"])
    w.sim.cfg.install_fail_prob = 1.0
    w.sim.apply_event(LINK_DOWN, edge_key(old[1], old[2]))
    for t in range(5, 15):
        step_world(w, t)
    assert w.controller.counters["install_failures"] >= 1
    assert w.controller.counters["errors"] == 0
    w.sim.cfg.install_fail_prob = 0.0
    for t in range(15, 25):
        step_world(w, t)
    assert w.controller.routes["flow_0"] is not None and w.sim.path_valid(w.controller.routes["flow_0"])


def test_verification_failure_rolls_back():
    w = quiet_world()
    for t in range(5):
        step_world(w, t)
    ctl = w.controller
    old = list(ctl.routes["flow_0"])
    w.sim.probe_path = lambda path: (False, None)       # data plane refuses every verification
    plan = ctl.plan_flow("flow_0")
    ctl.routes["flow_0"] = old
    res = ctl.apply_plan(plan.plan_id) if plan.candidate_path and plan.candidate_path != old else plan
    if plan.candidate_path and plan.candidate_path != old:
        assert res.status == "verify_failed" and ctl.routes["flow_0"] == old
        assert ctl.counters["rollbacks"] == 1


def test_manual_rollback_restores_previous_route():
    w = quiet_world()
    for t in range(5):
        step_world(w, t)
    ctl = w.controller
    old = list(ctl.routes["flow_0"])
    w.sim.apply_event(LINK_DOWN, edge_key(old[1], old[2]))
    for t in range(5, 12):
        step_world(w, t)
    assert ctl.routes["flow_0"] != old
    w.sim.apply_event(LINK_UP, edge_key(old[1], old[2]))
    for t in range(12, 25):
        step_world(w, t)
    # previous route (the original) is valid again, so rollback is allowed
    assert ctl.rollback_flow("flow_0") is True
    assert ctl.routes["flow_0"] == old


def test_plan_is_computed_server_side_and_revalidated():
    w = quiet_world(strategy="heuristic")
    for t in range(5):
        step_world(w, t)
    ctl = w.controller
    plan = ctl.plan_flow("flow_0")
    assert plan.status in ("proposed", "no_feasible_route") and plan.step == ctl.t
    if plan.candidate_path:
        k = edge_key(plan.candidate_path[0], plan.candidate_path[1])
        ctl.state.links[k].status = __import__("app.topology.graph_state", fromlist=["Health"]).Health.DOWN
        out = ctl.apply_plan(plan.plan_id)
        assert out.status == "rejected"      # state changed between plan and apply


def test_hysteresis_reduces_route_changes():
    churn = {}
    for hyst in (True, False):
        total = 0
        for seed in (1, 2, 3):
            r = run_experiment(RunConfig(scenario="hazard", strategy="heuristic", seed=seed, use_hysteresis=hyst,
                                         use_min_improvement=hyst))
            total += r.summary["route_churn"]
        churn[hyst] = total
    assert churn[True] <= churn[False]


def test_metrics_are_sane():
    r = run_experiment(RunConfig(scenario="single_link_failure", strategy="reactive", seed=2))
    s = r.summary
    assert 0 <= s["pdr"] <= 1 and 0 <= s["availability"] <= 1
    assert s["avg_latency_ms"] > 0 and s["p95_latency_ms"] >= s["avg_latency_ms"] * 0.5
    assert s["controller_errors"] == 0


def test_reactive_beats_static_on_availability():
    a = run_experiment(RunConfig(scenario="single_link_failure", strategy="static", seed=2)).summary
    b = run_experiment(RunConfig(scenario="single_link_failure", strategy="reactive", seed=2)).summary
    assert b["availability"] > a["availability"]
