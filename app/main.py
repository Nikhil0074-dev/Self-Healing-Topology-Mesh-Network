"""FastAPI management API for the self-healing mesh controller (simulator-backed).

Run:  uvicorn app.main:app --reload
State-changing endpoints require the header  X-API-Key  (env MESH_API_KEY).
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.experiments_manager import ExperimentManager
from app.core.config import Settings
from app.core.security import RateLimiter, check_api_key, valid_identifier
from app.routing.cost_function import route_cost
from app.storage.repository import Repository
from simulator.experiment_runner import STRATEGIES, RunConfig
from simulator.failure_injector import SCENARIO_NAMES
from simulator.network_state import edge_key
from simulator.topology_generator import TOPOLOGY_KINDS


class ExperimentRequest(BaseModel):
    topology_kind: str = "geometric"
    n_nodes: int = Field(20, ge=6, le=60)
    n_flows: int = Field(5, ge=1, le=20)
    seed: int = 0
    strategy: str = "heuristic"
    scenario: str = "hazard"
    steps: int = Field(200, ge=10, le=2000)
    warmup_steps: int = Field(5, ge=0, le=500)


class FailureRequest(BaseModel):
    experiment_id: Optional[str] = None
    failure_type: str
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    node: Optional[str] = None
    duration_seconds: Optional[int] = Field(None, ge=1, le=2000)


class EvaluateRequest(BaseModel):
    experiment_id: Optional[str] = None
    source: str
    destination: str
    k: int = Field(4, ge=1, le=10)


class PlanRequest(BaseModel):
    experiment_id: Optional[str] = None
    flow_id: str


class ApplyRequest(BaseModel):
    experiment_id: Optional[str] = None
    plan_id: str


class RollbackRequest(BaseModel):
    experiment_id: Optional[str] = None
    flow_id: str


class AdvanceRequest(BaseModel):
    steps: int = Field(10, ge=1, le=1000)


def create_app(settings: Optional[Settings] = None, predictor=None, repo: Optional[Repository] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if predictor is None and os.path.exists(settings.model_path):
        try:
            from gnn.inference import GNNPredictor
            predictor = GNNPredictor.load(settings.model_path)
        except Exception:  # an unusable model must never prevent the controller from starting
            predictor = None
    repo = repo or Repository(settings.db_path if settings.db_path == ":memory:" else _ensure_dir(settings.db_path))
    mgr = ExperimentManager(predictor, repo)
    limiter = RateLimiter(settings.rate_limit_per_min)
    app = FastAPI(title="Self-Healing Topology Mesh", version="1.0")
    app.state.manager, app.state.repo, app.state.settings = mgr, repo, settings

    def auth(request: Request, x_api_key: Optional[str] = Header(default=None)) -> str:
        if not check_api_key(x_api_key, settings.api_key):
            raise HTTPException(401, "invalid or missing API key")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(429, "rate limit exceeded")
        return client

    def exp_or_404(eid: Optional[str]):
        try:
            return mgr.get(eid)
        except KeyError:
            raise HTTPException(404, "experiment not found; create one with POST /api/v1/experiments")

    def check_ids(*ids: Optional[str]) -> None:
        for i in ids:
            if i is not None and not valid_identifier(i):
                raise HTTPException(422, f"invalid identifier {i!r}")

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "experiments": len(mgr.experiments), "model_loaded": predictor is not None,
                "model": getattr(predictor, "meta", {}).get("version") if predictor else None}

    @app.post("/api/v1/experiments", status_code=201)
    def create_experiment(req: ExperimentRequest, actor: str = Depends(auth)):
        if req.topology_kind not in TOPOLOGY_KINDS or req.strategy not in STRATEGIES \
                or req.scenario not in SCENARIO_NAMES:
            raise HTTPException(422, "unknown topology_kind, strategy or scenario")
        cfg = RunConfig(scenario=req.scenario, strategy=req.strategy, seed=req.seed, topo_kind=req.topology_kind,
                        n_nodes=req.n_nodes, n_flows=req.n_flows, n_steps=max(req.steps, 200))
        exp = mgr.create(cfg, req.warmup_steps)
        repo.audit(actor, "create_experiment", {"id": exp.experiment_id, **req.model_dump()})
        return {"experiment_id": exp.experiment_id, "scenario": exp.world.scenario.description, "step": exp.t}

    @app.post("/api/v1/experiments/{experiment_id}/advance")
    def advance(experiment_id: str, req: AdvanceRequest, actor: str = Depends(auth)):
        check_ids(experiment_id)
        exp_or_404(experiment_id)
        exp = mgr.advance(experiment_id, req.steps)
        repo.audit(actor, "advance", {"id": experiment_id, "steps": req.steps})
        return {"experiment_id": experiment_id, "step": exp.t, "state": exp.world.controller.sm.state.value}

    @app.get("/api/v1/experiments/{experiment_id}")
    def get_experiment(experiment_id: str):
        check_ids(experiment_id)
        exp = exp_or_404(experiment_id)
        ctl = exp.world.controller
        return {"experiment_id": experiment_id, "step": exp.t, "config": exp.world.cfg.__dict__,
                "scenario": exp.world.scenario.description, "controller": ctl.summary()}

    @app.get("/api/v1/topology")
    def topology(experiment_id: Optional[str] = None):
        exp = exp_or_404(experiment_id)
        ctl = exp.world.controller
        from app.topology.connectivity import connected_components
        comps = connected_components(ctl.state.build_graph(strict=False))
        return {"experiment_id": exp.experiment_id, "step": exp.t,
                "nodes": [{"id": n, "status": s.status.value} for n, s in ctl.state.nodes.items()],
                "links": [{"source": k[0], "target": k[1], "status": l.status.value} for k, l in ctl.state.links.items()],
                "components": [sorted(c) for c in comps]}

    @app.get("/api/v1/nodes")
    def nodes(experiment_id: Optional[str] = None):
        exp = exp_or_404(experiment_id)
        return [{"id": n, "status": s.status.value, "cpu": s.cpu, "mem": s.mem, "queue": s.queue, "load": s.load,
                 "last_seen": s.last_seen_step, "recent_failures": s.recent_failures(exp.t)}
                for n, s in exp.world.controller.state.nodes.items()]

    @app.get("/api/v1/links")
    def links(experiment_id: Optional[str] = None):
        exp = exp_or_404(experiment_id)
        return [{"source": k[0], "target": k[1], "status": l.status.value, "latency_ms": round(l.ewma_latency, 2),
                 "loss": round(l.ewma_loss, 4), "util": round(l.util, 3), "bandwidth_mbps": l.bandwidth,
                 "age": l.age(exp.t), "recent_failures": l.recent_failures(exp.t)}
                for k, l in exp.world.controller.state.links.items()]

    @app.post("/api/v1/failures", status_code=201)
    def inject_failure(req: FailureRequest, actor: str = Depends(auth)):
        check_ids(req.experiment_id, req.source_node, req.target_node, req.node)
        exp = exp_or_404(req.experiment_id)
        sim = exp.world.sim
        if req.failure_type in ("link_down", "link_up"):
            if not (req.source_node and req.target_node) or edge_key(req.source_node, req.target_node) not in sim.link:
                raise HTTPException(422, "unknown link")
            target = edge_key(req.source_node, req.target_node)
        elif req.failure_type in ("node_down", "node_up"):
            if not req.node or req.node not in sim.node:
                raise HTTPException(422, "unknown node")
            target = req.node
        else:
            raise HTTPException(422, "failure_type must be link_down, link_up, node_down or node_up")
        mgr.inject(req.experiment_id, req.failure_type, target, req.duration_seconds)
        repo.audit(actor, "inject_failure", req.model_dump())
        return {"status": "injected", "failure_type": req.failure_type, "target": target, "step": exp.t}

    @app.post("/api/v1/routes/evaluate")
    def evaluate_routes(req: EvaluateRequest):
        check_ids(req.experiment_id, req.source, req.destination)
        exp = exp_or_404(req.experiment_id)
        ctl = exp.world.controller
        if req.source not in ctl.state.nodes or req.destination not in ctl.state.nodes:
            raise HTTPException(422, "unknown source or destination")
        from app.routing.baseline_router import ScoringRouter
        from app.routing.candidate_paths import generate_candidate_paths
        router = ctl.strategy if isinstance(ctl.strategy, ScoringRouter) else ScoringRouter()
        router.prepare(ctl.state, exp.t)
        strict = ctl.state.build_graph(strict=True)
        from app.routing.cost_function import link_weight
        wf = lambda u, v, d: link_weight(router._est[edge_key(u, v)], router.weights)
        out = []
        for p in generate_candidate_paths(strict, req.source, req.destination, req.k, wf):
            c, parts = route_cost(p, router._est, router.weights)
            out.append({"path": p, "cost": c, "components": parts})
        out.sort(key=lambda x: x["cost"])
        return {"candidates": out, "status": "ok" if out else "no_feasible_route"}

    @app.post("/api/v1/healing/plan", status_code=201)
    def plan(req: PlanRequest, actor: str = Depends(auth)):
        check_ids(req.experiment_id, req.flow_id)
        exp = exp_or_404(req.experiment_id)
        if req.flow_id not in exp.world.controller.flows:
            raise HTTPException(422, "unknown flow")
        p = exp.world.controller.plan_flow(req.flow_id)
        repo.save_plan(exp.experiment_id, p.to_dict())
        return p.to_dict()

    @app.post("/api/v1/healing/apply")
    def apply(req: ApplyRequest, actor: str = Depends(auth)):
        check_ids(req.experiment_id, req.plan_id)
        exp = exp_or_404(req.experiment_id)
        if req.plan_id not in exp.world.controller.plans:
            raise HTTPException(404, "unknown plan")
        p = exp.world.controller.apply_plan(req.plan_id)
        repo.audit(actor, "apply_plan", {"plan": req.plan_id, "status": p.status})
        repo.save_plan(exp.experiment_id, p.to_dict())
        return p.to_dict()

    @app.post("/api/v1/healing/rollback")
    def rollback(req: RollbackRequest, actor: str = Depends(auth)):
        check_ids(req.experiment_id, req.flow_id)
        exp = exp_or_404(req.experiment_id)
        if req.flow_id not in exp.world.controller.flows:
            raise HTTPException(422, "unknown flow")
        ok = exp.world.controller.rollback_flow(req.flow_id)
        repo.audit(actor, "rollback", {"flow": req.flow_id, "ok": ok})
        if not ok:
            raise HTTPException(409, "nothing safe to roll back to")
        return {"status": "rolled_back", "route": exp.world.controller.routes[req.flow_id]}

    @app.get("/api/v1/metrics")
    def metrics(experiment_id: Optional[str] = None):
        exp = exp_or_404(experiment_id)
        m = mgr.finish_metrics(experiment_id)
        return {k: (None if isinstance(v, float) and v != v else v) for k, v in m.items()}

    @app.get("/api/v1/events")
    def events(experiment_id: Optional[str] = None, limit: int = 100):
        exp = exp_or_404(experiment_id)
        return [e.to_dict() for e in exp.world.controller.events[-max(1, min(limit, 1000)):]]

    return app


def _ensure_dir(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


app = create_app()
