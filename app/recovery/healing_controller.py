"""Self-healing controller: detect -> plan -> validate -> apply -> verify -> roll back."""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from app.core.events import Event
from app.detection.failure_detector import FailureDetector
from app.detection.heartbeat import DetectorConfig
from app.monitoring.telemetry_collector import Observation
from app.recovery.recovery_state import ControllerState, RecoveryPlan, StateMachine
from app.recovery.rollback import RollbackManager
from app.routing.baseline_router import Proposal, ReactiveRouter, RoutingStrategy
from app.routing.route_validator import ValidationPolicy, validate_route
from app.topology.connectivity import connected_components, is_reachable
from app.topology.graph_state import BROKEN, Health, TopologyState, edge_key
from simulator.traffic_generator import Flow


@dataclass
class ChangePolicy:
    use_hysteresis: bool = True         # persistence + cooldown + change-rate limit
    use_min_improvement: bool = True
    min_improvement: float = 0.15       # required fractional cost reduction
    persistence_steps: int = 3
    cooldown_steps: int = 10
    max_changes_per_window: int = 4
    window_steps: int = 60
    max_attempts: int = 3


class HealingController:
    def __init__(self, graph: nx.Graph, flows: Sequence[Flow], strategy: RoutingStrategy, data_plane,
                 detector_cfg: Optional[DetectorConfig] = None, change_policy: Optional[ChangePolicy] = None,
                 validation: Optional[ValidationPolicy] = None, experiment_id: str = "exp"):
        self.flows: Dict[str, Flow] = {f.flow_id: f for f in flows}
        self.strategy = strategy
        self.fallback = ReactiveRouter()
        self.dp = data_plane
        self.state = TopologyState(graph)
        self.detector = FailureDetector(detector_cfg)
        self.policy = change_policy or ChangePolicy()
        self.validation = validation or ValidationPolicy()
        self.sm = StateMachine()
        self.rollback = RollbackManager()
        self.experiment_id = experiment_id
        self.routes: Dict[str, Optional[List[str]]] = {fid: None for fid in self.flows}
        self.flow_status: Dict[str, str] = {fid: "OK" for fid in self.flows}
        self.plans: Dict[str, RecoveryPlan] = {}
        self.events: List[Event] = []
        self.fault_detect_steps: Dict[str, List[int]] = defaultdict(list)
        self.t = 0
        self._broken_prev: Dict[str, bool] = {}
        self._last_change: Dict[str, int] = {}
        self._change_times: Dict[str, List[int]] = defaultdict(list)
        self._pending: Dict[str, Tuple[tuple, int]] = {}
        self._n_components = 1
        self._plan_ids = itertools.count(1)
        self.counters = {"route_changes": 0, "rollbacks": 0, "install_failures": 0, "verify_failures": 0,
                         "rejected": 0, "control_msgs": 0, "errors": 0, "safe_mode_steps": 0,
                         "link_down_declared": 0}

    # ------------------------------------------------------------------ helpers
    def _log(self, ev: Event) -> None:
        self.events.append(ev)

    def _emit(self, kind: str, entity: str = "", **detail) -> None:
        self._log(Event(self.t, kind, entity, detail))

    def route_broken(self, path: Optional[Sequence[str]]) -> bool:
        if not path:
            return True
        if any(self.state.nodes[n].status in BROKEN for n in path):
            return True
        for a, b in zip(path, path[1:]):
            ls = self.state.links.get(edge_key(a, b))
            if ls is None or ls.status in BROKEN:
                return True
        return False

    def _failed_elements(self, path: Optional[Sequence[str]]) -> List[str]:
        out: List[str] = []
        if not path:
            return out
        out += [n for n in path if self.state.nodes[n].status in BROKEN]
        for a, b in zip(path, path[1:]):
            ls = self.state.links.get(edge_key(a, b))
            if ls is not None and ls.status in BROKEN:
                out.append(f"{a}|{b}")
        return out

    def _transition(self, new: ControllerState, reason: str) -> None:
        ev = self.sm.transition(new, self.t, reason)
        if ev:
            self._log(ev)

    # ------------------------------------------------------------------ lifecycle
    def bootstrap(self) -> None:
        strict = self.state.build_graph(strict=True)
        for fid, flow in self.flows.items():
            path = self.strategy.initial_route(flow, strict)
            if path and self.dp.install_route(fid, path):
                self.routes[fid] = path
            else:
                self.routes[fid] = None
                self.flow_status[fid] = "NO_ROUTE"
        self._emit("bootstrap", flows=len(self.flows))

    def step(self, obs: Observation) -> None:
        self.t = t = obs.step
        for ev in self.detector.update(self.state, obs):
            if ev.kind == "link_down":
                self.counters["link_down_declared"] += 1
            self._log(ev)
        self.state.snapshot(t)
        strict = self.state.build_graph(strict=True)
        loose = self.state.build_graph(strict=False)
        self._track_partitions(loose)

        active: RoutingStrategy = self.strategy
        try:
            active.prepare(self.state, t)
        except Exception as exc:  # ML or scoring failure must never stop healing
            self.counters["errors"] += 1
            self._emit("strategy_error", type(exc).__name__, message=str(exc)[:120])
            active = self.fallback
            active.prepare(self.state, t)
        if active.fallback_active or active is self.fallback:
            self.counters["safe_mode_steps"] += 1

        for fid, flow in self.flows.items():
            try:
                self._handle_flow(flow, strict, loose, active)
            except Exception as exc:
                self.counters["errors"] += 1
                self._emit("controller_error", fid, type=type(exc).__name__, message=str(exc)[:120])
        self._update_global_state(loose, active)

    # ------------------------------------------------------------------ per-flow logic
    def _handle_flow(self, flow: Flow, strict: nx.Graph, loose: nx.Graph, active: RoutingStrategy) -> None:
        t, fid = self.t, flow.flow_id
        cur = self.routes[fid]
        broken = self.route_broken(cur)
        if broken and not self._broken_prev.get(fid, False):
            self.fault_detect_steps[fid].append(t)
            self._emit("route_broken", fid, failed=self._failed_elements(cur))
        self._broken_prev[fid] = broken

        if broken:
            if not is_reachable(loose, flow.src, flow.dst):
                self._mark_status(fid, "UNREACHABLE", "no path between endpoints in current topology")
                if cur is not None:
                    self.dp.install_route(fid, None)
                    self.routes[fid] = None
                return
            self._transition(ControllerState.RECOVERY_PLANNING, f"flow {fid}")
            proposal = self._safe_propose(active, flow, cur, strict, broken=True)
            if proposal is None:
                self._mark_status(fid, "NO_ROUTE", "reachable but no feasible validated route")
                if cur is not None:
                    self.dp.install_route(fid, None)
                    self.routes[fid] = None
                return
            ok = self._apply_candidates(flow, proposal.ranked[: self.policy.max_attempts], proposal.reason, forced=True)
            self._mark_status(fid, "OK" if ok else "NO_ROUTE", "recovery result")
            return

        self._mark_status(fid, "OK", "")
        if active.proactive:
            proposal = self._safe_propose(active, flow, cur, strict, broken=False)
            if proposal and self._should_switch(fid, proposal, cur):
                self._apply_candidates(flow, [(proposal.path, proposal.cost)], proposal.reason, forced=False)

    def _safe_propose(self, active, flow, cur, strict, broken) -> Optional[Proposal]:
        try:
            return active.propose(flow, cur, self.state, self.t, self.validation, broken, strict)
        except Exception as exc:
            self.counters["errors"] += 1
            self._emit("strategy_error", flow.flow_id, type=type(exc).__name__, message=str(exc)[:120])
            self.fallback.prepare(self.state, self.t)
            return self.fallback.propose(flow, cur, self.state, self.t, self.validation, broken, strict) if broken else None

    def _mark_status(self, fid: str, status: str, reason: str) -> None:
        if self.flow_status.get(fid) != status:
            self.flow_status[fid] = status
            if status in ("UNREACHABLE", "NO_ROUTE"):
                self._emit("no_feasible_route" if status == "NO_ROUTE" else "flow_unreachable", fid, reason=reason)
            else:
                self._emit("flow_status", fid, status=status)

    def _should_switch(self, fid: str, prop: Proposal, cur: Optional[List[str]]) -> bool:
        pol, t = self.policy, self.t
        if cur is None or prop.path == cur:
            self._pending.pop(fid, None)
            return False
        if prop.current_cost is not None:
            gain = prop.current_cost - prop.cost
            need = pol.min_improvement * prop.current_cost if pol.use_min_improvement else 1e-9
            if gain < need:
                self._pending.pop(fid, None)
                return False
        if pol.use_hysteresis:
            key = tuple(prop.path)
            prev = self._pending.get(fid)
            count = prev[1] + 1 if prev and prev[0] == key else 1
            self._pending[fid] = (key, count)
            if count < pol.persistence_steps:
                return False
            if t - self._last_change.get(fid, -10**9) < pol.cooldown_steps:
                return False
            recent = [x for x in self._change_times[fid] if t - x <= pol.window_steps]
            if len(recent) >= pol.max_changes_per_window:
                return False
        return True

    # ------------------------------------------------------------------ apply / verify / roll back
    def _apply_candidates(self, flow: Flow, candidates: List[Tuple[List[str], float]], reason: str,
                          forced: bool) -> bool:
        fid, t = flow.flow_id, self.t
        old = self.routes[fid]
        for path, cost in candidates:
            plan = self._new_plan(flow, path, cost, old, reason)
            res = validate_route(path, flow, self.state, t, self.validation, current=old, relax_capacity=forced)
            if not res.ok:
                plan.status, plan.validation_status = "rejected", "invalid"
                plan.decision_reason = "; ".join(res.reasons)
                self.counters["rejected"] += 1
                self._emit("route_rejected", fid, reasons=res.reasons)
                continue
            plan.validation_status = "valid"
            if self._execute_plan(flow, plan, forced):
                return True
        return False

    def _new_plan(self, flow: Flow, path, cost, old, reason) -> RecoveryPlan:
        plan = RecoveryPlan(f"plan_{next(self._plan_ids):05d}", flow.flow_id, flow.src, flow.dst,
                            list(path) if path else None, cost, list(old) if old else None,
                            self._failed_elements(old), decision_reason=reason, step=self.t)
        self.plans[plan.plan_id] = plan
        return plan

    def _execute_plan(self, flow: Flow, plan: RecoveryPlan, forced: bool) -> bool:
        fid, t, path, old = flow.flow_id, self.t, plan.candidate_path, self.routes[flow.flow_id]
        self._transition(ControllerState.RECOVERY_APPLYING, plan.plan_id)
        self.rollback.record(fid, old, t)
        self.counters["control_msgs"] += len(path)
        if not self.dp.install_route(fid, path):
            plan.status = "install_failed"
            self.counters["install_failures"] += 1
            self.rollback.pop(fid)
            self._emit("install_failed", fid, plan=plan.plan_id)
            return False
        self._transition(ControllerState.RECOVERY_VERIFYING, plan.plan_id)
        ok, lat = self.dp.probe_path(path)
        self.counters["control_msgs"] += 5 * (len(path) - 1)
        if ok:
            self.routes[fid] = list(path)
            plan.status = "applied"
            self._last_change[fid] = t
            self._change_times[fid].append(t)
            self._pending.pop(fid, None)
            self.counters["route_changes"] += 1
            self._emit("route_changed", fid, plan=plan.plan_id, old=old, new=path, forced=forced,
                       reason=plan.decision_reason, verify_latency=round(lat or 0.0, 2))
            return True
        plan.status = "verify_failed"
        self.counters["verify_failures"] += 1
        self.counters["rollbacks"] += 1
        self.rollback.pop(fid)
        if old is not None and not self.route_broken(old):
            self.dp.install_route(fid, old)
            self.routes[fid] = list(old)
        else:
            self.dp.install_route(fid, None)
            self.routes[fid] = None
        self._emit("route_rollback", fid, plan=plan.plan_id, restored=self.routes[fid])
        return False

    # ------------------------------------------------------------------ external API helpers
    def plan_flow(self, flow_id: str) -> RecoveryPlan:
        """Create (but do not apply) a plan from the controller's own graph - never client supplied."""
        flow = self.flows[flow_id]
        cur = self.routes[flow_id]
        broken = self.route_broken(cur)
        strict = self.state.build_graph(strict=True)
        self.strategy.prepare(self.state, self.t)
        prop = self._safe_propose(self.strategy, flow, cur, strict, broken)
        if prop is None:
            plan = self._new_plan(flow, None, None, cur, "no feasible route")
            plan.status, plan.validation_status = "no_feasible_route", "invalid"
            return plan
        plan = self._new_plan(flow, prop.path, prop.cost, cur, prop.reason)
        res = validate_route(prop.path, flow, self.state, self.t, self.validation, current=cur, relax_capacity=broken)
        plan.validation_status = "valid" if res.ok else "invalid"
        if not res.ok:
            plan.decision_reason += " | " + "; ".join(res.reasons)
        return plan

    def apply_plan(self, plan_id: str) -> RecoveryPlan:
        plan = self.plans[plan_id]
        if plan.status != "proposed" or not plan.candidate_path:
            return plan
        flow = self.flows[plan.flow_id]
        cur = self.routes[flow.flow_id]
        res = validate_route(plan.candidate_path, flow, self.state, self.t, self.validation, current=cur,
                             relax_capacity=self.route_broken(cur))
        if not res.ok:
            plan.status, plan.validation_status = "rejected", "invalid"
            plan.decision_reason += " | revalidation failed: " + "; ".join(res.reasons)
            return plan
        plan.validation_status = "valid"
        self._execute_plan(flow, plan, forced=self.route_broken(cur))
        return plan

    def rollback_flow(self, flow_id: str) -> bool:
        last = self.rollback.last(flow_id)
        if last is None:
            return False
        _, path = last
        if path is not None and self.route_broken(path):
            return False
        if not self.dp.install_route(flow_id, path):
            return False
        self.rollback.pop(flow_id)
        self.routes[flow_id] = list(path) if path else None
        self.counters["rollbacks"] += 1
        self._emit("manual_rollback", flow_id, restored=self.routes[flow_id])
        return True

    # ------------------------------------------------------------------ global state
    def _track_partitions(self, loose: nx.Graph) -> None:
        n = len(connected_components(loose))
        if n > self._n_components:
            self._emit("partition_detected", components=n)
        elif n < self._n_components:
            self._emit("partition_healed", components=n)
        self._n_components = n

    def _update_global_state(self, loose: nx.Graph, active: RoutingStrategy) -> None:
        st = self.state
        links = [l.status for l in st.links.values()]
        nodes = [n.status for n in st.nodes.values()]
        if any(s == "UNREACHABLE" for s in self.flow_status.values()):
            new, why = ControllerState.PARTITIONED, "flow endpoints in different components"
        elif active.fallback_active or active is self.fallback:
            new, why = ControllerState.SAFE_MODE, "ML component bypassed"
        elif Health.PROBATION in links or Health.PROBATION in nodes:
            new, why = ControllerState.REINTEGRATING, "probing restored elements"
        elif Health.SUSPECTED in links or Health.SUSPECTED in nodes:
            new, why = ControllerState.FAILURE_SUSPECTED, "awaiting corroboration"
        elif (Health.DEGRADED in links or Health.DOWN in links or Health.DOWN in nodes):
            new, why = ControllerState.DEGRADED, "degraded or failed elements present"
        else:
            new, why = ControllerState.HEALTHY, "all monitored elements healthy"
        self._transition(new, why)

    def summary(self) -> dict:
        return {"state": self.sm.state.value, "routes": {k: v for k, v in self.routes.items()},
                "flow_status": dict(self.flow_status), **self.counters}
