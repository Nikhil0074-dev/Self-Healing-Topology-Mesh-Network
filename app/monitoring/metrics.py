"""Per-run network and recovery metrics."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class MetricsCollector:
    def __init__(self, flows, outage_confirm: int = 2):
        self.flows = {f.flow_id: f for f in flows}
        self.confirm = outage_confirm
        self.sent = self.delivered = self.late = self.expired = 0
        self.latencies: List[float] = []
        self.avail_steps = 0
        self.flow_steps = 0
        self.thr_sum = 0.0
        self.stretch: List[float] = []
        self._open: Dict[str, Optional[int]] = {fid: None for fid in self.flows}
        self._ok_run: Dict[str, int] = {fid: 0 for fid in self.flows}
        self.outages: List[Tuple[str, int, Optional[int]]] = []
        self.n_steps = 0

    def record(self, t: int, flow_steps: dict, sim) -> None:
        self.n_steps += 1
        for fid, fs in flow_steps.items():
            self.sent += fs.sent
            self.delivered += fs.delivered
            self.late += fs.late_delivered
            self.expired += fs.expired
            self.latencies.extend(fs.latencies)
            avail = fs.sent > 0 and fs.delivered >= 0.5 * fs.sent
            self.flow_steps += 1
            self.avail_steps += int(avail)
            self.thr_sum += self.flows[fid].demand_mbps * (fs.delivered / fs.sent if fs.sent else 0.0)
            if not avail:
                if self._open[fid] is None:
                    self._open[fid] = t
                self._ok_run[fid] = 0
            elif self._open[fid] is not None:
                self._ok_run[fid] += 1
                if self._ok_run[fid] >= self.confirm:
                    self.outages.append((fid, self._open[fid], t - self.confirm + 1))
                    self._open[fid], self._ok_run[fid] = None, 0
            if avail:
                path = sim.routes[fid]
                ref = sim.reference_latency(self.flows[fid].src, self.flows[fid].dst)
                cur = sim.route_latency_now(path) if path else None
                if ref and cur:
                    self.stretch.append(cur / ref)

    def finalize(self, controller, false_down: int, total_down: int, predictor=None) -> dict:
        occ, det, delay, transient = [], [], [], 0
        for fid, start, end in self.outages:
            ds = [x for x in controller.fault_detect_steps.get(fid, []) if start <= x <= end]
            if ds:
                occ.append(end - start)          # fault occurrence -> verified recovery
                det.append(end - ds[0])          # fault detection  -> verified recovery
                delay.append(ds[0] - start)      # detection delay
            else:
                transient += 1                   # quality loss that never broke the route
        unrecovered = sum(1 for v in self._open.values() if v is not None)
        lat = np.array(self.latencies, dtype=float)
        c = controller.counters
        nan = float("nan")
        return {
            "avg_latency_ms": float(lat.mean()) if lat.size else nan,
            "p95_latency_ms": float(np.percentile(lat, 95)) if lat.size else nan,
            "pdr": self.delivered / self.sent if self.sent else nan,
            "pdr_with_late": (self.delivered + self.late) / self.sent if self.sent else nan,
            "throughput_mbps": self.thr_sum / max(1, self.n_steps),
            "availability": self.avail_steps / self.flow_steps if self.flow_steps else nan,
            "recovery_time_occurrence": float(np.mean(occ)) if occ else nan,
            "recovery_time_detection": float(np.mean(det)) if det else nan,
            "detection_delay": float(np.mean(delay)) if delay else nan,
            "n_outages": len(self.outages) + unrecovered,
            "transient_outages": transient,
            "unrecovered_outages": unrecovered,
            "route_stretch": float(np.mean(self.stretch)) if self.stretch else nan,
            "route_churn": c["route_changes"],
            "control_msgs": c["control_msgs"],
            "rollbacks": c["rollbacks"],
            "controller_errors": c["errors"],
            "safe_mode_steps": c["safe_mode_steps"],
            "false_positive_rate": false_down / total_down if total_down else 0.0,
            "buffered_expired": self.expired,
            "inference_ms_total": predictor.total_ms if predictor is not None else 0.0,
            "inference_calls": predictor.calls if predictor is not None else 0,
        }
