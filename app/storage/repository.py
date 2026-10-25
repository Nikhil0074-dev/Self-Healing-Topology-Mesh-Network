"""SQLite persistence for experiments, events, plans, metrics and model versions."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments(experiment_id TEXT PRIMARY KEY, config TEXT, seed INTEGER, status TEXT, created REAL);
CREATE TABLE IF NOT EXISTS experiment_metrics(experiment_id TEXT, metric_name TEXT, metric_value REAL);
CREATE TABLE IF NOT EXISTS recovery_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, step INTEGER, kind TEXT, entity TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS route_plans(plan_id TEXT PRIMARY KEY, experiment_id TEXT, flow_id TEXT, source TEXT, destination TEXT, candidate_path TEXT, status TEXT, validation_status TEXT, reason TEXT, step INTEGER);
CREATE TABLE IF NOT EXISTS model_versions(model_id INTEGER PRIMARY KEY AUTOINCREMENT, version TEXT, artifact_path TEXT, feature_schema TEXT, created REAL);
CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, actor TEXT, action TEXT, detail TEXT);
"""


class Repository:
    def __init__(self, path: str = ":memory:"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(route_plans)").fetchall()}
            if "plan_id" in columns:
                indexes = self._db.execute("PRAGMA index_list(route_plans)").fetchall()
                has_unique_plan_id = any(
                    index[2] and "plan_id" in {
                        row[2] for row in self._db.execute(f"PRAGMA index_info('{index[1]}')").fetchall()
                    }
                    for index in indexes
                )
                if not has_unique_plan_id:
                    self._db.executescript("""
                        ALTER TABLE route_plans RENAME TO route_plans_legacy;
                        CREATE TABLE route_plans(
                            plan_id TEXT PRIMARY KEY, experiment_id TEXT, flow_id TEXT,
                            source TEXT, destination TEXT, candidate_path TEXT, status TEXT,
                            validation_status TEXT, reason TEXT, step INTEGER
                        );
                        INSERT OR REPLACE INTO route_plans
                        SELECT plan_id, experiment_id, flow_id, source, destination,
                               candidate_path, status, validation_status, reason, step
                        FROM route_plans_legacy;
                        DROP TABLE route_plans_legacy;
                    """)
            self._db.commit()

    def _exec(self, sql: str, args: Iterable[Any] = ()) -> None:
        with self._lock:
            self._db.execute(sql, tuple(args))
            self._db.commit()

    def _all(self, sql: str, args: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(args)).fetchall()]

    def save_experiment(self, experiment_id: str, config: dict, seed: int, status: str = "created") -> None:
        self._exec("INSERT OR REPLACE INTO experiments VALUES(?,?,?,?,?)",
                   (experiment_id, json.dumps(config, default=str), seed, status, time.time()))

    def set_status(self, experiment_id: str, status: str) -> None:
        self._exec("UPDATE experiments SET status=? WHERE experiment_id=?", (status, experiment_id))

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        rows = self._all("SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,))
        if not rows:
            return None
        r = rows[0]
        r["config"] = json.loads(r["config"])
        return r

    def save_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        with self._lock:
            self._db.executemany("INSERT INTO experiment_metrics VALUES(?,?,?)",
                                 [(experiment_id, k, float(v)) for k, v in metrics.items()
                                  if isinstance(v, (int, float))])
            self._db.commit()

    def get_metrics(self, experiment_id: str) -> Dict[str, float]:
        return {r["metric_name"]: r["metric_value"] for r in
                self._all("SELECT metric_name, metric_value FROM experiment_metrics WHERE experiment_id=?",
                          (experiment_id,))}

    def save_events(self, experiment_id: str, events: Iterable[dict]) -> None:
        with self._lock:
            self._db.executemany("INSERT INTO recovery_events(experiment_id,step,kind,entity,detail) VALUES(?,?,?,?,?)",
                                 [(experiment_id, e["step"], e["kind"], e["entity"], json.dumps(e["detail"], default=str))
                                  for e in events])
            self._db.commit()

    def get_events(self, experiment_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._all("SELECT step,kind,entity,detail FROM recovery_events WHERE experiment_id=? "
                         "ORDER BY event_id DESC LIMIT ?", (experiment_id, limit))
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows[::-1]

    def save_plan(self, experiment_id: str, plan: dict) -> None:
        self._exec("""INSERT INTO route_plans
            (plan_id, experiment_id, flow_id, source, destination, candidate_path, status,
             validation_status, reason, step)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(plan_id) DO UPDATE SET
                experiment_id=excluded.experiment_id,
                flow_id=excluded.flow_id,
                source=excluded.source,
                destination=excluded.destination,
                candidate_path=excluded.candidate_path,
                status=excluded.status,
                validation_status=excluded.validation_status,
                reason=excluded.reason,
                step=excluded.step""",
                   (plan["plan_id"], experiment_id, plan["flow_id"], plan["source"], plan["destination"],
                    json.dumps(plan["candidate_path"]), plan["status"], plan["validation_status"],
                    plan["decision_reason"], plan["step"]))

    def register_model(self, version: str, artifact_path: str, feature_schema: str) -> None:
        self._exec("INSERT INTO model_versions(version,artifact_path,feature_schema,created) VALUES(?,?,?,?)",
                   (version, artifact_path, feature_schema, time.time()))

    def list_models(self) -> List[Dict[str, Any]]:
        return self._all("SELECT * FROM model_versions ORDER BY model_id")

    def audit(self, actor: str, action: str, detail: dict) -> None:
        self._exec("INSERT INTO audit_log(ts,actor,action,detail) VALUES(?,?,?,?)",
                   (time.time(), actor, action, json.dumps(detail, default=str)))

    def get_audit(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._all("SELECT ts,actor,action,detail FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
