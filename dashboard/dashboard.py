"""Streamlit dashboard:  streamlit run dashboard/dashboard.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from simulator.experiment_runner import STRATEGIES, RunConfig, run_experiment
from simulator.failure_injector import SCENARIO_NAMES
from simulator.topology_generator import TOPOLOGY_KINDS

MODEL_PATH = os.getenv("MESH_MODEL_PATH", "models/link_gnn_v1.pt")
COLORS = {"HEALTHY": "#2e9e5b", "DEGRADED": "#e0a030", "SUSPECTED": "#d6c23a", "DOWN": "#c0392b",
          "PROBATION": "#2f7fc1", "UNKNOWN": "#999999"}

st.set_page_config(page_title="Self-Healing Mesh", layout="wide")
st.title("Self-Healing Topology Mesh Network")


@st.cache_resource
def load_predictor():
    if os.path.exists(MODEL_PATH):
        try:
            from gnn.inference import GNNPredictor
            return GNNPredictor.load(MODEL_PATH)
        except Exception as exc:  # noqa
            return None
    return None


predictor = load_predictor()
with st.sidebar:
    st.header("Experiment")
    scenario = st.selectbox("Scenario", SCENARIO_NAMES, index=0)
    strategy = st.selectbox("Routing strategy", STRATEGIES, index=3)
    topo = st.selectbox("Topology", TOPOLOGY_KINDS, index=0)
    n_nodes = st.slider("Nodes", 10, 40, 20)
    n_flows = st.slider("Flows", 1, 8, 4)
    seed = st.number_input("Seed", 0, 100000, 1000)
    run = st.button("Run experiment", type="primary")
    st.caption("GNN model: " + (predictor.meta.get("version", "loaded") if predictor else "not found (GNN runs in fallback mode)"))


@st.cache_data(show_spinner=True)
def run_cached(scenario, strategy, topo, n_nodes, n_flows, seed):
    cfg = RunConfig(scenario=scenario, strategy=strategy, seed=int(seed), topo_kind=topo, n_nodes=n_nodes,
                    n_flows=n_flows, n_steps=200)
    res = run_experiment(cfg, predictor, record_frames=True, keep_world=True)
    pos = {n: d["pos"] for n, d in res.world.graph.nodes(data=True)}
    edges = list(res.world.graph.edges)
    flows = {f.flow_id: (f.src, f.dst) for f in res.world.flows}
    return res.summary, res.events, res.frames, pos, edges, flows, res.world.scenario.description


if run or "result" not in st.session_state:
    st.session_state["result"] = run_cached(scenario, strategy, topo, n_nodes, n_flows, seed)
summary, events, frames, pos, edges, flows, desc = st.session_state["result"]
st.info(desc)

t1, t2, t3, t4 = st.tabs(["Topology", "Recovery timeline", "Metrics", "Compare strategies"])

with t1:
    step = st.slider("Time step", 0, len(frames) - 1, min(70, len(frames) - 1))
    fr = frames[step]
    fig, ax = plt.subplots(figsize=(8, 6))
    g = nx.Graph()
    g.add_nodes_from(pos)
    for u, v in edges:
        g.add_edge(u, v)
    nx.draw_networkx_nodes(g, pos, node_size=220, node_color="#34495e", ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: n[-2:] for n in g.nodes}, font_color="white", font_size=7, ax=ax)
    for (u, v) in edges:
        key = (u, v) if u <= v else (v, u)
        status = fr["links"].get(key, "UNKNOWN")
        phys_down = key in [tuple(x) for x in fr["physical_down"]]
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v)], edge_color=COLORS.get(status, "#999"), width=1.5,
                               style="dashed" if phys_down else "solid", ax=ax)
    palette = ["#8e44ad", "#16a085", "#d35400", "#2c3e50", "#c0392b", "#7f8c8d", "#27ae60", "#f39c12"]
    for i, (fid, path) in enumerate(fr["routes"].items()):
        if path:
            nx.draw_networkx_edges(g, pos, edgelist=list(zip(path, path[1:])), edge_color=palette[i % 8],
                                   width=4, alpha=0.35, ax=ax)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Link colour = controller's belief (green healthy, amber degraded, red down, blue probation). "
               "Dashed = physically down in the simulator (ground truth). Wide translucent lines = active routes.")
    st.write(f"Controller state: **{fr['state']}**")
    st.table(pd.DataFrame([{"flow": f, "src": flows[f][0], "dst": flows[f][1],
                            "route": " → ".join(p) if p else "(none)"} for f, p in fr["routes"].items()]))

with t2:
    kinds = sorted({e["kind"] for e in events})
    pick = st.multiselect("Event kinds", kinds, default=[k for k in kinds if k in (
        "route_changed", "route_broken", "link_down", "link_up", "node_down", "node_up", "state_change",
        "partition_detected", "partition_healed", "flow_unreachable", "no_feasible_route", "route_rollback")])
    rows = [{"step": e["step"], "event": e["kind"], "entity": e["entity"],
             "detail": ", ".join(f"{k}={v}" for k, v in e["detail"].items())} for e in events if e["kind"] in pick]
    st.dataframe(pd.DataFrame(rows), width="stretch", height=450)

with t3:
    cols = st.columns(4)
    for c, (k, label) in zip(cols, [("avg_latency_ms", "Avg latency (ms)"), ("p95_latency_ms", "p95 latency (ms)"),
                                    ("pdr", "Packet delivery"), ("availability", "Availability")]):
        c.metric(label, f"{summary[k]:.3f}")
    cols = st.columns(4)
    for c, (k, label) in zip(cols, [("recovery_time_detection", "Recovery (detect→verified)"),
                                    ("route_churn", "Route changes"), ("control_msgs", "Control msgs"),
                                    ("false_positive_rate", "False-positive rate")]):
        c.metric(label, f"{summary[k]:.3f}")
    st.json({k: (None if isinstance(v, float) and v != v else v) for k, v in summary.items()})

with t4:
    st.write("Runs every strategy on the same topology, traffic and failure pattern (same seed).")
    if st.button("Compare all strategies"):
        rows = []
        for s in STRATEGIES:
            cfg = RunConfig(scenario=scenario, strategy=s, seed=int(seed), topo_kind=topo, n_nodes=n_nodes,
                            n_flows=n_flows, n_steps=200)
            r = run_experiment(cfg, predictor).summary
            rows.append({"strategy": s, **{k: r[k] for k in ("avg_latency_ms", "p95_latency_ms", "pdr", "availability",
                                                              "recovery_time_detection", "route_churn", "control_msgs",
                                                              "safe_mode_steps")}})
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch")
        st.caption("One seed is anecdotal; use `python -m simulator.experiment_runner` for multi-seed statistics.")
