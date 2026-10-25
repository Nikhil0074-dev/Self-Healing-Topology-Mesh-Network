import math

from app.routing.baseline_router import ReactiveRouter, ScoringRouter, ShortestPathRouter, StaticRouter
from app.routing.candidate_paths import generate_candidate_paths
from app.routing.cost_function import CostWeights, EdgeEstimate, route_cost
from app.routing.route_validator import ValidationPolicy, validate_route
from app.topology.graph_state import Health, TopologyState, edge_key
from simulator.topology_generator import generate_topology
from simulator.traffic_generator import Flow


def setup():
    g = generate_topology("geometric", 20, 4)
    st = TopologyState(g)
    import networkx as nx
    nodes = sorted(g.nodes)
    src = nodes[0]
    dst = max(nodes, key=lambda n: nx.shortest_path_length(g, src, n))
    return g, st, Flow("f", src, dst, 5.0)


def test_candidates_are_simple_and_sorted():
    g, st, f = setup()
    paths = generate_candidate_paths(st.build_graph(), f.src, f.dst, 4, "base_latency")
    assert paths and all(len(set(p)) == len(p) for p in paths)
    cost = lambda p: sum(g.edges[a, b]["base_latency"] for a, b in zip(p, p[1:]))
    assert [cost(p) for p in paths] == sorted(cost(p) for p in paths)
    assert generate_candidate_paths(st.build_graph(), "zz", f.dst, 3) == []


def test_validator_rejects_bad_routes():
    g, st, f = setup()
    pol = ValidationPolicy()
    good = generate_candidate_paths(st.build_graph(), f.src, f.dst, 1, "base_latency")[0]
    assert validate_route(good, f, st, 0, pol).ok
    assert not validate_route(good + [good[0]], f, st, 0, pol).ok               # loop / endpoints
    assert not validate_route(good[:-1], f, st, 0, pol).ok                      # wrong destination
    assert not validate_route([f.src, f.dst], f, st, 0, pol).ok                 # no such link (not adjacent)
    k = edge_key(good[0], good[1])
    st.links[k].status = Health.DOWN
    assert not validate_route(good, f, st, 0, pol).ok
    st.links[k].status = Health.HEALTHY
    st.links[k].last_obs_step = 0
    assert not validate_route(good, f, st, 50, pol).ok                          # stale telemetry
    st.links[k].util = 0.99
    assert not validate_route(good, f, st, 0, pol).ok                           # over capacity
    assert validate_route(good, f, st, 0, pol, relax_capacity=True).ok
    assert not validate_route(good, f, st, 0, ValidationPolicy(forbidden_nodes={good[1]})).ok
    assert not validate_route(good, f, st, 0, pol, cost=math.inf).ok


def test_route_cost_prefers_reliable_path_even_if_longer():
    est = {edge_key("a", "b"): EdgeEstimate(10, 0.0, 0.1, 0.0), edge_key("b", "d"): EdgeEstimate(10, 0.0, 0.1, 0.6),
           edge_key("a", "c"): EdgeEstimate(15, 0.0, 0.1, 0.0), edge_key("c", "d"): EdgeEstimate(15, 0.0, 0.1, 0.0)}
    w = CostWeights()
    assert route_cost(["a", "c", "d"], est, w)[0] < route_cost(["a", "b", "d"], est, w)[0]
    assert math.isinf(route_cost(["a", "z"], est, w)[0])


def test_baselines_behaviour():
    g, st, f = setup()
    assert StaticRouter().propose(f, None, st, 0, ValidationPolicy(), True, st.build_graph()) is None
    cur = ShortestPathRouter().initial_route(f, st.build_graph())
    for R in (ShortestPathRouter, ReactiveRouter):
        r = R()
        r.prepare(st, 0)
        assert r.propose(f, cur, st, 0, ValidationPolicy(), False, st.build_graph()) is None   # route fine
    k = edge_key(cur[0], cur[1])
    st.links[k].status = Health.DOWN
    r = ReactiveRouter()
    p = r.propose(f, cur, st, 0, ValidationPolicy(), True, st.build_graph())
    assert p is not None and k not in {edge_key(a, b) for a, b in zip(p.path, p.path[1:])}


def test_scoring_router_avoids_risky_link():
    g, st, f = setup()
    r = ScoringRouter()
    r.prepare(st, 0)
    cur = r.propose(f, None, st, 0, ValidationPolicy(), True, st.build_graph()).path
    k = edge_key(cur[0], cur[1])
    st.links[k].ewma_loss = 0.4
    st.links[k].status = Health.DEGRADED
    r.prepare(st, 0)
    prop = r.propose(f, cur, st, 0, ValidationPolicy(), False, st.build_graph())
    assert prop is not None and prop.cost < prop.current_cost
    assert k not in {edge_key(a, b) for a, b in zip(prop.path, prop.path[1:])}
