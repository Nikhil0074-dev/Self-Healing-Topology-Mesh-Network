from app.core.security import RateLimiter, check_api_key, valid_identifier
from app.detection.failure_detector import FailureDetector
from app.detection.heartbeat import DetectorConfig
from app.monitoring.telemetry_collector import LinkProbe, NodeBeat, Observation, TelemetryCollector
from app.topology.graph_state import Health, TopologyState
from simulator.topology_generator import generate_topology


def make(g):
    st = TopologyState(g)
    return st, FailureDetector(DetectorConfig())


def obs(st, t, bad=(), loss=0.01, lat_mult=1.0, bad_nodes=()):
    o = Observation(t)
    for k, ls in st.links.items():
        if k in bad:
            o.link_probes[k] = LinkProbe(k, t, 10, 0, None, 1.0, 0, 0)
        else:
            o.link_probes[k] = LinkProbe(k, t, 10, 10 - int(loss * 10), ls.base_latency * lat_mult, loss, 0.2, 1.0)
    for n in st.nodes:
        o.node_beats[n] = NodeBeat(n, t, n not in bad_nodes, 0.1, 0.3, 0.0, 5.0)
    return o


def test_single_miss_only_suspects_then_recovers():
    g = generate_topology("geometric", 12, 1)
    st, det = make(g)
    k = st.link_list[0]
    det.update(st, obs(st, 1, bad=[k]))
    assert st.links[k].status == Health.SUSPECTED
    det.update(st, obs(st, 2))
    assert st.links[k].status == Health.HEALTHY


def test_three_misses_declare_down_and_probation_required():
    g = generate_topology("geometric", 12, 1)
    st, det = make(g)
    k = st.link_list[0]
    for t in (1, 2, 3):
        ev = det.update(st, obs(st, t, bad=[k]))
    assert st.links[k].status == Health.DOWN and any(e.kind == "link_down" for e in ev)
    kinds = []
    for t in range(4, 4 + 5):
        kinds += [e.kind for e in det.update(st, obs(st, t))]
        if t < 8:
            assert st.links[k].status == Health.PROBATION
    assert st.links[k].status == Health.HEALTHY and "link_up" in kinds


def test_probation_failure_returns_to_down():
    g = generate_topology("geometric", 12, 1)
    st, det = make(g)
    k = st.link_list[0]
    for t in (1, 2, 3):
        det.update(st, obs(st, t, bad=[k]))
    det.update(st, obs(st, 4))
    assert st.links[k].status == Health.PROBATION
    det.update(st, obs(st, 5, bad=[k]))
    assert st.links[k].status == Health.DOWN


def test_flapping_link_gets_longer_holddown():
    g = generate_topology("geometric", 12, 1)
    st, det = make(g)
    k = st.link_list[0]
    t = 0
    for _ in range(3):
        for _ in range(3):
            t += 1
            det.update(st, obs(st, t, bad=[k]))
        t += 1
        det.update(st, obs(st, t))
    assert st.links[k].probation_required == 15


def test_degradation_hysteresis():
    g = generate_topology("geometric", 12, 1)
    st, det = make(g)
    k = st.link_list[0]
    for t in range(1, 12):
        det.update(st, obs(st, t, loss=0.3))
    assert st.links[k].status == Health.DEGRADED
    for t in range(12, 40):
        det.update(st, obs(st, t, loss=0.09))   # between exit (0.06) and enter (0.12)
        if t < 16:
            continue
    # ewma of 0.09 stays above exit threshold -> remains degraded (no flapping)
    assert st.links[k].status == Health.DEGRADED
    for t in range(40, 70):
        det.update(st, obs(st, t, loss=0.0))
    assert st.links[k].status == Health.HEALTHY


def test_node_down_and_up():
    g = generate_topology("geometric", 12, 1)
    st, det = make(g)
    n = st.node_list[0]
    for t in (1, 2, 3):
        det.update(st, obs(st, t, bad_nodes=[n]))
    assert st.nodes[n].status == Health.DOWN
    for t in (4, 5, 6):
        det.update(st, obs(st, t))
    assert st.nodes[n].status == Health.HEALTHY


def test_collector_rejects_duplicates_stale_and_unknown():
    g = generate_topology("geometric", 8, 1)
    st = TopologyState(g)
    col = TelemetryCollector(st.node_list, st.link_list)
    k = st.link_list[0]
    p = LinkProbe(k, 10, 10, 9, 5.0, 0.1, 0.1, 1.0, seq=10)
    assert col.accept_link(p, 10)
    assert not col.accept_link(p, 10)                                   # duplicate
    assert not col.accept_link(LinkProbe(k, 1, 10, 9, 5.0, 0.1, 0.1, 1.0, seq=11), 20)   # too old
    assert not col.accept_link(LinkProbe(("x", "y"), 10, 10, 9, 5.0, 0.1, 0.1, 1.0, seq=1), 10)
    assert not col.accept_link(LinkProbe(k, 12, 10, 9, float("nan"), 0.1, 0.1, 1.0, seq=12), 12)
    assert col.rejected["stale"] == 2 and col.rejected["unknown"] == 1 and col.rejected["invalid"] == 1


def test_security_helpers():
    assert valid_identifier("node_01") and not valid_identifier("a b") and not valid_identifier("../x")
    assert check_api_key("k", "k") and not check_api_key("x", "k") and not check_api_key(None, "k")
    now = [0.0]
    rl = RateLimiter(2, 10.0, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")
    now[0] = 11.0
    assert rl.allow("a")
