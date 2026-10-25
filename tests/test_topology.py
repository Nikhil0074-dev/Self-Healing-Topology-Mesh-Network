import networkx as nx
import pytest

from app.topology.connectivity import connected_components, is_reachable, isolated_nodes, partition_summary
from app.topology.graph_state import Health, TopologyState
from simulator.topology_generator import TOPOLOGY_KINDS, generate_topology
from simulator.traffic_generator import generate_flows


@pytest.mark.parametrize("kind", TOPOLOGY_KINDS)
def test_generated_topologies_are_connected_and_attributed(kind):
    g = generate_topology(kind, 20, seed=5)
    assert nx.is_connected(g)
    for _, _, d in g.edges(data=True):
        assert d["base_latency"] > 0 and d["bandwidth"] > 0 and 0 <= d["base_loss"] < 0.05


def test_generation_is_deterministic():
    a, b = generate_topology("small_world", 18, 9), generate_topology("small_world", 18, 9)
    assert sorted(a.edges(data=True)) == sorted(b.edges(data=True))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        generate_topology("nope", 10, 0)
    with pytest.raises(ValueError):
        generate_topology("grid", 3, 0)


def test_grid_preserves_requested_node_count():
    for n in (6, 7, 10, 14, 20):
        g = generate_topology("grid", n, seed=1)
        assert g.number_of_nodes() == n
        assert nx.is_connected(g)


    g = generate_topology("geometric", 20, 1)
    flows = generate_flows(g, 5, 1)
    assert len(flows) == 5 and all(f.src != f.dst for f in flows)


def test_components_and_partition():
    g = nx.Graph([("a", "b"), ("c", "d")])
    g.add_node("e")
    assert [len(c) for c in connected_components(g)] == [2, 2, 1]
    assert isolated_nodes(g) == ["e"]
    assert partition_summary(g)["n_components"] == 3
    assert is_reachable(g, "a", "b") and not is_reachable(g, "a", "c") and not is_reachable(g, "a", "zzz")


def test_state_graph_strict_vs_loose():
    g = generate_topology("geometric", 12, 2)
    st = TopologyState(g)
    k = st.link_list[0]
    st.links[k].status = Health.SUSPECTED
    assert not st.build_graph(strict=True).has_edge(*k)
    assert st.build_graph(strict=False).has_edge(*k)
    st.links[k].status = Health.DOWN
    assert not st.build_graph(strict=False).has_edge(*k)
