from drone_swarm.model import Drone
from drone_swarm.topology import build_adjacency, bfs_reachable, connected_components


def _as_dict(drones):
    return {d.id: d for d in drones}


def test_build_adjacency_respects_comm_range():
    # A-B are 50 apart (in range), B-C are 70 apart (out of range).
    drones = _as_dict([
        Drone(id="A", x=0, y=0, priority=1),
        Drone(id="B", x=50, y=0, priority=1),
        Drone(id="C", x=120, y=0, priority=1),
    ])
    adjacency = build_adjacency(drones, comm_range=60)

    assert adjacency["A"] == {"B"}
    assert adjacency["B"] == {"A"}
    assert adjacency["C"] == set()


def test_build_adjacency_ignores_dead_drones():
    # D sits right on top of A but is dead: it must not appear anywhere.
    drones = _as_dict([
        Drone(id="A", x=0, y=0, priority=1),
        Drone(id="B", x=50, y=0, priority=1),
        Drone(id="D", x=0, y=0, priority=1, alive=False),
    ])
    adjacency = build_adjacency(drones, comm_range=60)

    assert "D" not in adjacency
    assert "D" not in adjacency["A"]
    assert "D" not in adjacency["B"]
    assert adjacency["A"] == {"B"}


def test_connected_components_finds_disjoint_clusters():
    # Two tight clusters placed far apart.
    drones = _as_dict([
        Drone(id="A", x=0, y=0, priority=1),
        Drone(id="B", x=30, y=0, priority=1),
        Drone(id="X", x=500, y=0, priority=1),
        Drone(id="Y", x=530, y=0, priority=1),
    ])
    adjacency = build_adjacency(drones, comm_range=50)
    components = connected_components(adjacency)

    as_frozensets = {frozenset(c) for c in components}
    assert as_frozensets == {frozenset({"A", "B"}), frozenset({"X", "Y"})}


def test_bfs_reachable_returns_component_including_start():
    # Line graph A-B-C plus an isolated node D.
    adjacency = {
        "A": {"B"},
        "B": {"A", "C"},
        "C": {"B"},
        "D": set(),
    }
    assert bfs_reachable(adjacency, "A") == {"A", "B", "C"}
    assert bfs_reachable(adjacency, "D") == {"D"}


def test_bfs_reachable_unknown_start_is_empty():
    adjacency = {"A": {"B"}, "B": {"A"}}
    assert bfs_reachable(adjacency, "Z") == set()
