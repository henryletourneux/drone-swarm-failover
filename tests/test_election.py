from drone_swarm.model import Drone
from drone_swarm.swarm import Swarm


def _tick(swarm, n):
    for _ in range(n):
        swarm.tick()


def _clique(priorities, spacing=30):
    """A fully-connected swarm: every drone within range of every other,
    so killing any single drone never disconnects the rest."""
    drones = []
    for i, prio in enumerate(priorities.items()):
        drone_id, priority = prio
        # Tight square-ish layout; all pairwise distances << comm_range.
        drones.append(Drone(id=drone_id, x=(i % 3) * spacing, y=(i // 3) * spacing, priority=priority))
    return Swarm(drones, comm_range=500)


def _alive_ids(swarm):
    return {d.id for d in swarm.drones.values() if d.alive}


def _assert_no_dead_nexus(swarm):
    for drone in swarm.drones.values():
        if drone.nexus_id is not None:
            assert swarm.drones[drone.nexus_id].alive, (
                f"{drone.id} points at dead nexus {drone.nexus_id}"
            )


def test_fresh_swarm_converges_to_highest_priority():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 10)

    for drone in swarm.drones.values():
        assert drone.nexus_id == "D4"
    assert swarm.drones["D4"].role == "nexus"
    _assert_no_dead_nexus(swarm)


def test_reelection_after_nexus_killed():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 10)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    swarm.kill("D4")
    _tick(swarm, 10)

    for drone_id in _alive_ids(swarm):
        assert swarm.drones[drone_id].nexus_id == "D3"
    _assert_no_dead_nexus(swarm)


def test_cascading_handoff_across_three_nexuses():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 10)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    swarm.kill("D4")
    _tick(swarm, 10)
    assert all(swarm.drones[i].nexus_id == "D3" for i in _alive_ids(swarm))

    swarm.kill("D3")
    _tick(swarm, 10)
    for drone_id in _alive_ids(swarm):
        assert swarm.drones[drone_id].nexus_id == "D2"
    _assert_no_dead_nexus(swarm)


def test_partition_yields_independent_nexuses():
    # A line A-B-M-C-D where only bridge M links the two halves.
    # comm_range=100 connects consecutive nodes but not across the gap.
    drones = [
        Drone(id="A", x=0, y=0, priority=10),
        Drone(id="B", x=0, y=50, priority=20),
        Drone(id="M", x=0, y=130, priority=99),
        Drone(id="C", x=0, y=210, priority=30),
        Drone(id="D", x=0, y=260, priority=40),
    ]
    swarm = Swarm(drones, comm_range=100)
    _tick(swarm, 10)
    # Bridge has the highest priority, so it leads the whole mesh first.
    assert all(d.nexus_id == "M" for d in swarm.drones.values())

    swarm.kill("M")
    _tick(swarm, 10)

    left = {"A", "B"}
    right = {"C", "D"}
    left_nexus = {swarm.drones[i].nexus_id for i in left}
    right_nexus = {swarm.drones[i].nexus_id for i in right}

    assert left_nexus == {"B"}   # highest priority in left half
    assert right_nexus == {"D"}  # highest priority in right half
    assert left_nexus.isdisjoint(right_nexus)
    # No drone points across the partition.
    for i in left:
        assert swarm.drones[i].nexus_id in left
    for i in right:
        assert swarm.drones[i].nexus_id in right
    _assert_no_dead_nexus(swarm)
