"""Election-engine tests for the asynchronous message-passing rewrite.

Every test builds a Swarm from hand-placed Drone() instances and drives it
with an explicit *fast* SwarmConfig so convergence happens in a handful of
ticks (the default config's 4s nexus timeout would need ~10 ticks just to
notice a dead nexus). packet_loss_rate is 0.0 here so correctness tests are
deterministic; a lossy convergence test lives in test_swarm.py.
"""
from drone_swarm.election import ElectionRole
from drone_swarm.model import Drone
from drone_swarm.swarm import Swarm, SwarmConfig

# Fast, lossless timing: ~4 ticks to notice a dead nexus, ~2-tick election
# window, negligible latency. Generous tick budgets below all clear this.
FAST = SwarmConfig(
    nexus_heartbeat_interval_s=0.3,
    nexus_timeout_s=0.8,
    comm_latency_s=0.05,
    tick_dt_s=0.2,
    packet_loss_rate=0.0,
)


def _tick(swarm, n):
    for _ in range(n):
        swarm.tick()


def _clique(priorities, comm_range=500, seed=1):
    """A fully-connected swarm: every drone within range of every other, so
    killing any single drone never disconnects the rest."""
    drones = []
    for i, (drone_id, priority) in enumerate(priorities.items()):
        drones.append(Drone(id=drone_id, x=(i % 3) * 30, y=(i // 3) * 30, priority=priority))
    return Swarm(drones, comm_range=comm_range, config=FAST, seed=seed)


def _alive_ids(swarm):
    return {d.id for d in swarm.drones.values() if d.alive}


def _assert_no_dead_nexus(swarm):
    for drone in swarm.drones.values():
        if drone.nexus_id is not None:
            assert swarm.drones[drone.nexus_id].alive, (
                f"{drone.id} points at dead nexus {drone.nexus_id}"
            )


def _assert_nexus_role_self_consistent(swarm):
    """A drone whose election believes it is NEXUS must name itself."""
    for drone_id, election in swarm.elections.items():
        if election.role == ElectionRole.NEXUS:
            assert election.known_nexus_id == drone_id, (
                f"{drone_id} is NEXUS but known_nexus_id={election.known_nexus_id}"
            )


def test_fresh_swarm_converges_to_highest_priority():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)

    for drone in swarm.drones.values():
        assert drone.nexus_id == "D4"
    assert swarm.drones["D4"].role == "nexus"
    _assert_no_dead_nexus(swarm)
    _assert_nexus_role_self_consistent(swarm)


def test_reelection_after_nexus_killed():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    swarm.kill("D4")
    _tick(swarm, 25)

    for drone_id in _alive_ids(swarm):
        assert swarm.drones[drone_id].nexus_id == "D3"
    _assert_no_dead_nexus(swarm)
    _assert_nexus_role_self_consistent(swarm)


def test_cascading_handoff_across_three_nexuses():
    """The core feature: kill the nexus, re-elect, kill the new one too, and
    a third drone must take over."""
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    swarm.kill("D4")
    _tick(swarm, 25)
    assert all(swarm.drones[i].nexus_id == "D3" for i in _alive_ids(swarm))

    swarm.kill("D3")
    _tick(swarm, 25)
    for drone_id in _alive_ids(swarm):
        assert swarm.drones[drone_id].nexus_id == "D2"
    _assert_no_dead_nexus(swarm)
    _assert_nexus_role_self_consistent(swarm)


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
    swarm = Swarm(drones, comm_range=100, config=FAST, seed=3)
    _tick(swarm, 20)
    # Bridge has the highest priority, so it leads the whole mesh first.
    assert all(d.nexus_id == "M" for d in swarm.drones.values())

    swarm.kill("M")
    _tick(swarm, 30)

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
    _assert_nexus_role_self_consistent(swarm)


def test_reconnecting_partitions_merge_onto_one_nexus():
    """Regression test for the split-brain merge bug.

    Two clusters start far apart and each independently elects its own nexus
    at the SAME term (both boot from term 0). L1 is deliberately the
    higher-priority incumbent (95 vs 50) so that if the merge were resolved
    by priority, L1 would win. It is NOT: same-term contact is broken by
    sender_id (string compare), and "R1" > "L1", so R1 must win. This asserts
    the actual _ingest_heartbeats tie-break rule, not a priority contest.

    The deeper regression is stability: a stale/duplicate same-term
    ElectionMessage arriving after a term is resolved must not knock an
    already-elected nexus back into a fresh campaign. We verify terms do not
    creep upward over a long post-merge stretch.
    """
    left = [
        Drone(id="L0", x=0, y=0, priority=10),
        Drone(id="L1", x=20, y=0, priority=95),   # highest priority overall
        Drone(id="L2", x=0, y=20, priority=15),
    ]
    right = [
        Drone(id="R0", x=1000, y=0, priority=40),
        Drone(id="R1", x=1020, y=0, priority=50),  # highest on the right only
        Drone(id="R2", x=1000, y=20, priority=30),
    ]
    swarm = Swarm(left + right, comm_range=50, config=FAST, seed=2)

    left_ids = {"L0", "L1", "L2"}
    right_ids = {"R0", "R1", "R2"}

    # Let each side settle independently, then confirm it's genuinely stable
    # (not perpetually re-electing) by sampling the same nexus twice a few
    # ticks apart — the pre-fix bug caused endless cycling, not a wrong pick.
    _tick(swarm, 20)
    left_nexus_a = {swarm.drones[i].nexus_id for i in left_ids}
    right_nexus_a = {swarm.drones[i].nexus_id for i in right_ids}
    assert left_nexus_a == {"L1"}
    assert right_nexus_a == {"R1"}

    _tick(swarm, 5)
    assert {swarm.drones[i].nexus_id for i in left_ids} == {"L1"}
    assert {swarm.drones[i].nexus_id for i in right_ids} == {"R1"}

    # Both sides are on the same term (both booted from scratch).
    assert swarm.elections["L1"].term == swarm.elections["R1"].term

    # Slide the right cluster into range of the left one.
    for drone_id in right_ids:
        swarm.drones[drone_id].x -= 990

    _tick(swarm, 30)

    # Everyone converges onto the single lexicographically-greater id, R1.
    for drone in swarm.drones.values():
        assert drone.nexus_id == "R1", f"{drone.id} -> {drone.nexus_id}, expected R1"
    assert any(e["type"] == "swarms_merged" for e in swarm.event_log)
    _assert_no_dead_nexus(swarm)
    _assert_nexus_role_self_consistent(swarm)

    # Settled, not oscillating: terms must not increment over a long stretch.
    terms_before = (swarm.elections["L1"].term, swarm.elections["R1"].term)
    _tick(swarm, 30)
    terms_after = (swarm.elections["L1"].term, swarm.elections["R1"].term)
    assert terms_after == terms_before, (
        f"terms drifted after merge: {terms_before} -> {terms_after} "
        "(stale same-term election re-triggered a campaign)"
    )
    # And still everyone on R1 afterwards.
    assert all(d.nexus_id == "R1" for d in swarm.drones.values())
