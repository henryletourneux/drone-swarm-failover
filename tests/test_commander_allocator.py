"""Tests for dynamic platoon membership (command.py's reassign_platoon)
and the heuristic commander allocator that drives it
(commander_allocator.py) -- deciding who guards which under-covered
mission zone and who's on patrol, restructuring platoons to match.

Same fast/lossless timing convention as test_command.py so convergence
happens in a handful of ticks."""
from drone_swarm.command import CommandConfig
from drone_swarm.commander_allocator import CommanderAllocatorConfig, HeuristicCommanderAllocator
from drone_swarm.election import ElectionRole
from drone_swarm.mission import MissionConfig, Zone
from drone_swarm.model import Drone
from drone_swarm.swarm import Swarm, SwarmConfig

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


def _platoon_of(n, size):
    return {f"D{i}": f"P{i // size}" for i in range(n)}


def _make_swarm(n_drones, platoon_size, zones, reallocation_interval_ticks=10, positions=None):
    mission_config = MissionConfig(zones=zones, reallocation_interval_ticks=5)
    command_config = CommandConfig(platoon_of=_platoon_of(n_drones, platoon_size))
    allocator = HeuristicCommanderAllocator(CommanderAllocatorConfig(reallocation_interval_ticks=reallocation_interval_ticks))
    config = SwarmConfig(**{
        **FAST.__dict__, "mission_config": mission_config, "command_config": command_config,
        "commander_allocator": allocator,
    })
    if positions is None:
        positions = [(50 + 10 * i, 50) for i in range(n_drones)]
    drones = [Drone(id=f"D{i}", x=x, y=y, priority=50 + i) for i, (x, y) in enumerate(positions)]
    swarm = Swarm(drones, comm_range=2000, width=1000, height=1000, config=config, seed=1)
    return swarm


# -- reassign_platoon: the core dynamic-membership mechanism -------------------

def test_reassign_platoon_is_a_noop_when_already_in_that_platoon():
    swarm = _make_swarm(6, 3, zones=())
    d_id = "D0"
    original_election = swarm.elections[d_id]
    original_platoon = swarm.command.platoon_of[d_id]

    swarm.command.reassign_platoon(swarm, d_id, original_platoon)

    assert swarm.elections[d_id] is original_election  # untouched, no fresh election issued


def test_reassign_platoon_moves_drone_and_issues_a_fresh_term0_election():
    swarm = _make_swarm(6, 3, zones=())
    d_id = "D0"
    old_platoon = swarm.command.platoon_of[d_id]
    new_platoon = "P1" if old_platoon != "P1" else "P0"

    swarm.command.reassign_platoon(swarm, d_id, new_platoon)

    assert swarm.command.platoon_of[d_id] == new_platoon
    assert swarm.drones[d_id].platoon_id == new_platoon
    election = swarm.elections[d_id]
    assert election.term == 0
    assert election.role == ElectionRole.FOLLOWER
    assert election._layer == f"platoon:{new_platoon}"


def test_platoon_size_reflects_current_membership_not_the_original_config():
    swarm = _make_swarm(6, 3, zones=())  # P0: D0,D1,D2 -- P1: D3,D4,D5
    assert swarm.command.platoon_size("P0") == 3
    assert swarm.command.platoon_size("P1") == 3

    swarm.command.reassign_platoon(swarm, "D0", "P1")

    assert swarm.command.platoon_size("P0") == 2
    assert swarm.command.platoon_size("P1") == 4


# -- HeuristicCommanderAllocator: guard/patrol duty decisions ------------------

def test_allocate_assigns_nearest_drones_to_guard_an_undersupplied_zone():
    zone = Zone(id="Z", x=500, y=500, radius=20, required_drones=2)
    positions = [(490, 500), (0, 0), (1000, 1000), (510, 500)]  # D0, D3 are nearest to the zone
    swarm = _make_swarm(4, 2, zones=(zone,), positions=positions)

    allocator = swarm.config.commander_allocator
    swarm.command.commander_elections["D0"] = _fake_nexus_election()  # ensure maybe_allocate's gate passes
    allocator._allocate(swarm)

    guards = {d.id for d in swarm.drones.values() if d.duty == "guard"}
    assert guards == {"D0", "D3"}
    assert swarm.drones["D0"].mission_zone_id == "Z"
    assert swarm.drones["D3"].mission_zone_id == "Z"
    patrol = {d.id for d in swarm.drones.values() if d.duty == "patrol"}
    assert patrol == {"D1", "D2"}


def test_allocate_excludes_drones_currently_investigating_a_disturbance():
    zone = Zone(id="Z", x=500, y=500, radius=20, required_drones=1)
    positions = [(490, 500), (495, 500)]  # D0 nearer, but investigating
    swarm = _make_swarm(2, 1, zones=(zone,), positions=positions)
    swarm.drones["D0"].investigating_disturbance_id = "some-disturbance"

    allocator = swarm.config.commander_allocator
    allocator._allocate(swarm)

    assert swarm.drones["D0"].duty is None  # left alone entirely
    assert swarm.drones["D0"].mission_zone_id is None
    assert swarm.drones["D1"].duty == "guard"


def test_repeated_allocate_calls_do_not_evict_a_successful_guard():
    """Targeted regression for the real oscillation bug found while
    building this: a drone already successfully guarding a now-secured
    zone was originally still included in the reallocation pool. Since a
    secured zone no longer appears in guard_needs (precisely BECAUSE this
    drone is the one securing it), it would get reassigned to "patrol"
    and walk away -- un-securing the zone, which would then re-trigger
    guard duty next pass, forever. Independently verified by removing the
    mission_zone_id exclusion from _allocate's `available` filter and
    confirming this test fails before restoring it."""
    zone = Zone(id="Z", x=500, y=500, radius=20, required_drones=1)
    swarm = _make_swarm(2, 1, zones=(zone,), positions=[(490, 500), (0, 0)])
    allocator = swarm.config.commander_allocator
    allocator._allocate(swarm)
    assert swarm.drones["D0"].duty == "guard"
    assert swarm.drones["D0"].mission_zone_id == "Z"

    # Simulate what mission.py's own _occupants_of would compute once D0
    # physically arrives and secures the zone.
    swarm.mission.zone_statuses["Z"].occupant_ids = ["D0"]
    swarm.mission.zone_statuses["Z"].secured = True

    allocator._allocate(swarm)  # a second pass must not evict D0

    assert swarm.drones["D0"].duty == "guard"
    assert swarm.drones["D0"].mission_zone_id == "Z"


def test_allocate_spreads_patrol_duty_across_remaining_platoon_slots():
    swarm = _make_swarm(9, 3, zones=())  # no guard needs at all -- 3 slots, all patrol
    allocator = swarm.config.commander_allocator
    allocator._allocate(swarm)

    assert all(d.duty == "patrol" for d in swarm.drones.values())
    used_platoons = {d.platoon_id for d in swarm.drones.values()}
    assert used_platoons == {"P0", "P1", "P2"}  # spread across all 3 slots, not dumped in one


def _fake_nexus_election():
    from drone_swarm.election import NexusElection
    return NexusElection(nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8)


# -- maybe_allocate: cadence + commander-elected gating -------------------------

def test_maybe_allocate_respects_the_reallocation_interval():
    allocator = HeuristicCommanderAllocator(CommanderAllocatorConfig(reallocation_interval_ticks=5))
    swarm = _make_swarm(4, 2, zones=())
    swarm.command.commander_elections["D0"] = _fake_nexus_election()
    swarm.command.commander_elections["D0"].role = ElectionRole.NEXUS

    for _ in range(4):
        assert allocator.maybe_allocate(swarm) is False
    assert allocator.maybe_allocate(swarm) is True


def test_maybe_allocate_does_nothing_without_an_elected_commander():
    allocator = HeuristicCommanderAllocator(CommanderAllocatorConfig(reallocation_interval_ticks=1))
    swarm = _make_swarm(4, 2, zones=())
    # No commander_elections seeded at all -- commander_ids() is empty.

    assert allocator.maybe_allocate(swarm) is False
    assert all(d.duty is None for d in swarm.drones.values())


# -- End-to-end through Swarm.tick() --------------------------------------------

def test_end_to_end_zones_get_guarded_and_the_rest_go_on_patrol():
    zone_a = Zone(id="A", x=100, y=100, radius=30, required_drones=3)
    zone_b = Zone(id="B", x=800, y=800, radius=30, required_drones=2)
    swarm = _make_swarm(
        12, 3, zones=(zone_a, zone_b), reallocation_interval_ticks=15,
        positions=[(50 + 20 * i, 50 + 20 * i) for i in range(12)],
    )

    _tick(swarm, 400)  # converge platoon/commander elections, then let the allocator run and zones fill

    assert swarm.mission.zone_statuses["A"].secured is True
    assert swarm.mission.zone_statuses["B"].secured is True
    duties = {d.duty for d in swarm.drones.values()}
    assert "guard" in duties
    assert "patrol" in duties
