"""Tests for patrol/disturbance investigation (patrol.py).

Covers: the module in isolation (spawn cadence/cap, spare-capacity-only
dispatch, resolved-disturbance pruning), and end-to-end through Swarm
(actual navigation to a disturbance, investigator death mid-investigation,
resolution freeing the drone back into the mission's reserve pool, and
patrol staying inert without a mission_config -- the same soft-dependency
shape mission_config/command_config already established for their own
inert-by-default behavior)."""
from __future__ import annotations

from types import SimpleNamespace

from drone_swarm.mission import MissionConfig, MissionState, Zone
from drone_swarm.model import Drone
from drone_swarm.patrol import Disturbance, PatrolConfig, PatrolState
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


class _FixedRng:
    """Deterministic stand-in for random.Random -- always returns the
    midpoint of the requested range, so spawn position is predictable in
    isolated PatrolState tests that don't go through Swarm's real rng."""

    def uniform(self, a, b):
        return (a + b) / 2.0


def _fake_swarm(drones: dict, mission: MissionState | None, width=300, height=300):
    return SimpleNamespace(drones=drones, mission=mission, width=width, height=height)


# -- Spawn cadence / cap ------------------------------------------------------

def test_disturbance_spawns_only_once_interval_elapses():
    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=5), _FixedRng())
    swarm = _fake_swarm({}, None)
    for t in range(1, 5):
        events = patrol.tick(swarm, t)
        assert events == []
    events = patrol.tick(swarm, 5)
    assert len(events) == 1
    assert events[0]["kind"] == "spawned"
    assert len(patrol.disturbances) == 1


def test_spawn_blocked_while_at_max_active_cap_then_resumes():
    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=1, max_active_disturbances=1), _FixedRng())
    swarm = _fake_swarm({}, None)
    patrol.tick(swarm, 1)
    assert len(patrol.disturbances) == 1
    first_id = next(iter(patrol.disturbances))

    # Interval keeps elapsing, but the cap holds firm since nothing resolves.
    for t in range(2, 6):
        patrol.tick(swarm, t)
    assert len(patrol.disturbances) == 1

    # Resolve the first, then the very next interval tick spawns a second.
    patrol.disturbances[first_id].resolved = True
    patrol.disturbances[first_id].resolved_at_tick = 6
    patrol.tick(swarm, 7)
    assert len(patrol.disturbances) == 2


# -- Dispatch: idle-reserve-pool rule -----------------------------------------

def test_dispatch_does_not_pull_a_zone_occupant_even_if_the_zone_is_oversupplied():
    zone = Zone(id="Z", x=0, y=0, radius=50, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))
    # Two drones physically inside a zone that only requires one -- this
    # WOULD have counted as "spare" under the old, buggy surplus-based
    # design (see module docstring). Under the real fix, occupying a zone
    # at all (regardless of surplus) excludes a drone from dispatch.
    drones = {
        "A": Drone(id="A", x=0, y=0, priority=1),
        "B": Drone(id="B", x=1, y=0, priority=1),
    }
    mission._occupants_of(drones)
    assert len(mission.zone_statuses["Z"].occupant_ids) == 2

    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=1), _FixedRng())
    swarm = _fake_swarm(drones, mission, width=100, height=100)
    patrol.tick(swarm, 1)

    disturbance = next(iter(patrol.disturbances.values()))
    assert disturbance.investigator_id is None
    assert set(mission.zone_statuses["Z"].occupant_ids) == {"A", "B"}  # untouched


def test_dispatch_pulls_from_the_idle_pool_not_zone_occupants():
    zone = Zone(id="Z", x=0, y=0, radius=50, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))
    drones = {
        "anchor": Drone(id="anchor", x=0, y=0, priority=1),  # occupies Z, not eligible
        "idle1": Drone(id="idle1", x=200, y=200, priority=1),  # outside Z, uncommitted
        "idle2": Drone(id="idle2", x=200, y=205, priority=1),
    }
    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].occupant_ids == ["anchor"]

    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=2), _FixedRng())
    swarm = _fake_swarm(drones, mission, width=1000, height=1000)
    patrol.disturbances["d0"] = Disturbance(id="d0", x=190, y=190, spawned_tick=0)
    patrol.disturbances["d1"] = Disturbance(id="d1", x=210, y=210, spawned_tick=0)

    patrol._dispatch(swarm)

    assigned_ids = {d.investigator_id for d in patrol.disturbances.values() if d.investigator_id}
    assert assigned_ids == {"idle1", "idle2"}  # both idle drones used, anchor untouched
    assert drones["anchor"].investigating_disturbance_id is None


def test_dispatch_picks_the_nearest_idle_drone():
    drones = {
        "near": Drone(id="near", x=10, y=0, priority=1),
        "far": Drone(id="far", x=40, y=0, priority=1),
    }
    mission = MissionState(MissionConfig(zones=()))
    mission._occupants_of(drones)

    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=1), _FixedRng())
    swarm = _fake_swarm(drones, mission, width=100, height=100)
    patrol.disturbances["d0"] = Disturbance(id="d0", x=12, y=0, spawned_tick=0)

    patrol._dispatch(swarm)

    assert patrol.disturbances["d0"].investigator_id == "near"


# -- Resolved-disturbance pruning ---------------------------------------------

def test_resolved_disturbance_pruned_only_after_display_window():
    patrol = PatrolState(PatrolConfig(resolved_display_ticks=5), _FixedRng())
    patrol.disturbances["d0"] = Disturbance(id="d0", x=0, y=0, spawned_tick=0, resolved=True, resolved_at_tick=10)

    patrol._prune_resolved(15)  # exactly at the boundary -- must NOT be pruned yet
    assert "d0" in patrol.disturbances

    patrol._prune_resolved(16)  # one tick past -- now it goes
    assert "d0" not in patrol.disturbances


# -- End-to-end through Swarm --------------------------------------------------

def test_investigation_requires_arrival_before_accruing_progress():
    zone = Zone(id="Z", x=50, y=50, radius=10, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=2)
    patrol_config = PatrolConfig(
        spawn_interval_ticks=2, max_active_disturbances=1,
        investigation_ticks_required=3, investigation_range=10, spawn_margin=200,
    )
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "patrol_config": patrol_config})
    # Wide, near-square arena so the fixed-midpoint-style rng spawn (actual
    # Random here, not _FixedRng) lands far from the anchor at (50, 50).
    drones = [
        Drone(id="anchor", x=50, y=50, priority=99),
        Drone(id="spare", x=60, y=60, priority=1),
    ]
    swarm = Swarm(drones, comm_range=500, width=400, height=400, config=config, seed=2)
    _tick(swarm, 4)  # let the zone secure and a disturbance spawn+dispatch

    investigator_id = next(
        (d.investigator_id for d in swarm.patrol.disturbances.values() if d.investigator_id), None,
    )
    assert investigator_id is not None
    disturbance = next(d for d in swarm.patrol.disturbances.values() if d.investigator_id == investigator_id)

    # Immediately after dispatch the investigator is still travelling --
    # spawn_margin=200 in a 400x400 arena guarantees real distance to cover.
    assert disturbance.ticks_investigated == 0
    assert disturbance.resolved is False

    _tick(swarm, 60)  # plenty of ticks to arrive and finish investigating
    assert disturbance.resolved is True


def test_investigator_death_returns_disturbance_to_market_and_clears_its_own_field():
    zone = Zone(id="Z", x=50, y=50, radius=10, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=2)
    patrol_config = PatrolConfig(
        spawn_interval_ticks=2, max_active_disturbances=1,
        investigation_ticks_required=20, investigation_range=10,
    )
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "patrol_config": patrol_config})
    drones = [
        Drone(id="anchor", x=50, y=50, priority=99),  # inside zone radius 10, secures it
        Drone(id="spare1", x=150, y=150, priority=1),  # well outside -- idle, dispatch-eligible
        Drone(id="spare2", x=145, y=145, priority=1),
    ]
    swarm = Swarm(drones, comm_range=500, width=200, height=200, config=config, seed=5)
    _tick(swarm, 4)

    disturbance = next(iter(swarm.patrol.disturbances.values()))
    investigator_id = disturbance.investigator_id
    assert investigator_id is not None

    swarm.kill(investigator_id)
    _tick(swarm, 1)

    assert swarm.drones[investigator_id].investigating_disturbance_id is None
    # The disturbance either already picked up the other spare drone this
    # same tick, or is waiting for one -- either way it must not still
    # think the dead drone is on the case.
    assert disturbance.investigator_id != investigator_id


def test_resolved_disturbance_frees_investigator_into_reserve_pool():
    zone_a = Zone(id="A", x=50, y=50, radius=10, required_drones=1)
    zone_b = Zone(id="B", x=350, y=350, radius=10, required_drones=1)
    mission_config = MissionConfig(zones=(zone_a, zone_b), reallocation_interval_ticks=2)
    patrol_config = PatrolConfig(
        spawn_interval_ticks=2, max_active_disturbances=1,
        investigation_ticks_required=3, investigation_range=10,
    )
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "patrol_config": patrol_config})
    drones = [
        Drone(id="anchor", x=50, y=50, priority=99),  # secures zone A, radius 10
        Drone(id="spare", x=200, y=200, priority=1),  # well outside both zones -- idle
    ]
    swarm = Swarm(drones, comm_range=500, width=400, height=400, config=config, seed=9)

    # Resolved disturbances are pruned from patrol.disturbances after only
    # resolved_display_ticks (8) -- purely a frontend display window -- so
    # this has to catch the moment of resolution rather than scanning at
    # the end of a long tick run, which would always find nothing pruned
    # away and (incorrectly) look like resolution never happens at all.
    resolved_investigator_id = None
    for _ in range(200):
        swarm.tick()
        just_resolved = [d for d in swarm.patrol.disturbances.values() if d.resolved]
        if just_resolved:
            resolved_investigator_id = just_resolved[0].investigator_id
            break

    assert resolved_investigator_id is not None, "expected at least one disturbance to resolve within 200 ticks"
    drone = swarm.drones[resolved_investigator_id]
    # Freed, not stuck -- available for mission.py's own allocator/
    # substitution machinery to pick up again, same as any other released
    # assignment in this codebase.
    assert drone.investigating_disturbance_id is None


def test_allocator_does_not_steal_a_drone_mid_investigation():
    """Targeted regression for the real cross-module bug found while
    building this: HeuristicAllocator.allocate() didn't originally check
    investigating_disturbance_id, so a drone dispatched to investigate
    could get assigned a mission_zone_id on the very next reallocation
    pass -- it would never actually travel there (movement priority
    favors the investigation), silently corrupting both. A weaker version
    of this test that only checked investigating_disturbance_id cleared
    on resolution passed even with the bug reintroduced (that field gets
    cleared by _advance() regardless of what allocate() does) -- this one
    checks mission_zone_id specifically, during the investigation, which
    is the part that actually breaks."""
    zone_a = Zone(id="A", x=50, y=50, radius=10, required_drones=1)
    zone_b = Zone(id="B", x=350, y=350, radius=10, required_drones=1)  # left unsecured on purpose
    mission_config = MissionConfig(zones=(zone_a, zone_b), reallocation_interval_ticks=2)
    patrol_config = PatrolConfig(
        spawn_interval_ticks=2, max_active_disturbances=1,
        investigation_ticks_required=50, investigation_range=10,  # slow, so investigation spans many reallocation passes
    )
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "patrol_config": patrol_config})
    drones = [
        Drone(id="anchor", x=50, y=50, priority=99),  # secures zone A
        Drone(id="spare", x=200, y=200, priority=1),  # the only drone left to either investigate or fill zone B
    ]
    swarm = Swarm(drones, comm_range=500, width=400, height=400, config=config, seed=9)

    dispatched = False
    for _ in range(40):
        swarm.tick()
        spare = swarm.drones["spare"]
        if spare.investigating_disturbance_id is not None:
            dispatched = True
            assert spare.mission_zone_id is None, (
                "allocator assigned a mission zone to a drone that is currently investigating a disturbance"
            )
    assert dispatched, "expected 'spare' to be dispatched to investigate within 40 ticks"


def test_patrol_inert_without_mission_config():
    patrol_config = PatrolConfig(spawn_interval_ticks=2, max_active_disturbances=2)
    config = SwarmConfig(**{**FAST.__dict__, "patrol_config": patrol_config})
    drones = [Drone(id=f"D{i}", x=10 * i, y=10 * i, priority=1) for i in range(4)]
    swarm = Swarm(drones, comm_range=500, width=200, height=200, config=config, seed=1)

    _tick(swarm, 30)  # would be plenty of time to dispatch/resolve if it could

    assert len(swarm.patrol.disturbances) > 0  # spawning still happens
    assert all(d.investigator_id is None for d in swarm.patrol.disturbances.values())
    assert all(d.resolved is False for d in swarm.patrol.disturbances.values())
    assert all(d.investigating_disturbance_id is None for d in swarm.drones.values())


# -- User-placed disturbances (add_disturbance) -------------------------------

def test_add_disturbance_bypasses_the_max_active_cap():
    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=1, max_active_disturbances=1), _FixedRng())
    swarm = _fake_swarm({}, None)
    patrol.tick(swarm, 1)
    assert len(patrol.disturbances) == 1  # cap reached via ordinary auto-spawn

    # A user-placed disturbance still goes through even though the cap is
    # already full -- it's a deliberate one-off action, not ambient spawn
    # traffic the cap is meant to throttle (see add_disturbance's
    # docstring).
    new_id = patrol.add_disturbance(12.0, 34.0, tick=2)
    assert len(patrol.disturbances) == 2
    placed = patrol.disturbances[new_id]
    assert (placed.x, placed.y) == (12.0, 34.0)
    assert placed.investigator_id is None


def test_add_disturbance_is_picked_up_by_the_next_dispatch():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))
    drones = {
        "anchor": Drone(id="anchor", x=0, y=0, priority=1),
        "idle": Drone(id="idle", x=100, y=100, priority=1),
    }
    mission._occupants_of(drones)

    patrol = PatrolState(PatrolConfig(spawn_interval_ticks=100), _FixedRng())
    swarm = _fake_swarm(drones, mission, width=200, height=200)
    new_id = patrol.add_disturbance(95.0, 95.0, tick=0)

    patrol._dispatch(swarm)

    assert patrol.disturbances[new_id].investigator_id == "idle"


def test_to_state_dict_includes_patrol_only_when_configured():
    plain = Swarm([Drone(id="A", x=0, y=0, priority=1)], comm_range=100, config=FAST, seed=1)
    assert "patrol" not in plain.to_state_dict()

    patrol_config = PatrolConfig(spawn_interval_ticks=100)
    config = SwarmConfig(**{**FAST.__dict__, "patrol_config": patrol_config})
    with_patrol = Swarm([Drone(id="A", x=0, y=0, priority=1)], comm_range=100, config=config, seed=1)
    state = with_patrol.to_state_dict()
    assert "patrol" in state
    assert "disturbances" in state["patrol"]
