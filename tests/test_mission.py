"""Tests for the zone-coverage resource-allocation system (mission.py).

Covers: the module in isolation (battery drain, zone security detection,
heuristic allocator scoring/assignment), and end-to-end through Swarm
(nexus-driven reallocation timing, actual navigation toward an assigned
zone -- not just the assignment decision, since a real bug during
development was drones being assigned but never actually moving).
"""
from drone_swarm.mission import (
    LOW_BATTERY_WARNING,
    MIN_BATTERY_TO_ASSIGN,
    HeuristicAllocator,
    MissionConfig,
    MissionState,
    Zone,
    ZoneStatus,
)
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


# -- MissionState: occupancy / security -------------------------------------

def test_zone_secured_once_required_drones_present():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=2)
    config = MissionConfig(zones=(zone,))
    mission = MissionState(config)

    drones = {
        "A": Drone(id="A", x=0, y=0, priority=1),
        "B": Drone(id="B", x=5, y=0, priority=1),
        "C": Drone(id="C", x=100, y=100, priority=1),  # far away, not in zone
    }
    mission._occupants_of(drones)
    status = mission.zone_statuses["Z"]
    assert set(status.occupant_ids) == {"A", "B"}
    assert status.secured is True


def test_zone_not_secured_below_requirement():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=3)
    mission = MissionState(MissionConfig(zones=(zone,)))
    drones = {"A": Drone(id="A", x=0, y=0, priority=1), "B": Drone(id="B", x=1, y=0, priority=1)}
    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].secured is False


def test_dead_drones_excluded_from_occupancy():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))
    drones = {"A": Drone(id="A", x=0, y=0, priority=1, alive=False)}
    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].occupant_ids == []
    assert mission.zone_statuses["Z"].secured is False


# -- MissionState: battery drain ---------------------------------------------

def test_battery_drains_from_base_rate_and_movement():
    zone = Zone(id="Z", x=1000, y=1000, radius=10, required_drones=99)  # far away, never occupied
    config = MissionConfig(zones=(zone,), base_drain_per_tick=1.0, move_drain_per_unit=0.5)
    mission = MissionState(config)
    drone = Drone(id="A", x=0, y=0, priority=1, battery=100.0)
    drones = {"A": drone}

    mission._drain_batteries(drones)  # no movement yet (first call establishes baseline position)
    assert drone.battery == 99.0  # only base drain

    drone.x += 4.0  # simulate 4 units of movement between calls
    mission._drain_batteries(drones)
    assert drone.battery == 99.0 - 1.0 - (4.0 * 0.5)  # base + movement drain


def test_battery_drains_faster_in_undersupported_contested_zone():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=5, threat_level=2.0)  # can't be secured by 1 drone
    config = MissionConfig(zones=(zone,), base_drain_per_tick=0.1, contested_drain_per_tick=1.0)
    mission = MissionState(config)
    drone = Drone(id="A", x=0, y=0, priority=1, battery=100.0)
    drones = {"A": drone}

    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].secured is False
    mission._drain_batteries(drones)

    expected_drain = 0.1 + (1.0 * 2.0)  # base + contested*threat_level
    assert drone.battery == 100.0 - expected_drain


def test_secured_occupancy_drain_defaults_to_zero_extra_cost():
    """Regression test for a real bug found via a live-scenario smoke
    test: an earlier attempt to make substitution demo-observable bumped
    base_drain_per_tick instead (which applies to every drone forever,
    with no recharge -- it nearly drained the entire live-demo fleet to
    zero within a few real minutes regardless of what any drone was
    doing, and zero zones ever finished securing). This field exists
    specifically so that lever only touches drones actively holding an
    already-secured zone, and it must default to inert."""
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,), base_drain_per_tick=0.0))
    drone = Drone(id="A", x=0, y=0, priority=1, battery=100.0)
    drones = {"A": drone}

    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].secured is True
    mission._drain_batteries(drones)
    assert drone.battery == 100.0  # secured, and secured_occupancy_drain_per_tick defaults to 0.0


def test_secured_occupancy_drain_applies_only_to_secured_occupants():
    secured_zone = Zone(id="S", x=0, y=0, radius=10, required_drones=1)
    unsecured_zone = Zone(id="U", x=1000, y=1000, radius=10, required_drones=5)  # can't be secured by 1 drone
    config = MissionConfig(
        zones=(secured_zone, unsecured_zone),
        base_drain_per_tick=0.0,
        contested_drain_per_tick=0.0,
        secured_occupancy_drain_per_tick=0.2,
    )
    mission = MissionState(config)
    secured_occupant = Drone(id="A", x=0, y=0, priority=1, battery=100.0)
    unsecured_occupant = Drone(id="B", x=1000, y=1000, priority=1, battery=100.0)
    idle_drone = Drone(id="C", x=500, y=500, priority=1, battery=100.0)
    drones = {"A": secured_occupant, "B": unsecured_occupant, "C": idle_drone}

    mission._occupants_of(drones)
    assert mission.zone_statuses["S"].secured is True
    assert mission.zone_statuses["U"].secured is False
    mission._drain_batteries(drones)

    assert secured_occupant.battery == 99.8  # 100 - secured_occupancy_drain_per_tick
    assert unsecured_occupant.battery == 100.0  # unsecured -- contested drain, not secured drain, and that's 0 here
    assert idle_drone.battery == 100.0  # not an occupant of anything


def test_battery_floor_at_zero():
    zone = Zone(id="Z", x=1000, y=1000, radius=10, required_drones=99)
    mission = MissionState(MissionConfig(zones=(zone,), base_drain_per_tick=50.0))
    drone = Drone(id="A", x=0, y=0, priority=1, battery=10.0)
    drones = {"A": drone}
    mission._drain_batteries(drones)
    assert drone.battery == 0.0  # clamped, not negative


# -- HeuristicAllocator -------------------------------------------------------

def test_allocator_prefers_higher_battery_and_closer_drone():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    allocator = HeuristicAllocator()

    near_full_battery = Drone(id="near_full", x=1, y=0, priority=1, battery=100.0)
    far_low_battery = Drone(id="far_low", x=500, y=0, priority=1, battery=10.0)
    drones = {"near_full": near_full_battery, "far_low": far_low_battery}

    assignments = allocator.allocate(drones, [status], arena_diagonal=1000.0)
    assert assignments == {"near_full": "Z"}


def test_allocator_discourages_pulling_a_relay_off_post():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    allocator = HeuristicAllocator()

    # Identical position/battery, only role differs.
    relay = Drone(id="relay", x=10, y=0, priority=1, battery=100.0, role="relay")
    leaf = Drone(id="leaf", x=10, y=0, priority=1, battery=100.0, role="leaf")
    drones = {"relay": relay, "leaf": leaf}

    assignments = allocator.allocate(drones, [status], arena_diagonal=1000.0)
    assert assignments == {"leaf": "Z"}


def test_allocator_ignores_low_battery_and_already_committed_drones():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=2)
    status = ZoneStatus(zone=zone)
    allocator = HeuristicAllocator()

    low_battery = Drone(id="low", x=1, y=0, priority=1, battery=5.0)  # below MIN_BATTERY_TO_ASSIGN
    committed = Drone(id="committed", x=1, y=0, priority=1, battery=100.0)
    available = Drone(id="available", x=1, y=0, priority=1, battery=100.0)
    drones = {"low": low_battery, "committed": committed, "available": available}

    other_zone_status = ZoneStatus(zone=Zone(id="Z2", x=999, y=999, radius=5, required_drones=1))
    other_zone_status.occupant_ids = ["committed"]  # already holding a different zone

    assignments = allocator.allocate(drones, [status, other_zone_status], arena_diagonal=1000.0)
    assert assignments == {"available": "Z"}


def test_allocator_prioritizes_most_threatened_zone_first():
    low_threat = ZoneStatus(zone=Zone(id="low", x=0, y=0, radius=10, required_drones=1, threat_level=0.0))
    high_threat = ZoneStatus(zone=Zone(id="high", x=0, y=0, radius=10, required_drones=1, threat_level=5.0))
    allocator = HeuristicAllocator()

    only_drone = Drone(id="A", x=0, y=0, priority=1, battery=100.0)
    assignments = allocator.allocate({"A": only_drone}, [low_threat, high_threat], arena_diagonal=1000.0)
    assert assignments == {"A": "high"}


def test_allocator_stops_when_no_eligible_drones_remain():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=5)
    status = ZoneStatus(zone=zone)
    allocator = HeuristicAllocator()
    drones = {"A": Drone(id="A", x=0, y=0, priority=1, battery=100.0)}
    assignments = allocator.allocate(drones, [status], arena_diagonal=1000.0)
    assert assignments == {"A": "Z"}  # only 1 of the 5 required got assigned, no crash


# -- End to end through Swarm -------------------------------------------------

def _mission_swarm(zones, n=6, reallocation_interval_ticks=3, seed=1):
    config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0,
        mission_config=MissionConfig(zones=zones, reallocation_interval_ticks=reallocation_interval_ticks),
    )
    drones = [Drone(id=f"D{i}", x=(i % 3) * 30, y=(i // 3) * 30, priority=10 + i) for i in range(n)]
    return Swarm(drones, comm_range=500, config=config, seed=seed)


def test_mission_is_none_when_not_configured():
    drones = [Drone(id="D0", x=0, y=0, priority=1)]
    swarm = Swarm(drones, comm_range=100)
    assert swarm.mission is None
    swarm.tick()  # must not crash without a mission_config
    assert swarm.mission is None


def test_reallocation_only_happens_after_a_nexus_is_elected():
    zone = Zone(id="Z", x=500, y=500, radius=50, required_drones=1)
    swarm = _mission_swarm((zone,))
    # Before any nexus exists, no drone should have a mission assignment yet.
    swarm.tick()
    assert all(d.mission_zone_id is None for d in swarm.drones.values())


def test_drone_actually_navigates_to_its_assigned_zone_and_secures_it():
    """Regression test for a real bug found during development: a drone
    could be ASSIGNED a zone (mission_zone_id set) without ever actually
    moving toward it, since normal movement is just constant-velocity
    drift, unrelated to mission_zone_id, unless Swarm._move specifically
    steers assigned drones -- so this checks the zone actually becomes
    secured, not just that an assignment was made."""
    # Drones start clustered around (0,0)-(60,30) (see _mission_swarm) --
    # keep the zone within realistic reach of the fallback mission cruise
    # speed (4 units/tick, since these drones are stationary/wander-speed
    # 0 by default) within a fast, non-flaky tick budget.
    zone = Zone(id="Z", x=100, y=50, radius=30, required_drones=2)
    swarm = _mission_swarm((zone,), n=6, reallocation_interval_ticks=3)

    _tick(swarm, 150)

    assert swarm.mission.zone_statuses["Z"].secured is True
    assert swarm.mission.all_secured() is True
    # At least the required drones actually ended up physically in range.
    assigned = [d for d in swarm.drones.values() if d.mission_zone_id == "Z"]
    assert len(assigned) >= zone.required_drones


def test_mission_assigned_events_logged():
    zone = Zone(id="Z", x=500, y=500, radius=40, required_drones=1)
    swarm = _mission_swarm((zone,), n=4, reallocation_interval_ticks=3)
    _tick(swarm, 30)
    assert any(e["type"] == "mission_assigned" for e in swarm.event_log)


def test_to_state_dict_includes_mission_when_configured_and_omits_otherwise():
    zone = Zone(id="Z", x=500, y=500, radius=40, required_drones=1)
    with_mission = _mission_swarm((zone,), n=2)
    with_mission.tick()
    assert "mission" in with_mission.to_state_dict()
    assert with_mission.to_state_dict()["mission"]["zones"][0]["id"] == "Z"

    without_mission = Swarm([Drone(id="D0", x=0, y=0, priority=1)], comm_range=100)
    without_mission.tick()
    assert "mission" not in without_mission.to_state_dict()


def test_low_battery_drone_loses_its_assignment():
    zone = Zone(id="Z", x=500, y=500, radius=40, required_drones=1)
    swarm = _mission_swarm((zone,), n=2)
    drone = swarm.drones["D0"]
    drone.mission_zone_id = "Z"
    drone.battery = 5.0  # below MIN_BATTERY_TO_ASSIGN, and not actually in the zone
    swarm.tick()
    assert drone.mission_zone_id is None


# -- Battery substitution / reserve pool ---------------------------------------

def test_plan_substitutions_dispatches_reserve_to_relieve_draining_occupant():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone, occupant_ids=["tired"], secured=True)
    allocator = HeuristicAllocator()

    tired = Drone(id="tired", x=0, y=0, priority=1, battery=LOW_BATTERY_WARNING - 1.0)
    reserve = Drone(id="reserve", x=5, y=0, priority=1, battery=100.0)
    drones = {"tired": tired, "reserve": reserve}

    assignments = allocator.plan_substitutions(drones, [status], arena_diagonal=1000.0)
    assert assignments == {"reserve": "Z"}


def test_plan_substitutions_prioritizes_most_urgent_zone_first():
    """The 'AI prioritizes which zones get priority substitution' part:
    with only one reserve drone available and two zones both needing
    relief, the more urgent one (lower remaining battery) wins."""
    urgent = ZoneStatus(zone=Zone(id="urgent", x=0, y=0, radius=10, required_drones=1), occupant_ids=["low"], secured=True)
    less_urgent = ZoneStatus(zone=Zone(id="less", x=0, y=0, radius=10, required_drones=1), occupant_ids=["mid"], secured=True)
    allocator = HeuristicAllocator()

    low_battery_occupant = Drone(id="low", x=0, y=0, priority=1, battery=5.0)
    mid_battery_occupant = Drone(id="mid", x=0, y=0, priority=1, battery=LOW_BATTERY_WARNING - 1.0)
    reserve = Drone(id="reserve", x=0, y=0, priority=1, battery=100.0)
    drones = {"low": low_battery_occupant, "mid": mid_battery_occupant, "reserve": reserve}

    assignments = allocator.plan_substitutions(drones, [urgent, less_urgent], arena_diagonal=1000.0)
    assert assignments == {"reserve": "urgent"}


def test_plan_substitutions_only_pulls_from_idle_healthy_reserve():
    """Committed drones (busy elsewhere) and drones too low on battery to
    be reserve material themselves are never poached for a substitution."""
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone, occupant_ids=["tired"], secured=True)
    allocator = HeuristicAllocator()

    tired = Drone(id="tired", x=0, y=0, priority=1, battery=10.0)
    already_assigned = Drone(id="assigned", x=0, y=0, priority=1, battery=100.0, mission_zone_id="other")
    low_battery_reserve = Drone(id="low_reserve", x=0, y=0, priority=1, battery=40.0)  # below the 50% reserve floor
    drones = {"tired": tired, "assigned": already_assigned, "low_reserve": low_battery_reserve}

    assignments = allocator.plan_substitutions(drones, [status], arena_diagonal=1000.0)
    assert assignments == {}


def test_plan_substitutions_ignores_zones_that_are_not_secured_or_not_draining():
    allocator = HeuristicAllocator()
    reserve = Drone(id="reserve", x=0, y=0, priority=1, battery=100.0)

    unsecured = ZoneStatus(zone=Zone(id="unsecured", x=0, y=0, radius=10, required_drones=2), occupant_ids=["a"], secured=False)
    healthy = ZoneStatus(zone=Zone(id="healthy", x=0, y=0, radius=10, required_drones=1), occupant_ids=["b"], secured=True)
    drones = {
        "a": Drone(id="a", x=0, y=0, priority=1, battery=5.0),  # low battery, but zone isn't secured
        "b": Drone(id="b", x=0, y=0, priority=1, battery=90.0),  # secured, but not draining
        "reserve": reserve,
    }

    assignments = allocator.plan_substitutions(drones, [unsecured, healthy], arena_diagonal=1000.0)
    assert assignments == {}


def test_release_relieved_occupants_only_after_replacement_physically_arrives():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))

    tired = Drone(id="tired", x=0, y=0, priority=1, battery=10.0, mission_zone_id="Z")
    drones = {"tired": tired}
    mission._occupants_of(drones)
    mission._release_relieved_occupants(drones)
    # Alone in the zone and still needed -- held despite very low battery,
    # exactly like before this feature existed, since no relief has arrived.
    assert tired.mission_zone_id == "Z"
    assert mission.zone_statuses["Z"].occupant_ids == ["tired"]

    fresh = Drone(id="fresh", x=1, y=0, priority=1, battery=100.0, mission_zone_id="Z")
    drones["fresh"] = fresh
    mission._occupants_of(drones)  # both now physically present -- surplus of 1
    mission._release_relieved_occupants(drones)
    assert tired.mission_zone_id is None
    assert mission.zone_statuses["Z"].occupant_ids == ["fresh"]
    assert mission.zone_statuses["Z"].secured is True


def test_stale_assignment_released_once_zone_secured_without_it():
    """Regression test for a real bug found via a live-scenario smoke test:
    a drone inbound to a zone that gets secured by OTHER, physically-closer
    drones before it arrives never becomes an occupant (so the 'arrived,
    still holding' path never fires) and might never drop low enough on
    battery to lose eligibility either -- so it kept mission_zone_id set
    forever, permanently unavailable as substitution reserve capacity for
    any other zone. It should be released as soon as its target zone is
    secured without it."""
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))

    already_secured_it = Drone(id="occupant", x=0, y=0, priority=1, battery=90.0, mission_zone_id="Z")
    still_inbound = Drone(id="inbound", x=500, y=500, priority=1, battery=90.0, mission_zone_id="Z")
    drones = {"occupant": already_secured_it, "inbound": still_inbound}

    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].secured is True

    class _FakeSwarm:
        pass
    fake_swarm = _FakeSwarm()
    fake_swarm.drones = drones
    fake_swarm.width, fake_swarm.height = 1000.0, 1000.0

    mission.tick(fake_swarm)
    assert still_inbound.mission_zone_id is None
    assert already_secured_it.mission_zone_id == "Z"  # the actual occupant keeps holding


def test_inbound_substitute_not_released_before_it_arrives():
    """Regression test for a real bug found via a live-scenario smoke test:
    a substitution target is by definition an already-secured (but
    draining) zone -- exactly the state the stale-assignment release
    above (test_stale_assignment_released_once_zone_secured_without_it)
    is looking for. Without distinguishing "secured because someone else
    already has it covered" from "secured but the occupant needs relief",
    a freshly-dispatched substitute got released the very next tick,
    before it could ever physically arrive, then immediately redispatched
    -- an infinite same-tick substitution-event loop that never actually
    delivered relief."""
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    mission = MissionState(MissionConfig(zones=(zone,)))

    tired_occupant = Drone(id="tired", x=0, y=0, priority=1, battery=LOW_BATTERY_WARNING - 1.0, mission_zone_id="Z")
    inbound_substitute = Drone(id="incoming", x=500, y=500, priority=1, battery=90.0, mission_zone_id="Z")
    drones = {"tired": tired_occupant, "incoming": inbound_substitute}

    mission._occupants_of(drones)
    assert mission.zone_statuses["Z"].secured is True  # tired alone already meets required_drones=1

    class _FakeSwarm:
        pass
    fake_swarm = _FakeSwarm()
    fake_swarm.drones = drones
    fake_swarm.width, fake_swarm.height = 1000.0, 1000.0

    mission.tick(fake_swarm)
    assert inbound_substitute.mission_zone_id == "Z"  # still en route, not released
    assert tired_occupant.mission_zone_id == "Z"  # still holding, hasn't been relieved yet


def test_battery_substitution_event_fires_without_waiting_for_periodic_reallocation():
    """Regression proof that substitution is the fast reactive layer: it
    must dispatch relief long before the coarse reallocation_interval_ticks
    cadence would otherwise get around to it."""
    zone = Zone(id="Z", x=0, y=0, radius=5, required_drones=1)
    config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0,
        mission_config=MissionConfig(zones=(zone,), reallocation_interval_ticks=1000),
    )
    occupant = Drone(id="occupant", x=0, y=0, priority=10, battery=LOW_BATTERY_WARNING - 1.0)
    reserve = Drone(id="reserve", x=200, y=200, priority=11, battery=100.0)
    swarm = Swarm([occupant, reserve], comm_range=500, config=config, seed=1)

    _tick(swarm, 20)

    assert any(e["type"] == "battery_substitution" for e in swarm.event_log)
    assert reserve.mission_zone_id == "Z"
