"""Tests for the zone-coverage resource-allocation system (mission.py).

Covers: the module in isolation (battery drain, zone security detection,
heuristic allocator scoring/assignment), and end-to-end through Swarm
(nexus-driven reallocation timing, actual navigation toward an assigned
zone -- not just the assignment decision, since a real bug during
development was drones being assigned but never actually moving).
"""
from drone_swarm.mission import HeuristicAllocator, MissionConfig, MissionState, Zone, ZoneStatus
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
