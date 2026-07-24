"""Tests for boid-style flocking steering (flocking.py), in isolation from
Swarm -- flock_velocity is a pure function of a drone, its neighbors, a
config, and an optional target, so it's tested directly against small
hand-built scenarios rather than through a full simulation. End-to-end
tests through Swarm are at the bottom, covering the parts that only show
up once movement actually runs on a tick loop: neighbor-finding via
SpatialGrid, and that mission/investigating drones are correctly left
alone by flocking rather than getting dragged off course."""
import math

from drone_swarm.flocking import FlockingConfig, flock_velocity
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


def test_separation_pushes_apart_when_too_close():
    config = FlockingConfig(separation_radius=30.0, separation_weight=2.0, alignment_weight=0.0, cohesion_weight=0.0)
    drone = Drone(id="A", x=0, y=0, priority=1)
    neighbor = Drone(id="B", x=10, y=0, priority=1)  # well within separation_radius

    vx, vy = flock_velocity(drone, [neighbor], config)

    assert vx < 0  # steers away from the neighbor, i.e. in -x
    assert abs(vy) < 1e-9


def test_no_separation_force_outside_separation_radius():
    config = FlockingConfig(separation_radius=10.0, separation_weight=2.0, alignment_weight=0.0, cohesion_weight=0.0)
    drone = Drone(id="A", x=0, y=0, priority=1)
    neighbor = Drone(id="B", x=100, y=0, priority=1)  # far outside separation_radius

    vx, vy = flock_velocity(drone, [neighbor], config)

    # No separation, no alignment/cohesion weight, no target -- nothing
    # pulling it anywhere, so it keeps its prior (zero) velocity.
    assert (vx, vy) == (0.0, 0.0)


def test_alignment_steers_toward_average_neighbor_heading():
    config = FlockingConfig(separation_weight=0.0, alignment_weight=1.0, cohesion_weight=0.0)
    drone = Drone(id="A", x=0, y=0, priority=1, vx=0, vy=0)
    neighbors = [
        Drone(id="B", x=200, y=200, priority=1, vx=3.0, vy=0.0),  # far enough away: no separation/cohesion noise
        Drone(id="C", x=-200, y=-200, priority=1, vx=3.0, vy=0.0),
    ]

    vx, vy = flock_velocity(drone, neighbors, config)

    assert vx > 0  # picks up the shared +x heading of its neighbors
    assert abs(vy) < 1e-6


def test_cohesion_pulls_a_straggler_toward_the_group_centroid():
    config = FlockingConfig(separation_weight=0.0, alignment_weight=0.0, cohesion_weight=1.0)
    drone = Drone(id="A", x=0, y=0, priority=1)
    neighbors = [Drone(id="B", x=300, y=0, priority=1), Drone(id="C", x=300, y=0, priority=1)]  # centroid at (300, 0)

    vx, vy = flock_velocity(drone, neighbors, config)

    assert vx > 0  # steers toward the centroid, i.e. +x
    assert abs(vy) < 1e-9


def test_target_steering_is_direction_only_not_distance_scaled():
    config = FlockingConfig(separation_weight=0.0, alignment_weight=0.0, cohesion_weight=0.0, target_weight=2.0)
    drone = Drone(id="A", x=0, y=0, priority=1)

    near_vx, near_vy = flock_velocity(drone, [], config, target=(10, 0))
    far_vx, far_vy = flock_velocity(drone, [], config, target=(10_000, 0))

    # Same direction, same magnitude either way -- a steady compass
    # bearing, not a force that overwhelms everything when far and fades
    # to nothing right before arrival.
    assert math.isclose(near_vx, far_vx, rel_tol=1e-9)
    assert math.isclose(near_vy, far_vy, rel_tol=1e-9)
    assert near_vx > 0


def test_speed_is_clamped_to_max_speed():
    config = FlockingConfig(separation_radius=50.0, separation_weight=10.0, max_speed=3.0)
    drone = Drone(id="A", x=0, y=0, priority=1)
    neighbor = Drone(id="B", x=1, y=0, priority=1)  # extremely close -- huge raw separation force

    vx, vy = flock_velocity(drone, [neighbor], config)

    assert math.hypot(vx, vy) <= config.max_speed + 1e-9


def test_alone_with_no_target_keeps_previous_heading_instead_of_stopping():
    config = FlockingConfig()
    drone = Drone(id="A", x=0, y=0, priority=1, vx=1.5, vy=-2.0)

    vx, vy = flock_velocity(drone, [], config, target=None)

    assert (vx, vy) == (1.5, -2.0)


# -- End-to-end through Swarm --------------------------------------------------

def test_flocking_disabled_by_default_matches_old_straight_line_bounce():
    """flocking_config=None (the default) must leave movement byte-for-byte
    the old behavior -- regression guard for the additive-layer promise
    stated in SwarmConfig's docstring."""
    drone = Drone(id="A", x=10, y=10, priority=1, vx=3.0, vy=0.0)
    swarm = Swarm([drone], comm_range=100, width=100, height=100, config=FAST, seed=1)

    swarm.tick()

    assert swarm.drones["A"].x == 13.0
    assert swarm.drones["A"].y == 10.0
    assert swarm.drones["A"].vx == 3.0  # untouched -- no flocking recompute happened


def test_free_drones_cohere_toward_each_other_over_time():
    config = SwarmConfig(**{**FAST.__dict__, "flocking_config": FlockingConfig(neighbor_radius=500, target_weight=0)})
    drones = [
        Drone(id="A", x=50, y=250, priority=1),
        Drone(id="B", x=450, y=250, priority=1),
    ]
    swarm = Swarm(drones, comm_range=500, width=500, height=500, config=config, seed=1)
    start_gap = abs(swarm.drones["A"].x - swarm.drones["B"].x)

    _tick(swarm, 40)

    end_gap = abs(swarm.drones["A"].x - swarm.drones["B"].x)
    assert end_gap < start_gap  # cohesion actually pulled them together


def test_mission_drone_is_not_dragged_off_by_flocking():
    """A drone travelling to hold a mission zone must keep steering there
    even with flocking enabled and neighbors pulling in a different
    direction -- flocking only ever applies to drones with nothing else
    assigned (see Swarm._move's ordering)."""
    zone = Zone(id="Z", x=480, y=250, radius=10, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=1000)
    flocking_config = FlockingConfig(neighbor_radius=500, cohesion_weight=5.0, target_weight=0)
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "flocking_config": flocking_config})

    on_mission = Drone(id="M", x=20, y=250, priority=1, mission_zone_id="Z")
    # A cluster of free drones far in the OPPOSITE direction from the
    # zone -- strong cohesion would pull "M" back toward them if the
    # mission-check didn't take priority over flocking.
    lures = [Drone(id=f"L{i}", x=0, y=0, priority=1) for i in range(5)]
    swarm = Swarm([on_mission, *lures], comm_range=500, width=500, height=500, config=config, seed=1)

    _tick(swarm, 60)

    m = swarm.drones["M"]
    assert m.mission_zone_id == "Z"
    assert math.hypot(m.x - zone.x, m.y - zone.y) < math.hypot(20 - zone.x, 250 - zone.y)  # net progress toward the zone
    assert m.x > 20  # moved toward +x (the zone), not toward the x=0 lure cluster


def test_patrol_route_advances_through_full_swarm_ticks():
    from drone_swarm.patrol import PatrolConfig

    patrol_config = PatrolConfig(route_waypoint_count=4, route_edge_margin=20, route_arrival_radius=40.0, spawn_interval_ticks=100000)
    flocking_config = FlockingConfig(max_speed=8.0)
    mission_config = MissionConfig(zones=(), reallocation_interval_ticks=1000)
    config = SwarmConfig(**{
        **FAST.__dict__, "mission_config": mission_config,
        "patrol_config": patrol_config, "flocking_config": flocking_config,
    })
    drones = [Drone(id=f"D{i}", x=150, y=150, priority=1) for i in range(6)]
    swarm = Swarm(drones, comm_range=500, width=300, height=300, config=config, seed=3)

    start_index = swarm.patrol.route_index
    advanced = False
    for _ in range(300):
        swarm.tick()
        if swarm.patrol.route_index != start_index:
            advanced = True
            break

    assert advanced, "expected the patrol route to advance to a new waypoint within 300 ticks"
