"""Tests for topology/obstacles (obstacles.py). Pure geometry functions at
the top, no Swarm needed; end-to-end movement tests through Swarm at the
bottom, covering what only shows up once real steering runs on a tick
loop -- in particular the head-on stall this module's tangential-avoidance
component was specifically added to fix."""
import math

from drone_swarm.mission import MissionConfig, Zone
from drone_swarm.model import Drone
from drone_swarm.obstacles import Obstacle, obstacle_avoidance, point_inside_any, push_point_outside_all
from drone_swarm.swarm import Swarm, SwarmConfig

FAST = SwarmConfig(
    nexus_heartbeat_interval_s=0.3,
    nexus_timeout_s=0.8,
    comm_latency_s=0.05,
    tick_dt_s=0.2,
    packet_loss_rate=0.0,
)


def test_no_avoidance_force_far_from_any_obstacle():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=20)]
    dx, dy = obstacle_avoidance(500, 500, obstacles, margin=30)
    assert (dx, dy) == (0.0, 0.0)


def test_avoidance_has_a_radial_component_away_from_obstacle_center():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=20)]
    dx, dy = obstacle_avoidance(40, 0, obstacles, margin=30, heading=None)  # surface_dist=20, within margin 30
    assert dx > 0  # net push is still further in +x, away from the obstacle
    assert dy != 0  # plus a tangential component (see the "stall" bug this fixes)


def test_avoidance_tangential_direction_follows_heading_not_fixed():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=20)]
    # Same position, opposite intended headings -- the tangential
    # component should flip sign to curve toward whichever way the drone
    # was already trying to go, not a fixed rotation regardless of intent.
    _, dy_up = obstacle_avoidance(40, 0, obstacles, margin=30, heading=(0, 1))
    _, dy_down = obstacle_avoidance(40, 0, obstacles, margin=30, heading=(0, -1))
    assert dy_up > 0
    assert dy_down < 0


def test_avoidance_is_stronger_when_deeper_inside_the_margin():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=20)]
    near_edge_dx, _ = obstacle_avoidance(49, 0, obstacles, margin=30)   # surface_dist=29, barely inside margin
    deep_dx, _ = obstacle_avoidance(25, 0, obstacles, margin=30)        # surface_dist=5, deep inside margin
    assert deep_dx > near_edge_dx


def test_avoidance_sums_across_multiple_nearby_obstacles():
    obstacles = [Obstacle(id="O1", x=-30, y=0, radius=10), Obstacle(id="O2", x=30, y=0, radius=10)]
    # Sitting between two obstacles pushing from opposite sides -- net
    # horizontal force should roughly cancel (symmetric setup).
    dx, dy = obstacle_avoidance(0, 0, obstacles, margin=30)
    assert abs(dx) < 1e-6


def test_point_inside_any_true_within_radius():
    obstacles = [Obstacle(id="O1", x=100, y=100, radius=15)]
    assert point_inside_any(105, 100, obstacles) is True


def test_point_inside_any_false_outside_radius():
    obstacles = [Obstacle(id="O1", x=100, y=100, radius=15)]
    assert point_inside_any(200, 200, obstacles) is False


def test_point_inside_any_respects_extra_margin():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=10)]
    assert point_inside_any(15, 0, obstacles, extra_margin=0.0) is False
    assert point_inside_any(15, 0, obstacles, extra_margin=10.0) is True


def test_push_point_outside_all_clears_a_single_obstacle():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=20)]
    px, py = push_point_outside_all(5, 0, obstacles, extra_margin=5.0)
    assert math.hypot(px, py) >= 20 + 5.0 - 1e-6
    assert not point_inside_any(px, py, obstacles, extra_margin=5.0)


def test_push_point_outside_all_clears_overlapping_obstacles():
    obstacles = [Obstacle(id="O1", x=0, y=0, radius=20), Obstacle(id="O2", x=25, y=0, radius=20)]
    px, py = push_point_outside_all(10, 0, obstacles, extra_margin=2.0)
    assert not point_inside_any(px, py, obstacles, extra_margin=2.0)


# -- End-to-end through Swarm --------------------------------------------------

def test_drone_bends_around_an_obstacle_between_it_and_its_zone():
    zone = Zone(id="Z", x=400, y=200, radius=15, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=1000)
    obstacles = (Obstacle(id="O1", x=200, y=200, radius=40),)
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "obstacles": obstacles})
    drone = Drone(id="M", x=0, y=200, priority=1, mission_zone_id="Z")
    swarm = Swarm([drone], comm_range=500, width=500, height=400, config=config, seed=1)

    closest_approach = float("inf")
    for _ in range(150):
        swarm.tick()
        m = swarm.drones["M"]
        closest_approach = min(closest_approach, math.hypot(m.x - 200, m.y - 200))

    m = swarm.drones["M"]
    assert math.hypot(m.x - zone.x, m.y - zone.y) < 20  # actually arrived
    assert closest_approach >= 40 - 1e-6  # never actually entered the obstacle


def test_drone_does_not_stall_head_on_into_an_obstacle():
    """The specific bug this module's tangential-avoidance component
    fixes: a drone travelling in a dead-straight line with an obstacle
    directly on its path used to freeze exactly where the obstacle's
    (purely radial, before this fix) repulsion equalled the drone's own
    pull toward its target -- a classic potential-field local minimum.
    Regression-verified by temporarily reverting obstacle_avoidance to a
    radial-only version and confirming this test fails (the drone gets
    stuck around x=127 and never reaches the zone) before restoring the
    tangential-component fix."""
    zone = Zone(id="Z", x=400, y=200, radius=15, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=1000)
    # Obstacle centered EXACTLY on the straight line from drone to zone --
    # the worst case for a purely radial potential field.
    obstacles = (Obstacle(id="O1", x=200, y=200, radius=40),)
    config = SwarmConfig(**{**FAST.__dict__, "mission_config": mission_config, "obstacles": obstacles})
    drone = Drone(id="M", x=0, y=200, priority=1, mission_zone_id="Z")
    swarm = Swarm([drone], comm_range=500, width=500, height=400, config=config, seed=1)

    for _ in range(150):
        swarm.tick()

    m = swarm.drones["M"]
    assert math.hypot(m.x - zone.x, m.y - zone.y) < 20, (
        f"drone stalled at ({m.x:.1f}, {m.y:.1f}) instead of reaching the zone at ({zone.x}, {zone.y})"
    )


def test_flocking_drones_also_avoid_obstacles():
    from drone_swarm.flocking import FlockingConfig

    obstacles = (Obstacle(id="O1", x=250, y=200, radius=50),)
    flocking_config = FlockingConfig(neighbor_radius=500)
    config = SwarmConfig(**{**FAST.__dict__, "flocking_config": flocking_config, "obstacles": obstacles})
    drones = [Drone(id=f"D{i}", x=0, y=190 + i * 5, priority=1) for i in range(4)]
    swarm = Swarm(drones, comm_range=500, width=500, height=400, config=config, seed=1)

    closest_approach = float("inf")
    for _ in range(80):
        swarm.tick()
        for d in swarm.drones.values():
            closest_approach = min(closest_approach, math.hypot(d.x - 250, d.y - 200))

    assert closest_approach >= 50 - 1e-6  # no flocking drone ever entered the obstacle


def test_to_state_dict_includes_obstacles_only_when_configured():
    plain = Swarm([Drone(id="A", x=0, y=0, priority=1)], comm_range=100, config=FAST, seed=1)
    assert "obstacles" not in plain.to_state_dict()

    obstacles = (Obstacle(id="O1", x=10, y=10, radius=5),)
    config = SwarmConfig(**{**FAST.__dict__, "obstacles": obstacles})
    with_obstacles = Swarm([Drone(id="A", x=0, y=0, priority=1)], comm_range=100, config=config, seed=1)
    state = with_obstacles.to_state_dict()
    assert state["obstacles"] == [{"id": "O1", "x": 10, "y": 10, "radius": 5}]
