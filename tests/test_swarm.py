from drone_swarm.model import Drone
from drone_swarm.simulation import create_random_swarm
from drone_swarm.swarm import Swarm


def _tick(swarm, n):
    for _ in range(n):
        swarm.tick()


def test_every_alive_drone_gets_a_live_nexus():
    swarm = create_random_swarm(seed=42)
    _tick(swarm, 30)

    for drone in swarm.drones.values():
        if drone.alive:
            assert drone.nexus_id is not None
            assert swarm.drones[drone.nexus_id].alive


def test_event_log_records_an_election_win():
    swarm = create_random_swarm(seed=42)
    _tick(swarm, 30)

    types = [event["type"] for event in swarm.event_log]
    assert "election_won" in types


def test_state_dict_shape():
    swarm = create_random_swarm(seed=42)
    _tick(swarm, 5)
    state = swarm.to_state_dict()

    assert set(state.keys()) >= {"tick", "drones", "edges", "event_log"}
    for drone in state["drones"]:
        assert {"id", "alive", "role", "nexus_id"} <= set(drone.keys())


def test_killing_dead_drone_is_idempotent():
    swarm = create_random_swarm(seed=42)
    _tick(swarm, 5)

    victim = "D0"
    assert swarm.kill(victim) is True
    assert swarm.kill(victim) is False  # already dead

    down_events = [
        e for e in swarm.event_log
        if e["type"] == "drone_down" and e.get("drone") == victim
    ]
    assert len(down_events) == 1


def test_moving_drone_travels_and_bounces_off_boundary():
    drone = Drone(id="D0", x=95, y=50, priority=10, vx=10, vy=0)
    swarm = Swarm([drone], comm_range=100, width=100, height=100)

    swarm.tick()
    assert drone.x == 100  # clamped at the right edge
    assert drone.vx == -10  # velocity flipped

    swarm.tick()
    assert drone.x == 90  # now heading back left


def test_stationary_drones_default_to_zero_velocity():
    swarm = create_random_swarm(seed=42, speed=0.0)
    positions_before = {d.id: (d.x, d.y) for d in swarm.drones.values()}
    swarm.tick()
    positions_after = {d.id: (d.x, d.y) for d in swarm.drones.values()}
    assert positions_before == positions_after
