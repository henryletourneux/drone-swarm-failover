from drone_swarm.simulation import create_random_swarm


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
