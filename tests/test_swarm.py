"""Integration-level tests driving whole swarms from create_random_swarm.

Uses a fast SwarmConfig so convergence lands in tens of ticks rather than
the default config's hundreds. The default comm_range=180 leaves the random
layout partitioned into several connected components, which is fine for the
"every drone has *a* live nexus" invariants; the single-nexus convergence
tests widen comm_range so the whole swarm is one connected mesh.
"""
from drone_swarm.model import Drone
from drone_swarm.simulation import create_random_swarm
from drone_swarm.swarm import Swarm, SwarmConfig

FAST = SwarmConfig(
    nexus_heartbeat_interval_s=0.3,
    nexus_timeout_s=0.8,
    comm_latency_s=0.05,
    tick_dt_s=0.2,
    packet_loss_rate=0.0,
)

# Same timing as FAST but with realistic loss, for the eventual-convergence
# test. Loss only slows delivery; it must never break correctness.
LOSSY = SwarmConfig(
    nexus_heartbeat_interval_s=0.3,
    nexus_timeout_s=0.8,
    comm_latency_s=0.05,
    tick_dt_s=0.2,
    packet_loss_rate=0.1,
)


def _tick(swarm, n):
    for _ in range(n):
        swarm.tick()


def test_every_alive_drone_gets_a_live_nexus():
    swarm = create_random_swarm(seed=42, config=FAST)
    _tick(swarm, 50)

    for drone in swarm.drones.values():
        if drone.alive:
            assert drone.nexus_id is not None
            assert swarm.drones[drone.nexus_id].alive


def test_event_log_records_an_election_win():
    swarm = create_random_swarm(seed=42, config=FAST)
    _tick(swarm, 50)

    types = [event["type"] for event in swarm.event_log]
    assert "election_won" in types


def test_state_dict_shape():
    swarm = create_random_swarm(seed=42, config=FAST)
    _tick(swarm, 10)
    state = swarm.to_state_dict()

    assert set(state.keys()) >= {"tick", "drones", "edges", "event_log"}
    for drone in state["drones"]:
        assert {"id", "alive", "role", "nexus_id"} <= set(drone.keys())


def test_killing_dead_drone_is_idempotent():
    swarm = create_random_swarm(seed=42, config=FAST)
    _tick(swarm, 10)

    victim = "D0"
    assert swarm.kill(victim) is True
    assert swarm.kill(victim) is False  # already dead

    down_events = [
        e for e in swarm.event_log
        if e["type"] == "drone_down" and e.get("drone") == victim
    ]
    assert len(down_events) == 1


def test_moving_drone_travels_and_bounces_off_boundary():
    # Uses the old-style constructor (no config/seed) to confirm the added
    # kwargs stay optional; movement/bounce logic was untouched by the rewrite.
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


def test_packet_loss_still_converges_to_single_nexus():
    """A realistic 10% packet-loss rate must not silently break election —
    given a connected mesh and generous ticks, the swarm still settles on
    exactly one live nexus. Timing is not asserted, only eventual outcome,
    so this stays non-flaky."""
    swarm = create_random_swarm(seed=42, comm_range=1000, config=LOSSY)
    _tick(swarm, 200)

    nexus_ids = {d.nexus_id for d in swarm.drones.values() if d.alive}
    assert nexus_ids != {None}
    assert len(nexus_ids) == 1, f"expected one nexus, got {nexus_ids}"
    (nexus_id,) = nexus_ids
    assert swarm.drones[nexus_id].alive
