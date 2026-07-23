"""Tests for the observability/metrics layer (drone_swarm/metrics.py plus the
mesh/election counters and Swarm._track_recovery that feed it).

Timing/helpers mirror the established conventions in test_election.py and
test_swarm.py: a fast lossless SwarmConfig and hand-placed Drone() cliques so
convergence lands in a handful of ticks.
"""
from drone_swarm.election import NexusElection
from drone_swarm.metrics import SwarmMetrics, _distribution
from drone_swarm.model import Drone
from drone_swarm.protocol import NexusHeartbeat
from drone_swarm.simulation import create_random_swarm
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


def _clique(priorities, comm_range=500, seed=1, config=FAST):
    """A fully-connected swarm: every drone within range of every other, so
    killing any single drone never disconnects the rest."""
    drones = []
    for i, (drone_id, priority) in enumerate(priorities.items()):
        drones.append(Drone(id=drone_id, x=(i % 3) * 30, y=(i // 3) * 30, priority=priority))
    return Swarm(drones, comm_range=comm_range, config=config, seed=seed)


def _alive_ids(swarm):
    return {d.id for d in swarm.drones.values() if d.alive}


def _count_events(swarm, event_type):
    return len([e for e in swarm.event_log if e["type"] == event_type])


# -- 1. bootstrap election is not a "recovery" -----------------------------

def test_bootstrap_election_records_no_recovery():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    # Nothing to "recover" from on first boot: no drone ever lost a nexus it
    # previously had, so recovery_times_s stays empty.
    assert swarm.metrics.recovery_times_s == []
    assert swarm.metrics_snapshot()["recovery"]["count"] == 0


# -- 2. a real failover records one recovery per survivor -------------------

def test_failover_records_one_recovery_per_survivor():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    assert swarm.metrics_snapshot()["recovery"]["count"] == 0

    n_alive_before = len(_alive_ids(swarm))
    swarm.kill("D4")
    _tick(swarm, 30)
    assert all(swarm.drones[i].nexus_id == "D3" for i in _alive_ids(swarm))

    survivors = n_alive_before - 1
    recovery = swarm.metrics_snapshot()["recovery"]
    assert recovery["count"] == survivors

    # Durations are real gaps: strictly positive, and loosely bounded by the
    # configured timing (a survivor notices the loss ~nexus_timeout_s later and
    # re-converges within a few ticks). Generous bound -> not flaky.
    for d in swarm.metrics.recovery_times_s:
        assert d > 0.0
        assert d < 10.0
    assert recovery["max_s"] < 10.0


# -- 3. cascading failures grow the recovery/election counters -------------

def test_cascading_failures_grow_metrics_monotonically():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)
    m0 = swarm.metrics_snapshot()
    assert m0["recovery"]["count"] == 0

    swarm.kill("D4")
    _tick(swarm, 30)
    assert all(swarm.drones[i].nexus_id == "D3" for i in _alive_ids(swarm))
    m1 = swarm.metrics_snapshot()

    swarm.kill("D3")
    _tick(swarm, 30)
    assert all(swarm.drones[i].nexus_id == "D2" for i in _alive_ids(swarm))
    m2 = swarm.metrics_snapshot()

    # Each kill strictly adds recoveries and elections.
    assert m1["recovery"]["count"] > m0["recovery"]["count"]
    assert m2["recovery"]["count"] > m1["recovery"]["count"]
    assert m1["elections_started"] > m0["elections_started"]
    assert m2["elections_started"] > m1["elections_started"]
    assert m1["elections_won"] > m0["elections_won"]
    assert m2["elections_won"] > m1["elections_won"]


# -- 4. counters cross-check against the event_log (real merge scenario) ----

def test_election_counts_cross_check_event_log_through_a_merge():
    """Drives a genuine partition -> merge (mirrors test_election.py's
    reconnecting-partitions scenario) so the run contains election_started,
    election_won AND swarms_merged events, then asserts the metrics counters
    equal the event_log tallies for each -- an independent consistency check."""
    left = [
        Drone(id="L0", x=0, y=0, priority=10),
        Drone(id="L1", x=20, y=0, priority=95),
        Drone(id="L2", x=0, y=20, priority=15),
    ]
    right = [
        Drone(id="R0", x=1000, y=0, priority=40),
        Drone(id="R1", x=1020, y=0, priority=50),
        Drone(id="R2", x=1000, y=20, priority=30),
    ]
    swarm = Swarm(left + right, comm_range=50, config=FAST, seed=2)

    _tick(swarm, 25)
    # Slide the right cluster into range of the left one to force a merge.
    for drone_id in {"R0", "R1", "R2"}:
        swarm.drones[drone_id].x -= 990
    _tick(swarm, 30)

    assert all(d.nexus_id == "R1" for d in swarm.drones.values())
    # This scenario genuinely exercised a merge, not just failover.
    assert _count_events(swarm, "swarms_merged") >= 1
    assert _count_events(swarm, "election_started") >= 1
    assert _count_events(swarm, "election_won") >= 1

    metrics = swarm.metrics_snapshot()
    assert metrics["elections_started"] == _count_events(swarm, "election_started")
    assert metrics["elections_won"] == _count_events(swarm, "election_won")
    assert metrics["merges"] == _count_events(swarm, "swarms_merged")


# -- 5. message counters -----------------------------------------------------

def test_message_counters_increase_and_delivered_never_exceeds_sent():
    swarm = create_random_swarm(seed=42, comm_range=1000, config=FAST)

    prev = swarm.metrics_snapshot()
    assert prev["messages_delivered"] <= prev["messages_sent"]
    saw_growth = False
    for _ in range(10):
        _tick(swarm, 5)
        cur = swarm.metrics_snapshot()
        # Monotonic non-decreasing counters.
        assert cur["messages_sent"] >= prev["messages_sent"]
        assert cur["messages_delivered"] >= prev["messages_delivered"]
        assert cur["messages_dropped_loss"] >= prev["messages_dropped_loss"]
        # Delivery can never outrun what was actually sent.
        assert cur["messages_delivered"] <= cur["messages_sent"]
        if cur["messages_sent"] > prev["messages_sent"]:
            saw_growth = True
        prev = cur
    assert saw_growth


def test_higher_packet_loss_produces_more_drops():
    lossless = create_random_swarm(
        seed=7, comm_range=1000,
        config=SwarmConfig(
            nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8,
            comm_latency_s=0.05, tick_dt_s=0.2, packet_loss_rate=0.0,
        ),
    )
    lossy = create_random_swarm(
        seed=7, comm_range=1000,
        config=SwarmConfig(
            nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8,
            comm_latency_s=0.05, tick_dt_s=0.2, packet_loss_rate=0.3,
        ),
    )
    _tick(lossless, 150)
    _tick(lossy, 150)

    assert lossless.metrics_snapshot()["messages_dropped_loss"] == 0
    assert lossy.metrics_snapshot()["messages_dropped_loss"] > 0


# -- 6. security rejections --------------------------------------------------

def test_security_rejections_zero_in_plain_mode():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)
    swarm.kill("D4")
    _tick(swarm, 30)
    # Verification always passes when bft_mode is off, so nothing is rejected.
    assert swarm.metrics_snapshot()["security_rejections"] == 0


def test_forged_message_is_rejected_under_bft():
    bft = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8,
        comm_latency_s=0.05, tick_dt_s=0.2, packet_loss_rate=0.0,
        bft_mode=True,
    )
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, config=bft)
    _tick(swarm, 25)
    # Honest signed traffic verifies cleanly: no rejections yet.
    assert swarm.metrics_snapshot()["security_rejections"] == 0

    # Inject an obviously-forged heartbeat: impersonates a real (positioned)
    # drone so the mesh will relay it, but its signature is garbage. Every
    # other in-range drone must verify-and-drop it.
    forged = NexusHeartbeat(sender_id="D0", sent_at_s=swarm.time_s, term=999, signature=b"garbage")
    swarm.mesh.broadcast(forged, swarm.time_s)
    _tick(swarm, 5)

    assert swarm.metrics_snapshot()["security_rejections"] > 0


# -- 7. _distribution() hand-verified ---------------------------------------

def test_distribution_matches_hand_computation():
    # 9 values, deliberately passed unsorted to also exercise the sort.
    values = [5.5, 1.0, 9.9, 3.0, 7.7, 2.5, 8.3, 4.2, 6.1]
    # sorted -> [1.0, 2.5, 3.0, 4.2, 5.5, 6.1, 7.7, 8.3, 9.9], n = 9
    #   mean = 48.2 / 9 = 5.35555... -> 5.356
    #   p50: idx = min(8, round(0.50*8)) = round(4.0) = 4 -> ordered[4] = 5.5
    #   p95: idx = min(8, round(0.95*8)) = round(7.6) = 8 -> ordered[8] = 9.9
    #   max = 9.9
    dist = _distribution(values)
    assert dist == {
        "count": 9,
        "mean_s": 5.356,
        "p50_s": 5.5,
        "p95_s": 9.9,
        "max_s": 9.9,
    }


def test_distribution_empty_is_all_none():
    assert _distribution([]) == {
        "count": 0, "mean_s": None, "p50_s": None, "p95_s": None, "max_s": None,
    }


# -- 8. snapshot shape + state-dict wiring ----------------------------------

def test_metrics_snapshot_shape_and_state_dict_wiring():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90})
    _tick(swarm, 20)

    snap = swarm.metrics_snapshot()
    assert set(snap.keys()) == {
        "recovery", "elections_started", "elections_won", "merges",
        "messages_sent", "messages_delivered", "messages_dropped_loss",
        "security_rejections",
    }
    assert set(snap["recovery"].keys()) == {"count", "mean_s", "p50_s", "p95_s", "max_s"}

    state = swarm.to_state_dict()
    assert "metrics" in state
    assert state["metrics"] == snap


def test_swarm_metrics_defaults():
    m = SwarmMetrics()
    assert m.recovery_times_s == []
    assert m.elections_started == 0
    assert m.elections_won == 0
    assert m.merges == 0
    m.record_recovery(1.5)
    assert m.recovery_times_s == [1.5]
