"""Tests for the learned commander allocator (commander_policy.py) and
its training pipeline (commander_train.py).

Deliberately does NOT assert "the trained policy beats
HeuristicCommanderAllocator" -- same discipline as test_policy.py: that's
an empirical finding to report honestly (see README), not something to
bake into a test as if guaranteed. What IS tested: the mechanics are
correct (the drop-in interface, the same oscillation-prevention fix
HeuristicCommanderAllocator needed, the reset-between-episodes lifecycle),
and that a real REINFORCE update actually shifts probability toward a
better decision on an unambiguous, isolated scenario -- the same
surgical, environment-independent way test_policy.py validates the
gradient mechanic, not a beat-the-baseline claim."""
import torch

from drone_swarm.command import CommandConfig
from drone_swarm.commander_allocator import CommanderAllocatorConfig, HeuristicCommanderAllocator
from drone_swarm.commander_policy import LearnedCommanderAllocator
from drone_swarm.commander_train import run_episode, train
from drone_swarm.mission import MissionConfig, Zone, ZoneStatus
from drone_swarm.model import Drone
from drone_swarm.policy import AllocatorPolicy, drone_zone_features
from drone_swarm.swarm import Swarm, SwarmConfig


def _platoon_of(n, size):
    return {f"D{i}": f"P{i // size}" for i in range(n)}


# -- Drop-in interface / mechanics ---------------------------------------------

def test_learned_commander_allocator_is_a_valid_drop_in_replacement():
    """Same interface contract as HeuristicCommanderAllocator: a swarm
    running with a LearnedCommanderAllocator must not crash, and drones
    it assigns guard duty to must respect the same eligibility rules
    (alive, not investigating)."""
    zone = Zone(id="Z", x=50, y=50, radius=30, required_drones=2)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=1000)
    command_config = CommandConfig(platoon_of=_platoon_of(8, 2))
    allocator = LearnedCommanderAllocator(AllocatorPolicy(), CommanderAllocatorConfig(reallocation_interval_ticks=5), sample=False)
    swarm_config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0,
        mission_config=mission_config, command_config=command_config, commander_allocator=allocator,
    )
    drones = [Drone(id=f"D{i}", x=(i % 4) * 20, y=(i // 4) * 20, priority=10 + i) for i in range(8)]
    swarm = Swarm(drones, comm_range=2000, config=swarm_config, seed=1)

    for _ in range(100):
        swarm.tick()  # must not raise

    for drone in swarm.drones.values():
        if drone.duty == "guard":
            assert drone.alive
            assert drone.mission_zone_id is not None


def test_greedy_inference_is_deterministic():
    """sample=False must be reproducible -- same policy, same state, same
    decision every time, since this is the mode used for live-demo
    inference."""
    policy = AllocatorPolicy()
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    drones = [Drone(id="A", x=5, y=0, priority=1, battery=80.0), Drone(id="B", x=50, y=0, priority=1, battery=80.0)]

    def pick():
        feats = torch.tensor([drone_zone_features(d, status, arena_diagonal=100.0) for d in drones])
        with torch.no_grad():
            scores = policy(feats)
        return int(torch.argmax(scores).item())

    assert pick() == pick()


def test_episode_log_probs_reset_between_episodes():
    """Regression for the same bug class LearnedAllocator's own
    reset_episode() already guards against in policy.py: episode N+1
    starting with stale log_prob tensors from episode N still attached
    would crash on the next backward() with 'trying to backward through
    the graph a second time'."""
    policy = AllocatorPolicy()
    allocator = LearnedCommanderAllocator(policy, CommanderAllocatorConfig(reallocation_interval_ticks=1), sample=True)

    zone = Zone(id="Z", x=50, y=50, radius=30, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=1000)
    command_config = CommandConfig(platoon_of=_platoon_of(4, 2))
    swarm_config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0,
        mission_config=mission_config, command_config=command_config, commander_allocator=allocator,
    )
    # Well outside the zone's radius (30) so none of them starts inside it
    # by coincidence -- guard duty must come from an actual _allocate
    # decision, not free physical luck.
    drones = [Drone(id=f"D{i}", x=300 + 10 * i, y=300 + 10 * i, priority=10 + i) for i in range(4)]
    swarm = Swarm(drones, comm_range=2000, config=swarm_config, seed=1)

    allocator.reset_episode()
    for _ in range(60):
        swarm.tick()
    assert len(allocator.episode_log_probs) >= 1

    allocator.reset_episode()
    assert allocator.episode_log_probs == []
    assert allocator.episode_entropies == []


def test_repeated_allocate_calls_do_not_evict_a_successful_guard():
    """The learned allocator must carry the same oscillation-prevention
    fix HeuristicCommanderAllocator needed (see that class's own
    docstring for the real bug this guards against): a drone already
    successfully guarding a now-secured zone must not get reassigned to
    patrol just because that zone no longer appears in guard_needs."""
    zone = Zone(id="Z", x=500, y=500, radius=20, required_drones=1)
    mission_config = MissionConfig(zones=(zone,), reallocation_interval_ticks=1000)
    command_config = CommandConfig(platoon_of=_platoon_of(2, 1))
    allocator = LearnedCommanderAllocator(AllocatorPolicy(), CommanderAllocatorConfig(), sample=False)
    swarm_config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0,
        mission_config=mission_config, command_config=command_config, commander_allocator=allocator,
    )
    drones = [Drone(id="D0", x=490, y=500, priority=1), Drone(id="D1", x=0, y=0, priority=1)]
    swarm = Swarm(drones, comm_range=2000, config=swarm_config, seed=1)

    allocator._allocate(swarm)
    # An untrained policy has no guaranteed "nearest wins" preference the
    # way the heuristic does -- assert on whichever drone it actually
    # picked, not a hardcoded one.
    guard_id = next(d.id for d in swarm.drones.values() if d.duty == "guard")
    assert swarm.drones[guard_id].mission_zone_id == "Z"

    swarm.mission.zone_statuses["Z"].occupant_ids = [guard_id]
    swarm.mission.zone_statuses["Z"].secured = True
    allocator._allocate(swarm)

    assert swarm.drones[guard_id].duty == "guard"
    assert swarm.drones[guard_id].mission_zone_id == "Z"


def test_allocate_handles_no_guard_needs_and_no_available_drones_gracefully():
    command_config = CommandConfig(platoon_of=_platoon_of(3, 3))
    allocator = LearnedCommanderAllocator(AllocatorPolicy(), CommanderAllocatorConfig(), sample=False)
    swarm_config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0,
        command_config=command_config, commander_allocator=allocator,  # no mission_config at all
    )
    drones = [Drone(id=f"D{i}", x=0, y=0, priority=1) for i in range(3)]
    swarm = Swarm(drones, comm_range=2000, config=swarm_config, seed=1)

    allocator._allocate(swarm)  # must not raise despite no mission/no guard needs

    assert all(d.duty == "patrol" for d in swarm.drones.values())


# -- Training pipeline: a real learning signal, not a beat-the-baseline claim --

def test_reinforce_update_increases_probability_of_rewarded_drone_choice():
    """Directly validates the core REINFORCE mechanic in isolation from
    the full multi-tick environment, same surgical shape as
    test_policy.py's own version -- here the decision is "which candidate
    DRONE should fill this zone slot" (commander_policy.py's actual
    per-slot decision), not "which zone should this drone join." Given an
    unambiguous scenario (a very close drone vs a very far one for the
    same zone), repeated REINFORCE updates must increase probability mass
    on picking the closer one. If this doesn't hold, the gradient math
    itself is wrong, independent of any environment noise."""
    torch.manual_seed(3)
    policy = AllocatorPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)

    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    near_drone = Drone(id="near", x=10, y=0, priority=1, battery=100.0)
    far_drone = Drone(id="far", x=900, y=0, priority=1, battery=100.0)

    def build_probs():
        feats = torch.tensor([
            drone_zone_features(near_drone, status, arena_diagonal=1000.0),
            drone_zone_features(far_drone, status, arena_diagonal=1000.0),
        ])
        scores = policy(feats)
        return torch.softmax(scores, dim=0)

    with torch.no_grad():
        prob_near_before = build_probs()[0].item()

    reward_for_action = {0: 1.0, 1: -1.0}
    for _ in range(200):
        probs = build_probs()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        reward = reward_for_action[int(action.item())]
        loss = -dist.log_prob(action) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prob_near_after = build_probs()[0].item()

    assert prob_near_after > prob_near_before + 0.1, (
        f"REINFORCE update did not shift probability toward the rewarded drone: "
        f"P(near) before={prob_near_before:.3f} after={prob_near_after:.3f}"
    )


def test_train_and_evaluate_pipeline_runs_end_to_end_without_crashing():
    """A softer, complementary check to the isolated gradient-mechanic
    test above: the full train() -> evaluate() pipeline (real
    environment, real reward shaping, real randomized episodes) must run
    to completion and produce a well-formed comparison report. Doesn't
    assert who wins -- see commander_train.py's module docstring and the
    README for why that's an honest empirical finding, not a guarantee."""
    from drone_swarm.commander_train import evaluate

    policy = train(n_episodes=12, seed=1, verbose=False)
    results = evaluate(policy, n_episodes=3, seed=2)

    assert set(results.keys()) == {"heuristic", "learned"}
    for stats in results.values():
        assert 0.0 <= stats["fully_secured_fraction"] <= 1.0
        assert stats["mean_zones_secured_at_end"] >= 0


def test_run_episode_returns_consistent_final_swarm_state():
    """Regression shape matching test_policy.py's own version: run_episode
    must return the actually-simulated swarm (real progress), not a fresh
    unticked one."""
    from drone_swarm.commander_train import _random_episode_config
    import random

    n_drones, width, height, zones, platoon_size = _random_episode_config(random.Random(7))
    allocator = HeuristicCommanderAllocator(CommanderAllocatorConfig(reallocation_interval_ticks=5))
    ret, _, _, secured_at, swarm = run_episode(allocator, n_drones, width, height, zones, platoon_size, seed=1, collect_log_probs=False)

    assert swarm.tick_count > 0
    if secured_at is not None:
        assert swarm.mission.all_secured() is True
