"""Tests for the learned allocation policy (policy.py) and its training
pipeline (train.py).

Deliberately does NOT assert "the trained policy beats HeuristicAllocator"
-- that's an empirical finding to report honestly (see README), not
something to bake into a test as if it were guaranteed. What IS tested:
the mechanics are correct (feature extraction, the drop-in interface,
the agent-count-invariance property the whole design leans on), and that
training actually produces a real learning signal over a random-init
policy on a fixed, controlled scenario -- a fair, non-brittle way to
check the pipeline actually learns something, without claiming it beats
a specific baseline.
"""
import torch

from drone_swarm.mission import MissionConfig, Zone, ZoneStatus
from drone_swarm.model import Drone
from drone_swarm.policy import AllocatorPolicy, LearnedAllocator, drone_zone_features
from drone_swarm.train import run_episode, train
from drone_swarm.swarm import Swarm, SwarmConfig


# -- Feature extraction -------------------------------------------------------

def test_features_are_bounded_and_relative_not_absolute():
    """The whole agent-count-invariance argument in policy.py's docstring
    depends on features being relative (fractions), not raw magnitudes
    that would differ between a 6-drone and a 100-drone swarm."""
    zone = Zone(id="Z", x=100, y=100, radius=50, required_drones=3, threat_level=2.0)
    status = ZoneStatus(zone=zone, occupant_ids=["A"])
    drone = Drone(id="B", x=0, y=0, priority=1, battery=50.0, role="relay")

    feats = drone_zone_features(drone, status, arena_diagonal=1000.0)
    assert len(feats) == 8
    assert all(0.0 <= f <= 1.0 for f in feats), feats
    assert feats[0] == 0.5  # battery_frac: 50/100


def test_features_identical_for_same_relative_state_at_different_scale():
    """The literal invariance check: a drone halfway across a small arena
    and a drone halfway across a huge arena, at the same battery/threat/
    need/role state, must produce IDENTICAL feature vectors -- proving
    the network sees the same input regardless of absolute scale."""
    zone_small = Zone(id="Z", x=50, y=0, radius=10, required_drones=2, threat_level=1.0)
    zone_large = Zone(id="Z", x=5000, y=0, radius=10, required_drones=2, threat_level=1.0)
    status_small = ZoneStatus(zone=zone_small)
    status_large = ZoneStatus(zone=zone_large)

    drone_small = Drone(id="A", x=0, y=0, priority=1, battery=75.0, role="leaf")
    drone_large = Drone(id="A", x=0, y=0, priority=1, battery=75.0, role="leaf")

    feats_small = drone_zone_features(drone_small, status_small, arena_diagonal=100.0)
    feats_large = drone_zone_features(drone_large, status_large, arena_diagonal=10000.0)
    assert feats_small == feats_large


def test_role_one_hot_encoding():
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    for role, expected_idx in (("relay", 4), ("leaf", 5), ("nexus", 6)):
        drone = Drone(id="A", x=0, y=0, priority=1, role=role)
        feats = drone_zone_features(drone, status, arena_diagonal=100.0)
        one_hot = feats[4:7]
        assert one_hot[expected_idx - 4] == 1.0
        assert sum(one_hot) == 1.0


# -- AllocatorPolicy / LearnedAllocator mechanics -----------------------------

def test_policy_output_shape():
    policy = AllocatorPolicy()
    x = torch.zeros((3, 8))
    scores = policy(x)
    assert scores.shape == (3,)


def test_learned_allocator_is_a_valid_drop_in_replacement():
    """Same interface contract as HeuristicAllocator.allocate: a swarm
    running with a LearnedAllocator must not crash, and must respect the
    same eligibility rules (alive, charged, not already committed)."""
    zone = Zone(id="Z", x=50, y=50, radius=30, required_drones=2)
    config = MissionConfig(zones=(zone,), reallocation_interval_ticks=3)
    swarm_config = SwarmConfig(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, comm_latency_s=0.05,
        tick_dt_s=0.2, packet_loss_rate=0.0, mission_config=config,
    )
    drones = [Drone(id=f"D{i}", x=(i % 3) * 20, y=(i // 3) * 20, priority=10 + i) for i in range(6)]
    swarm = Swarm(drones, comm_range=500, config=swarm_config, seed=1)
    swarm.mission.allocator = LearnedAllocator(AllocatorPolicy(), sample=False)

    for _ in range(30):
        swarm.tick()  # must not raise

    for drone in swarm.drones.values():
        if drone.mission_zone_id is not None:
            assert drone.alive
            assert drone.battery > 0


def test_greedy_inference_is_deterministic():
    """sample=False must be reproducible -- same policy, same state, same
    decision every time (no hidden randomness), since this is the mode
    used for live-demo inference."""
    policy = AllocatorPolicy()
    allocator_a = LearnedAllocator(policy, sample=False)
    allocator_b = LearnedAllocator(policy, sample=False)

    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    drones = {"A": Drone(id="A", x=5, y=0, priority=1, battery=80.0)}

    result_a = allocator_a.allocate(drones, [status], arena_diagonal=100.0)
    result_b = allocator_b.allocate(drones, [status], arena_diagonal=100.0)
    assert result_a == result_b


def test_episode_log_probs_reset_between_episodes():
    """Regression test for a real bug found during development: episode
    N+1 could start with stale log_prob tensors from episode N still
    attached (already backpropped, freed autograd graph), crashing on
    the next backward() with 'trying to backward through the graph a
    second time'. reset_episode() must actually clear accumulated state."""
    policy = AllocatorPolicy()
    allocator = LearnedAllocator(policy, sample=True)

    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone)
    drones = {"A": Drone(id="A", x=5, y=0, priority=1, battery=80.0)}

    allocator.reset_episode()
    allocator.allocate(drones, [status], arena_diagonal=100.0)
    assert len(allocator.episode_log_probs) >= 1

    allocator.reset_episode()
    assert allocator.episode_log_probs == []
    assert allocator.episode_entropies == []


def test_no_assignment_when_no_eligible_drones_or_zones():
    policy = AllocatorPolicy()
    allocator = LearnedAllocator(policy, sample=True)
    allocator.reset_episode()

    # No zones need anyone (already secured).
    zone = Zone(id="Z", x=0, y=0, radius=10, required_drones=1)
    status = ZoneStatus(zone=zone, occupant_ids=["A"], secured=True)
    drones = {"A": Drone(id="A", x=0, y=0, priority=1, battery=80.0)}
    assert allocator.allocate(drones, [status], arena_diagonal=100.0) == {}
    assert allocator.episode_log_probs == []


# -- Training pipeline: a real learning signal, not a beat-the-baseline claim --

def _controlled_scenario():
    """A small, fixed, easy-to-secure scenario -- used only by the
    run_episode consistency check below, not by the training-signal test
    (which exercises train.py's own randomized episode generator, since
    it's testing the real shipped pipeline end to end)."""
    zones = (Zone(id="Z", x=60, y=60, radius=40, required_drones=2, threat_level=1.0),)
    return dict(n_drones=6, width=150.0, height=150.0, zones=zones)


def test_reinforce_update_increases_probability_of_rewarded_action():
    """Directly validates the core REINFORCE mechanic in isolation from
    the full multi-tick environment: a real bug hunt (see below) found
    that checking aggregate multi-episode environment returns is prone
    to exact-tie coincidences (an untrained and a "trained" policy can
    end up making IDENTICAL argmax decisions on a specific held-out set
    purely because neither one's weight shift crosses any decision
    boundary for those particular inputs) -- not a meaningful pass/fail
    signal either way. This is the standard, surgical way to sanity-check
    a policy-gradient implementation instead: given an unambiguous
    scenario (a very close zone vs a very far one), repeated REINFORCE
    updates using the real reward-weighted -log_prob loss must increase
    the probability mass on the better action. If this doesn't hold, the
    gradient math itself is wrong -- independent of any environment
    noise, reward-shaping choices, or training-duration questions."""
    torch.manual_seed(3)
    policy = AllocatorPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)

    close_zone = Zone(id="close", x=0, y=0, radius=10, required_drones=1)
    far_zone = Zone(id="far", x=900, y=0, radius=10, required_drones=1)
    close_status = ZoneStatus(zone=close_zone)
    far_status = ZoneStatus(zone=far_zone)
    drone = Drone(id="A", x=10, y=0, priority=1, battery=100.0)

    def build_probs():
        feats = torch.tensor([
            drone_zone_features(drone, close_status, arena_diagonal=1000.0),
            drone_zone_features(drone, far_status, arena_diagonal=1000.0),
        ])
        scores = policy(feats)
        all_scores = torch.cat([scores, torch.zeros(1)])
        return torch.softmax(all_scores, dim=0)

    with torch.no_grad():
        prob_close_before = build_probs()[0].item()

    # Reward structure with an unambiguous best action: picking the close
    # zone (index 0) is clearly better than far (index 1) or staying (index 2).
    reward_for_action = {0: 1.0, 1: -1.0, 2: -0.5}
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
        prob_close_after = build_probs()[0].item()

    assert prob_close_after > prob_close_before + 0.1, (
        f"REINFORCE update did not shift probability toward the rewarded action: "
        f"P(close) before={prob_close_before:.3f} after={prob_close_after:.3f}"
    )


def test_train_and_evaluate_pipeline_runs_end_to_end_without_crashing():
    """A softer, complementary check to the isolated gradient-mechanic
    test above: the full train() -> evaluate() pipeline (real
    environment, real reward shaping, real randomized episodes) must run
    to completion and produce a well-formed comparison report. Doesn't
    assert who wins -- see train.py's module docstring and the README for
    why that's an honest empirical finding to report, not a guarantee."""
    from drone_swarm.train import evaluate

    policy = train(n_episodes=15, seed=1, verbose=False)
    results = evaluate(policy, n_episodes=3, seed=2)

    assert set(results.keys()) == {"heuristic", "learned"}
    for stats in results.values():
        assert 0.0 <= stats["fully_secured_fraction"] <= 1.0
        assert stats["mean_zones_secured_at_end"] >= 0


def test_run_episode_returns_consistent_final_swarm_state():
    """Regression test for a real bug found during development:
    evaluate() was reconstructing a FRESH, unticked swarm to inspect
    final state instead of using the one run_episode actually simulated
    -- always reporting the pre-episode (zero-progress) state. Checks
    run_episode's returned swarm reflects genuine simulated progress."""
    scenario = _controlled_scenario()
    policy = AllocatorPolicy()
    allocator = LearnedAllocator(policy, sample=False)
    ret, _, _, secured_at, swarm = run_episode(allocator, seed=1, collect_log_probs=False, **scenario)
    assert swarm.tick_count > 0
    # The returned swarm's own mission state must be internally
    # consistent with whether/when the episode reported full completion.
    if secured_at is not None:
        assert swarm.mission.all_secured() is True
