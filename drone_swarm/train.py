"""Trains `AllocatorPolicy` via REINFORCE (vanilla policy gradient) against
the actual zone-coverage mission environment (mission.py), then evaluates
it honestly against `HeuristicAllocator` on held-out configs. Standalone:

    python3 -m drone_swarm.train                 # train + evaluate + save
    python3 -m drone_swarm.train --evaluate-only  # load models/allocator_policy.pt and just compare

Training runs on small, fast-to-simulate swarms (see policy.py's module
docstring for why that's a legitimate choice, not a corner cut: the
network's features are agent-count-invariant, so a policy learned at
small scale is learning the same decision the 100-drone live demo needs).

Reward per episode: +10 the tick a zone first becomes secured, +20 bonus
for securing every zone, -5 the first time any drone's battery hits zero,
a small per-tick cost for each zone still unsecured (rewards speed), and
a small bonus for average battery remaining at episode end (rewards
efficiency, not just brute-forcing every zone with every drone). A
running average of past returns is used as the baseline for variance
reduction, in place of a learned critic -- simpler, and sufficient at
this problem size.
"""
from __future__ import annotations

import argparse
import random
import statistics

import torch

from .mission import HeuristicAllocator, MissionConfig, Zone
from .model import Drone
from .policy import AllocatorPolicy, LearnedAllocator, load_policy, save_policy
from .swarm import Swarm, SwarmConfig

MODEL_PATH = "models/allocator_policy.pt"

TRAIN_TICK_DT_S = 0.2
TRAIN_N_TICKS = 220
REALLOCATION_INTERVAL = 8


def _random_episode_config(rng: random.Random):
    """A fresh, randomized small-scale scenario each episode -- varied
    zone counts/positions/threat and swarm layout, so the policy learns
    something that generalizes rather than memorizing one map."""
    n_drones = rng.randint(8, 14)
    width, height = 500.0, 400.0
    n_zones = rng.randint(2, 3)
    zones = tuple(
        Zone(
            id=f"Z{i}",
            x=rng.uniform(60, width - 60),
            y=rng.uniform(60, height - 60),
            radius=40.0,
            required_drones=rng.randint(2, 4),
            threat_level=rng.uniform(0.0, 3.0),
        )
        for i in range(n_zones)
    )
    return n_drones, width, height, zones


def _build_swarm(allocator, n_drones, width, height, zones, seed) -> Swarm:
    config = SwarmConfig(
        nexus_heartbeat_interval_s=0.4, nexus_timeout_s=1.0, comm_latency_s=0.05,
        tick_dt_s=TRAIN_TICK_DT_S, packet_loss_rate=0.0,
        mission_config=MissionConfig(zones=zones, reallocation_interval_ticks=REALLOCATION_INTERVAL),
    )
    rng = random.Random(seed)
    drones = [
        Drone(id=f"D{i}", x=rng.uniform(0, width), y=rng.uniform(0, height), priority=rng.uniform(1, 100))
        for i in range(n_drones)
    ]
    swarm = Swarm(drones, comm_range=180.0, width=width, height=height, config=config, seed=seed)
    swarm.mission.allocator = allocator
    return swarm


def run_episode(allocator, n_drones, width, height, zones, seed, collect_log_probs: bool):
    """Runs one full episode, returns (total_reward, log_probs,
    fully_secured_at, swarm) -- the final swarm is returned too so
    callers (evaluate()) can inspect end-of-episode state without
    re-simulating from scratch. log_probs is only populated (and only
    meaningful) when `allocator` is a LearnedAllocator with sample=True;
    empty for evaluation/heuristic runs."""
    if collect_log_probs and hasattr(allocator, "reset_episode"):
        allocator.reset_episode()

    swarm = _build_swarm(allocator, n_drones, width, height, zones, seed)
    n_zones = len(zones)

    reward = 0.0
    prev_secured = 0
    battery_penalized: set = set()
    fully_secured_at = None

    for tick in range(TRAIN_N_TICKS):
        swarm.tick()

        secured = sum(1 for s in swarm.mission.zone_statuses.values() if s.secured)
        if secured > prev_secured:
            reward += 10.0 * (secured - prev_secured)
        prev_secured = secured
        reward -= 0.01 * (n_zones - secured)

        for d in swarm.drones.values():
            if d.battery <= 0.0 and d.id not in battery_penalized:
                reward -= 5.0
                battery_penalized.add(d.id)

        if fully_secured_at is None and swarm.mission.all_secured():
            fully_secured_at = tick
            reward += 20.0
            break

    avg_battery = sum(d.battery for d in swarm.drones.values()) / len(swarm.drones)
    reward += 0.05 * avg_battery

    if collect_log_probs and hasattr(allocator, "episode_log_probs"):
        log_probs = list(allocator.episode_log_probs)
        entropies = list(allocator.episode_entropies)
    else:
        log_probs, entropies = [], []
    return reward, log_probs, entropies, fully_secured_at, swarm


def train(n_episodes: int = 400, lr: float = 5e-3, entropy_coef: float = 0.02, seed: int = 0, verbose: bool = True) -> AllocatorPolicy:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = AllocatorPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    allocator = LearnedAllocator(policy, sample=True)

    baseline = 0.0
    baseline_momentum = 0.9
    returns_window: list = []

    for episode in range(n_episodes):
        n_drones, width, height, zones = _random_episode_config(rng)
        episode_seed = rng.randint(0, 2**31 - 1)
        ret, log_probs, entropies, _, _ = run_episode(allocator, n_drones, width, height, zones, episode_seed, collect_log_probs=True)

        if log_probs:
            advantage = ret - baseline
            policy_loss = -torch.stack(log_probs).sum() * advantage
            entropy_bonus = torch.stack(entropies).sum() if entropies else torch.tensor(0.0)
            loss = policy_loss - entropy_coef * entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * ret
        returns_window.append(ret)
        if len(returns_window) > 20:
            returns_window.pop(0)

        if verbose and (episode % 20 == 0 or episode == n_episodes - 1):
            print(f"episode {episode:4d}  return={ret:7.2f}  baseline={baseline:7.2f}  avg20={statistics.mean(returns_window):7.2f}")

    return policy


def evaluate(policy: AllocatorPolicy, n_episodes: int = 30, seed: int = 12345) -> dict:
    """Compares the trained policy (greedy, no exploration) against
    HeuristicAllocator on the SAME held-out set of episode configs
    (different seed range than training, so this isn't just replaying
    memorized maps). Reports honestly -- if the policy loses, that's
    what gets printed."""
    rng = random.Random(seed)
    configs = [(*_random_episode_config(rng), rng.randint(0, 2**31 - 1)) for _ in range(n_episodes)]

    results = {}
    for name, allocator_factory in (
        ("heuristic", lambda: HeuristicAllocator()),
        ("learned", lambda: LearnedAllocator(policy, sample=False)),
    ):
        secured_counts, times, battery_left, drones_lost = [], [], [], []
        for n_drones, width, height, zones, episode_seed in configs:
            allocator = allocator_factory()
            # Use the swarm run_episode already simulated -- rebuilding a
            # fresh one here would just be an unticked, all-zeros-progress
            # swarm, not the actual end-of-episode state.
            ret, _, _, secured_at, swarm = run_episode(allocator, n_drones, width, height, zones, episode_seed, collect_log_probs=False)
            secured_counts.append(sum(1 for s in swarm.mission.zone_statuses.values() if s.secured))
            times.append(secured_at if secured_at is not None else TRAIN_N_TICKS)
            battery_left.append(sum(d.battery for d in swarm.drones.values()) / len(swarm.drones))
            drones_lost.append(sum(1 for d in swarm.drones.values() if d.battery <= 0.0))
        results[name] = {
            "mean_zones_secured_at_end": statistics.mean(secured_counts),
            "mean_ticks_to_fully_secure_or_budget": statistics.mean(times),
            "fully_secured_fraction": sum(1 for t in times if t < TRAIN_N_TICKS) / len(times),
            "mean_battery_remaining": statistics.mean(battery_left),
            "mean_drones_depleted": statistics.mean(drones_lost),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the mission allocation policy")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()

    import os
    os.makedirs("models", exist_ok=True)

    if args.evaluate_only:
        policy = load_policy(MODEL_PATH)
    else:
        print(f"Training for {args.episodes} episodes...")
        policy = train(n_episodes=args.episodes)
        save_policy(policy, MODEL_PATH)
        print(f"Saved trained policy to {MODEL_PATH}")

    print(f"\nEvaluating over {args.eval_episodes} held-out episodes...")
    results = evaluate(policy, n_episodes=args.eval_episodes)
    print("\n" + "=" * 60)
    print("  LEARNED vs HEURISTIC -- held-out evaluation")
    print("=" * 60)
    for name, stats in results.items():
        print(f"  {name:10s}  zones_secured={stats['mean_zones_secured_at_end']:.2f}  "
              f"ticks_to_secure={stats['mean_ticks_to_fully_secure_or_budget']:.1f}  "
              f"fully_secured_rate={stats['fully_secured_fraction']:.0%}  "
              f"avg_battery_left={stats['mean_battery_remaining']:.1f}  "
              f"avg_drones_depleted={stats['mean_drones_depleted']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
