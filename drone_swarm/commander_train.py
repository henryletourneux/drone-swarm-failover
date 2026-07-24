"""Trains commander_policy.py's LearnedCommanderAllocator via REINFORCE
against the actual hierarchical-command + mission environment (command.py
+ commander_allocator.py + mission.py), then evaluates it honestly
against HeuristicCommanderAllocator on held-out configs. Standalone,
mirroring train.py's own CLI exactly:

    python3 -m drone_swarm.commander_train                 # train + evaluate + save
    python3 -m drone_swarm.commander_train --evaluate-only  # load models/commander_policy.pt and just compare

Training runs on small, fast-to-simulate swarms, same legitimacy argument
as train.py's own choice (policy.py's features are agent-count-invariant,
so a policy learned at small scale is learning the same decision the
100-drone live demo needs) -- reinforced by the fact that this reuses
policy.py's own AllocatorPolicy/drone_zone_features verbatim (see
commander_policy.py's module docstring for why that's honest reuse, not
padding).

The mission's OWN periodic reallocation (mission.py's HeuristicAllocator,
which MissionState always has SOME instance of) is deliberately disabled
during training (reallocation_interval_ticks set far longer than any
episode) so zone coverage can only ever come from the commander
allocator under training -- otherwise the mission-level allocator could
quietly secure zones on its own, muddying which system actually deserves
credit for the outcome the reward is measuring.

Reward shape deliberately mirrors train.py's own (same underlying "get
zones secured, don't deplete batteries" problem, now decided by a
commander instead of a flat nexus): +10 the tick a zone first becomes
secured, +20 bonus for securing every zone, -5 the first time any drone's
battery hits zero, a small per-tick cost for each zone still unsecured, a
small bonus for average battery remaining at episode end. Running-average
baseline for variance reduction, same as train.py.
"""
from __future__ import annotations

import argparse
import random
import statistics

import torch

from .command import CommandConfig
from .commander_allocator import CommanderAllocatorConfig, HeuristicCommanderAllocator
from .commander_policy import LearnedCommanderAllocator
from .mission import MissionConfig, Zone
from .model import Drone
from .policy import AllocatorPolicy, load_policy, save_policy
from .swarm import Swarm, SwarmConfig

MODEL_PATH = "models/commander_policy.pt"

TRAIN_TICK_DT_S = 0.2
TRAIN_N_TICKS = 260  # generous enough for platoon+commander convergence AND several reallocation passes after it
ALLOCATOR_REALLOCATION_INTERVAL = 8
MISSION_REALLOCATION_INTERVAL = 100_000  # effectively "never", see module docstring


def _random_episode_config(rng: random.Random):
    """A fresh, randomized small-scale scenario each episode -- varied
    zone counts/positions/threat, swarm layout, AND platoon sizing, so
    the policy learns something that generalizes rather than memorizing
    one map or one platoon shape."""
    n_drones = rng.randint(9, 16)
    width, height = 500.0, 400.0
    n_zones = rng.randint(1, 2)
    zones = tuple(
        Zone(
            id=f"Z{i}",
            x=rng.uniform(60, width - 60),
            y=rng.uniform(60, height - 60),
            radius=35.0,
            required_drones=rng.randint(2, 4),
            threat_level=rng.uniform(0.0, 3.0),
        )
        for i in range(n_zones)
    )
    platoon_size = rng.randint(2, 4)
    return n_drones, width, height, zones, platoon_size


def _build_swarm(commander_allocator, n_drones, width, height, zones, platoon_size, seed) -> Swarm:
    platoon_of = {f"D{i}": f"P{i // platoon_size}" for i in range(n_drones)}
    config = SwarmConfig(
        nexus_heartbeat_interval_s=0.4, nexus_timeout_s=1.0, comm_latency_s=0.05,
        tick_dt_s=TRAIN_TICK_DT_S, packet_loss_rate=0.0,
        mission_config=MissionConfig(zones=zones, reallocation_interval_ticks=MISSION_REALLOCATION_INTERVAL),
        command_config=CommandConfig(platoon_of=platoon_of),
        commander_allocator=commander_allocator,
    )
    rng = random.Random(seed)
    drones = [
        Drone(id=f"D{i}", x=rng.uniform(0, width), y=rng.uniform(0, height), priority=rng.uniform(1, 100))
        for i in range(n_drones)
    ]
    return Swarm(drones, comm_range=220.0, width=width, height=height, config=config, seed=seed)


def run_episode(commander_allocator, n_drones, width, height, zones, platoon_size, seed, collect_log_probs: bool):
    """Runs one full episode, returns (total_reward, log_probs, entropies,
    fully_secured_at, swarm) -- mirrors train.py's run_episode exactly in
    shape, for the same reasons (the final swarm returned too, so
    evaluate() can inspect end-of-episode state without re-simulating)."""
    if collect_log_probs and hasattr(commander_allocator, "reset_episode"):
        commander_allocator.reset_episode()

    swarm = _build_swarm(commander_allocator, n_drones, width, height, zones, platoon_size, seed)
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

    if collect_log_probs and hasattr(commander_allocator, "episode_log_probs"):
        log_probs = list(commander_allocator.episode_log_probs)
        entropies = list(commander_allocator.episode_entropies)
    else:
        log_probs, entropies = [], []
    return reward, log_probs, entropies, fully_secured_at, swarm


def train(n_episodes: int = 400, lr: float = 5e-3, entropy_coef: float = 0.02, seed: int = 0, verbose: bool = True) -> AllocatorPolicy:
    torch.manual_seed(seed)
    rng = random.Random(seed)

    policy = AllocatorPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    allocator_config = CommanderAllocatorConfig(reallocation_interval_ticks=ALLOCATOR_REALLOCATION_INTERVAL)
    commander_allocator = LearnedCommanderAllocator(policy, allocator_config, sample=True)

    baseline = 0.0
    baseline_momentum = 0.9
    returns_window: list = []

    for episode in range(n_episodes):
        n_drones, width, height, zones, platoon_size = _random_episode_config(rng)
        episode_seed = rng.randint(0, 2**31 - 1)
        ret, log_probs, entropies, _, _ = run_episode(
            commander_allocator, n_drones, width, height, zones, platoon_size, episode_seed, collect_log_probs=True,
        )

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
    HeuristicCommanderAllocator on the SAME held-out set of episode
    configs (different seed range than training). Reports honestly --
    if the policy loses, that's what gets printed, same discipline as
    train.py's own evaluate()."""
    rng = random.Random(seed)
    configs = [(*_random_episode_config(rng), rng.randint(0, 2**31 - 1)) for _ in range(n_episodes)]

    results = {}
    for name, allocator_factory in (
        ("heuristic", lambda: HeuristicCommanderAllocator(CommanderAllocatorConfig(reallocation_interval_ticks=ALLOCATOR_REALLOCATION_INTERVAL))),
        ("learned", lambda: LearnedCommanderAllocator(policy, CommanderAllocatorConfig(reallocation_interval_ticks=ALLOCATOR_REALLOCATION_INTERVAL), sample=False)),
    ):
        secured_counts, times, battery_left, drones_lost = [], [], [], []
        for n_drones, width, height, zones, platoon_size, episode_seed in configs:
            allocator = allocator_factory()
            ret, _, _, secured_at, swarm = run_episode(
                allocator, n_drones, width, height, zones, platoon_size, episode_seed, collect_log_probs=False,
            )
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
    parser = argparse.ArgumentParser(description="Train/evaluate the commander guard/patrol allocation policy")
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
    print("  LEARNED vs HEURISTIC (commander allocator) -- held-out evaluation")
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
