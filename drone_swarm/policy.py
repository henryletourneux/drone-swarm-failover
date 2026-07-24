"""A learned resource-allocation policy -- Phase 2 of mission.py, trained
against Phase 1's environment rather than hand-tuned like
`HeuristicAllocator`.

Design choice worth stating plainly: the network scores a single
(drone, zone) pair at a time from purely *relative* features (battery
fraction, distance as a fraction of the arena diagonal, normalized
threat/need, role, secured status) -- never absolute swarm size or drone
count. That makes the input space agent-count-invariant: a policy
trained on a small, fast-to-simulate swarm (see train.py) is making the
same kind of decision a 100-drone deployment needs, not a smaller
version of a different problem. `tests/test_policy.py` includes an
explicit check of this transfer property, not just an assumption.

`LearnedAllocator` implements the exact same `.allocate(drones,
zone_statuses, arena_diagonal) -> dict` interface as
`HeuristicAllocator` -- a true drop-in swap for `MissionState.allocator`,
not a parallel system. Whether it's actually any *good* is a question
for train.py's evaluation, not something asserted here.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .mission import MIN_BATTERY_TO_ASSIGN

FEATURE_NAMES = (
    "battery_frac", "distance_frac", "threat_frac", "need_frac",
    "is_relay", "is_leaf", "is_nexus", "zone_secured",
)
FEATURE_DIM = len(FEATURE_NAMES)
THREAT_NORMALIZER = 5.0  # threat_level values in this project run roughly 0-3; keeps the feature in a sane range without hard-clipping most real configs


def drone_zone_features(drone, zone_status, arena_diagonal: float) -> list:
    zone = zone_status.zone
    distance = math.hypot(drone.x - zone.x, drone.y - zone.y)
    still_needed = max(0, zone.required_drones - len(zone_status.occupant_ids))
    return [
        drone.battery / 100.0,
        min(1.0, distance / max(arena_diagonal, 1.0)),
        min(1.0, zone.threat_level / THREAT_NORMALIZER),
        still_needed / max(zone.required_drones, 1),
        1.0 if drone.role == "relay" else 0.0,
        1.0 if drone.role == "leaf" else 0.0,
        1.0 if drone.role == "nexus" else 0.0,
        1.0 if zone_status.secured else 0.0,
    ]


class AllocatorPolicy(nn.Module):
    """Small MLP: one (drone, zone) feature vector in, one scalar
    affinity score out. Deliberately small -- this is a low-dimensional
    decision problem, not an image/language task, and a small network
    trains faster and is easier to sanity-check than an oversized one
    would be."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LearnedAllocator:
    """Drop-in replacement for HeuristicAllocator, backed by a (trained
    or untrained) AllocatorPolicy.

    `sample=True` (training rollouts): actions are sampled from the
    softmax distribution over {each still-needed zone, stay-idle}, for
    exploration, and each decision's log-prob is appended to
    `episode_log_probs` for a REINFORCE-style trainer to read at episode
    end. Deliberately an explicit accumulate/reset lifecycle rather than
    a "last call's decisions" snapshot: `allocate()` isn't called every
    tick (only on reallocation ticks), so a snapshot would silently go
    stale between calls -- and since one allocator instance gets reused
    across many training episodes, a stale snapshot read at the start of
    episode N+1 would still be holding onto episode N's already-freed
    autograd graph, which crashes on the next backward() with "trying to
    backward through the graph a second time." Call `reset_episode()`
    once at the start of each rollout.

    `sample=False` (evaluation / live demo inference): greedy argmax, no
    randomness, no gradient needed, episode_log_probs untouched.
    """

    def __init__(self, policy: AllocatorPolicy, sample: bool = False):
        self.policy = policy
        self.sample = sample
        self.episode_log_probs: list = []
        self.episode_entropies: list = []  # for an entropy bonus during training -- encourages continued exploration rather than collapsing to a deterministic policy too early

    def reset_episode(self) -> None:
        self.episode_log_probs = []
        self.episode_entropies = []

    def allocate(self, drones: dict, zone_statuses: list, arena_diagonal: float) -> dict:
        committed = {d_id for s in zone_statuses for d_id in s.occupant_ids}
        eligible = [
            d for d in drones.values()
            if d.alive and d.battery > MIN_BATTERY_TO_ASSIGN and d.id not in committed
        ]
        needy = [s for s in zone_statuses if not s.secured]
        assignments: dict = {}
        if not eligible or not needy:
            return assignments

        remaining_need = {s.zone.id: max(0, s.zone.required_drones - len(s.occupant_ids)) for s in needy}

        grad_context = torch.enable_grad() if self.sample else torch.no_grad()
        with grad_context:
            for drone in eligible:
                candidates = [s for s in needy if remaining_need[s.zone.id] > 0]
                if not candidates:
                    break
                features = [drone_zone_features(drone, s, arena_diagonal) for s in candidates]
                x = torch.tensor(features, dtype=torch.float32)
                zone_scores = self.policy(x)
                stay_score = torch.zeros(1)
                all_scores = torch.cat([zone_scores, stay_score])
                probs = torch.softmax(all_scores, dim=0)

                if self.sample:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    self.episode_log_probs.append(dist.log_prob(action))
                    self.episode_entropies.append(dist.entropy())
                else:
                    action = torch.argmax(probs)

                action_idx = int(action.item())
                if action_idx < len(candidates):
                    zone = candidates[action_idx].zone
                    assignments[drone.id] = zone.id
                    remaining_need[zone.id] -= 1
                # else: policy chose "stay idle" for this drone

        return assignments


def save_policy(policy: AllocatorPolicy, path: str) -> None:
    torch.save(policy.state_dict(), path)


def load_policy(path: str) -> AllocatorPolicy:
    policy = AllocatorPolicy()
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy
