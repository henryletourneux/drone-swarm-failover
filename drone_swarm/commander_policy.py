"""Learned commander allocator -- Phase 3b of hierarchical command
(command.py / commander_allocator.py), trained via the same REINFORCE
approach train.py/policy.py already established for zone allocation.

Reuses `AllocatorPolicy` and `drone_zone_features` from policy.py
directly, rather than inventing a parallel architecture for an identical
problem: "which drone should fill this zone slot" is the same decision
shape at both tiers. The commander's guard-duty assignment genuinely IS
that problem, just triggered by the commander's own cadence
(`CommanderAllocatorConfig.reallocation_interval_ticks`) over the
commander's own candidate pool (drones not already committed to a zone
or investigating a disturbance), rather than the flat mission-level
nexus's periodic pass. A *fresh* policy instance is trained on it here,
not the mission-level one's weights reused as-is -- nothing here assumes
the zone-allocation policy's own training generalizes to this different
triggering cadence and candidate pool without actually being asked to.

Same two-phase discipline as mission.py/policy.py:
`HeuristicCommanderAllocator` (commander_allocator.py) is the honest
baseline; this is evaluated against it, not assumed better -- see
commander_train.py's `evaluate()`.
"""
from __future__ import annotations

import math

import torch

from .commander_allocator import CommanderAllocatorConfig
from .policy import AllocatorPolicy, drone_zone_features


class LearnedCommanderAllocator:
    """Drop-in replacement for HeuristicCommanderAllocator -- same
    `maybe_allocate(swarm) -> bool` contract, so it's a true swap for
    `SwarmConfig.commander_allocator`, not a parallel system.

    `sample=True` (training rollouts): each drone-to-zone-slot decision
    is sampled from the softmax over currently-available drones for exploration,
    with its log-prob appended to `episode_log_probs` for a REINFORCE
    trainer to read at episode end -- same explicit accumulate/reset
    lifecycle `LearnedAllocator` already established in policy.py, for
    the same reason (this allocator instance is reused across many
    training episodes; a stale snapshot would leak a freed autograd graph
    into the next episode's backward() call).

    `sample=False` (evaluation / live demo inference): greedy argmax, no
    randomness, no gradient needed.
    """

    def __init__(self, policy: AllocatorPolicy, config: CommanderAllocatorConfig | None = None, sample: bool = False):
        self.policy = policy
        self.config = config or CommanderAllocatorConfig()
        self.sample = sample
        self._ticks_since_reallocation = 0
        self.episode_log_probs: list = []
        self.episode_entropies: list = []

    def reset_episode(self) -> None:
        self.episode_log_probs = []
        self.episode_entropies = []

    def maybe_allocate(self, swarm) -> bool:
        self._ticks_since_reallocation += 1
        if self._ticks_since_reallocation < self.config.reallocation_interval_ticks:
            return False
        if swarm.command is None or not swarm.command.commander_ids():
            return False
        self._ticks_since_reallocation = 0
        self._allocate(swarm)
        return True

    def _allocate(self, swarm) -> None:
        command = swarm.command
        # Same "idle" definition and the same real bug this guards
        # against as HeuristicCommanderAllocator._allocate -- see that
        # method's docstring for the oscillation this mission_zone_id
        # check independently prevents here too.
        available = [
            swarm.drones[d_id] for d_id in sorted(swarm.drones)
            if swarm.drones[d_id].alive
            and swarm.drones[d_id].investigating_disturbance_id is None
            and swarm.drones[d_id].mission_zone_id is None
        ]
        if not available:
            return

        guard_needs = []  # [(zone_status, still_needed)]
        if swarm.mission is not None:
            for status in swarm.mission.zone_statuses.values():
                need = status.zone.required_drones - len(status.occupant_ids)
                if need > 0:
                    guard_needs.append((status, need))
        guard_needs.sort(key=lambda pair: pair[1], reverse=True)

        platoon_ids = list(command.platoon_ids)
        assignments: dict = {}  # drone_id -> (platoon_id, duty, zone_id | None)
        slot_index = 0
        arena_diagonal = math.hypot(swarm.width, swarm.height)

        grad_context = torch.enable_grad() if self.sample else torch.no_grad()
        with grad_context:
            for status, need in guard_needs:
                if slot_index >= len(platoon_ids) or not available:
                    break
                pid = platoon_ids[slot_index]
                slot_index += 1
                chosen = []
                for _ in range(min(need, len(available))):
                    features = [drone_zone_features(d, status, arena_diagonal) for d in available]
                    x = torch.tensor(features, dtype=torch.float32)
                    scores = self.policy(x)
                    probs = torch.softmax(scores, dim=0)

                    if self.sample:
                        dist = torch.distributions.Categorical(probs)
                        action = dist.sample()
                        self.episode_log_probs.append(dist.log_prob(action))
                        self.episode_entropies.append(dist.entropy())
                    else:
                        action = torch.argmax(probs)

                    chosen.append(available.pop(int(action.item())))
                for drone in chosen:
                    assignments[drone.id] = (pid, "guard", status.zone.id)

        remaining_slots = platoon_ids[slot_index:] or [platoon_ids[-1]]
        for i, drone in enumerate(available):
            pid = remaining_slots[i % len(remaining_slots)]
            assignments[drone.id] = (pid, "patrol", None)

        for d_id, (pid, duty, zone_id) in assignments.items():
            drone = swarm.drones[d_id]
            command.reassign_platoon(swarm, d_id, pid)
            drone.duty = duty
            if duty == "guard":
                drone.mission_zone_id = zone_id
            elif drone.mission_zone_id is not None:
                drone.mission_zone_id = None
