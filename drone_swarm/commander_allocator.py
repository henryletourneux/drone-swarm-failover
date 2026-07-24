"""Heuristic commander allocator: decides, periodically, how alive drones
should be grouped into platoons and what each drone's DUTY is -- "guard"
(hold a specific under-covered mission zone) or "patrol" (join the shared
flocking/patrol-route population, see flocking.py/patrol.py). This is the
"structure and allocate members to platoons to patrol, stand guard"
half of hierarchical command (drone_swarm/command.py) that a static,
config-assigned `platoon_of` alone can't provide.

Deliberately mirrors `mission.py`'s `HeuristicAllocator` in spirit -- the
honest, explainable, non-learned baseline a future learned commander
policy needs to clear to be worth using at all, the same two-phase
pattern `mission.py`/`policy.py` already established for zone allocation
(Phase 1 heuristic baseline, Phase 2 a trained policy evaluated against
it). Same "omniscient call made by the simulation" honest simplification
`mission.py`'s own allocator already uses, too: this reads swarm state
directly rather than routing decisions through the mesh's own
partial-information message-passing.

Design: platoon SLOTS are fixed (`CommandState.platoon_ids` never grows
or shrinks here -- only which drones sit in which slot changes, via
`CommandState.reassign_platoon`). One slot per still-undersupplied
mission zone gets filled with exactly the nearest drones needed to
secure it ("guard" duty, tied to that zone); every remaining drone gets
"patrol" duty and is spread evenly across whatever slots are left --
patrol duty doesn't care which platoon number it's nominally in, only
guard duty ties a platoon's identity to a specific zone. A drone
currently investigating a disturbance is left alone entirely (mirrors
`patrol.py`'s own dispatch excluding it from the idle pool for the same
reason: pulling it off an active investigation to restructure platoons
around it would be actively counterproductive).

Deliberately restricted to plain (non-BFT) mode -- see command.py's
module docstring ("Dynamic platoon membership") for why. `server.py`
never constructs one of these for a `bft_mode` config.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CommanderAllocatorConfig:
    reallocation_interval_ticks: int = 30


class HeuristicCommanderAllocator:
    def __init__(self, config: CommanderAllocatorConfig | None = None) -> None:
        self.config = config or CommanderAllocatorConfig()
        self._ticks_since_reallocation = 0

    def maybe_allocate(self, swarm) -> bool:
        """Runs at most once every `reallocation_interval_ticks`, and
        only once at least one commander is elected (an unelected swarm
        has nobody to attribute the decision to -- mirrors mission.py's
        own "allocation is driven by the currently-elected nexus" idiom,
        one tier up). Returns whether it actually ran, mostly so tests
        don't have to reverse-engineer the tick cadence."""
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
        # Same "idle" definition mission.py's plan_substitutions and
        # patrol.py's dispatch already use: alive, not investigating, not
        # currently committed to a zone. That last check matters here
        # specifically -- a real bug during development omitted it, and a
        # drone already successfully guarding a now-secured zone got
        # scooped back into the reallocation pool on the very next pass
        # (that zone no longer appears in guard_needs below, precisely
        # BECAUSE this drone is the one filling it), got reassigned to
        # "patrol," and walked away -- un-securing the zone it had just
        # secured, which then re-triggered guard duty next pass:
        # perpetual oscillation, never a stable secured state. Caught by
        # test_end_to_end_zones_get_guarded_and_the_rest_go_on_patrol,
        # independently re-verified by removing the mission_zone_id
        # check and confirming that test fails before restoring it.
        available = [
            swarm.drones[d_id] for d_id in sorted(swarm.drones)
            if swarm.drones[d_id].alive
            and swarm.drones[d_id].investigating_disturbance_id is None
            and swarm.drones[d_id].mission_zone_id is None
        ]
        if not available:
            return

        guard_needs = []  # [(zone, still_needed)]
        if swarm.mission is not None:
            for status in swarm.mission.zone_statuses.values():
                need = status.zone.required_drones - len(status.occupant_ids)
                if need > 0:
                    guard_needs.append((status.zone, need))
        # Most-undersupplied first -- same "most urgent gap first" idea
        # as HeuristicAllocator.allocate's own most-threatened-first
        # ordering, so a bigger shortfall doesn't lose out to a smaller
        # one just because platoon slots ran out first.
        guard_needs.sort(key=lambda pair: pair[1], reverse=True)

        platoon_ids = list(command.platoon_ids)
        assignments: dict = {}  # drone_id -> (platoon_id, duty, zone_id | None)
        slot_index = 0
        for zone, need in guard_needs:
            if slot_index >= len(platoon_ids) or not available:
                break
            pid = platoon_ids[slot_index]
            slot_index += 1
            # Nearest-to-the-zone first, same distance-weighted spirit as
            # HeuristicAllocator.score -- minimizes travel, not just
            # "whichever drone id sorts first."
            available.sort(key=lambda d: math.hypot(d.x - zone.x, d.y - zone.y))
            chosen, available = available[:need], available[need:]
            for drone in chosen:
                assignments[drone.id] = (pid, "guard", zone.id)

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
                # Pulled off guard duty (or held a stray assignment some
                # other way) -- free it back to idle so it can actually
                # go patrol, rather than staying stuck holding a job
                # that's no longer its own.
                drone.mission_zone_id = None
