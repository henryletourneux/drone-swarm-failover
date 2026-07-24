"""Patrol / disturbance investigation: dynamically-spawned "disturbance"
sites that idle patrol drones break off to investigate -- built directly
on top of the existing zone-coverage mission machinery (mission.py) rather
than as a parallel system, per the project's own roadmap note that this
was "a natural extension of the existing zone/mission machinery."

A disturbance is picked up from the swarm's genuinely idle/uncommitted
drones -- alive, not currently occupying a zone, not already investigating
something else -- the exact same reserve-pool notion `HeuristicAllocator.
plan_substitutions()` already uses for battery-substitution relief.

An earlier version of this dispatched from a zone's "surplus" occupants
(beyond `required_drones`) instead. That was a real design bug, not just
a naming choice: `HeuristicAllocator.allocate()` never assigns more than a
zone's `required_drones` to begin with, so genuine surplus essentially
never occurs in practice outside of incidental starting-position overlap
-- meaning disturbances would spawn and simply never get investigated
under any realistic mission configuration. Caught by
`test_investigation_requires_arrival_before_accruing_progress` and
`test_resolved_disturbance_frees_investigator_into_reserve_pool`, both of
which failed against the surplus-based version and pass against this one.
Pulling from the idle pool instead is also strictly safer: it can never
pull a drone off a zone that needs it, full stop, rather than relying on
a surplus computation to stay non-negative.

Once dispatched, the investigating drone is simply excluded from the idle
pool (via `investigating_disturbance_id`) until it resolves -- no
`mission_zone_id` bookkeeping needed on the way out, since idle drones
never had one set. On resolution it's freed back to plain idle (`
investigating_disturbance_id = None`), and ordinary `MissionState`
allocation/substitution picks it up wherever it's next needed, the same
"release it, the existing machinery re-absorbs it" idiom mission.py
already established for battery substitution.

This module is a second additive layer stacked on `MissionState`, same
principle as `bft_mode` and `mission_config` before it: `SwarmConfig.
patrol_config = None` (default) makes it entirely inert. It also has a
soft dependency on `mission_config` being set too -- without an active
mission there's no `zone_statuses` to compute "committed" from, so
`PatrolState.tick()` simply spawns and ages disturbances but never
assigns an investigator; nothing breaks, it just never resolves anything.

Honest limitation, stated plainly: dispatch is greedy and single-pass, not
globally optimal -- each unresolved disturbance (oldest first) claims the
single nearest currently-idle drone, one at a time. Two disturbances that
spawn on opposite sides of the arena in the same tick can each grab a
suboptimal drone if a truly optimal assignment would have crossed them --
the same honest tradeoff `HeuristicAllocator` already makes for zone
assignment (greedy and explainable over globally optimal).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .mission import MIN_BATTERY_TO_ASSIGN


@dataclass
class Disturbance:
    id: str
    x: float
    y: float
    spawned_tick: int
    investigator_id: str | None = None
    ticks_investigated: int = 0
    resolved: bool = False
    resolved_at_tick: int | None = None


@dataclass(frozen=True)
class PatrolConfig:
    spawn_interval_ticks: int = 40
    max_active_disturbances: int = 2
    investigation_range: float = 30.0
    investigation_ticks_required: int = 12
    # Cosmetic margin so a spawned disturbance never renders flush against
    # (or past) the arena edge.
    spawn_margin: float = 40.0
    # How many ticks a resolved disturbance stays in to_state_dict() after
    # resolving, purely so the live frontend has a moment to show a
    # "resolved" state before it disappears, rather than vanishing the
    # instant the last investigation tick lands.
    resolved_display_ticks: int = 8


class PatrolState:
    """Owns disturbance spawn/dispatch/investigate/resolve bookkeeping for
    one swarm's mission. Call `.tick(swarm, tick)` once per `Swarm.tick()`,
    after `MissionState.tick()` has updated zone occupancy for this tick --
    dispatch reads `swarm.mission.zone_statuses` and needs it current."""

    def __init__(self, config: PatrolConfig, rng) -> None:
        self.config = config
        self._rng = rng
        self._ticks_since_spawn = 0
        self._next_id = 0
        self.disturbances: dict[str, Disturbance] = {}

    def _new_id(self) -> str:
        disturbance_id = f"disturbance-{self._next_id}"
        self._next_id += 1
        return disturbance_id

    def _spawn(self, swarm, tick: int) -> str | None:
        active = sum(1 for d in self.disturbances.values() if not d.resolved)
        if active >= self.config.max_active_disturbances:
            return None
        m = self.config.spawn_margin
        x = self._rng.uniform(m, max(m, swarm.width - m))
        y = self._rng.uniform(m, max(m, swarm.height - m))
        disturbance_id = self._new_id()
        self.disturbances[disturbance_id] = Disturbance(id=disturbance_id, x=x, y=y, spawned_tick=tick)
        return disturbance_id

    def add_disturbance(self, x: float, y: float, tick: int) -> str:
        """User-placed disturbance (e.g. a live-demo click), as opposed to
        the ambient auto-spawn above. Deliberately bypasses
        `max_active_disturbances` -- that cap exists to keep unattended,
        randomly-timed spawns from cluttering the arena over time, not to
        limit an explicit, one-at-a-time user action, the same way
        `_launch_random_attack` in server.py always fires regardless of
        ambient swarm state."""
        disturbance_id = self._new_id()
        self.disturbances[disturbance_id] = Disturbance(id=disturbance_id, x=x, y=y, spawned_tick=tick)
        return disturbance_id

    def _dispatch(self, swarm) -> list:
        """Sends the nearest idle drone after each unassigned disturbance,
        oldest disturbance first. "Idle" is exactly mission.py's own
        reserve-pool definition (see module docstring for why this isn't
        zone-surplus): alive, not currently occupying any zone, not
        already investigating something else."""
        mission = swarm.mission
        if mission is None:
            return []
        unassigned = sorted(
            (d for d in self.disturbances.values() if not d.resolved and d.investigator_id is None),
            key=lambda d: d.spawned_tick,
        )
        if not unassigned:
            return []

        committed = {
            drone_id
            for status in mission.zone_statuses.values()
            for drone_id in status.occupant_ids
        }
        idle = [
            d for d in swarm.drones.values()
            if d.alive and d.id not in committed and d.mission_zone_id is None
            and d.investigating_disturbance_id is None
            and d.battery > MIN_BATTERY_TO_ASSIGN
        ]

        events = []
        for disturbance in unassigned:
            if not idle:
                break
            nearest = min(idle, key=lambda d: math.hypot(d.x - disturbance.x, d.y - disturbance.y))
            idle.remove(nearest)
            nearest.investigating_disturbance_id = disturbance.id
            disturbance.investigator_id = nearest.id
            events.append({"disturbance_id": disturbance.id, "drone_id": nearest.id, "kind": "dispatched"})
        return events

    def _advance(self, swarm, tick: int) -> list:
        events = []
        for disturbance in self.disturbances.values():
            if disturbance.resolved or disturbance.investigator_id is None:
                continue
            drone = swarm.drones.get(disturbance.investigator_id)
            if drone is None or not drone.alive:
                # Investigator died mid-investigation: the disturbance goes
                # back on the market for the next spare drone, same as any
                # other abandoned assignment in this codebase (mirrors
                # MissionState.tick()'s own dead-drone handling).
                if drone is not None:
                    drone.investigating_disturbance_id = None
                disturbance.investigator_id = None
                disturbance.ticks_investigated = 0
                continue
            distance = math.hypot(drone.x - disturbance.x, drone.y - disturbance.y)
            if distance > self.config.investigation_range:
                continue  # still travelling in
            disturbance.ticks_investigated += 1
            if disturbance.ticks_investigated >= self.config.investigation_ticks_required:
                disturbance.resolved = True
                disturbance.resolved_at_tick = tick
                drone.investigating_disturbance_id = None
                events.append({"disturbance_id": disturbance.id, "drone_id": drone.id, "kind": "resolved"})
        return events

    def _prune_resolved(self, tick: int) -> None:
        to_drop = [
            d_id for d_id, d in self.disturbances.items()
            if d.resolved and tick - d.resolved_at_tick > self.config.resolved_display_ticks
        ]
        for d_id in to_drop:
            del self.disturbances[d_id]

    def tick(self, swarm, tick: int) -> list:
        """Advances spawn/dispatch/investigate/resolve bookkeeping for one
        tick. Returns a list of {"disturbance_id", "drone_id"?, "kind"} for
        the caller to log/narrate ("spawned" | "dispatched" | "resolved"),
        empty most ticks."""
        events = []
        self._prune_resolved(tick)

        self._ticks_since_spawn += 1
        if self._ticks_since_spawn >= self.config.spawn_interval_ticks:
            self._ticks_since_spawn = 0
            spawned_id = self._spawn(swarm, tick)
            if spawned_id is not None:
                events.append({"disturbance_id": spawned_id, "kind": "spawned"})

        events.extend(self._dispatch(swarm))
        events.extend(self._advance(swarm, tick))
        return events

    def to_state_dict(self) -> dict:
        return {
            "disturbances": [
                {
                    "id": d.id,
                    "x": d.x,
                    "y": d.y,
                    "investigator_id": d.investigator_id,
                    "progress": min(1.0, d.ticks_investigated / self.config.investigation_ticks_required),
                    "resolved": d.resolved,
                }
                for d in self.disturbances.values()
            ],
        }
