"""Patrol / disturbance investigation: dynamically-spawned "disturbance"
sites that idle patrol drones break off to investigate -- built directly
on top of the existing zone-coverage mission machinery (mission.py) rather
than as a parallel system, per the project's own roadmap note that this
was "a natural extension of the existing zone/mission machinery."

A disturbance is picked up from the swarm's genuinely idle/uncommitted
drones -- alive, not currently occupying a zone, not already investigating
something else -- the exact same reserve-pool notion `HeuristicAllocator.
plan_substitutions()` already uses for battery-substitution relief.

Each disturbance has a random `severity` (1-3), and resolving it takes
`severity` drones' worth of *combined* presence, not just any one drone
showing up: `_advance` accrues `min(currently-present-investigators,
severity)` effort per tick against a `severity * investigation_ticks_
required` total, so under-resourcing it (fewer investigators than
severity calls for) genuinely slows resolution down rather than being
free, and over-resourcing it past severity doesn't speed things up
further -- this is the "properly allocating resources" half of the
response, not just "someone eventually looks into it." `_dispatch`
mirrors `HeuristicAllocator.allocate()`'s own shape almost exactly:
keep assigning the nearest still-idle drone to whichever disturbance
needs it most (most severe, then longest-waiting) until every
disturbance's headcount is met or idle drones run out.

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

## Patrol route

This module also owns *where idle drones actually go* while they wait for
something to investigate, rather than leaving them to drift aimlessly
(`Swarm._move`'s old default) or freeze in place. Without hierarchical
command (no `swarm.command`), `patrol_target(swarm)` returns a single
shared destination -- the current waypoint on a ring of points auto-
generated inset from the arena edges (`_ring_waypoints`), so the whole
idle population tours the arena's perimeter together as one flock
(`flocking.py` does the actual steering; this only supplies the
destination). A route advances to its next waypoint once the relevant
population's own centroid arrives, not any single drone -- so a straggler
at the back doesn't yank the target away from drones still converging on
the current one. `route_enabled` is a plain mutable flag (not part of the
frozen `PatrolConfig`) so it can be toggled live from the running demo,
same reasoning as `FlockingConfig` not being frozen (see flocking.py).

## Per-platoon patrol routes

With hierarchical command active, `patrol_target(swarm, platoon_id=...)`
gives each platoon its OWN small loop -- circling whichever defendable
mission zone it's nearest to in the platoon-id ordering (round-robin if
there are more platoons than zones), sized just outside that zone's own
radius (`zone_patrol_margin`), rather than the whole idle population
sharing one arena-spanning ring. This was a real, live-observed problem
with the single-shared-route design, not a hypothetical: a hundred
drones all converging on ONE distant point at a time meant the "flock"
routinely spanned nearly the entire arena width (measured live: >1500
units wide in a 2000-unit arena, comm_range 210) rather than moving as a
cohesive group -- reading as one shapeless blob rather than an organized
patrol, AND scattering same-platoon drones far enough apart that they'd
lose radio contact with each other entirely, fragmenting a single
platoon's election into several simultaneous nexuses (the live demo's
"too many nexuses, no relays" symptom traced directly back to this).
Small, zone-anchored loops keep each platoon's own patrol population
physically close together -- both a better read as an actual patrol
pattern and healthier for the platoon's own mesh connectivity. Falls back
to the same shared-ring behavior above if there's no mission (no zones to
anchor to) or no `swarm.command` at all, so `patrol_config` alone (no
`command_config`) is completely unaffected by this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .mission import MIN_BATTERY_TO_ASSIGN
from .obstacles import push_point_outside_all

MIN_SEVERITY = 1
MAX_SEVERITY = 3


@dataclass
class Disturbance:
    id: str
    x: float
    y: float
    spawned_tick: int
    severity: int = 1
    investigator_ids: list = field(default_factory=list)
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

    # Patrol route (see "Patrol route" in the module docstring).
    route_waypoint_count: int = 8
    route_edge_margin: float = 120.0
    route_arrival_radius: float = 90.0
    # Per-platoon patrol loops (see "Per-platoon patrol routes" below):
    # how far outside a defendable zone's own radius the loop circling it
    # sits, and how many points that loop has (smaller than the
    # arena-perimeter route above -- a tight loop around one zone doesn't
    # need as many points to read as a real patrol pattern).
    zone_patrol_margin: float = 70.0
    zone_patrol_waypoint_count: int = 5


def _ring_waypoints(width: float, height: float, count: int, margin: float) -> list:
    """An elliptical ring of `count` points inset `margin` from the arena
    edges -- auto-derived from arena size rather than hardcoded, so a
    bigger arena (more room to patrol) just produces a bigger ring for
    free, no separate per-mode tuning needed."""
    cx, cy = width / 2.0, height / 2.0
    rx = max(1.0, width / 2.0 - margin)
    ry = max(1.0, height / 2.0 - margin)
    return [
        (cx + rx * math.cos(2 * math.pi * i / count), cy + ry * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ]


def platoon_zone_anchor(platoon_ids, zones: list, platoon_id: str):
    """The zone a given platoon patrols around, via a stable round-robin
    index%zones mapping (more platoons than zones is the common case) --
    factored out here (rather than left inline in `_ensure_platoon_route`)
    so `commander_allocator.py` can use the exact same mapping to send a
    newly-available patrol drone to the slot that's actually nearest it,
    not just whichever slot has the fewest members (see that module's own
    "real bug found live" note for why that distinction matters). None if
    there are no zones to anchor to."""
    if not zones:
        return None
    sorted_platoon_ids = sorted(platoon_ids)
    sorted_zones = sorted(zones, key=lambda z: z.id)
    index = sorted_platoon_ids.index(platoon_id) if platoon_id in sorted_platoon_ids else 0
    return sorted_zones[index % len(sorted_zones)]


def _zone_loop_waypoints(zone, count: int, margin: float) -> list:
    """A small ring of `count` points circling `zone` at `margin` outside
    its own radius -- see "Per-platoon patrol routes" in the module
    docstring for why an anchored, tight loop replaces one arena-spanning
    shared ring."""
    r = zone.radius + margin
    return [
        (zone.x + r * math.cos(2 * math.pi * i / count), zone.y + r * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ]


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
        # Route waypoints need arena dimensions, which this class doesn't
        # have at construction time (only Swarm does) -- populated lazily
        # on first tick() instead, same pattern as _spawn already uses.
        self.route_waypoints: list = []
        self.route_index: int = 0
        self.route_enabled: bool = True
        # Per-platoon loops (see "Per-platoon patrol routes"), populated
        # lazily per platoon_id the first time it's actually needed.
        self.platoon_routes: dict = {}
        self.platoon_route_index: dict = {}

    def _ensure_route(self, swarm) -> None:
        if self.route_waypoints:
            return
        waypoints = _ring_waypoints(
            swarm.width, swarm.height, self.config.route_waypoint_count, self.config.route_edge_margin,
        )
        obstacles = swarm.config.obstacles
        if obstacles:
            # A waypoint landing inside an obstacle would mean the idle
            # flock's own target is somewhere physically unreachable --
            # nudge any such waypoint to the nearest clear point instead.
            waypoints = [push_point_outside_all(x, y, obstacles) for x, y in waypoints]
        self.route_waypoints = waypoints

    def _ensure_platoon_route(self, swarm, platoon_id: str) -> None:
        if platoon_id in self.platoon_routes:
            return
        zones = [status.zone for status in swarm.mission.zone_statuses.values()] if swarm.mission is not None else []
        zone = platoon_zone_anchor(swarm.command.platoon_ids, zones, platoon_id)
        if zone is None:
            # No defendable areas to anchor to -- fall back to the same
            # arena-perimeter ring every platoon would share anyway.
            self._ensure_route(swarm)
            self.platoon_routes[platoon_id] = self.route_waypoints
            self.platoon_route_index[platoon_id] = 0
            return

        waypoints = _zone_loop_waypoints(zone, self.config.zone_patrol_waypoint_count, self.config.zone_patrol_margin)
        obstacles = swarm.config.obstacles
        if obstacles:
            waypoints = [push_point_outside_all(x, y, obstacles) for x, y in waypoints]
        self.platoon_routes[platoon_id] = waypoints
        self.platoon_route_index[platoon_id] = 0

    def patrol_target(self, swarm, platoon_id: str | None = None) -> tuple | None:
        """The current destination the relevant population should steer
        toward, or None if patrol routing is off. Without hierarchical
        command (or when `platoon_id` isn't given), this is the single
        shared arena-perimeter destination the WHOLE idle population
        tours together. With it, `platoon_id` selects that platoon's own
        small zone-anchored loop instead -- see "Per-platoon patrol
        routes" in the module docstring for why. Either way, the route
        advances once the relevant population's own centroid arrives, not
        any single drone, so a straggler at the back doesn't yank the
        target away from drones still converging on the current one."""
        if not self.route_enabled:
            return None
        if platoon_id is None or swarm.command is None:
            return self._global_patrol_target(swarm)
        return self._platoon_patrol_target(swarm, platoon_id)

    def _global_patrol_target(self, swarm) -> tuple | None:
        self._ensure_route(swarm)
        if not self.route_waypoints:
            return None
        idle = [
            d for d in swarm.drones.values()
            if d.alive and d.mission_zone_id is None and d.investigating_disturbance_id is None
        ]
        if not idle:
            return self.route_waypoints[self.route_index]

        centroid_x = sum(d.x for d in idle) / len(idle)
        centroid_y = sum(d.y for d in idle) / len(idle)
        target = self.route_waypoints[self.route_index]
        if math.hypot(centroid_x - target[0], centroid_y - target[1]) <= self.config.route_arrival_radius:
            self.route_index = (self.route_index + 1) % len(self.route_waypoints)
            target = self.route_waypoints[self.route_index]
        return target

    def _platoon_patrol_target(self, swarm, platoon_id: str) -> tuple | None:
        self._ensure_platoon_route(swarm, platoon_id)
        waypoints = self.platoon_routes.get(platoon_id)
        if not waypoints:
            return None
        index = self.platoon_route_index.get(platoon_id, 0)
        target = waypoints[index]

        members = [
            d for d in swarm.drones.values()
            if d.alive and d.platoon_id == platoon_id
            and d.mission_zone_id is None and d.investigating_disturbance_id is None
        ]
        if not members:
            return target

        centroid_x = sum(d.x for d in members) / len(members)
        centroid_y = sum(d.y for d in members) / len(members)
        if math.hypot(centroid_x - target[0], centroid_y - target[1]) <= self.config.route_arrival_radius:
            index = (index + 1) % len(waypoints)
            self.platoon_route_index[platoon_id] = index
            target = waypoints[index]
        return target

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
        if swarm.config.obstacles:
            x, y = push_point_outside_all(x, y, swarm.config.obstacles)
        disturbance_id = self._new_id()
        severity = self._rng.randint(MIN_SEVERITY, MAX_SEVERITY)
        self.disturbances[disturbance_id] = Disturbance(id=disturbance_id, x=x, y=y, spawned_tick=tick, severity=severity)
        return disturbance_id

    def add_disturbance(self, x: float, y: float, tick: int, obstacles=()) -> str:
        """User-placed disturbance (e.g. a live-demo click), as opposed to
        the ambient auto-spawn above. Deliberately bypasses
        `max_active_disturbances` -- that cap exists to keep unattended,
        randomly-timed spawns from cluttering the arena over time, not to
        limit an explicit, one-at-a-time user action, the same way
        `_launch_random_attack` in server.py always fires regardless of
        ambient swarm state. Severity is still randomized rather than
        fixed, same as an auto-spawned one -- the operator is flagging
        *where* to look, not diagnosing how serious it'll turn out to be.
        `obstacles` defaults to empty rather than reading from a `swarm`
        object (unlike _spawn) since the caller here is server.py's WS
        handler, which already has the click coordinates in hand and
        doesn't need this method to reach back into swarm state for
        anything else."""
        if obstacles:
            x, y = push_point_outside_all(x, y, obstacles)
        disturbance_id = self._new_id()
        severity = self._rng.randint(MIN_SEVERITY, MAX_SEVERITY)
        self.disturbances[disturbance_id] = Disturbance(id=disturbance_id, x=x, y=y, spawned_tick=tick, severity=severity)
        return disturbance_id

    def _dispatch(self, swarm) -> list:
        """Keeps assigning the nearest still-idle drone to whichever
        under-resourced disturbance needs it most (most severe first, then
        longest-waiting) until every disturbance's investigator headcount
        matches its severity or idle drones run out -- the same
        keep-assigning-until-met-or-out-of-candidates shape as
        `HeuristicAllocator.allocate()`, just for investigator headcount
        instead of zone headcount. "Idle" is exactly mission.py's own
        reserve-pool definition: alive, not currently occupying any zone,
        not already investigating something else."""
        mission = swarm.mission
        if mission is None:
            return []
        needing = sorted(
            (d for d in self.disturbances.values() if not d.resolved and len(d.investigator_ids) < d.severity),
            key=lambda d: (-d.severity, d.spawned_tick),
        )
        if not needing:
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
        for disturbance in needing:
            still_needed = disturbance.severity - len(disturbance.investigator_ids)
            for _ in range(still_needed):
                if not idle:
                    return events
                nearest = min(idle, key=lambda d: math.hypot(d.x - disturbance.x, d.y - disturbance.y))
                idle.remove(nearest)
                nearest.investigating_disturbance_id = disturbance.id
                disturbance.investigator_ids.append(nearest.id)
                events.append({"disturbance_id": disturbance.id, "drone_id": nearest.id, "kind": "dispatched"})
        return events

    def _advance(self, swarm, tick: int) -> list:
        events = []
        for disturbance in self.disturbances.values():
            if disturbance.resolved:
                continue

            present = 0
            for investigator_id in list(disturbance.investigator_ids):
                drone = swarm.drones.get(investigator_id)
                if drone is None or not drone.alive:
                    # Investigator died mid-investigation: it's dropped
                    # from the roster (progress made so far is NOT reset --
                    # only the accrual rate suffers until _dispatch backfills
                    # the vacancy), same "abandoned assignment, existing
                    # machinery re-absorbs/replaces it" idiom used elsewhere
                    # in this codebase.
                    if drone is not None:
                        drone.investigating_disturbance_id = None
                    disturbance.investigator_ids.remove(investigator_id)
                    continue
                if math.hypot(drone.x - disturbance.x, drone.y - disturbance.y) <= self.config.investigation_range:
                    present += 1  # arrived and actively contributing this tick

            if not disturbance.investigator_ids:
                continue  # nobody assigned (yet, or anymore) -- nothing to accrue

            # Capped at severity: sending more than the situation calls for
            # doesn't resolve it any faster, only sending FEWER does real
            # harm (partial credit, proportionally slower).
            disturbance.ticks_investigated += min(present, disturbance.severity)
            required_effort = disturbance.severity * self.config.investigation_ticks_required
            if disturbance.ticks_investigated >= required_effort:
                disturbance.resolved = True
                disturbance.resolved_at_tick = tick
                resolved_by = list(disturbance.investigator_ids)
                for investigator_id in resolved_by:
                    drone = swarm.drones.get(investigator_id)
                    if drone is not None:
                        drone.investigating_disturbance_id = None
                events.append({"disturbance_id": disturbance.id, "drone_ids": resolved_by, "kind": "resolved"})
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
                    "severity": d.severity,
                    "investigator_ids": list(d.investigator_ids),
                    "progress": min(1.0, d.ticks_investigated / (d.severity * self.config.investigation_ticks_required)),
                    "resolved": d.resolved,
                }
                for d in self.disturbances.values()
            ],
            "route": [[x, y] for x, y in self.route_waypoints],
            "route_index": self.route_index,
            "route_enabled": self.route_enabled,
            "platoon_routes": {
                pid: {"route": [[x, y] for x, y in waypoints], "route_index": self.platoon_route_index.get(pid, 0)}
                for pid, waypoints in self.platoon_routes.items()
            },
        }
