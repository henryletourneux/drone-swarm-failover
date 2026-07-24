"""Zone-coverage resource allocation: "Zone Coverage Under Threat."

The environment a resource-allocation policy (heuristic here; a learned
one in a later phase) actually has to solve: the arena has designated
**zones** drones need to reach and hold. A zone is "secured" once enough
drones are physically present in it; an undersupported *contested* zone
(`threat_level > 0`) drains the battery of whoever's camped there faster
than normal flight does. Every drone has a finite `battery` (see
model.py) that only depletes here -- running out doesn't destroy a drone
(that's a separate, adversarial concept handled by kill()/antagonist),
it just makes the drone ineligible for further zone assignments. That
decoupling is deliberate: this whole module is an additive layer on top
of the core election/mesh mechanics, the same way bft_mode was -- nothing
in election.py or swarm.py's core tick logic needs to know this exists.

Allocation decisions are driven by the currently-elected nexus, tying
this back into the failover story: lose the nexus mid-mission and the
newly-elected one has to pick up allocation duties. Honest simplification
stated plainly: this first phase computes allocation with full,
instantaneous knowledge of every drone's position/battery (an omniscient
call made by the simulation, mirroring how `Swarm._assign_roles` already
works) rather than routing it through the mesh's own realistic,
partial-information message-passing. Routing allocation through actual
NexusHeartbeat-style status reports -- with the genuine possibility of
stale or missing data -- is a natural, valuable next refinement, and
exactly the kind of uncertainty a learned policy (Phase 2) would have a
real reason to exist for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

MIN_BATTERY_TO_ASSIGN = 20.0
ROLE_REASSIGN_BONUS = {"leaf": 0.2, "relay": -0.3, "nexus": -1.0, "unassigned": 0.0}


@dataclass(frozen=True)
class Zone:
    id: str
    x: float
    y: float
    radius: float
    required_drones: int
    threat_level: float = 0.0  # 0 = uncontested; higher drains battery faster while undersupported


@dataclass(frozen=True)
class MissionConfig:
    zones: tuple
    base_drain_per_tick: float = 0.01
    move_drain_per_unit: float = 0.003
    contested_drain_per_tick: float = 0.2
    reallocation_interval_ticks: int = 10


@dataclass
class ZoneStatus:
    zone: Zone
    occupant_ids: list = field(default_factory=list)
    secured: bool = False


class HeuristicAllocator:
    """Greedy, weighted, fully explainable baseline -- the honest bar any
    learned policy (Phase 2) needs to clear to be worth using at all.

    For each under-resourced zone, most-threatened first, repeatedly pick
    the best-scoring still-available drone until the zone's requirement
    is met or eligible drones run out. Score rewards high battery and
    short distance, and discourages pulling a `relay` (structurally load
    -bearing for mesh connectivity) off its post in favor of reassigning
    `leaf` drones, which are less connectivity-critical.
    """

    def __init__(self, w_battery: float = 1.0, w_distance: float = 1.0, w_role: float = 0.5):
        self.w_battery = w_battery
        self.w_distance = w_distance
        self.w_role = w_role

    def score(self, drone, zone: Zone, arena_diagonal: float) -> float:
        distance = math.hypot(drone.x - zone.x, drone.y - zone.y)
        battery_frac = drone.battery / 100.0
        distance_frac = min(1.0, distance / max(arena_diagonal, 1.0))
        role_term = ROLE_REASSIGN_BONUS.get(drone.role, 0.0)
        return (
            self.w_battery * battery_frac
            - self.w_distance * distance_frac
            + self.w_role * role_term
        )

    def allocate(self, drones: dict, zone_statuses: list, arena_diagonal: float) -> dict:
        """Returns {drone_id: zone_id} for every NEW assignment decided
        this call -- drones already adequately covering a secured zone
        aren't reassigned, and only eligible (alive, charged) drones not
        already committed elsewhere are considered."""
        committed = {
            drone_id
            for status in zone_statuses
            for drone_id in status.occupant_ids
        }
        eligible = [
            d for d in drones.values()
            if d.alive and d.battery > MIN_BATTERY_TO_ASSIGN and d.id not in committed
        ]

        assignments: dict = {}
        # Most-threatened first, then most-still-needed as a tiebreaker --
        # both descending, so reverse=True alone is correct here. (An
        # earlier version negated threat_level AND passed reverse=True,
        # which cancel out to ascending-by-threat -- least urgent first,
        # backwards from the intent. Caught by
        # test_allocator_prioritizes_most_threatened_zone_first.)
        needy = sorted(
            (s for s in zone_statuses if not s.secured),
            key=lambda s: (s.zone.threat_level, s.zone.required_drones - len(s.occupant_ids)),
            reverse=True,
        )
        for status in needy:
            still_needed = status.zone.required_drones - len(status.occupant_ids)
            for _ in range(max(0, still_needed)):
                if not eligible:
                    return assignments
                best = max(eligible, key=lambda d: self.score(d, status.zone, arena_diagonal))
                assignments[best.id] = status.zone.id
                eligible.remove(best)
        return assignments


class MissionState:
    """Owns battery drain and zone occupancy/security bookkeeping for one
    swarm. Call `.tick(swarm)` once per `Swarm.tick()`, after positions
    have moved for that tick."""

    def __init__(self, config: MissionConfig, allocator: HeuristicAllocator | None = None) -> None:
        self.config = config
        self.allocator = allocator if allocator is not None else HeuristicAllocator()
        self._last_positions: dict = {}
        self._ticks_since_reallocation = 0
        self.zone_statuses: dict = {z.id: ZoneStatus(zone=z) for z in config.zones}

    def _occupants_of(self, drones: dict) -> None:
        for status in self.zone_statuses.values():
            status.occupant_ids = []
        for drone in drones.values():
            if not drone.alive:
                continue
            for status in self.zone_statuses.values():
                z = status.zone
                if math.hypot(drone.x - z.x, drone.y - z.y) <= z.radius:
                    status.occupant_ids.append(drone.id)
        for status in self.zone_statuses.values():
            status.secured = len(status.occupant_ids) >= status.zone.required_drones

    def _drain_batteries(self, drones: dict) -> None:
        for drone in drones.values():
            if not drone.alive:
                continue
            prev = self._last_positions.get(drone.id, (drone.x, drone.y))
            moved = math.hypot(drone.x - prev[0], drone.y - prev[1])
            self._last_positions[drone.id] = (drone.x, drone.y)
            drone.battery = max(0.0, drone.battery - self.config.base_drain_per_tick - moved * self.config.move_drain_per_unit)

        for status in self.zone_statuses.values():
            if status.secured:
                continue
            for drone_id in status.occupant_ids:
                drone = drones[drone_id]
                drone.battery = max(0.0, drone.battery - self.config.contested_drain_per_tick * status.zone.threat_level)

    def tick(self, swarm) -> list:
        """Advances battery/occupancy bookkeeping, and reallocates via the
        current nexus every `reallocation_interval_ticks`. Returns a list
        of (drone_id, zone_id) newly-made assignments this call, for the
        caller to log/narrate -- empty most ticks."""
        self._drain_batteries(swarm.drones)
        self._occupants_of(swarm.drones)

        for drone in swarm.drones.values():
            if drone.mission_zone_id is not None:
                status = self.zone_statuses.get(drone.mission_zone_id)
                if status is not None and drone.id in status.occupant_ids:
                    continue  # arrived, still holding its assignment
            if drone.battery <= MIN_BATTERY_TO_ASSIGN or not drone.alive:
                drone.mission_zone_id = None

        self._ticks_since_reallocation += 1
        nexus_id = next((d.id for d in swarm.drones.values() if d.alive and d.role == "nexus"), None)
        if nexus_id is None or self._ticks_since_reallocation < self.config.reallocation_interval_ticks:
            return []
        self._ticks_since_reallocation = 0

        arena_diagonal = math.hypot(swarm.width, swarm.height)
        new_assignments = self.allocator.allocate(swarm.drones, list(self.zone_statuses.values()), arena_diagonal)
        for drone_id, zone_id in new_assignments.items():
            swarm.drones[drone_id].mission_zone_id = zone_id
        return list(new_assignments.items())

    def all_secured(self) -> bool:
        return all(status.secured for status in self.zone_statuses.values())

    def to_state_dict(self) -> dict:
        return {
            "zones": [
                {
                    "id": status.zone.id,
                    "x": status.zone.x,
                    "y": status.zone.y,
                    "radius": status.zone.radius,
                    "required_drones": status.zone.required_drones,
                    "threat_level": status.zone.threat_level,
                    "occupants": len(status.occupant_ids),
                    "secured": status.secured,
                }
                for status in self.zone_statuses.values()
            ],
        }
