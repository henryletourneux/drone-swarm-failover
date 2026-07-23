"""Flood-based leader ("nexus") election.

This is the heart of the project. There is no central authority — when a
drone notices its connected component has no confirmed nexus, every drone
in that component nominates itself, and on every following tick each
drone adopts the best candidate (highest priority, id as tiebreaker) seen
among its immediate neighbors so far. That "best so far" value spreads
outward one hop per tick — like ripples through a web — until every drone
in the component has independently converged on the same winner.

This is a from-scratch implementation of flooding-based maximum
consensus, the same family of algorithm as leader election in general
(non-ring) graphs.
"""
from __future__ import annotations


def _candidate_key(candidate_id: str, candidate_priority: float) -> tuple:
    # Higher priority wins; id is a deterministic tiebreaker so results
    # are reproducible in tests and identical across all drones.
    return (candidate_priority, candidate_id)


def resolve_component(drones: dict, adjacency: dict, component: set, event_log: list, tick: int) -> None:
    """Bring one connected component one step closer to (or into) agreement
    on a nexus. Mutates the drones in `component` in place."""

    incumbent = _find_incumbent(drones, component)

    if incumbent is not None:
        _sync_to_incumbent(drones, component, incumbent, event_log, tick)
        return

    _run_election_step(drones, adjacency, component, event_log, tick)


def _find_incumbent(drones: dict, component: set) -> str | None:
    """A drone in this component is the incumbent nexus if it is alive,
    present in the component, and self-confirms (nexus_id == its own id)."""
    for drone_id in component:
        drone = drones[drone_id]
        if drone.alive and drone.nexus_id == drone.id:
            return drone_id
    return None


def _sync_to_incumbent(drones: dict, component: set, incumbent: str, event_log: list, tick: int) -> None:
    for drone_id in component:
        drone = drones[drone_id]
        if drone.nexus_id != incumbent:
            drone.nexus_id = incumbent
        drone.candidate_id = None
        drone.candidate_priority = -1.0


def _run_election_step(drones: dict, adjacency: dict, component: set, event_log: list, tick: int) -> None:
    # Anyone in this component not already mid-election nominates itself.
    newly_started = []
    for drone_id in component:
        drone = drones[drone_id]
        if drone.candidate_id is None:
            drone.candidate_id = drone.id
            drone.candidate_priority = drone.priority
            newly_started.append(drone_id)

    if newly_started:
        event_log.append({
            "tick": tick,
            "type": "election_started",
            "detail": f"{len(newly_started)} drone(s) lost their nexus and started an election",
            "drones": sorted(newly_started),
        })

    # Synchronous flood step: compute everyone's next candidate from a
    # snapshot of this tick's values, so update order can't bias the result.
    snapshot = {d_id: (drones[d_id].candidate_id, drones[d_id].candidate_priority) for d_id in component}

    next_candidates = {}
    for drone_id in component:
        best_id, best_priority = snapshot[drone_id]
        for neighbor_id in adjacency.get(drone_id, ()):
            if neighbor_id not in snapshot:
                continue
            n_id, n_priority = snapshot[neighbor_id]
            if _candidate_key(n_id, n_priority) > _candidate_key(best_id, best_priority):
                best_id, best_priority = n_id, n_priority
        next_candidates[drone_id] = (best_id, best_priority)

    for drone_id in component:
        drone = drones[drone_id]
        drone.candidate_id, drone.candidate_priority = next_candidates[drone_id]

    # Converged once every drone in the component agrees on the same candidate.
    winners = {next_candidates[d_id][0] for d_id in component}
    if len(winners) == 1:
        winner_id = next(iter(winners))
        for drone_id in component:
            drone = drones[drone_id]
            drone.nexus_id = winner_id
            drone.candidate_id = None
            drone.candidate_priority = -1.0
        event_log.append({
            "tick": tick,
            "type": "election_won",
            "detail": f"{winner_id} elected as new nexus for a {len(component)}-drone group",
            "winner": winner_id,
        })
