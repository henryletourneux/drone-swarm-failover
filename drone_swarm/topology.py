"""Pure graph algorithms over the swarm's mesh network.

The mesh isn't a fixed structure — it's derived every tick from which
alive drones happen to be within radio range of each other. These
functions are written from scratch (no networkx) so the algorithm is
easy to read end to end.
"""
from __future__ import annotations

from .spatial_grid import SpatialGrid


def build_adjacency(drones: dict, comm_range: float) -> dict:
    """Return {drone_id: set(neighbor_ids)} for alive drones within comm_range.

    Uses a SpatialGrid rather than an all-pairs distance check -- the
    naive version is O(n^2) per call, which is fine at a handful of
    drones but measured catastrophically bad at 100 (this is called
    twice per tick, on top of MeshNetwork's own per-message neighbor
    queries -- see spatial_grid.py for the full profiling story).
    """
    alive = {d.id: (d.x, d.y) for d in drones.values() if d.alive}
    grid = SpatialGrid(cell_size=max(comm_range, 1.0))
    grid.rebuild(alive)

    adjacency = {drone_id: set() for drone_id in alive}
    for drone_id in alive:
        adjacency[drone_id] = set(grid.neighbors_within(drone_id, comm_range))
    return adjacency


def bfs_reachable(adjacency: dict, start: str) -> set:
    """All node ids reachable from `start`, including itself."""
    if start not in adjacency:
        return set()

    seen = {start}
    frontier = [start]
    while frontier:
        next_frontier = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return seen


def connected_components(adjacency: dict) -> list:
    """All connected components of the graph, as a list of node-id sets."""
    unvisited = set(adjacency.keys())
    components = []
    while unvisited:
        start = next(iter(unvisited))
        component = bfs_reachable(adjacency, start)
        components.append(component)
        unvisited -= component
    return components
