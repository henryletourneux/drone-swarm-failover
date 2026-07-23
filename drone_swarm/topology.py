"""Pure graph algorithms over the swarm's mesh network.

The mesh isn't a fixed structure — it's derived every tick from which
alive drones happen to be within radio range of each other. These
functions are written from scratch (no networkx) so the algorithm is
easy to read end to end.
"""
from __future__ import annotations

import math


def build_adjacency(drones: dict, comm_range: float) -> dict:
    """Return {drone_id: set(neighbor_ids)} for alive drones within comm_range."""
    alive_ids = [d.id for d in drones.values() if d.alive]
    adjacency = {drone_id: set() for drone_id in alive_ids}

    for i, a_id in enumerate(alive_ids):
        a = drones[a_id]
        for b_id in alive_ids[i + 1:]:
            b = drones[b_id]
            dist = math.hypot(a.x - b.x, a.y - b.y)
            if dist <= comm_range:
                adjacency[a_id].add(b_id)
                adjacency[b_id].add(a_id)

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
