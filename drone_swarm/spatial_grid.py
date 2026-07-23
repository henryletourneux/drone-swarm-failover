"""Uniform spatial grid for fast proximity queries.

Both `MeshNetwork.neighbors_of` and `topology.build_adjacency` need to
answer the same question, over and over, every tick: "which drones are
within `comm_range` of this one?" Answered naively (check every drone
against every other drone), that's O(n^2) per tick -- fine at 14 drones,
catastrophic at 100 (measured: ~0.05ms/tick at n=14 vs ~193ms/tick at
n=100 in profiling, worse than quadratic once you account for message
volume itself also scaling with n).

A uniform grid fixes this the standard way: bucket drones into cells
sized to `comm_range`, so a neighbor query only has to look at the ~9
cells around a drone instead of the whole population. Rebuilding the
grid is O(n); each query is O(local density), so a full tick's worth of
neighbor queries is O(n) overall for a reasonably-distributed swarm,
instead of O(n^2).
"""
from __future__ import annotations

import math


class SpatialGrid:
    def __init__(self, cell_size: float) -> None:
        # Cells sized to comm_range means any two drones within range are
        # always found by checking a drone's own cell plus its 8 neighbors.
        self.cell_size = max(cell_size, 1e-6)
        self._cells: dict = {}
        self._positions: dict = {}

    def _cell_of(self, x: float, y: float) -> tuple:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def rebuild(self, positions: dict) -> None:
        """`positions`: {drone_id: (x, y)} for every drone that should be
        queryable -- callers filter to "alive" themselves."""
        self._positions = positions
        self._cells = {}
        for drone_id, (x, y) in positions.items():
            self._cells.setdefault(self._cell_of(x, y), []).append(drone_id)

    def neighbors_within(self, drone_id: str, radius: float) -> list:
        """Every OTHER drone within `radius` of `drone_id`'s current
        position. Empty if `drone_id` isn't in the grid (e.g. not alive)."""
        origin = self._positions.get(drone_id)
        if origin is None:
            return []
        return self.neighbors_of_point(origin[0], origin[1], radius, exclude=drone_id)

    def neighbors_of_point(self, x: float, y: float, radius: float, exclude: str = None) -> list:
        cx, cy = self._cell_of(x, y)
        cell_span = int(radius // self.cell_size) + 1
        radius_sq = radius * radius

        out = []
        for dx in range(-cell_span, cell_span + 1):
            for dy in range(-cell_span, cell_span + 1):
                for candidate_id in self._cells.get((cx + dx, cy + dy), ()):
                    if candidate_id == exclude:
                        continue
                    other_x, other_y = self._positions[candidate_id]
                    dist_sq = (x - other_x) ** 2 + (y - other_y) ** 2
                    if dist_sq <= radius_sq:
                        out.append(candidate_id)
        return out
