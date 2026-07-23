from __future__ import annotations

from .election import resolve_component
from .model import Drone
from .topology import build_adjacency, connected_components

MAX_EVENT_LOG = 200


class Swarm:
    """Owns the drones and advances the simulation one tick at a time.

    Nothing here is pre-scripted: on every tick the mesh is rebuilt from
    current positions/aliveness, split into connected components, and any
    component lacking a confirmed nexus runs one more step of the election
    flood. Kill enough drones (including repeatedly killing whoever just
    became nexus) and you get the cascading handoff / self-healing "web"
    behavior this project is named for.
    """

    def __init__(self, drones: list, comm_range: float, width: float = 800.0, height: float = 500.0):
        self.drones = {d.id: d for d in drones}
        self.comm_range = comm_range
        self.width = width
        self.height = height
        self.tick_count = 0
        self.event_log: list = []

    def kill(self, drone_id: str) -> bool:
        drone = self.drones.get(drone_id)
        if drone is None or not drone.alive:
            return False
        drone.alive = False
        drone.role = "unassigned"
        drone.nexus_id = None
        drone.candidate_id = None
        drone.candidate_priority = -1.0
        self.event_log.append({
            "tick": self.tick_count,
            "type": "drone_down",
            "detail": f"{drone_id} went down",
            "drone": drone_id,
        })
        return True

    def tick(self) -> None:
        self.tick_count += 1
        self._move()
        adjacency = build_adjacency(self.drones, self.comm_range)
        components = connected_components(adjacency)

        for component in components:
            resolve_component(self.drones, adjacency, component, self.event_log, self.tick_count)

        self._assign_roles(adjacency)
        if len(self.event_log) > MAX_EVENT_LOG:
            self.event_log = self.event_log[-MAX_EVENT_LOG:]

    def _move(self) -> None:
        for drone in self.drones.values():
            if not drone.alive or (drone.vx == 0.0 and drone.vy == 0.0):
                continue
            drone.x += drone.vx
            drone.y += drone.vy
            if drone.x < 0.0 or drone.x > self.width:
                drone.vx = -drone.vx
                drone.x = max(0.0, min(self.width, drone.x))
            if drone.y < 0.0 or drone.y > self.height:
                drone.vy = -drone.vy
                drone.y = max(0.0, min(self.height, drone.y))

    def _assign_roles(self, adjacency: dict) -> None:
        for drone in self.drones.values():
            if not drone.alive:
                drone.role = "unassigned"
            elif drone.nexus_id == drone.id:
                drone.role = "nexus"
            else:
                degree = len(adjacency.get(drone.id, ()))
                drone.role = "relay" if degree >= 2 else "leaf"

    def to_state_dict(self) -> dict:
        adjacency = build_adjacency(self.drones, self.comm_range)
        edges = sorted({tuple(sorted((a, b))) for a, neighbors in adjacency.items() for b in neighbors})
        return {
            "tick": self.tick_count,
            "drones": [d.to_dict() for d in self.drones.values()],
            "edges": [list(e) for e in edges],
            "event_log": self.event_log[-30:],
        }
