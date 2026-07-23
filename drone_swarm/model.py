from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Drone:
    """One node in the swarm.

    `priority` is a fixed score (think: signal strength / processing
    headroom) used to break ties during an election — the reachable drone
    with the highest priority becomes the new nexus.
    """

    id: str
    x: float
    y: float
    priority: float
    alive: bool = True

    # Which drone this one currently believes is the nexus (coordinator).
    nexus_id: str | None = None

    # Election bookkeeping — the best (id, priority) this drone has seen
    # so far during the current flood, and which round that flood is.
    candidate_id: str | None = None
    candidate_priority: float = -1.0
    election_round: int = 0

    # Cosmetic, derived from graph position each tick: "nexus" | "relay" | "leaf" | "unassigned"
    role: str = "unassigned"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "priority": round(self.priority, 1),
            "alive": self.alive,
            "nexus_id": self.nexus_id,
            "role": self.role,
        }
