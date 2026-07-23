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

    # Constant-velocity drift per tick; zero by default (stationary), so
    # every hand-built Drone() in existing tests is unaffected.
    vx: float = 0.0
    vy: float = 0.0

    # Which drone this one currently believes is the nexus (coordinator).
    # Synced each tick from this drone's NexusElection.known_nexus_id — the
    # actual election bookkeeping (candidacy, term, timeouts) lives there,
    # not here, since it's a message-passing state machine, not static data.
    nexus_id: str | None = None

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
