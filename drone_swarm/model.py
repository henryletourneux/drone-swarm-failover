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

    # Mission-system fields (drone_swarm/mission.py) -- inert unless a
    # SwarmConfig.mission_config is actually supplied, so every hand-built
    # Drone() in existing tests is unaffected. battery=100.0 (full) and
    # mission_zone_id=None (unassigned) are the natural "not in a mission"
    # defaults.
    battery: float = 100.0
    mission_zone_id: str | None = None

    # Hierarchical-command fields (drone_swarm/command.py) -- inert unless
    # SwarmConfig.command_config is supplied, so every hand-built Drone()
    # in existing tests is unaffected. platoon_id is static, set once at
    # swarm construction. commander_id is this drone's OWN belief about
    # who the commander is, populated only while it holds a commander
    # election engine (i.e. while it's currently its platoon's nexus) --
    # deliberately not propagated down to ordinary platoon members, since
    # there's no message type carrying that information to them.
    platoon_id: str | None = None
    commander_id: str | None = None

    # Patrol/disturbance-investigation field (drone_swarm/patrol.py) -- inert
    # unless SwarmConfig.patrol_config is supplied, same additive-layer
    # principle as mission_zone_id/platoon_id above. Set while this drone is
    # travelling to or holding at a disturbance. Only ever dispatched from
    # the idle pool (mission_zone_id already None), so a drone investigates
    # OR holds a mission zone, never both at once.
    investigating_disturbance_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "priority": round(self.priority, 1),
            "alive": self.alive,
            "nexus_id": self.nexus_id,
            "role": self.role,
            "battery": round(self.battery, 1),
            "mission_zone_id": self.mission_zone_id,
            "platoon_id": self.platoon_id,
            "commander_id": self.commander_id,
            "investigating_disturbance_id": self.investigating_disturbance_id,
        }
