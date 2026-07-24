from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .election import ElectionRole, NexusElection
from .identity import DroneIdentity, IdentityRegistry, SwarmAuthority
from .mesh_network import MeshNetwork
from .metrics import SwarmMetrics
from .mission import MissionState
from .topology import build_adjacency

MAX_EVENT_LOG = 200


@dataclass(frozen=True)
class SwarmConfig:
    """Tunable parameters for the mesh network and election timing.

    `tick_dt_s` is how much simulated time one `Swarm.tick()` call
    represents — it's independent of real wall-clock time, so how "fast"
    a failover feels in a live demo depends on how often the caller
    actually calls tick() (see TICK_SECONDS in server.py).

    `bft_mode` turns on cryptographic signing/verification of every
    election message (see election.py's module docstring) — off by
    default so the base coordination mechanism stays exactly as tested
    without it. Turn it on to run the swarm hardened against the
    antagonist/ package's attacks.

    `mission_config` turns on zone-coverage resource allocation (see
    mission.py) — None (default) means the mission system is entirely
    inert, same principle as bft_mode: an additive layer, off by default,
    zero effect on the base election/mesh tests without it.
    """

    max_relay_hops: int = 4
    packet_loss_rate: float = 0.02
    comm_latency_s: float = 0.1
    nexus_heartbeat_interval_s: float = 1.5
    nexus_timeout_s: float = 4.0
    tick_dt_s: float = 0.4
    bft_mode: bool = False
    mission_config: object = None


class Swarm:
    """Owns the drones, the mesh network they communicate over, and each
    drone's election state machine, and advances the simulation one tick
    at a time.

    Unlike an earlier version of this project, no drone has instant global
    knowledge of the swarm here — coordination is a genuine emergent
    property of messages actually delivered (with latency and loss) over
    `MeshNetwork`. See `election.py` for the convergence argument,
    including how partitions merging back together are handled without
    any special-cased logic.
    """

    def __init__(
        self,
        drones: list,
        comm_range: float,
        width: float = 800.0,
        height: float = 500.0,
        config: SwarmConfig | None = None,
        seed: int | None = None,
    ):
        self.drones = {d.id: d for d in drones}
        self.comm_range = comm_range
        self.width = width
        self.height = height
        self.config = config if config is not None else SwarmConfig()
        self.time_s = 0.0
        self.tick_count = 0
        self.event_log: list = []
        self.metrics = SwarmMetrics()
        self._nexus_gap_start: dict = {}

        self.mesh = MeshNetwork(
            comm_range=comm_range,
            max_relay_hops=self.config.max_relay_hops,
            packet_loss_rate=self.config.packet_loss_rate,
            latency_s=self.config.comm_latency_s,
            rng=random.Random(seed),
        )

        # Identities are only real cryptographic objects in bft_mode; left
        # as empty dicts otherwise so callers can always check `swarm.identities`
        # / `swarm.registry` without a None check, whether or not it's populated.
        self.authority = None
        self.registry = None
        self.identities: dict = {}
        self.credentials: dict = {}
        if self.config.bft_mode:
            self.authority = SwarmAuthority()
            self.registry = IdentityRegistry(self.authority.public_key)
            for d in drones:
                identity = DroneIdentity(d.id)
                self.identities[d.id] = identity
                self.credentials[d.id] = self.authority.issue_credential(d.id, d.priority)
                self.registry.register(d.id, identity.public_key)

        self.elections = {
            d.id: NexusElection(
                nexus_heartbeat_interval_s=self.config.nexus_heartbeat_interval_s,
                nexus_timeout_s=self.config.nexus_timeout_s,
                bft_mode=self.config.bft_mode,
                identity=self.identities.get(d.id),
                credential=self.credentials.get(d.id),
                registry=self.registry,
                total_swarm_size=len(drones),
            )
            for d in drones
        }
        for d in drones:
            self.mesh.update_known_position(d.id, d.x, d.y, d.alive)

        # Cached by tick() and reused by to_state_dict() -- both used to
        # independently call build_adjacency, computing the same thing
        # twice every tick for no reason. Computed once here too so a
        # to_state_dict() call before the first tick() (e.g. serving the
        # very first WebSocket message) still returns real edges.
        self._last_adjacency: dict = build_adjacency(self.drones, self.comm_range)

        self.mission = MissionState(self.config.mission_config) if self.config.mission_config is not None else None

    def kill(self, drone_id: str) -> bool:
        drone = self.drones.get(drone_id)
        if drone is None or not drone.alive:
            return False
        drone.alive = False
        drone.role = "unassigned"
        drone.nexus_id = None
        self.mesh.update_known_position(drone_id, drone.x, drone.y, False)
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

        for drone in self.drones.values():
            self.mesh.update_known_position(drone.id, drone.x, drone.y, drone.alive)

        inboxes = self.mesh.deliver_due_messages(self.time_s)

        previous_states = {}
        all_outgoing = []
        for drone_id, drone in self.drones.items():
            if not drone.alive:
                continue
            election = self.elections[drone_id]
            previous_states[drone_id] = (election.role, election.known_nexus_id)
            inbox = inboxes.get(drone_id, [])
            _, outgoing = election.step(self.time_s, drone_id, drone.priority, inbox)
            all_outgoing.append(outgoing)

        for outgoing in all_outgoing:
            for message in outgoing:
                self.mesh.broadcast(message, self.time_s)

        self.time_s += self.config.tick_dt_s

        for drone_id, drone in self.drones.items():
            if not drone.alive:
                continue
            election = self.elections[drone_id]
            drone.nexus_id = election.known_nexus_id
            previous = previous_states.get(drone_id)
            self._log_transition(drone_id, previous, election)
            self._track_recovery(drone_id, previous, election)

        self._last_adjacency = build_adjacency(self.drones, self.comm_range)
        self._assign_roles(self._last_adjacency)

        if self.mission is not None:
            # Runs after role assignment: allocation needs to know who's
            # currently nexus, and the relay/leaf distinction it weighs
            # reassignment decisions by.
            new_assignments = self.mission.tick(self)
            for drone_id, zone_id in new_assignments:
                self.event_log.append({
                    "tick": self.tick_count,
                    "type": "mission_assigned",
                    "detail": f"{drone_id} assigned to zone {zone_id}",
                    "drone": drone_id,
                    "zone": zone_id,
                })

        if len(self.event_log) > MAX_EVENT_LOG:
            self.event_log = self.event_log[-MAX_EVENT_LOG:]

    def _log_transition(self, drone_id: str, previous, election: NexusElection) -> None:
        if previous is None:
            return
        prev_role, _ = previous

        if prev_role != ElectionRole.CANDIDATE and election.role == ElectionRole.CANDIDATE:
            self.metrics.elections_started += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "election_started",
                "detail": f"{drone_id} lost contact with its nexus and started a campaign (term {election.term})",
                "drones": [drone_id],
            })
        elif prev_role == ElectionRole.CANDIDATE and election.role == ElectionRole.NEXUS:
            self.metrics.elections_won += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "election_won",
                "detail": f"{drone_id} elected nexus (term {election.term})",
                "winner": drone_id,
            })
        elif prev_role == ElectionRole.NEXUS and election.role == ElectionRole.FOLLOWER:
            self.metrics.merges += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "swarms_merged",
                "detail": f"{drone_id} yielded nexus to {election.known_nexus_id} (newer term {election.term})",
                "drones": [drone_id],
            })

    def _track_recovery(self, drone_id: str, previous, election: NexusElection) -> None:
        """Per-drone time-without-a-known-nexus -- see metrics.py for why
        this is the metric used instead of a swarm-wide MTTR figure."""
        if previous is None:
            return
        _, prev_nexus = previous
        curr_nexus = election.known_nexus_id
        if prev_nexus is not None and curr_nexus is None:
            self._nexus_gap_start[drone_id] = self.time_s
        elif prev_nexus is None and curr_nexus is not None:
            start = self._nexus_gap_start.pop(drone_id, None)
            if start is not None:
                self.metrics.record_recovery(self.time_s - start)

    def _move(self) -> None:
        for drone in self.drones.values():
            if not drone.alive:
                continue
            if self.mission is not None and drone.mission_zone_id is not None:
                self._move_toward_zone(drone)
                continue
            if drone.vx == 0.0 and drone.vy == 0.0:
                continue
            drone.x += drone.vx
            drone.y += drone.vy
            if drone.x < 0.0 or drone.x > self.width:
                drone.vx = -drone.vx
                drone.x = max(0.0, min(self.width, drone.x))
            if drone.y < 0.0 or drone.y > self.height:
                drone.vy = -drone.vy
                drone.y = max(0.0, min(self.height, drone.y))

    def _move_toward_zone(self, drone) -> None:
        """A drone with an active mission assignment steers directly for
        its zone instead of drifting -- at the same speed magnitude it was
        already moving at (falling back to a sane default for drones that
        started stationary), holding position once comfortably inside the
        zone's radius rather than orbiting or overshooting through it."""
        status = self.mission.zone_statuses.get(drone.mission_zone_id)
        if status is None:
            return
        zone = status.zone
        dx, dy = zone.x - drone.x, zone.y - drone.y
        distance = math.hypot(dx, dy)
        speed = math.hypot(drone.vx, drone.vy) or 4.0
        if distance <= max(zone.radius * 0.6, speed):
            return
        drone.x += dx / distance * speed
        drone.y += dy / distance * speed

    def _assign_roles(self, adjacency: dict) -> None:
        for drone in self.drones.values():
            if not drone.alive:
                drone.role = "unassigned"
            elif drone.nexus_id == drone.id:
                drone.role = "nexus"
            else:
                degree = len(adjacency.get(drone.id, ()))
                drone.role = "relay" if degree >= 2 else "leaf"

    def metrics_snapshot(self) -> dict:
        return self.metrics.snapshot(self.mesh, self.elections)

    def to_state_dict(self) -> dict:
        edges = sorted({tuple(sorted((a, b))) for a, neighbors in self._last_adjacency.items() for b in neighbors})
        state = {
            "tick": self.tick_count,
            "world": {"width": self.width, "height": self.height},
            "drones": [d.to_dict() for d in self.drones.values()],
            "edges": [list(e) for e in edges],
            "event_log": self.event_log[-30:],
            "metrics": self.metrics_snapshot(),
        }
        if self.mission is not None:
            state["mission"] = self.mission.to_state_dict()
        return state
