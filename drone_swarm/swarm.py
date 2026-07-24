from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .command import CommandState
from .election import ElectionRole, NexusElection
from .identity import DroneIdentity, IdentityRegistry, SwarmAuthority
from .mesh_network import MeshNetwork
from .metrics import SwarmMetrics
from .mission import MissionState
from .patrol import PatrolState
from .topology import build_adjacency

MAX_EVENT_LOG = 200


def _transition_kind(prev_role: ElectionRole, new_role: ElectionRole) -> str | None:
    """Classifies a role change shared by both the flat/platoon layer
    (_log_transition) and the commander layer (_log_commander_transition)
    -- same election mechanic, same three transitions worth narrating,
    at either tier."""
    if prev_role != ElectionRole.CANDIDATE and new_role == ElectionRole.CANDIDATE:
        return "started"
    if prev_role == ElectionRole.CANDIDATE and new_role == ElectionRole.NEXUS:
        return "won"
    if prev_role == ElectionRole.NEXUS and new_role == ElectionRole.FOLLOWER:
        return "merged"
    return None


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

    `command_config` turns on hierarchical command (see command.py):
    drones are grouped into platoons, each electing its own nexus, with
    a second election among the current platoon nexuses picking an
    overall commander — the same principle again, None (default) means
    every drone runs a single flat, swarm-wide election exactly as
    before, zero effect on the base tests without it.

    `patrol_config` turns on disturbance investigation (see patrol.py):
    dynamically-spawned disturbance sites that spare mission drones break
    off to investigate — same additive principle, None (default) is
    entirely inert. Has a soft dependency on `mission_config` also being
    set (see patrol.py's module docstring for what happens without it).
    """

    max_relay_hops: int = 4
    packet_loss_rate: float = 0.02
    comm_latency_s: float = 0.1
    nexus_heartbeat_interval_s: float = 1.5
    nexus_timeout_s: float = 4.0
    tick_dt_s: float = 0.4
    bft_mode: bool = False
    mission_config: object = None
    command_config: object = None
    patrol_config: object = None


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

        # command_config replaces the flat, swarm-wide election below with
        # one scoped to each drone's own platoon (same NexusElection
        # class, different layer/total_swarm_size -- see command.py).
        # Either way self.elections ends up {drone_id: NexusElection},
        # one per drone, and the rest of tick() doesn't need to know
        # which mode it's in.
        self.command = CommandState(self.config.command_config, self.drones) if self.config.command_config is not None else None
        election_kwargs = dict(
            nexus_heartbeat_interval_s=self.config.nexus_heartbeat_interval_s,
            nexus_timeout_s=self.config.nexus_timeout_s,
            bft_mode=self.config.bft_mode,
        )
        if self.command is not None:
            self.elections = self.command.build_platoon_elections(
                self.drones, self.identities, self.credentials, self.registry, election_kwargs,
            )
        else:
            self.elections = {
                d.id: NexusElection(
                    **election_kwargs,
                    identity=self.identities.get(d.id),
                    credential=self.credentials.get(d.id),
                    registry=self.registry,
                    total_swarm_size=len(drones),
                )
                for d in drones
            }
        # Always initialized, whether or not command_config is set, so
        # tick() never needs a None-check to decide whether to touch
        # them -- empty/unused in flat mode, same idiom as
        # self.identities/self.credentials above.
        self.commander_elections: dict = self.command.commander_elections if self.command is not None else {}
        self._commander_gap_start: dict = {}

        for d in drones:
            self.mesh.update_known_position(d.id, d.x, d.y, d.alive)

        # Cached by tick() and reused by to_state_dict() -- both used to
        # independently call build_adjacency, computing the same thing
        # twice every tick for no reason. Computed once here too so a
        # to_state_dict() call before the first tick() (e.g. serving the
        # very first WebSocket message) still returns real edges.
        self._last_adjacency: dict = build_adjacency(self.drones, self.comm_range)

        self.mission = MissionState(self.config.mission_config) if self.config.mission_config is not None else None
        # Separate Random instance from the mesh's own (both seeded from
        # the same `seed` param is fine -- they're independent objects
        # drawing from unrelated call sequences, so this doesn't correlate
        # disturbance placement with packet-loss rolls).
        self.patrol = PatrolState(self.config.patrol_config, random.Random(seed)) if self.config.patrol_config is not None else None

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

        if self.command is not None:
            # Same inboxes as above -- commander-layer messages arrive
            # over the exact same mesh, just filtered by layer inside
            # NexusElection.step() itself. Must run before the broadcast
            # loop below so commander-layer outgoing messages go out the
            # same tick they're produced, same as every other layer.
            self.command.reconcile_and_step_commander_layer(self, inboxes, all_outgoing)

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
            for result in new_assignments:
                drone_id, zone_id = result["drone_id"], result["zone_id"]
                if result["kind"] == "substitution":
                    self.event_log.append({
                        "tick": self.tick_count,
                        "type": "battery_substitution",
                        "detail": f"{drone_id} dispatched to relieve a draining occupant in zone {zone_id}",
                        "drone": drone_id,
                        "zone": zone_id,
                    })
                else:
                    self.event_log.append({
                        "tick": self.tick_count,
                        "type": "mission_assigned",
                        "detail": f"{drone_id} assigned to zone {zone_id}",
                        "drone": drone_id,
                        "zone": zone_id,
                    })

        if self.patrol is not None:
            # Runs after mission.tick(): dispatch reads this tick's fresh
            # zone_statuses (occupancy/surplus) to decide who's spare.
            for result in self.patrol.tick(self, self.tick_count):
                kind, disturbance_id = result["kind"], result["disturbance_id"]
                if kind == "spawned":
                    self.event_log.append({
                        "tick": self.tick_count,
                        "type": "disturbance_spawned",
                        "detail": f"disturbance {disturbance_id} detected",
                        "disturbance": disturbance_id,
                    })
                elif kind == "dispatched":
                    self.event_log.append({
                        "tick": self.tick_count,
                        "type": "disturbance_dispatched",
                        "detail": f"{result['drone_id']} dispatched to investigate {disturbance_id}",
                        "drone": result["drone_id"],
                        "disturbance": disturbance_id,
                    })
                else:
                    self.event_log.append({
                        "tick": self.tick_count,
                        "type": "disturbance_resolved",
                        "detail": f"{result['drone_id']} resolved disturbance {disturbance_id}",
                        "drone": result["drone_id"],
                        "disturbance": disturbance_id,
                    })

        if len(self.event_log) > MAX_EVENT_LOG:
            self.event_log = self.event_log[-MAX_EVENT_LOG:]

    def _log_transition(self, drone_id: str, previous, election: NexusElection) -> None:
        if previous is None:
            return
        prev_role, _ = previous
        kind = _transition_kind(prev_role, election.role)

        if kind == "started":
            self.metrics.elections_started += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "election_started",
                "detail": f"{drone_id} lost contact with its nexus and started a campaign (term {election.term})",
                "drones": [drone_id],
            })
        elif kind == "won":
            self.metrics.elections_won += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "election_won",
                "detail": f"{drone_id} elected nexus (term {election.term})",
                "winner": drone_id,
            })
        elif kind == "merged":
            self.metrics.merges += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "swarms_merged",
                "detail": f"{drone_id} yielded nexus to {election.known_nexus_id} (newer term {election.term})",
                "drones": [drone_id],
            })

    def _log_commander_transition(self, drone_id: str, previous, election: NexusElection) -> None:
        """Same classification as _log_transition, distinct event types
        and counters -- see command.py for why this is a second,
        independent election rather than the same one relabeled."""
        if previous is None:
            return
        prev_role, _ = previous
        kind = _transition_kind(prev_role, election.role)

        if kind == "started":
            self.metrics.commander_elections_started += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "commander_election_started",
                "detail": f"{drone_id} (platoon nexus) lost contact with the commander and started a campaign (term {election.term})",
                "drones": [drone_id],
            })
        elif kind == "won":
            self.metrics.commander_elections_won += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "commander_elected",
                "detail": f"{drone_id} elected commander (term {election.term})",
                "winner": drone_id,
            })
        elif kind == "merged":
            self.metrics.commander_merges += 1
            self.event_log.append({
                "tick": self.tick_count,
                "type": "commander_merged",
                "detail": f"{drone_id} yielded commander to {election.known_nexus_id} (newer term {election.term})",
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

    def _track_commander_recovery(self, drone_id: str, previous, election: NexusElection) -> None:
        """Same shape as _track_recovery, a separate gap-start dict
        (self._commander_gap_start) so a drone that's simultaneously
        tracked at both tiers (it's a platoon nexus running a commander
        campaign) never has one tier's bookkeeping clobber the other's."""
        if previous is None:
            return
        _, prev_commander = previous
        curr_commander = election.known_nexus_id
        if prev_commander is not None and curr_commander is None:
            self._commander_gap_start[drone_id] = self.time_s
        elif prev_commander is None and curr_commander is not None:
            start = self._commander_gap_start.pop(drone_id, None)
            if start is not None:
                self.metrics.record_commander_recovery(self.time_s - start)

    def _move(self) -> None:
        for drone in self.drones.values():
            if not drone.alive:
                continue
            if self.patrol is not None and drone.investigating_disturbance_id is not None:
                self._move_toward_disturbance(drone)
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

    def _move_toward_disturbance(self, drone) -> None:
        """Same steer-and-hold shape as `_move_toward_zone`, targeting a
        point rather than a zone -- holds once within the investigation
        range instead of overshooting through it, since that's the range
        `PatrolState._advance` itself checks to accrue investigation
        progress."""
        disturbance = self.patrol.disturbances.get(drone.investigating_disturbance_id)
        if disturbance is None:
            drone.investigating_disturbance_id = None
            return
        dx, dy = disturbance.x - drone.x, disturbance.y - drone.y
        distance = math.hypot(dx, dy)
        speed = math.hypot(drone.vx, drone.vy) or 4.0
        if distance <= max(self.patrol.config.investigation_range * 0.6, speed):
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
        snapshot = self.metrics.snapshot(self.mesh, self.elections)
        if self.command is not None:
            snapshot["commander"] = self.metrics.commander_snapshot(self.commander_elections)
        return snapshot

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
        if self.command is not None:
            state["command"] = self.command.to_state_dict(self)
        if self.patrol is not None:
            state["patrol"] = self.patrol.to_state_dict()
        return state
