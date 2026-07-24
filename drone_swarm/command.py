"""Hierarchical command: drones are grouped into static **platoons**, each
independently electing its own nexus via the exact same Bully/heartbeat
mechanism as a flat swarm (see election.py), and a second, dynamically-
recomposing election runs among whichever drones currently hold a
platoon-nexus seat to elect an overall **commander** — "the same
election/failover mechanism, applied recursively," not a second algorithm.

Message-layer tagging (`NexusHeartbeat.layer`/`ElectionMessage.layer`, see
protocol.py) is what keeps the two tiers from cross-contaminating on the
wire: every drone runs one `NexusElection` for its own platoon
(`layer=f"platoon:{platoon_id}"`), and drones that currently hold that
platoon's nexus seat additionally run a second, independent
`NexusElection` for the commander tier (`layer="commander"`). Both travel
over the exact same `MeshNetwork` — no separate long-range radio channel
exists or is needed: `mesh_network.py` relays any message through any
alive drone regardless of role, so commander-layer traffic between
physically-scattered platoon nexuses is already carried by ordinary
platoon members acting as relays, for free.

Dynamic membership, worked through plainly: a platoon-nexus seat can
change hands (failover, or a fresh election after a partition), so
`CommandState.commander_elections` is rebuilt every tick rather than
being fixed at construction like every other engine dict in this
codebase. When a drone stops holding a platoon-nexus seat (demoted or
killed), its commander engine is discarded outright — from every OTHER
commander-layer engine's point of view this is indistinguishable from
that participant going silent, a case the underlying Bully+recency
mechanism already handles (and already has test coverage for at the
flat layer): the followers of a vanished sitting commander simply time
out and re-campaign; a leading candidate that vanishes mid-campaign
leaves already-superseded rivals to self-heal after one more timeout
window; a drone that later regains its platoon-nexus seat cold-joins
with a brand-new, term-0 commander engine, which is always safe to
absorb (a fresh follower unconditionally adopts any newer-term
heartbeat, and a stale low-term candidacy from it is ignored by anyone
already converged).

Honest limitation, stated plainly (same principle as the quorum-
threshold limitation already documented in election.py): if the current
platoon-nexus subgraph isn't fully connected within `max_relay_hops`
(widely dispersed platoons, a small comm_range, or an actual partition),
no single commander converges -- each mutually-reachable cluster of
platoon nexuses independently elects its own, exactly mirroring the flat
layer's own already-tested multi-nexus-under-partition behavior. That's
why `commander_ids()` returns a list, not a single id: normally length
0 or 1, honestly >= 2 under a real partition.
"""
from __future__ import annotations

from dataclasses import dataclass

from .election import ElectionRole, NexusElection


@dataclass(frozen=True)
class CommandConfig:
    """`platoon_of` must map EVERY drone id in the swarm to a platoon id --
    static, required, validated eagerly at `Swarm.__init__` (fail fast
    rather than silently leaving a drone unassigned). Platoon formation
    itself is deliberately not derived from topology/position here; that
    would be a much bigger, separate feature (coverage-path-planning-
    style area decomposition), not this one."""

    platoon_of: dict


class CommandState:
    def __init__(self, config: CommandConfig, drones: dict) -> None:
        missing = sorted(d_id for d_id in drones if d_id not in config.platoon_of)
        if missing:
            raise ValueError(f"CommandConfig.platoon_of is missing drone id(s): {missing}")
        self.config = config
        self.platoon_ids = sorted({config.platoon_of[d_id] for d_id in drones})
        # drone_id -> NexusElection, populated/depopulated every tick to
        # track whichever drones currently hold a platoon-nexus seat --
        # the one engine dict in this codebase that's dynamically
        # rebuilt rather than fixed at construction.
        self.commander_elections: dict = {}

    def build_platoon_elections(self, drones: dict, identities: dict, credentials: dict, registry, base_kwargs: dict) -> dict:
        """Returns {drone_id: NexusElection}, one per drone, scoped to
        that drone's own platoon -- the drop-in replacement for the flat
        swarm's `self.elections` when command_config is set. Also stamps
        `drone.platoon_id` (static for the swarm's lifetime).

        `identities`/`credentials` are the same per-drone dicts
        `Swarm.__init__` already builds for bft_mode (empty dicts
        otherwise) -- passed in explicitly rather than reached for on a
        `swarm` object, since this runs during `Swarm.__init__` itself,
        before `self` is fully constructed."""
        platoon_sizes: dict = {}
        for pid in self.platoon_ids:
            platoon_sizes[pid] = sum(1 for d_id in drones if self.config.platoon_of[d_id] == pid)

        elections = {}
        for d_id, drone in drones.items():
            pid = self.config.platoon_of[d_id]
            drone.platoon_id = pid
            elections[d_id] = NexusElection(
                **base_kwargs,
                identity=identities.get(d_id),
                credential=credentials.get(d_id),
                registry=registry,
                total_swarm_size=platoon_sizes[pid],
                layer=f"platoon:{pid}",
            )
        return elections

    def commander_total_swarm_size(self) -> int:
        # Fixed: exactly one nexus seat per platoon is guaranteed by the
        # Bully tie-break at any settled moment, even though WHICH drone
        # fills each seat churns constantly.
        return len(self.platoon_ids)

    def reconcile_and_step_commander_layer(self, swarm, inboxes: dict, all_outgoing: list) -> None:
        current_nexus_ids = {
            d_id for d_id, election in swarm.elections.items()
            if swarm.drones[d_id].alive and election.role == ElectionRole.NEXUS
        }

        for d_id in list(self.commander_elections):
            if d_id not in current_nexus_ids:
                del self.commander_elections[d_id]
                swarm.drones[d_id].commander_id = None
                swarm._commander_gap_start.pop(d_id, None)

        for d_id in current_nexus_ids:
            if d_id in self.commander_elections:
                continue
            self.commander_elections[d_id] = NexusElection(
                nexus_heartbeat_interval_s=swarm.config.nexus_heartbeat_interval_s,
                nexus_timeout_s=swarm.config.nexus_timeout_s,
                bft_mode=swarm.config.bft_mode,
                identity=swarm.identities.get(d_id),
                credential=swarm.credentials.get(d_id),
                registry=swarm.registry,
                total_swarm_size=self.commander_total_swarm_size(),
                layer="commander",
            )

        previous_states = {}
        for d_id, election in self.commander_elections.items():
            drone = swarm.drones[d_id]
            previous_states[d_id] = (election.role, election.known_nexus_id)
            inbox = inboxes.get(d_id, [])
            _, outgoing = election.step(swarm.time_s, d_id, drone.priority, inbox)
            all_outgoing.append(outgoing)

        for d_id, election in self.commander_elections.items():
            drone = swarm.drones[d_id]
            drone.commander_id = election.known_nexus_id
            previous = previous_states.get(d_id)
            swarm._log_commander_transition(d_id, previous, election)
            swarm._track_commander_recovery(d_id, previous, election)

    def commander_ids(self) -> list:
        return sorted(
            d_id for d_id, election in self.commander_elections.items()
            if election.role == ElectionRole.NEXUS
        )

    def to_state_dict(self, swarm) -> dict:
        platoons = {}
        for pid in self.platoon_ids:
            member_ids = sorted(d_id for d_id in swarm.drones if self.config.platoon_of[d_id] == pid)
            nexus_id = next(
                (
                    d_id for d_id in member_ids
                    if swarm.drones[d_id].alive and swarm.elections[d_id].role == ElectionRole.NEXUS
                ),
                None,
            )
            platoons[pid] = {"member_ids": member_ids, "nexus_id": nexus_id}
        return {"platoons": platoons, "commander_ids": self.commander_ids()}
