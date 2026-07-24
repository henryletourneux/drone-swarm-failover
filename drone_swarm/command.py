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

## Dynamic platoon membership

`platoon_of` starts as `CommandConfig`'s static seed but is no longer the
source of truth after construction -- `CommandState.platoon_of` (a plain
mutable dict) is, and `reassign_platoon()` is how it changes: move a
drone to a different platoon slot, and it gets a brand-new, term-0
election scoped to that platoon's layer. This is exactly the same
"cold-join is always safe to absorb" reasoning already established above
for the commander tier, one level down -- from the OLD platoon's point of
view, the drone just went silent (a case Bully+recency already handles),
and the NEW platoon absorbs a fresh follower the same way it would absorb
any drone that just came back into range. `commander_allocator.py`'s
`HeuristicCommanderAllocator` is what actually decides who moves where
and why (see that module).

**Deliberately restricted to plain (non-BFT) mode.** `election.py`'s own
module docstring already states, in its own words, that this project
"prioritizes defending against a minority of malicious/rogue nodes within
a partition -- not ... truly dynamic membership," calling dynamic BFT
membership "an open problem in real BFT systems too." Reassigning a
drone gives its new platoon a correct `total_swarm_size` for the quorum
threshold, but does NOT retroactively fix the OTHER members' elections,
whose `total_swarm_size` was fixed at their own construction -- in plain
mode that value is never consulted for anything (`_verify_quorum_
certificate` is the only reader, and it's bft_mode-gated), so the
staleness is inert; under bft_mode it would matter. Rather than solve
that open problem, `server.py` simply never wires a commander allocator
into a bft_mode config -- Security mode's platoons stay static, exactly
as before this feature existed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .election import ElectionRole, NexusElection


@dataclass(frozen=True)
class CommandConfig:
    """`platoon_of` must map EVERY drone id in the swarm to a platoon id --
    required, validated eagerly at `Swarm.__init__` (fail fast rather
    than silently leaving a drone unassigned). This is only the STARTING
    assignment now -- see "Dynamic platoon membership" above for how (and
    under what restriction) it can change afterward. Platoon formation
    itself is deliberately not derived from topology/position here; that
    would be a much bigger, separate feature (coverage-path-planning-
    style area decomposition), not this one."""

    platoon_of: dict


class CommandState:
    def __init__(self, config: CommandConfig, drones: dict, allocator=None) -> None:
        missing = sorted(d_id for d_id in drones if d_id not in config.platoon_of)
        if missing:
            raise ValueError(f"CommandConfig.platoon_of is missing drone id(s): {missing}")
        self.config = config
        # The live, mutable source of truth from here on -- config.platoon_of
        # is only ever read again as this dict's initial seed, in
        # build_platoon_elections below.
        self.platoon_of: dict = dict(config.platoon_of)
        self.platoon_ids = sorted({config.platoon_of[d_id] for d_id in drones})
        # None (default) means platoons stay exactly as configured,
        # forever -- the original, fully backward-compatible behavior.
        # Only set for modes that explicitly opt in (see module docstring
        # for why bft_mode never does).
        self.allocator = allocator
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
            platoon_sizes[pid] = sum(1 for d_id in drones if self.platoon_of[d_id] == pid)

        elections = {}
        for d_id, drone in drones.items():
            pid = self.platoon_of[d_id]
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

    def platoon_size(self, platoon_id: str) -> int:
        return sum(1 for pid in self.platoon_of.values() if pid == platoon_id)

    def reassign_platoon(self, swarm, drone_id: str, new_platoon_id: str) -> None:
        """Moves `drone_id` into `new_platoon_id`, giving it a fresh,
        term-0 election scoped to the new platoon's layer -- see
        "Dynamic platoon membership" in the module docstring for why this
        is safe. A no-op if the drone is already in that platoon (so
        callers can call this unconditionally every reallocation pass
        without churning elections for drones that didn't actually
        move)."""
        if self.platoon_of.get(drone_id) == new_platoon_id:
            return
        self.platoon_of[drone_id] = new_platoon_id
        drone = swarm.drones[drone_id]
        drone.platoon_id = new_platoon_id
        swarm.elections[drone_id] = NexusElection(
            nexus_heartbeat_interval_s=swarm.config.nexus_heartbeat_interval_s,
            nexus_timeout_s=swarm.config.nexus_timeout_s,
            bft_mode=swarm.config.bft_mode,
            identity=swarm.identities.get(drone_id),
            credential=swarm.credentials.get(drone_id),
            registry=swarm.registry,
            total_swarm_size=self.platoon_size(new_platoon_id),
            layer=f"platoon:{new_platoon_id}",
        )

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
            member_ids = sorted(d_id for d_id in swarm.drones if self.platoon_of[d_id] == pid)
            nexus_id = next(
                (
                    d_id for d_id in member_ids
                    if swarm.drones[d_id].alive and swarm.elections[d_id].role == ElectionRole.NEXUS
                ),
                None,
            )
            platoons[pid] = {"member_ids": member_ids, "nexus_id": nexus_id}
        return {"platoons": platoons, "commander_ids": self.commander_ids()}
