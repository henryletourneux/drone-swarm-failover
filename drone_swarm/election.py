"""Heartbeat/term-based Bully leader ("nexus") election, run per-drone over
a lossy, latent mesh network.

This replaces an earlier version of this module that did a synchronous
global recompute every tick — every drone effectively had instant,
perfect knowledge of the whole swarm, which made convergence trivial but
wasn't a very honest simulation of a real mesh network. Here, no drone
ever has global knowledge: everything a drone believes is purely a
function of which `protocol.py` messages actually reached it, and when,
via `MeshNetwork`.

Convergence sketch
-------------------
* A drone with no live nexus (never seen one, or its heartbeats have gone
  stale for longer than `nexus_timeout_s`) opens an election for the next
  *term* and broadcasts its candidacy (its priority).
* Candidacies are ordered by `(priority, drone_id)` — higher priority
  wins, id breaks ties deterministically so two equal-priority drones can
  never both win the same term.
* A candidate that hears a *better* candidacy for its term is superseded
  and steps back to waiting; the sole un-superseded candidate declares
  victory when its election window closes, and starts heartbeating.
* Two independently-elected nexuses meeting (e.g. a partition
  reconnecting) are resolved the way real consensus protocols do it: on
  contact, the *newer term* always wins, not priority — priority only
  ever decides a single fresh election, never a stale-vs-fresh conflict
  (an old, possibly-stale leader's priority claim isn't trustworthy
  evidence about what's happened on the other side of a partition; recency
  is). That makes merges an emergent property of ordinary heartbeat
  handling — no special-cased merge/runoff logic required, unlike the
  synchronous version this replaced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from .protocol import ElectionMessage, NexusHeartbeat


class ElectionRole(Enum):
    NEXUS = auto()
    FOLLOWER = auto()
    CANDIDATE = auto()


@dataclass
class _Campaign:
    """Book-keeping for an in-progress election this drone is running."""

    term: int
    deadline_s: float
    superseded: bool = False


class NexusElection:
    """Bully-algorithm election state machine for a single drone.

    `step()` is given the current time and whatever messages arrived this
    tick, and returns this drone's (possibly changed) role plus any
    messages it wants broadcast onto the mesh. It never reaches into
    another drone's state directly.
    """

    def __init__(
        self,
        nexus_heartbeat_interval_s: float,
        nexus_timeout_s: float,
        election_window_s: float | None = None,
    ) -> None:
        self._heartbeat_interval_s = nexus_heartbeat_interval_s
        self._timeout_s = nexus_timeout_s
        self._election_window_s = (
            election_window_s if election_window_s is not None else nexus_heartbeat_interval_s
        )

        self.role: ElectionRole = ElectionRole.FOLLOWER
        self.known_nexus_id: str | None = None
        self.term: int = 0
        # Last time we had positive contact with a live nexus (or -inf if
        # we've never seen one, so the very first tick opens an election).
        self._last_nexus_contact_s: float = -math.inf
        self._campaign: _Campaign | None = None
        self._last_heartbeat_emitted_s: float = -math.inf

    def step(self, now_s: float, self_id: str, self_priority: float, incoming: list) -> tuple:
        outgoing: list = []

        self._ingest_heartbeats(now_s, self_id, incoming)
        self._ingest_elections(now_s, self_id, self_priority, incoming, outgoing)
        self._maybe_start_election(now_s, self_id, self_priority, outgoing)
        self._maybe_resolve_campaign(now_s, self_id, outgoing)
        self._re_announce_if_candidate(now_s, self_id, self_priority, outgoing)
        self._emit_heartbeat_if_nexus(now_s, self_id, outgoing)

        return self.role, outgoing

    # -- message ingestion ------------------------------------------------

    def _ingest_heartbeats(self, now_s: float, self_id: str, incoming: list) -> None:
        """Adopt the nexus advertised by the best heartbeat we heard.

        A heartbeat is authoritative if its term is newer than ours, or the
        same term from a higher-id sender (the deterministic tie-break that
        collapses any same-term split-brain onto a single nexus).
        """
        for msg in incoming:
            if not isinstance(msg, NexusHeartbeat) or msg.sender_id == self_id:
                continue
            newer_term = msg.term > self.term
            same_term_ok = msg.term == self.term and (
                self.known_nexus_id is None or msg.sender_id >= self.known_nexus_id
            )
            if newer_term or same_term_ok:
                self.term = msg.term
                self.known_nexus_id = msg.sender_id
                self._last_nexus_contact_s = now_s
                self.role = ElectionRole.FOLLOWER
                self._campaign = None

    def _ingest_elections(self, now_s: float, self_id: str, self_priority: float, incoming: list, outgoing: list) -> None:
        """React to competing candidacies: join the election, and if we
        hear a superior candidate, mark our own campaign superseded so we
        won't wrongly declare victory."""
        for msg in incoming:
            if not isinstance(msg, ElectionMessage) or msg.sender_id == self_id:
                continue
            if msg.term < self.term:
                continue  # stale election for an already-decided term

            # A same-term candidacy that arrives *after* we've already
            # resolved that exact term (we won it, or we already know who
            # did) is a stale duplicate, not new information — e.g. a
            # multi-hop relay straggler, or a drone that hadn't yet heard
            # our victory heartbeat when it broadcast its own candidacy.
            # Without this check such a message would wrongly knock an
            # already-elected nexus (or an already-settled follower) back
            # into a fresh campaign for a term that's already decided.
            already_resolved_for_this_term = msg.term == self.term and (
                self.role == ElectionRole.NEXUS
                or (self.role == ElectionRole.FOLLOWER and self.known_nexus_id is not None)
            )
            if already_resolved_for_this_term:
                continue

            self._join_election(now_s, msg.term, self_id, self_priority, outgoing)

            if (msg.priority, msg.sender_id) > (self_priority, self_id):
                if self._campaign is not None:
                    self._campaign.superseded = True

    # -- election lifecycle ------------------------------------------------

    def _maybe_start_election(self, now_s: float, self_id: str, self_priority: float, outgoing: list) -> None:
        if self.role == ElectionRole.NEXUS or self._campaign is not None:
            return
        nexus_lost = (now_s - self._last_nexus_contact_s) > self._timeout_s
        if nexus_lost:
            self._begin_campaign(now_s, self.term + 1, self_id, self_priority, outgoing)

    def _join_election(self, now_s: float, term: int, self_id: str, self_priority: float, outgoing: list) -> None:
        if self._campaign is not None and self._campaign.term >= term:
            return
        self._begin_campaign(now_s, max(term, self.term), self_id, self_priority, outgoing)

    def _begin_campaign(self, now_s: float, term: int, self_id: str, self_priority: float, outgoing: list) -> None:
        self.term = term
        self.role = ElectionRole.CANDIDATE
        self.known_nexus_id = None
        self._campaign = _Campaign(term=term, deadline_s=now_s + self._election_window_s)
        outgoing.append(ElectionMessage(sender_id=self_id, sent_at_s=now_s, term=term, priority=self_priority))

    def _maybe_resolve_campaign(self, now_s: float, self_id: str, outgoing: list) -> None:
        campaign = self._campaign
        if campaign is None or now_s < campaign.deadline_s:
            return

        if not campaign.superseded:
            self.role = ElectionRole.NEXUS
            self.known_nexus_id = self_id
            self.term = campaign.term
            self._last_nexus_contact_s = now_s
            self._emit_heartbeat(now_s, self_id, outgoing)
        else:
            # Lost. Step back and grant the winner a fresh timeout window
            # to announce itself before we'd consider re-electing.
            self.role = ElectionRole.FOLLOWER
            self._last_nexus_contact_s = now_s
        self._campaign = None

    def _re_announce_if_candidate(self, now_s: float, self_id: str, self_priority: float, outgoing: list) -> None:
        """Re-broadcast candidacy each tick while the election is open so
        the eventual winner still asserts itself under packet loss."""
        campaign = self._campaign
        if campaign is None or self.role != ElectionRole.CANDIDATE:
            return
        already_sent = any(isinstance(m, ElectionMessage) and m.sender_id == self_id for m in outgoing)
        if already_sent:
            return
        outgoing.append(ElectionMessage(sender_id=self_id, sent_at_s=now_s, term=campaign.term, priority=self_priority))

    # -- nexus duties -------------------------------------------------------

    def _emit_heartbeat_if_nexus(self, now_s: float, self_id: str, outgoing: list) -> None:
        if self.role != ElectionRole.NEXUS:
            return
        self.known_nexus_id = self_id
        if (now_s - self._last_heartbeat_emitted_s) >= self._heartbeat_interval_s:
            self._emit_heartbeat(now_s, self_id, outgoing)

    def _emit_heartbeat(self, now_s: float, self_id: str, outgoing: list) -> None:
        outgoing.append(NexusHeartbeat(sender_id=self_id, sent_at_s=now_s, term=self.term))
        self._last_heartbeat_emitted_s = now_s
        self._last_nexus_contact_s = now_s
