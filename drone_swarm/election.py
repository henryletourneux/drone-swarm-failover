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

Byzantine fault tolerance (`bft_mode`)
---------------------------------------
Everything above trusts message content at face value — fine against
clean failures, not against a rogue transmitter. When `bft_mode` is on:

* Every outgoing message is signed with this drone's own key, and every
  incoming one is verified against the sender's known public key before
  it's trusted at all. An unsigned or badly-signed message is dropped,
  full stop — this defeats impersonation.
* Every ElectionMessage carries a Credential — this drone's priority,
  signed once by the SwarmAuthority at setup. A receiver checks the
  credential matches the claimed priority *and* is validly authority-
  signed, so a compromised drone can lie about lots of things but can't
  unilaterally claim a higher priority than it was actually issued.
* A heartbeat claiming a term more than one step ahead of what a receiver
  already knows must carry a QuorumCertificate — real, verified
  candidacies from a majority of the swarm, proving an election actually
  happened rather than one node just asserting a huge term number. Small,
  ordinary increments (the common case — routine cascading failover)
  never need one, so this doesn't change day-to-day behavior at all; it
  only closes the term-inflation attack specifically.

Honest limitation, stated plainly: the quorum threshold is a majority of
`total_swarm_size` (the swarm's original member count, fixed at setup).
That's a deliberate simplification, and it means a connected component
smaller than a majority of the original swarm can't *cryptographically
certify* a big term jump for itself, even though the underlying election
mechanism would otherwise let it operate independently. This project
prioritizes defending against a minority of malicious/rogue nodes within
a partition — not an unbounded majority, and not truly dynamic
membership, both of which are open problems in real BFT systems too.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

from .protocol import ElectionMessage, NexusHeartbeat, election_message_payload, heartbeat_payload


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
    # bft_mode only: verified candidacies seen for this exact term, keyed
    # by sender so a straggler/duplicate doesn't get double-counted —
    # the raw material for this campaign's quorum certificate if it wins.
    participants: dict = field(default_factory=dict)


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
        bft_mode: bool = False,
        identity=None,
        credential=None,
        registry=None,
        total_swarm_size: int = 1,
    ) -> None:
        self._heartbeat_interval_s = nexus_heartbeat_interval_s
        self._timeout_s = nexus_timeout_s
        self._election_window_s = (
            election_window_s if election_window_s is not None else nexus_heartbeat_interval_s
        )

        # bft_mode=False (default): none of this is touched, zero overhead,
        # behavior identical to before BFT existed.
        self._bft_mode = bft_mode
        self._identity = identity  # this drone's own DroneIdentity
        self._credential = credential  # this drone's authority-signed Credential
        self._registry = registry  # IdentityRegistry, shared read-only across the swarm
        self._total_swarm_size = total_swarm_size
        self._quorum_certificate: tuple = ()
        # bft_mode only: how many incoming messages this drone has dropped
        # for failing signature/credential/quorum verification. Public
        # (not underscore-prefixed) since metrics.py reads it directly.
        self.rejected_message_count = 0

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

    # -- BFT verification ---------------------------------------------------

    def _verify_election_message(self, msg: ElectionMessage) -> bool:
        if not self._bft_mode:
            return True
        if msg.signature is None or msg.credential is None:
            return False
        payload = election_message_payload(msg.sender_id, msg.sent_at_s, msg.term, msg.priority)
        if not self._registry.verify_signature(msg.sender_id, payload, msg.signature):
            return False
        credential = msg.credential
        if credential.drone_id != msg.sender_id or credential.priority != msg.priority:
            return False  # claiming a priority its credential doesn't actually back
        return self._registry.verify_credential(credential)

    def _verify_heartbeat(self, msg: NexusHeartbeat) -> bool:
        if not self._bft_mode:
            return True
        if msg.signature is None:
            return False
        payload = heartbeat_payload(msg.sender_id, msg.sent_at_s, msg.term)
        if not self._registry.verify_signature(msg.sender_id, payload, msg.signature):
            return False

        # Small, routine term increments (ordinary cascading failover)
        # never need a quorum certificate. Only a suspiciously large jump
        # does -- that's the actual term-inflation attack shape.
        if msg.term <= self.term + 1:
            return True
        return self._verify_quorum_certificate(msg)

    def _verify_quorum_certificate(self, msg: NexusHeartbeat) -> bool:
        distinct_signers = {msg.sender_id}
        for candidacy in msg.quorum_certificate:
            if not isinstance(candidacy, ElectionMessage):
                continue
            if candidacy.term != msg.term or candidacy.sender_id == msg.sender_id:
                continue
            if not self._verify_election_message(candidacy):
                continue  # doesn't count toward quorum -- couldn't be verified
            distinct_signers.add(candidacy.sender_id)
        return len(distinct_signers) > self._total_swarm_size / 2

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
            if not self._verify_heartbeat(msg):
                self.rejected_message_count += 1
                continue  # unsigned, forged, or an unproven term jump -- dropped
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
            if not self._verify_election_message(msg):
                self.rejected_message_count += 1
                continue  # unsigned, forged, or an uncertified priority claim -- dropped
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

            if self._bft_mode and self._campaign is not None and self._campaign.term == msg.term:
                self._campaign.participants[msg.sender_id] = msg

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
        outgoing.append(self._make_election_message(self_id, now_s, term, self_priority))

    def _maybe_resolve_campaign(self, now_s: float, self_id: str, outgoing: list) -> None:
        campaign = self._campaign
        if campaign is None or now_s < campaign.deadline_s:
            return

        if not campaign.superseded:
            self.role = ElectionRole.NEXUS
            self.known_nexus_id = self_id
            self.term = campaign.term
            self._last_nexus_contact_s = now_s
            self._quorum_certificate = tuple(campaign.participants.values())
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
        outgoing.append(self._make_election_message(self_id, now_s, campaign.term, self_priority))

    # -- nexus duties -------------------------------------------------------

    def _emit_heartbeat_if_nexus(self, now_s: float, self_id: str, outgoing: list) -> None:
        if self.role != ElectionRole.NEXUS:
            return
        self.known_nexus_id = self_id
        if (now_s - self._last_heartbeat_emitted_s) >= self._heartbeat_interval_s:
            self._emit_heartbeat(now_s, self_id, outgoing)

    def _emit_heartbeat(self, now_s: float, self_id: str, outgoing: list) -> None:
        signature = None
        if self._bft_mode:
            signature = self._identity.sign(heartbeat_payload(self_id, now_s, self.term))
        outgoing.append(NexusHeartbeat(
            sender_id=self_id,
            sent_at_s=now_s,
            term=self.term,
            signature=signature,
            quorum_certificate=self._quorum_certificate,
        ))
        self._last_heartbeat_emitted_s = now_s
        self._last_nexus_contact_s = now_s

    # -- message construction -------------------------------------------------

    def _make_election_message(self, self_id: str, now_s: float, term: int, self_priority: float) -> ElectionMessage:
        signature = None
        if self._bft_mode:
            signature = self._identity.sign(election_message_payload(self_id, now_s, term, self_priority))
        return ElectionMessage(
            sender_id=self_id,
            sent_at_s=now_s,
            term=term,
            priority=self_priority,
            signature=signature,
            credential=self._credential,
        )
