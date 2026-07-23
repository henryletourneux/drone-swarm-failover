"""Concrete attacks a rogue transmitter can mount against a BFT-mode swarm.

Every attack constructs a plausible-looking message with attacker-controlled
crypto and injects it through the ONLY public injection point a real radio
transmitter would have: `swarm.mesh.broadcast()`. None of them read a real
drone's private key or a real issued credential -- that would prove nothing.

Each method returns an `Injection` describing exactly what was put on the
wire, the `claim` a receiver would have to adopt for the attack to have
worked, and the specific `defense` (with code reference) expected to stop it.
Whether it actually worked is decided by `campaign.py` purely from observed
swarm behaviour, never from an internal logging hook.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from drone_swarm.protocol import ElectionMessage, NexusHeartbeat
from .identity import forged_credential, forged_identity, rogue_authority

GARBAGE_SIGNATURE_LEN = 64  # Ed25519 signatures are 64 bytes


@dataclass
class Injection:
    """A record of one attack that was placed on the mesh."""

    name: str
    description: str
    defense: str
    # What a live drone would have to end up believing for the forgery to
    # count as adopted: {"sender_id": ..., "term": ...}. None means the
    # attack has no distinct "adopted" state (replay/flood), so the campaign
    # instead treats ANY change to the stable swarm as a successful effect.
    claim: dict | None
    messages: list = field(default_factory=list)


class MeshWiretap:
    """Passive RF eavesdropper: wraps `mesh.broadcast` so every message that
    crosses the mesh is recorded, without altering delivery in any way. This
    models a listener with an antenna -- it only ever reads what is already
    being transmitted in the clear, which is how the attacker learns the
    current nexus/term and captures genuinely-signed messages to replay.

    Capped at `MAX_CAPTURED` (keeping the most recent) so this stays safe to
    attach to a long-running swarm (e.g. a live server), not just a short
    one-shot CLI campaign -- an unbounded capture list would otherwise grow
    for as long as the process stays up."""

    MAX_CAPTURED = 200

    def __init__(self, swarm) -> None:
        self._swarm = swarm
        self.captured: list = []
        self._real_broadcast = swarm.mesh.broadcast

        def tapped(message, now_s):
            self.captured.append(message)
            if len(self.captured) > self.MAX_CAPTURED:
                del self.captured[: len(self.captured) - self.MAX_CAPTURED]
            return self._real_broadcast(message, now_s)

        swarm.mesh.broadcast = tapped

    def detach(self) -> None:
        self._swarm.mesh.broadcast = self._real_broadcast

    @property
    def heartbeats(self) -> list:
        return [m for m in self.captured if isinstance(m, NexusHeartbeat)]

    @property
    def election_messages(self) -> list:
        return [m for m in self.captured if isinstance(m, ElectionMessage)]

    def latest_heartbeat(self):
        hbs = self.heartbeats
        return hbs[-1] if hbs else None

    def observed_term(self) -> int:
        """Highest term overheard on any message (what the attacker knows
        about the swarm's progress purely from listening)."""
        terms = [m.term for m in self.captured if hasattr(m, "term")]
        return max(terms) if terms else 0


class Antagonist:
    """The adversary. Holds a reference to the target swarm and a passive
    wiretap, and exposes one method per attack. Every method injects onto
    `swarm.mesh` and returns an `Injection`."""

    def __init__(self, swarm) -> None:
        self.swarm = swarm
        self.wiretap = MeshWiretap(swarm)

    # -- helpers ----------------------------------------------------------

    def _inject(self, message) -> None:
        self.swarm.mesh.broadcast(message, self.swarm.time_s)

    def _garbage_signature(self) -> bytes:
        return os.urandom(GARBAGE_SIGNATURE_LEN)

    # -- attack 1: impersonation -----------------------------------------

    def impersonate_nexus(self, victim_id: str, term: int) -> Injection:
        """Forge a NexusHeartbeat claiming to be `victim_id` at a fresh term,
        signed with an attacker-generated key (NOT victim's real key)."""
        forger = forged_identity(victim_id)
        now = self.swarm.time_s
        from drone_swarm.protocol import heartbeat_payload
        hb = NexusHeartbeat(
            sender_id=victim_id,
            sent_at_s=now,
            term=term,
            signature=forger.sign(heartbeat_payload(victim_id, now, term)),
        )
        self._inject(hb)
        return Injection(
            name="impersonation",
            description=f"Forged heartbeat claiming to be nexus {victim_id} at term {term}, "
            f"signed with an attacker key.",
            defense="election._verify_heartbeat -> IdentityRegistry.verify_signature: the "
            f"attacker key is not {victim_id}'s registered public key, so verify() raises "
            "InvalidSignature and the heartbeat is dropped before it can be adopted.",
            claim={"sender_id": victim_id, "term": term},
            messages=[hb],
        )

    # -- attack 2: unsigned / garbage-signature --------------------------

    def inject_garbage_heartbeat(self, victim_id: str, term: int) -> Injection:
        """The crudest attack: a heartbeat with no valid signature at all
        (random bytes). Should be trivially rejected."""
        now = self.swarm.time_s
        hb = NexusHeartbeat(
            sender_id=victim_id,
            sent_at_s=now,
            term=term,
            signature=self._garbage_signature(),
        )
        self._inject(hb)
        return Injection(
            name="unsigned/garbage-signature",
            description=f"Heartbeat impersonating {victim_id} at term {term} with a random "
            "64-byte garbage signature.",
            defense="election._verify_heartbeat -> verify_signature: random bytes are not a "
            "valid Ed25519 signature over the payload, so verify() raises InvalidSignature "
            "and the message is dropped. (A signature=None message is dropped one line earlier.)",
            claim={"sender_id": victim_id, "term": term},
            messages=[hb],
        )

    # -- attack 3: priority forgery --------------------------------------

    def forge_priority(self, victim_id: str, term: int, inflated_priority: float) -> Injection:
        """Impersonate `victim_id` announcing a candidacy with an inflated
        priority, backed by a credential the attacker self-issued from its
        own rogue authority (not the swarm's real SwarmAuthority)."""
        forger = forged_identity(victim_id)
        authority = rogue_authority()
        credential = forged_credential(authority, victim_id, inflated_priority)
        now = self.swarm.time_s
        from drone_swarm.protocol import election_message_payload
        em = ElectionMessage(
            sender_id=victim_id,
            sent_at_s=now,
            term=term,
            priority=inflated_priority,
            signature=forger.sign(
                election_message_payload(victim_id, now, term, inflated_priority)
            ),
            credential=credential,
        )
        self._inject(em)
        return Injection(
            name="priority-forgery",
            description=f"Candidacy impersonating {victim_id} claiming inflated priority "
            f"{inflated_priority}, backed by a credential self-signed by a rogue authority.",
            defense="election._verify_election_message: verify_signature fails first (attacker "
            "key != registered key). Even past that, verify_credential checks the credential "
            "against the REAL authority's public key, and the rogue authority's signature does "
            "not verify -- so the inflated priority is never trusted.",
            claim={"sender_id": victim_id, "term": term},
            messages=[em],
        )

    # -- attack 4: term inflation ----------------------------------------

    def inflate_term(self, victim_id: str, term: int) -> Injection:
        """Impersonate a nexus asserting a huge term jump with an empty
        quorum certificate -- claiming an election happened that never did."""
        forger = forged_identity(victim_id)
        now = self.swarm.time_s
        from drone_swarm.protocol import heartbeat_payload
        hb = NexusHeartbeat(
            sender_id=victim_id,
            sent_at_s=now,
            term=term,
            signature=forger.sign(heartbeat_payload(victim_id, now, term)),
            quorum_certificate=(),
        )
        self._inject(hb)
        return Injection(
            name="term-inflation",
            description=f"Heartbeat impersonating {victim_id} asserting a huge term jump to "
            f"{term} with an EMPTY quorum certificate.",
            defense="election._verify_heartbeat: the forged signature is rejected first. Even "
            "with a valid signature, term > known+1 forces _verify_quorum_certificate, and an "
            "empty cert yields only 1 distinct signer, which is not > total_swarm_size/2 -- "
            "so the unproven term jump is dropped.",
            claim={"sender_id": victim_id, "term": term},
            messages=[hb],
        )

    # -- attack 5: replay / repurposed evidence --------------------------

    def replay_captured_heartbeat(self) -> Injection:
        """Re-broadcast a genuinely-valid heartbeat the attacker overheard,
        verbatim, at a later (stale) time. The signature is real, so this
        tests whether a valid-but-stale message can gain the attacker
        anything."""
        captured = self.wiretap.latest_heartbeat()
        if captured is None:
            return Injection(
                name="replay",
                description="No heartbeat was overheard to replay.",
                defense="n/a -- nothing captured.",
                claim=None,
                messages=[],
            )
        self._inject(captured)  # verbatim: same signature, same sent_at_s, same term
        return Injection(
            name="replay",
            description=f"Re-broadcast a genuine captured heartbeat from {captured.sender_id} "
            f"(term {captured.term}) verbatim at a later time.",
            defense="The signature is valid, but the payload binds sender+term+time, so the "
            "attacker cannot alter it. In _ingest_heartbeats the term is not newer than what "
            "receivers already know, so nothing changes -- replay gains no ground.",
            claim=None,
            messages=[captured],
        )

    def repurpose_certificate(self, victim_id: str, term: int) -> Injection:
        """Bundle genuine captured ElectionMessages into a quorum certificate
        for a DIFFERENT term than they were actually signed for, wrapped in a
        forged heartbeat -- reusing real evidence out of context."""
        captured = list(self.wiretap.election_messages)
        forger = forged_identity(victim_id)
        now = self.swarm.time_s
        from drone_swarm.protocol import heartbeat_payload
        hb = NexusHeartbeat(
            sender_id=victim_id,
            sent_at_s=now,
            term=term,
            signature=forger.sign(heartbeat_payload(victim_id, now, term)),
            quorum_certificate=tuple(captured),
        )
        self._inject(hb)
        return Injection(
            name="repurposed-certificate",
            description=f"Forged term-{term} heartbeat for {victim_id} carrying {len(captured)} "
            "genuine captured candidacies that were signed for OTHER terms.",
            defense="election._verify_quorum_certificate skips every candidacy whose term != the "
            "heartbeat's term, so the out-of-context evidence counts for nothing (and the wrapping "
            "heartbeat's forged signature is rejected regardless).",
            claim={"sender_id": victim_id, "term": term},
            messages=[hb],
        )

    # -- attack 6: flood / spam (honestly, not mitigated) ----------------

    def flood(self, victim_id: str, count: int = 200) -> Injection:
        """A burst of garbage messages. There is no rate limiting, so this is
        NOT specifically defended -- reported honestly rather than as blocked."""
        now = self.swarm.time_s
        sent = []
        for i in range(count):
            hb = NexusHeartbeat(
                sender_id=victim_id,
                sent_at_s=now,
                term=self.wiretap.observed_term() + 1,
                signature=self._garbage_signature(),
            )
            self._inject(hb)
            sent.append(hb)
        return Injection(
            name="flood/spam",
            description=f"Burst of {count} garbage heartbeats impersonating {victim_id}.",
            defense="NOT specifically mitigated: no rate limiting exists. Each individual message "
            "is still rejected by the signature check so the ELECTION is not disrupted, but the "
            "resource/bandwidth cost of processing the burst is unmitigated -- an honest gap.",
            claim=None,
            messages=sent,
        )
