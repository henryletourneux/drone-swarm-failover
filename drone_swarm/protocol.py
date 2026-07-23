"""Message schemas exchanged over the simulated mesh network.

This is the wire-format contract the election state machine runs on: it
only ever produces/consumes these message types, and never reaches into
another drone's internal state directly. That boundary is what makes the
mesh's latency and packet loss "real" in the simulation rather than every
drone quietly cheating by peeking at global state.

`signature` / `credential` / `quorum_certificate` are optional and unused
unless `SwarmConfig.bft_mode` is on (see election.py and identity.py) --
plain messages work exactly as before otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    sender_id: str
    sent_at_s: float


@dataclass(frozen=True)
class NexusHeartbeat(Message):
    """Periodic 'I am still alive and coordinating' announcement from the
    current nexus."""

    term: int
    signature: bytes = None
    # A bundle of the ElectionMessages that legitimately elected this
    # nexus for this term -- required to trust the term claim in BFT mode.
    quorum_certificate: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class ElectionMessage(Message):
    """Bully-algorithm candidacy: a drone announcing itself for a term
    after missing nexus heartbeats."""

    term: int
    priority: float
    signature: bytes = None
    credential: object = None  # identity.Credential, when bft_mode is on


def election_message_payload(sender_id: str, sent_at_s: float, term: int, priority: float) -> bytes:
    return f"election:{sender_id}:{sent_at_s}:{term}:{priority}".encode()


def heartbeat_payload(sender_id: str, sent_at_s: float, term: int) -> bytes:
    return f"heartbeat:{sender_id}:{sent_at_s}:{term}".encode()
