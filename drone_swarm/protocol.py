"""Message schemas exchanged over the simulated mesh network.

This is the wire-format contract the election state machine runs on: it
only ever produces/consumes these message types, and never reaches into
another drone's internal state directly. That boundary is what makes the
mesh's latency and packet loss "real" in the simulation rather than every
drone quietly cheating by peeking at global state.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    sender_id: str
    sent_at_s: float


@dataclass(frozen=True)
class NexusHeartbeat(Message):
    """Periodic 'I am still alive and coordinating' announcement from the
    current nexus."""

    term: int


@dataclass(frozen=True)
class ElectionMessage(Message):
    """Bully-algorithm candidacy: a drone announcing itself for a term
    after missing nexus heartbeats."""

    term: int
    priority: float
