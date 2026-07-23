"""Simulated mesh network: range-limited, lossy, latent message delivery
between drones, with true multi-hop relay.

This is the one place in the simulation allowed to consult ground-truth
positions (it's standing in for physical radio propagation, which in
reality "knows" the true distance between two antennas). Every other
module — in particular the election state machine — only ever sees
message contents that actually arrived through here, on their own delayed
schedule, possibly not at all.

Without this, a swarm's coordination is only ever tested against instant,
perfect delivery. With it, a message reaches every drone within
`comm_range` of its *original* sender in one hop, and is then
flood-relayed onward from each receiving node to its own neighbors, up to
`max_relay_hops`, with per-hop latency and packet loss and a `seen_by`
set so the same message never loops back to a node that already has it.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class _InFlightMessage:
    message: object
    recipient_id: str
    deliver_at_s: float
    hops_remaining: int
    seen_by: frozenset


class MeshNetwork:
    def __init__(
        self,
        comm_range: float,
        max_relay_hops: int = 4,
        packet_loss_rate: float = 0.02,
        latency_s: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        self.comm_range = comm_range
        self.max_relay_hops = max_relay_hops
        self.packet_loss_rate = packet_loss_rate
        self.latency_s = latency_s
        self._rng = rng if rng is not None else random.Random()
        self._positions: dict = {}
        self._alive: set = set()
        self._queue: list = []

    def update_known_position(self, drone_id: str, x: float, y: float, alive: bool) -> None:
        self._positions[drone_id] = (x, y)
        if alive:
            self._alive.add(drone_id)
        else:
            self._alive.discard(drone_id)

    def neighbors_of(self, drone_id: str) -> list:
        origin = self._positions.get(drone_id)
        if origin is None:
            return []
        ox, oy = origin
        out = []
        for other_id in self._alive:
            if other_id == drone_id:
                continue
            other_x, other_y = self._positions[other_id]
            if math.hypot(ox - other_x, oy - other_y) <= self.comm_range:
                out.append(other_id)
        return out

    def broadcast(self, message, now_s: float) -> None:
        """Send `message` to every drone in range of its sender, then let
        it propagate onward through `deliver_due_messages` up to the
        configured hop limit."""
        origin_id = message.sender_id
        self._enqueue_to_neighbors(
            message,
            relaying_from=origin_id,
            already_seen=frozenset({origin_id}),
            hops_remaining=self.max_relay_hops,
            now_s=now_s,
        )

    def _enqueue_to_neighbors(self, message, relaying_from, already_seen, hops_remaining, now_s) -> None:
        for recipient_id in self.neighbors_of(relaying_from):
            if recipient_id in already_seen:
                continue
            if self._rng.random() < self.packet_loss_rate:
                continue  # dropped packet
            self._queue.append(_InFlightMessage(
                message=message,
                recipient_id=recipient_id,
                deliver_at_s=now_s + self.latency_s,
                hops_remaining=hops_remaining,
                seen_by=already_seen,
            ))

    def deliver_due_messages(self, now_s: float) -> dict:
        """Advance delivery: return everything that has arrived by `now_s`,
        keyed by recipient. Anything delivered with hops remaining is also
        re-queued for relay onward from the recipient, so a message keeps
        propagating outward tick by tick until it either exhausts its hop
        budget or has reached every reachable node."""
        delivered: dict = {}
        remaining = []
        for item in self._queue:
            if item.deliver_at_s <= now_s and item.recipient_id in self._alive:
                delivered.setdefault(item.recipient_id, []).append(item.message)
                if item.hops_remaining > 0:
                    self._enqueue_to_neighbors(
                        item.message,
                        relaying_from=item.recipient_id,
                        already_seen=item.seen_by | {item.recipient_id},
                        hops_remaining=item.hops_remaining - 1,
                        now_s=now_s,
                    )
            elif item.deliver_at_s > now_s:
                remaining.append(item)
            # else: recipient no longer alive -> message silently dropped
        self._queue = remaining
        return delivered
