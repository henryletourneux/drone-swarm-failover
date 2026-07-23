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
`max_relay_hops`, with per-hop latency and packet loss.

`seen_by` is a single set shared (and mutated in place) across every relay
path for one original broadcast, not recomputed per-path -- this matters:
in a densely-connected mesh, a node can be reachable via several different
relay paths, and without a *global* record of who's already gotten a copy,
each path would independently re-deliver to the same recipients, and each
of those redundant copies would itself keep relaying further. That's
combinatorial, not linear, in a dense enough graph (measured: one broadcast
in a 14-drone fully-connected mesh with the default hop limit produced
millions of redundant sends before this was global). A recipient is only
marked "seen" once a delivery to them actually succeeds (not on a
loss-dropped attempt), so redundant paths still provide their real
benefit -- resilience against packet loss -- without the blowup.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .spatial_grid import SpatialGrid


@dataclass
class _InFlightMessage:
    message: object
    recipient_id: str
    deliver_at_s: float
    hops_remaining: int
    seen_by: set  # shared across every in-flight copy of this one broadcast


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

        # Neighbor lookups are the hot path (called once per relay hop per
        # recipient, every tick) -- a naive check-every-drone scan is O(n)
        # per call and O(n^2) per tick, which measured catastrophically bad
        # past a few dozen drones. The grid turns that into ~O(n) per tick
        # by only checking drones in nearby cells. Rebuilt lazily, once per
        # tick, the first time a neighbor query is made after positions
        # changed -- callers never need to know this optimization exists.
        self._grid = SpatialGrid(cell_size=max(comm_range, 1.0))
        self._grid_dirty = True

        # Message-volume counters for metrics.py -- counted per (message,
        # recipient) transmission attempt, including relay hops, since
        # that's what actually reflects radio traffic/bandwidth cost.
        self.sent_count = 0
        self.delivered_count = 0
        self.dropped_loss_count = 0

    def update_known_position(self, drone_id: str, x: float, y: float, alive: bool) -> None:
        self._positions[drone_id] = (x, y)
        if alive:
            self._alive.add(drone_id)
        else:
            self._alive.discard(drone_id)
        self._grid_dirty = True

    def _ensure_grid(self) -> None:
        if not self._grid_dirty:
            return
        self._grid.rebuild({did: self._positions[did] for did in self._alive})
        self._grid_dirty = False

    def neighbors_of(self, drone_id: str) -> list:
        self._ensure_grid()
        return self._grid.neighbors_within(drone_id, self.comm_range)

    def broadcast(self, message, now_s: float) -> None:
        """Send `message` to every drone in range of its sender, then let
        it propagate onward through `deliver_due_messages` up to the
        configured hop limit."""
        origin_id = message.sender_id
        seen_by = {origin_id}
        self._enqueue_to_neighbors(
            message,
            relaying_from=origin_id,
            seen_by=seen_by,
            hops_remaining=self.max_relay_hops,
            now_s=now_s,
        )

    def _enqueue_to_neighbors(self, message, relaying_from, seen_by, hops_remaining, now_s) -> None:
        for recipient_id in self.neighbors_of(relaying_from):
            if recipient_id in seen_by:
                continue
            self.sent_count += 1
            if self._rng.random() < self.packet_loss_rate:
                self.dropped_loss_count += 1
                continue  # dropped packet -- NOT marked seen, so another relay path may still reach them
            seen_by.add(recipient_id)  # marked only once a delivery actually succeeds
            self._queue.append(_InFlightMessage(
                message=message,
                recipient_id=recipient_id,
                deliver_at_s=now_s + self.latency_s,
                hops_remaining=hops_remaining,
                seen_by=seen_by,
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
                self.delivered_count += 1
                if item.hops_remaining > 0:
                    self._enqueue_to_neighbors(
                        item.message,
                        relaying_from=item.recipient_id,
                        seen_by=item.seen_by,
                        hops_remaining=item.hops_remaining - 1,
                        now_s=now_s,
                    )
            elif item.deliver_at_s > now_s:
                remaining.append(item)
            # else: recipient no longer alive -> message silently dropped
        self._queue = remaining
        return delivered
