"""Live statistics derived from swarm activity — the difference between
"a demo you watch" and "a system you can measure."

`recovery_times_s` is deliberately defined as a *per-drone* quantity: how
long an individual drone went without a known nexus, from the moment its
belief was cleared (it lost contact, or started a fresh campaign) to the
moment it next had one (it won, or adopted a heartbeat). This is precise
and honestly computable straight from each drone's own state transitions.

A swarm-wide "time from this death to that election" metric was
deliberately NOT used instead: with multiple drones, partitions, and
concurrent campaigns, there's no single unambiguous way to match a
specific nexus death to the specific election that "recovered" from it.
The per-drone gap is the metric that's actually well-defined.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _distribution(values: list) -> dict:
    if not values:
        return {"count": 0, "mean_s": None, "p50_s": None, "p95_s": None, "max_s": None}
    ordered = sorted(values)
    n = len(ordered)

    def percentile(p: float) -> float:
        idx = min(n - 1, int(round(p * (n - 1))))
        return ordered[idx]

    return {
        "count": n,
        "mean_s": round(sum(ordered) / n, 3),
        "p50_s": round(percentile(0.50), 3),
        "p95_s": round(percentile(0.95), 3),
        "max_s": round(ordered[-1], 3),
    }


@dataclass
class SwarmMetrics:
    recovery_times_s: list = field(default_factory=list)
    elections_started: int = 0
    elections_won: int = 0
    merges: int = 0

    def record_recovery(self, duration_s: float) -> None:
        self.recovery_times_s.append(duration_s)

    def snapshot(self, mesh, elections: dict) -> dict:
        rejected = sum(getattr(e, "rejected_message_count", 0) for e in elections.values())
        return {
            "recovery": _distribution(self.recovery_times_s),
            "elections_started": self.elections_started,
            "elections_won": self.elections_won,
            "merges": self.merges,
            "messages_sent": mesh.sent_count,
            "messages_delivered": mesh.delivered_count,
            "messages_dropped_loss": mesh.dropped_loss_count,
            "security_rejections": rejected,
        }
