"""Scripted adversarial campaign against a BFT-mode swarm.

Boots a small hardened swarm, lets it elect a legitimate nexus, then runs
each attack in turn. Before and after each attack it snapshots every live
drone's (known_nexus_id, term) and ticks the sim forward so any effect can
propagate, then decides BLOCKED vs SUCCEEDED purely from that observed
behaviour -- there is no internal logging hook, by design.
"""
from __future__ import annotations

from dataclasses import dataclass

from drone_swarm.model import Drone
from drone_swarm.swarm import Swarm, SwarmConfig
from .attacks import Antagonist, Injection

CONVERGE_TICKS = 40
PROPAGATE_TICKS = 10


@dataclass
class AttackOutcome:
    injection: Injection
    blocked: bool
    before: dict
    after: dict


def build_swarm(seed: int = 7) -> Swarm:
    """Five stationary drones in mutual comm range, BFT hardened, zero packet
    loss so the baseline is a crisp, stable single-nexus swarm and any state
    change after an attack is unambiguously attributable to the attack."""
    priorities = [9.0, 7.0, 5.0, 3.0, 1.0]
    drones = [
        Drone(id=f"D{i}", x=400.0 + i * 20.0, y=250.0, priority=p)
        for i, p in enumerate(priorities)
    ]
    config = SwarmConfig(bft_mode=True, packet_loss_rate=0.0)
    return Swarm(drones, comm_range=400.0, config=config, seed=seed)


def snapshot(swarm: Swarm) -> dict:
    return {
        drone_id: (election.known_nexus_id, election.term)
        for drone_id, election in swarm.elections.items()
        if swarm.drones[drone_id].alive
    }


def _adopted(injection: Injection, before: dict, after: dict) -> bool:
    """Did the swarm's observed behaviour change in the way the attack
    intended? For a forgery with a concrete claim, success means some live
    drone actually came to believe the forged (nexus, term). For replay/flood
    (claim=None), success means the previously-stable swarm changed at all."""
    claim = injection.claim
    if claim is None:
        return after != before
    return any(
        nexus == claim["sender_id"] and term >= claim["term"]
        for nexus, term in after.values()
    )


def _tick(swarm: Swarm, n: int) -> None:
    for _ in range(n):
        swarm.tick()


def run_campaign(seed: int = 7) -> tuple:
    swarm = build_swarm(seed)
    adversary = Antagonist(swarm)  # installs the passive wiretap

    _tick(swarm, CONVERGE_TICKS)
    baseline = snapshot(swarm)
    observed_nexus = adversary.wiretap.latest_heartbeat()
    observed_nexus_id = observed_nexus.sender_id if observed_nexus else None
    observed_term = adversary.wiretap.observed_term()

    # Aim the impersonation attacks at a real drone that is NOT the current
    # nexus, so "some drone now believes victim is nexus" is a clean success
    # signal. Picked as the lowest-priority live non-nexus drone.
    victims = sorted(
        (d for d in swarm.drones.values() if d.alive and d.id != observed_nexus_id),
        key=lambda d: d.priority,
    )
    victim_id = victims[0].id

    outcomes = []

    def run(make_injection):
        before = snapshot(swarm)
        injection = make_injection()
        _tick(swarm, PROPAGATE_TICKS)
        after = snapshot(swarm)
        blocked = not _adopted(injection, before, after)
        outcomes.append(AttackOutcome(injection, blocked, before, after))

    run(lambda: adversary.impersonate_nexus(victim_id, observed_term + 1))
    run(lambda: adversary.inject_garbage_heartbeat(victim_id, observed_term + 1))
    run(lambda: adversary.forge_priority(victim_id, observed_term + 1, inflated_priority=999.0))
    run(lambda: adversary.inflate_term(victim_id, observed_term + 500))
    run(lambda: adversary.replay_captured_heartbeat())
    run(lambda: adversary.repurpose_certificate(victim_id, observed_term + 500))
    run(lambda: adversary.flood(victim_id, count=200))

    meta = {
        "nexus_id": observed_nexus_id,
        "observed_term": observed_term,
        "victim_id": victim_id,
        "baseline": baseline,
        "swarm_size": len(swarm.drones),
    }
    return outcomes, meta


def format_report(outcomes: list, meta: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("  ANTAGONIST CAMPAIGN REPORT -- BFT-mode drone swarm red team")
    lines.append("=" * 78)
    lines.append(
        f"  Swarm: {meta['swarm_size']} drones, BFT hardened. Legitimate nexus elected: "
        f"{meta['nexus_id']} (term {meta['observed_term']})."
    )
    lines.append(f"  Impersonation target ('victim'): {meta['victim_id']}")
    lines.append("=" * 78)
    lines.append("")

    n_blocked = sum(1 for o in outcomes if o.blocked)
    for i, outcome in enumerate(outcomes, 1):
        inj = outcome.injection
        # The flood is honestly a non-defense: it never changes the election
        # (so 'blocked' by our behavioural test) but is NOT actually mitigated.
        if inj.name == "flood/spam":
            verdict = "NOT MITIGATED (no election effect, but no rate limiting)"
        else:
            verdict = "BLOCKED" if outcome.blocked else "*** SUCCEEDED ***"
        lines.append(f"[{i}] {inj.name.upper()}  ->  {verdict}")
        lines.append(f"    Attempted : {inj.description}")
        lines.append(f"    Defense   : {inj.defense}")
        lines.append("")

    lines.append("-" * 78)
    lines.append(
        f"  Summary: {n_blocked}/{len(outcomes)} attacks produced no adverse state change. "
        "The signed-message gate is the workhorse -- an external adversary with no"
    )
    lines.append(
        "  registered key cannot pass it, so impersonation, garbage, priority forgery and"
    )
    lines.append(
        "  term inflation are all stopped at the signature check. Credential and quorum-"
    )
    lines.append(
        "  certificate checks are deeper defenses (vs a compromised insider). Flooding is"
    )
    lines.append("  the one honest gap: unmitigated at the resource level (no rate limiting).")
    lines.append("-" * 78)
    return "\n".join(lines)
