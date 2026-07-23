# antagonist

A "red team" tool for the drone-swarm-failover project's Byzantine-fault-tolerant hardened mode (`SwarmConfig(bft_mode=True)`). It plays a rogue radio transmitter with no legitimate cryptographic keys, and demonstrates that the swarm's real defenses actually hold against it — not just that they exist on paper.

## Run it

```bash
source ../.venv/bin/activate   # from this directory, or just activate the project venv
python3 -m antagonist.cli
```

## Design boundary

This package only ever touches a target `Swarm` through the same channel a real rogue transmitter would have — `swarm.mesh.broadcast()` — plus whatever it can passively overhear on the mesh (`MeshWiretap`, a non-invasive wrapper around `broadcast` that records everything without altering delivery). It never reaches into election internals, and it never borrows a real drone's private key or a genuinely authority-issued credential to construct an attack. Every forged message uses attacker-generated cryptographic material (`identity.py`) — the same publicly-available `cryptography` library anyone could use, producing signatures that are worthless against the real swarm's registered keys.

That boundary is deliberate, for two reasons: it's the only way an attack against the defenses actually proves anything, and it means this package doesn't depend on anything private to `drone_swarm` beyond its public message-injection surface and public protocol format. It's scoped so it could be extracted into its own general-purpose "adversarial mesh-network testing tool" later — the attack techniques (impersonation, forged credentials, unproven authority claims, replay) aren't specific to drones or to this project.

## What's implemented

| Attack | What it attempts | Result |
|---|---|---|
| Impersonation | Forge a heartbeat/candidacy claiming a real drone's id, signed with an attacker key | Blocked — signature doesn't match the real drone's registered public key |
| Unsigned / garbage signature | The crudest case: no signature, or random bytes | Blocked trivially |
| Priority forgery | Claim an inflated priority backed by a self-issued (rogue) credential | Blocked — credential doesn't verify against the real `SwarmAuthority` |
| Term inflation | Assert a huge term jump with an empty quorum certificate | Blocked — no valid majority of corroborating signatures |
| Replay | Re-broadcast a genuinely-valid captured heartbeat, verbatim, later | Blocked — not a newer term, gains no ground |
| Repurposed certificate | Bundle real captured candidacies into a certificate for a *different* term than they were signed for | Blocked — certificate verification checks term match per entry |
| Flood / spam | A burst of garbage messages | **Not mitigated** — each message is individually rejected so the election itself isn't disrupted, but there's no rate limiting, so the resource cost of processing a flood is a real, honestly-reported gap |

`campaign.py` decides BLOCKED vs. SUCCEEDED purely from observed swarm behavior before/after each injection (did any live drone actually come to believe the forged claim?) — never from an internal logging hook, so the report reflects what the swarm actually did, not what the attack code assumes happened.
