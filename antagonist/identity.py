"""The attacker's own cryptographic material.

A real external adversary can freely generate Ed25519 keys with the same
`cryptography` library everyone else uses -- so we reuse `DroneIdentity` and
`SwarmAuthority` directly. What the adversary CANNOT do is obtain a real
drone's private key or a genuinely authority-signed credential. These helpers
make that boundary explicit: everything here is attacker-generated, so any
signature it produces is worthless to a receiver that only trusts the real
swarm's registered keys.
"""
from __future__ import annotations

from drone_swarm.identity import Credential, DroneIdentity, SwarmAuthority


def forged_identity(claimed_drone_id: str) -> DroneIdentity:
    """A keypair that CLAIMS to be `claimed_drone_id` but is not the real
    drone's key. This is exactly what impersonation is: the id is spoofable,
    the signing key is not."""
    return DroneIdentity(claimed_drone_id)


def rogue_authority() -> SwarmAuthority:
    """The attacker's own fake fleet-operator key. It can self-issue any
    credential it likes, but the swarm trusts only the REAL authority's
    public key, so nothing this signs will verify."""
    return SwarmAuthority()


def forged_credential(authority: SwarmAuthority, drone_id: str, priority: float) -> Credential:
    """An inflated-priority credential signed by a rogue authority."""
    return authority.issue_credential(drone_id, priority)
