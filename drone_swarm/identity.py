"""Cryptographic identity for drones and the swarm's issuing authority.

This is the foundation the Byzantine-fault-tolerant mode (`SwarmConfig.bft_mode`)
is built on. Two distinct signatures back every election message:

1. Each drone signs its own outgoing messages with its own Ed25519 key --
   proves "whoever sent this holds this drone's private key," which is
   what defeats a rogue transmitter impersonating a real drone's id.
2. Each drone's claimed `priority` is backed by a Credential signed once,
   at swarm setup, by a SwarmAuthority key -- standing in for a fleet
   operator / manufacturer. A compromised drone can forge its OWN
   messages (it has its own key) but cannot forge a higher priority than
   it was actually issued, because it doesn't hold the authority's key.

Uses the `cryptography` library's Ed25519 implementation (the modern
standard used in SSH, TLS 1.3, and Signal) rather than hand-rolled crypto.
"""
from __future__ import annotations

from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _credential_payload(drone_id: str, priority: float) -> bytes:
    return f"credential:{drone_id}:{priority}".encode()


@dataclass(frozen=True)
class Credential:
    """Proof, signed by the SwarmAuthority, that `drone_id` was actually
    issued `priority`."""

    drone_id: str
    priority: float
    signature: bytes

    def is_valid(self, authority_public_key: Ed25519PublicKey) -> bool:
        try:
            authority_public_key.verify(self.signature, _credential_payload(self.drone_id, self.priority))
            return True
        except InvalidSignature:
            return False


class SwarmAuthority:
    """Stands in for the fleet operator / manufacturer: the one party
    every drone trusts to certify priorities at setup time. Only this
    class ever touches the authority's private key."""

    def __init__(self) -> None:
        self._private_key = Ed25519PrivateKey.generate()
        self.public_key: Ed25519PublicKey = self._private_key.public_key()

    def issue_credential(self, drone_id: str, priority: float) -> Credential:
        signature = self._private_key.sign(_credential_payload(drone_id, priority))
        return Credential(drone_id=drone_id, priority=priority, signature=signature)


class DroneIdentity:
    """A single drone's own signing key, used to prove a message actually
    came from it, not from a rogue transmitter claiming its id."""

    def __init__(self, drone_id: str) -> None:
        self.drone_id = drone_id
        self._private_key = Ed25519PrivateKey.generate()
        self.public_key: Ed25519PublicKey = self._private_key.public_key()

    def sign(self, payload: bytes) -> bytes:
        return self._private_key.sign(payload)


class IdentityRegistry:
    """The pre-shared trust every drone in the swarm has at setup time:
    the authority's public key, and every legitimate drone's public key.
    Modeled as established out-of-band before deployment (the same
    assumption real device-certificate systems make) -- distributing it
    isn't itself part of the attack surface this project defends.
    """

    def __init__(self, authority_public_key: Ed25519PublicKey) -> None:
        self.authority_public_key = authority_public_key
        self._drone_public_keys: dict = {}

    def register(self, drone_id: str, public_key: Ed25519PublicKey) -> None:
        self._drone_public_keys[drone_id] = public_key

    def public_key_for(self, drone_id: str):
        return self._drone_public_keys.get(drone_id)

    def verify_signature(self, drone_id: str, payload: bytes, signature: bytes) -> bool:
        public_key = self.public_key_for(drone_id)
        if public_key is None:
            return False  # unknown id -- can't be a legitimate drone
        try:
            public_key.verify(signature, payload)
            return True
        except InvalidSignature:
            return False

    def verify_credential(self, credential: Credential) -> bool:
        return credential.is_valid(self.authority_public_key)
