"""Adversarial ("red team") testing package for the drone swarm.

`antagonist` plays the role of a rogue radio transmitter with no legitimate
cryptographic keys. It only ever touches a target `Swarm` through its public
surface -- `swarm.mesh.broadcast()` to inject messages, and readable state
(drone ids, positions, `swarm.authority.public_key`, and whatever it can
overhear on the mesh) to aim them. It never reaches into election internals
and never borrows a real drone's private key or a real issued credential.

The point is to demonstrate that the BFT-mode defenses in `drone_swarm`
actually hold against something a real external attacker could do.
"""
from .attacks import Antagonist, Injection, MeshWiretap

__all__ = ["Antagonist", "Injection", "MeshWiretap"]
