"""Byzantine-fault-tolerance tests for `bft_mode`.

Two halves:

* **Liveness under hardening** (tests 1, 2, 8): turning `bft_mode` on must
  not change legitimate behaviour -- the swarm still converges on the
  highest-priority nexus and still cascades on failover, exactly like the
  plain-mode tests in test_election.py; and with `bft_mode` off, messages
  carry no signatures at all.
* **Safety against a rogue transmitter** (tests 3-7): forged, unsigned,
  priority-inflated, and term-inflated messages injected straight onto the
  mesh via `swarm.mesh.broadcast()` (the only injection point a real radio
  attacker has) must never move the elected nexus -- while a *genuinely*
  signed majority quorum certificate still IS accepted (test 7), proving the
  defense isn't so strict it would reject a legitimate large term jump.

Attacker crypto is always attacker-generated (`DroneIdentity` /
`SwarmAuthority` the test makes itself) -- never the swarm's real keys from
`swarm.identities` -- mirroring the antagonist/ package. The two exceptions
are commented inline: tests 5 and 6 additionally sign with a real drone key
to model a *compromised insider*, the only way to reach the credential /
quorum-certificate checks that sit behind the signature gate.
"""
from drone_swarm.election import ElectionRole, NexusElection
from drone_swarm.identity import (
    Credential,
    DroneIdentity,
    IdentityRegistry,
    SwarmAuthority,
)
from drone_swarm.model import Drone
from drone_swarm.protocol import (
    ElectionMessage,
    NexusHeartbeat,
    election_message_payload,
    heartbeat_payload,
)
from drone_swarm.swarm import Swarm, SwarmConfig

# Same fast, deterministic timing as tests/test_election.py's FAST, plus the
# BFT hardening switched on. Zero packet loss keeps the baseline crisp so any
# post-attack state change is unambiguously the attack's doing.
FAST_BFT = SwarmConfig(
    nexus_heartbeat_interval_s=0.3,
    nexus_timeout_s=0.8,
    comm_latency_s=0.05,
    tick_dt_s=0.2,
    packet_loss_rate=0.0,
    bft_mode=True,
)


def _tick(swarm, n):
    for _ in range(n):
        swarm.tick()


def _clique(priorities, config, comm_range=500, seed=1):
    """A fully-connected swarm: every drone within range of every other, so
    killing any single drone never disconnects the rest."""
    drones = []
    for i, (drone_id, priority) in enumerate(priorities.items()):
        drones.append(Drone(id=drone_id, x=(i % 3) * 30, y=(i // 3) * 30, priority=priority))
    return Swarm(drones, comm_range=comm_range, config=config, seed=seed)


def _alive_ids(swarm):
    return {d.id for d in swarm.drones.values() if d.alive}


def _snapshot(swarm):
    """Per-live-drone (believed nexus, term) -- the only observable state the
    attack tests judge success or failure from."""
    return {
        drone_id: (election.known_nexus_id, election.term)
        for drone_id, election in swarm.elections.items()
        if swarm.drones[drone_id].alive
    }


def _broadcast(swarm, message):
    swarm.mesh.broadcast(message, swarm.time_s)


class _CaptureMesh:
    """Minimal passive wiretap: records every message crossing the mesh
    without altering delivery. Used only by test 8 to inspect signatures."""

    def __init__(self, swarm):
        self.captured = []
        self._real = swarm.mesh.broadcast

        def tapped(message, now_s):
            self.captured.append(message)
            return self._real(message, now_s)

        swarm.mesh.broadcast = tapped


# ---------------------------------------------------------------------------
# 1. Legitimate convergence still works under bft_mode.
# ---------------------------------------------------------------------------
def test_bft_swarm_converges_to_highest_priority():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, FAST_BFT)
    _tick(swarm, 30)

    for drone in swarm.drones.values():
        assert drone.nexus_id == "D4"
    assert swarm.elections["D4"].role == ElectionRole.NEXUS
    # Sanity: this really is the hardened path, not silently plain-mode.
    assert swarm.config.bft_mode is True
    assert set(swarm.identities) == set(swarm.drones)


# ---------------------------------------------------------------------------
# 2. Cascading failover still works under bft_mode.
# ---------------------------------------------------------------------------
def test_bft_cascading_handoff_across_three_nexuses():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, FAST_BFT)
    _tick(swarm, 30)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    swarm.kill("D4")
    _tick(swarm, 30)
    assert all(swarm.drones[i].nexus_id == "D3" for i in _alive_ids(swarm))

    swarm.kill("D3")
    _tick(swarm, 30)
    for drone_id in _alive_ids(swarm):
        assert swarm.drones[drone_id].nexus_id == "D2"
        assert swarm.drones[swarm.drones[drone_id].nexus_id].alive


# ---------------------------------------------------------------------------
# 3. Unsigned / garbage-signature heartbeat is rejected.
# ---------------------------------------------------------------------------
def test_unsigned_heartbeat_is_rejected():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, FAST_BFT)
    _tick(swarm, 30)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    before = _snapshot(swarm)

    now = swarm.time_s
    # Impersonate non-nexus D0 as a fresh nexus, with (a) no signature and
    # (b) random garbage bytes. Neither is a valid Ed25519 signature.
    _broadcast(swarm, NexusHeartbeat(sender_id="D0", sent_at_s=now, term=99, signature=None))
    _broadcast(swarm, NexusHeartbeat(sender_id="D0", sent_at_s=now, term=99, signature=b"\x00" * 64))
    _tick(swarm, 10)

    # D4 still nexus, nobody adopted the forged (D0, term 99), terms didn't jump.
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    after = _snapshot(swarm)
    assert all(term < 99 for _, term in after.values())
    assert all(nexus != "D0" for nexus, _ in after.values())
    # The election didn't churn -- terms are unchanged by the injection.
    assert {t for _, t in after.values()} == {t for _, t in before.values()}


# ---------------------------------------------------------------------------
# 4. Impersonation (attacker keypair claiming a real drone's id) is rejected.
# ---------------------------------------------------------------------------
def test_impersonation_with_attacker_key_is_rejected():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, FAST_BFT)
    _tick(swarm, 30)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    # Attacker's OWN keypair that merely CLAIMS to be D0 -- not the real
    # swarm.identities["D0"] key. The id is spoofable; the signing key is not.
    forger = DroneIdentity("D0")
    now = swarm.time_s
    observed_term = max(e.term for e in swarm.elections.values())
    hb = NexusHeartbeat(
        sender_id="D0",
        sent_at_s=now,
        term=observed_term + 1,  # small jump: isolates the signature check
        signature=forger.sign(heartbeat_payload("D0", now, observed_term + 1)),
    )
    # The forged key must genuinely differ from the swarm's registered one.
    assert forger.public_key.public_bytes_raw() != swarm.identities["D0"].public_key.public_bytes_raw()

    _broadcast(swarm, hb)
    _tick(swarm, 10)

    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    assert all(nexus != "D0" for nexus, _ in _snapshot(swarm).values())


# ---------------------------------------------------------------------------
# 5. Priority forgery (fabricated credential, inflated priority) is rejected.
# ---------------------------------------------------------------------------
def test_priority_forgery_is_rejected():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, FAST_BFT)
    _tick(swarm, 30)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    now = swarm.time_s
    observed_term = max(e.term for e in swarm.elections.values())

    # A credential claiming D0 has priority 999, self-issued by a ROGUE
    # authority (not swarm.authority) -- it does not verify against the real
    # authority public key the registry trusts.
    rogue_authority = SwarmAuthority()
    fake_credential = rogue_authority.issue_credential("D0", 999.0)

    # (a) Attacker keypair + fake credential: rejected at the signature gate.
    attacker = DroneIdentity("D0")
    em_attacker = ElectionMessage(
        sender_id="D0",
        sent_at_s=now,
        term=observed_term,
        priority=999.0,
        signature=attacker.sign(election_message_payload("D0", now, observed_term, 999.0)),
        credential=fake_credential,
    )
    # (b) Compromised-insider variant: signed with D0's REAL key so it passes
    # the signature check, forcing the *credential* defense to be what stops
    # the inflated priority. This is the only way to exercise that branch.
    insider_key = swarm.identities["D0"]
    em_insider = ElectionMessage(
        sender_id="D0",
        sent_at_s=now,
        term=observed_term,
        priority=999.0,
        signature=insider_key.sign(election_message_payload("D0", now, observed_term, 999.0)),
        credential=fake_credential,
    )

    _broadcast(swarm, em_attacker)
    _broadcast(swarm, em_insider)
    _tick(swarm, 12)

    # If the inflated 999 priority had been trusted, D0 would have seized the
    # nexus. It didn't: D4 (real priority 90) still leads, D0 never wins.
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    assert all(nexus != "D0" for nexus, _ in _snapshot(swarm).values())

    # Prove the credential defense is really what rejects (b), independent of
    # the mesh: a receiver's verifier drops it even with a valid signature.
    verifier = swarm.elections["D4"]
    assert verifier._verify_election_message(em_insider) is False
    # And the real, correctly-issued credential for D0 DOES verify -- so the
    # rejection above is the forged authority signature, not a broken check.
    assert swarm.registry.verify_credential(swarm.credentials["D0"]) is True
    assert swarm.registry.verify_credential(fake_credential) is False


# ---------------------------------------------------------------------------
# 6. Term inflation without a quorum certificate is rejected.
# ---------------------------------------------------------------------------
def test_term_inflation_without_quorum_is_rejected():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, FAST_BFT)
    _tick(swarm, 30)
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())
    before = _snapshot(swarm)
    now = swarm.time_s
    observed_term = max(e.term for e in swarm.elections.values())
    huge_term = observed_term + 500

    # Sign with D0's REAL key (compromised insider) so the signature check
    # passes and the *quorum-certificate* check is specifically what must
    # reject this: a huge term jump with an empty certificate proves no
    # election actually happened.
    insider_key = swarm.identities["D0"]
    hb = NexusHeartbeat(
        sender_id="D0",
        sent_at_s=now,
        term=huge_term,
        signature=insider_key.sign(heartbeat_payload("D0", now, huge_term)),
        quorum_certificate=(),
    )
    _broadcast(swarm, hb)
    _tick(swarm, 10)

    # No drone jumped to term 500+ or adopted D0; D4 still leads.
    after = _snapshot(swarm)
    assert all(term < huge_term for _, term in after.values())
    assert all(nexus != "D0" for nexus, _ in after.values())
    assert all(d.nexus_id == "D4" for d in swarm.drones.values())

    # Directly: even validly signed, an empty cert fails the quorum check.
    verifier = swarm.elections["D4"]
    assert verifier._verify_heartbeat(hb) is False
    # Terms didn't creep from the injection.
    assert {t for _, t in after.values()} == {t for _, t in before.values()}


# ---------------------------------------------------------------------------
# 7. A GENUINE majority quorum certificate IS accepted (positive case).
# ---------------------------------------------------------------------------
def _isolated_bft_members(priorities):
    """Build real, mutually-trusting swarm crypto material for an isolated
    unit test: one authority, a shared registry, and per-member signing keys
    with authority-issued credentials. These stand in for legitimate swarm
    members -- none of it is attacker material."""
    authority = SwarmAuthority()
    registry = IdentityRegistry(authority.public_key)
    identities, credentials = {}, {}
    for drone_id, priority in priorities.items():
        identity = DroneIdentity(drone_id)
        identities[drone_id] = identity
        credentials[drone_id] = authority.issue_credential(drone_id, priority)
        registry.register(drone_id, identity.public_key)
    return authority, registry, identities, credentials


def _signed_candidacy(identities, credentials, sender, term, priority, now=0.0):
    return ElectionMessage(
        sender_id=sender,
        sent_at_s=now,
        term=term,
        priority=priority,
        signature=identities[sender].sign(election_message_payload(sender, now, term, priority)),
        credential=credentials[sender],
    )


def test_genuine_quorum_certificate_is_accepted():
    priorities = {"D0": 10.0, "D1": 20.0, "D2": 30.0, "D3": 40.0, "D4": 50.0}
    total = len(priorities)  # 5 -> majority is > 2.5, i.e. 3 distinct signers
    _authority, registry, identities, credentials = _isolated_bft_members(priorities)

    now, term = 1.0, 10  # a big jump from a receiver at term 0
    # A real, signed heartbeat from D0 for a large term, carrying genuine
    # candidacies from D1 and D2 for that SAME term. Signers {D0, D1, D2} = 3.
    cand_d1 = _signed_candidacy(identities, credentials, "D1", term, 20.0, now)
    cand_d2 = _signed_candidacy(identities, credentials, "D2", term, 30.0, now)
    hb = NexusHeartbeat(
        sender_id="D0",
        sent_at_s=now,
        term=term,
        signature=identities["D0"].sign(heartbeat_payload("D0", now, term)),
        quorum_certificate=(cand_d1, cand_d2),
    )

    # Receiver at term 0: term 10 forces the quorum path (10 > 0 + 1).
    receiver = NexusElection(
        nexus_heartbeat_interval_s=0.3,
        nexus_timeout_s=0.8,
        bft_mode=True,
        identity=identities["D0"],
        credential=credentials["D0"],
        registry=registry,
        total_swarm_size=total,
    )
    assert receiver.term == 0

    # The certificate is genuinely signed -- prove each candidacy verifies on
    # its own, so this positive case is not accidentally trivial.
    assert receiver._verify_election_message(cand_d1) is True
    assert receiver._verify_election_message(cand_d2) is True
    # And it is accepted, both directly and through the full heartbeat path.
    assert receiver._verify_quorum_certificate(hb) is True
    assert receiver._verify_heartbeat(hb) is True

    # Negative control A -- one signer short of a majority (only {D0, D1} = 2)
    # must be rejected, confirming the threshold actually bites.
    hb_short = NexusHeartbeat(
        sender_id="D0",
        sent_at_s=now,
        term=term,
        signature=identities["D0"].sign(heartbeat_payload("D0", now, term)),
        quorum_certificate=(cand_d1,),
    )
    assert receiver._verify_quorum_certificate(hb_short) is False
    assert receiver._verify_heartbeat(hb_short) is False

    # Negative control B -- padding the cert with a FORGED candidacy (attacker
    # key claiming D3) must not count toward quorum, so {D0, D1} + forged = 2
    # is still rejected. Proves the quorum check verifies signatures, not just
    # counts distinct ids.
    forged_d3 = ElectionMessage(
        sender_id="D3",
        sent_at_s=now,
        term=term,
        priority=40.0,
        signature=DroneIdentity("D3").sign(election_message_payload("D3", now, term, 40.0)),
        credential=credentials["D3"],
    )
    assert receiver._verify_election_message(forged_d3) is False
    hb_forged_pad = NexusHeartbeat(
        sender_id="D0",
        sent_at_s=now,
        term=term,
        signature=identities["D0"].sign(heartbeat_payload("D0", now, term)),
        quorum_certificate=(cand_d1, forged_d3),
    )
    assert receiver._verify_quorum_certificate(hb_forged_pad) is False


# ---------------------------------------------------------------------------
# 8. bft_mode=False is completely unaffected: no signatures, still converges.
# ---------------------------------------------------------------------------
def test_plain_mode_has_no_signatures_and_still_converges():
    swarm = _clique({"D0": 50, "D1": 60, "D2": 70, "D3": 80, "D4": 90}, SwarmConfig(
        nexus_heartbeat_interval_s=0.3,
        nexus_timeout_s=0.8,
        comm_latency_s=0.05,
        tick_dt_s=0.2,
        packet_loss_rate=0.0,
    ))
    assert swarm.config.bft_mode is False
    assert swarm.identities == {} and swarm.credentials == {}
    assert swarm.registry is None and swarm.authority is None

    tap = _CaptureMesh(swarm)
    _tick(swarm, 30)

    for drone in swarm.drones.values():
        assert drone.nexus_id == "D4"

    # Every protocol message crossing the mesh is unsigned in plain mode.
    protocol_msgs = [m for m in tap.captured if isinstance(m, (NexusHeartbeat, ElectionMessage))]
    assert protocol_msgs  # the swarm actually communicated
    for message in protocol_msgs:
        assert message.signature is None
    for message in protocol_msgs:
        if isinstance(message, ElectionMessage):
            assert message.credential is None
