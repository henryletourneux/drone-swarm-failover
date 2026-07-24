"""Tests for hierarchical command (command.py): platoons independently
electing their own nexus, and a second, dynamically-recomposing election
among the current platoon nexuses picking an overall commander -- the
same NexusElection mechanism reused at both tiers (see command.py's
module docstring for the full design rationale).

Same fast/lossless timing convention as test_election.py/test_bft.py so
convergence happens in a handful of ticks.
"""
from drone_swarm.command import CommandConfig, CommandState
from drone_swarm.election import ElectionRole, NexusElection
from drone_swarm.identity import DroneIdentity, IdentityRegistry, SwarmAuthority
from drone_swarm.model import Drone
from drone_swarm.protocol import ElectionMessage, NexusHeartbeat, election_message_payload, heartbeat_payload
from drone_swarm.swarm import Swarm, SwarmConfig

import pytest

FAST = SwarmConfig(
    nexus_heartbeat_interval_s=0.3,
    nexus_timeout_s=0.8,
    comm_latency_s=0.05,
    tick_dt_s=0.2,
    packet_loss_rate=0.0,
)

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


def _platoon_swarm(platoon_priorities: dict, config=FAST, comm_range=500, seed=1):
    """platoon_priorities: {platoon_id: {drone_id: priority}}. Builds a
    single fully-connected clique (every drone in range of every other,
    across platoons too) so commander-layer traffic between platoon
    nexuses is always deliverable unless a test deliberately separates
    them."""
    drones = []
    platoon_of = {}
    i = 0
    for platoon_id, members in platoon_priorities.items():
        for drone_id, priority in members.items():
            drones.append(Drone(id=drone_id, x=(i % 3) * 30, y=(i // 3) * 30, priority=priority))
            platoon_of[drone_id] = platoon_id
            i += 1
    command_config = CommandConfig(platoon_of=platoon_of)
    full_config = SwarmConfig(**{**config.__dict__, "command_config": command_config})
    return Swarm(drones, comm_range=comm_range, config=full_config, seed=seed)


def _alive_ids(swarm):
    return {d.id for d in swarm.drones.values() if d.alive}


# -- Backward compatibility: command_config=None is fully inert -------------

def test_command_config_none_is_fully_inert():
    swarm = Swarm([Drone(id="D0", x=0, y=0, priority=1)], comm_range=100, config=FAST)
    assert swarm.command is None
    assert swarm.commander_elections == {}
    _tick(swarm, 10)
    assert swarm.commander_elections == {}
    assert "command" not in swarm.to_state_dict()
    assert "commander" not in swarm.metrics_snapshot()
    assert swarm.drones["D0"].platoon_id is None
    assert swarm.drones["D0"].commander_id is None


# -- Platoon-layer isolation --------------------------------------------------

def test_each_platoon_converges_to_its_own_highest_priority_nexus():
    """The swarm-wide-highest-priority drone (C=95) sits in P2, deliberately
    NOT P1 -- if the layer filter were broken (a flat election leaking
    across platoons), P1's members would wrongly end up pointing at C."""
    swarm = _platoon_swarm({
        "P1": {"A": 70.0, "B": 50.0},
        "P2": {"C": 95.0, "D": 60.0},
    })
    _tick(swarm, 20)

    assert swarm.drones["A"].nexus_id == "A"
    assert swarm.drones["B"].nexus_id == "A"
    assert swarm.drones["C"].nexus_id == "C"
    assert swarm.drones["D"].nexus_id == "C"
    assert swarm.drones["A"].platoon_id == "P1"
    assert swarm.drones["C"].platoon_id == "P2"


# -- Commander-layer convergence ----------------------------------------------

def test_commander_converges_among_platoon_nexuses():
    swarm = _platoon_swarm({
        "P1": {"A": 70.0, "B": 50.0},
        "P2": {"C": 95.0, "D": 60.0},
        "P3": {"E": 40.0, "F": 30.0},
    })
    _tick(swarm, 30)

    assert swarm.drones["A"].nexus_id == "A"
    assert swarm.drones["C"].nexus_id == "C"
    assert swarm.drones["E"].nexus_id == "E"

    commander_ids = swarm.command.commander_ids()
    assert len(commander_ids) == 1
    commander_id = commander_ids[0]
    # Every platoon nexus's own belief about who the commander is agrees.
    assert swarm.drones["A"].commander_id == commander_id
    assert swarm.drones["C"].commander_id == commander_id
    assert swarm.drones["E"].commander_id == commander_id
    # Non-nexus drones never ran a commander engine at all.
    assert swarm.drones["B"].commander_id is None
    assert swarm.drones["D"].commander_id is None


def test_commander_elections_membership_tracks_current_platoon_nexuses():
    swarm = _platoon_swarm({
        "P1": {"A": 70.0, "B": 50.0},
        "P2": {"C": 95.0, "D": 60.0},
    })
    _tick(swarm, 20)

    current_nexus_ids = {
        d.id for d in swarm.drones.values()
        if d.alive and swarm.elections[d.id].role == ElectionRole.NEXUS
    }
    assert set(swarm.commander_elections.keys()) == current_nexus_ids
    assert current_nexus_ids == {"A", "C"}


# -- Dynamic membership: the two hard scenarios named in the design ----------

def test_killing_a_platoon_nexus_drops_and_refreshes_commander_engine():
    swarm = _platoon_swarm({
        "P1": {"A": 70.0, "B": 50.0},
        "P2": {"C": 95.0, "D": 60.0},
    })
    _tick(swarm, 20)
    assert "A" in swarm.commander_elections

    swarm.kill("A")
    _tick(swarm, 20)

    assert "A" not in swarm.commander_elections
    assert swarm.drones["A"].commander_id is None
    # B took over the platoon, and gets a brand-new (term-0-born) commander engine.
    assert swarm.drones["B"].nexus_id == "B"
    assert "B" in swarm.commander_elections


def test_killing_the_sitting_commander_triggers_new_commander_election():
    swarm = _platoon_swarm({
        "P1": {"A": 70.0, "B": 50.0},
        "P2": {"C": 95.0, "D": 60.0},
        "P3": {"E": 40.0, "F": 30.0},
    })
    _tick(swarm, 30)
    commander_id = swarm.command.commander_ids()[0]
    assert commander_id == "C"  # C has the highest priority swarm-wide

    swarm.kill("C")
    _tick(swarm, 40)

    # D takes over P2's nexus seat; the remaining platoon nexuses converge
    # on a new commander, and no live drone's belief points at dead C.
    assert swarm.drones["D"].nexus_id == "D"
    new_commander_ids = swarm.command.commander_ids()
    assert len(new_commander_ids) == 1
    assert new_commander_ids[0] != "C"
    for drone_id in _alive_ids(swarm):
        drone = swarm.drones[drone_id]
        if drone.commander_id is not None:
            assert drone.commander_id != "C"
            assert swarm.drones[drone.commander_id].alive


# -- Layer-isolation unit test, directly against NexusElection ---------------

def test_nexus_election_ignores_messages_from_a_different_layer():
    platoon_engine = NexusElection(nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, layer="platoon:P1")
    commander_engine = NexusElection(nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, layer="commander")

    _, outgoing = platoon_engine.step(0.0, "A", 90.0, [])
    assert outgoing  # A immediately opens a platoon-layer campaign (no known nexus yet)

    # Feed the platoon-layer candidacy straight into the commander-layer
    # engine: it must be completely ignored (wrong layer).
    commander_engine.step(0.1, "B", 10.0, outgoing)
    assert commander_engine.known_nexus_id is None
    assert commander_engine.role != ElectionRole.FOLLOWER or commander_engine.term == 0

    # Sanity: a MATCHING layer does exchange state (the filter isn't just
    # silently dropping everything).
    same_layer_engine = NexusElection(nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, layer="platoon:P1")
    same_layer_engine.step(0.1, "B", 10.0, outgoing)
    assert same_layer_engine.term >= 1


# -- BFT: relabeling and the quorum-certificate layer gap --------------------

def test_bft_message_relabeled_to_wrong_layer_is_rejected():
    authority = SwarmAuthority()
    registry = IdentityRegistry(authority.public_key)
    identity = DroneIdentity("A")
    credential = authority.issue_credential("A", 90.0)
    registry.register("A", identity.public_key)

    receiver = NexusElection(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, bft_mode=True,
        identity=DroneIdentity("R"), credential=authority.issue_credential("R", 1.0),
        registry=registry, layer="commander",
    )

    now = 0.0
    # Genuinely signed for layer="platoon:P1" ...
    real_payload = election_message_payload("A", now, 1, 90.0, "platoon:P1")
    signature = identity.sign(real_payload)
    # ... but relabeled to claim layer="commander" on the wire.
    relabeled = ElectionMessage(
        sender_id="A", sent_at_s=now, term=1, priority=90.0,
        signature=signature, credential=credential, layer="commander",
    )
    assert receiver._verify_election_message(relabeled) is False


def test_bft_quorum_certificate_rejects_wrong_layer_candidacies():
    """Regression for the real gap found while designing this feature:
    genuinely-signed platoon-layer candidacies, sniffed straight off the
    mesh, must not count toward a commander-layer heartbeat's quorum."""
    authority = SwarmAuthority()
    registry = IdentityRegistry(authority.public_key)
    identities, credentials = {}, {}
    for drone_id in ("D0", "D1", "D2"):
        identity = DroneIdentity(drone_id)
        identities[drone_id] = identity
        credentials[drone_id] = authority.issue_credential(drone_id, 10.0)
        registry.register(drone_id, identity.public_key)

    now = 0.0
    # Real, validly-signed PLATOON-layer candidacies for D1/D2.
    platoon_cand_d1 = ElectionMessage(
        sender_id="D1", sent_at_s=now, term=5, priority=10.0,
        signature=identities["D1"].sign(election_message_payload("D1", now, 5, 10.0, "platoon:P1")),
        credential=credentials["D1"], layer="platoon:P1",
    )
    platoon_cand_d2 = ElectionMessage(
        sender_id="D2", sent_at_s=now, term=5, priority=10.0,
        signature=identities["D2"].sign(election_message_payload("D2", now, 5, 10.0, "platoon:P1")),
        credential=credentials["D2"], layer="platoon:P1",
    )
    # Forged COMMANDER-layer heartbeat from D0, padded with the real
    # platoon-layer candidacies above to try to fake a quorum.
    forged_commander_hb = NexusHeartbeat(
        sender_id="D0", sent_at_s=now, term=5,
        signature=identities["D0"].sign(heartbeat_payload("D0", now, 5, "commander")),
        quorum_certificate=(platoon_cand_d1, platoon_cand_d2),
        layer="commander",
    )

    receiver = NexusElection(
        nexus_heartbeat_interval_s=0.3, nexus_timeout_s=0.8, bft_mode=True,
        identity=identities["D0"], credential=credentials["D0"],
        registry=registry, total_swarm_size=3, layer="commander",
    )
    # Each candidacy verifies fine ON ITS OWN (it's genuinely signed) --
    # proving this isn't accidentally trivial -- but must not count here.
    assert receiver._verify_election_message(platoon_cand_d1) is True
    assert receiver._verify_quorum_certificate(forged_commander_hb) is False
    assert receiver._verify_heartbeat(forged_commander_hb) is False


def test_bft_full_hierarchy_converges_like_the_non_bft_case():
    swarm = _platoon_swarm({
        "P1": {"A": 70.0, "B": 50.0},
        "P2": {"C": 95.0, "D": 60.0},
    }, config=FAST_BFT)
    _tick(swarm, 30)

    assert swarm.config.bft_mode is True
    assert swarm.drones["A"].nexus_id == "A"
    assert swarm.drones["C"].nexus_id == "C"
    assert len(swarm.command.commander_ids()) == 1


# -- Config validation ---------------------------------------------------------

def test_platoon_of_missing_a_drone_raises_value_error():
    drones = [Drone(id="A", x=0, y=0, priority=1), Drone(id="B", x=0, y=0, priority=1)]
    command_config = CommandConfig(platoon_of={"A": "P1"})  # "B" missing
    with pytest.raises(ValueError):
        Swarm(drones, comm_range=100, config=SwarmConfig(command_config=command_config))


# -- State/metrics shape -------------------------------------------------------

def test_to_state_dict_and_metrics_shape_when_configured():
    swarm = _platoon_swarm({"P1": {"A": 70.0, "B": 50.0}, "P2": {"C": 95.0, "D": 60.0}})
    _tick(swarm, 20)

    state = swarm.to_state_dict()
    assert set(state["command"].keys()) == {"platoons", "commander_ids"}
    assert set(state["command"]["platoons"].keys()) == {"P1", "P2"}
    for platoon in state["command"]["platoons"].values():
        assert set(platoon.keys()) == {"member_ids", "nexus_id"}

    snap = swarm.metrics_snapshot()
    assert "commander" in snap
    assert set(snap["commander"].keys()) == {"recovery", "elections_started", "elections_won", "merges", "security_rejections"}


def test_metrics_snapshot_key_set_unchanged_for_plain_swarm():
    """Explicit regression guard: adding commander metrics must not touch
    the flat-mode snapshot's exact key set."""
    swarm = Swarm([Drone(id="D0", x=0, y=0, priority=1)], comm_range=100, config=FAST)
    _tick(swarm, 5)
    assert set(swarm.metrics_snapshot().keys()) == {
        "recovery", "elections_started", "elections_won", "merges",
        "messages_sent", "messages_delivered", "messages_dropped_loss",
        "security_rejections",
    }


# -- Honest limitation: unreachable platoon-nexus subgraphs ------------------

def test_unreachable_platoons_each_elect_their_own_commander():
    """Two platoons placed far enough apart that no relay chain bridges
    them within max_relay_hops -- documented, expected behavior, not a
    bug: each mutually-reachable cluster elects its own commander,
    mirroring the flat layer's own partition behavior."""
    drones = [
        Drone(id="A", x=0, y=0, priority=70.0),
        Drone(id="B", x=10, y=0, priority=50.0),
        Drone(id="C", x=5000, y=5000, priority=95.0),
        Drone(id="D", x=5010, y=5000, priority=60.0),
    ]
    command_config = CommandConfig(platoon_of={"A": "P1", "B": "P1", "C": "P2", "D": "P2"})
    config = SwarmConfig(**{**FAST.__dict__, "command_config": command_config})
    swarm = Swarm(drones, comm_range=50, config=config, seed=1)  # short range -- P1/P2 physically unreachable

    _tick(swarm, 40)

    assert swarm.drones["A"].nexus_id == "A"
    assert swarm.drones["C"].nexus_id == "C"
    commander_ids = swarm.command.commander_ids()
    assert len(commander_ids) == 2
    assert set(commander_ids) == {"A", "C"}
