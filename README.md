# Drone Swarm Failover

A mesh network of simulated drones that elects a new coordinator ("nexus") whenever the current one goes down, and keeps doing it ad nauseum ad infinitum, no matter how many times you take out the new one. There's no central authority: the election spreads outward through the mesh one hop per tick, like ripples across a spiders web or static electricity through your hair, until every drone independently converges on the same winner.

Kill the nexus. Watch the swarm heal itself. Kill the replacement. Watch it happen again, pretty cool if i do say so myself.

## Why this exists

This is a simplified version of a real, hard distributed-systems problem: how do you keep a decentralized network coordinated when the node everyone was relying on disappears without warning? It's a small relative of leader-election protocols that run in real production infrastructure (etcd, CockroachDB, and friends all solve versions of this). Here it's applied to something visual and intuitive — a swarm of drones — so the behavior is easy to watch and reason about.

## How the algorithm works

No drone here has instant, perfect knowledge of the swarm. Every drone runs its own independent election state machine (`NexusElection` in `drone_swarm/election.py`), and everything it believes is purely a function of which messages have actually reached it — over `MeshNetwork` (`drone_swarm/mesh_network.py`), a genuinely simulated radio mesh with realistic per-hop **latency**, **packet loss**, and a **hop limit**, not instant perfect delivery.

1. A drone with no live nexus — never seen one, or its heartbeats have gone stale for longer than `nexus_timeout_s` — opens an election for the next **term** and broadcasts its candidacy (an `ElectionMessage` carrying its `priority`).
2. Candidacies are ordered by `(priority, drone_id)` — highest priority wins, id breaks ties deterministically.
3. A candidate that hears a better candidacy for its term steps back; the sole un-superseded candidate declares victory once its election window closes, and starts broadcasting `NexusHeartbeat`s.
4. Followers refresh their "nexus is alive" clock on every heartbeat they receive. No heartbeat for longer than `nexus_timeout_s` and they conclude the nexus is gone and open a new election themselves.

This is a proper Bully-algorithm implementation, not a shortcut — messages actually propagate hop-by-hop through `MeshNetwork.deliver_due_messages`, with real relay (up to `max_relay_hops`) and a `seen_by` set so a flooded message can't loop forever.

Cascading failure, partitions, and merges all fall out of this same mechanism, no special-casing required:
- **Cascading handoff**: kill the nexus, and every drone that was following it eventually times out and re-elects — automatically, repeatedly, however many times you do it.
- **Partition tolerance**: sever the only bridge between two clusters and each side independently notices (or keeps) its own nexus — you can end up with two nexuses at once, one per island.
- **Merge**: when two islands drift back into range, the two nexuses' heartbeats reach each other, and the tie resolves the way real consensus protocols do it — **the newer term always wins on contact, and same-term ties go to the higher drone id**, not priority. Priority only ever decides a single fresh election, never a stale-vs-fresh conflict — an old, possibly-stale nexus's priority claim isn't trustworthy evidence about what happened on the other side of a partition; recency is. So merges are just ordinary heartbeat handling, with no dedicated merge/runoff code path at all.

An earlier version of this project used a synchronous global recompute each tick instead — every drone had instant, perfect knowledge of the whole mesh, which made convergence trivial but wasn't an honest simulation of a real network. Rebuilding it on real message-passing surfaced a genuine bug in the process: a stale, late-arriving candidacy for a term that was already decided could wrongly knock an already-elected nexus back into a fresh campaign, forever. `tests/test_election.py` has a regression test for it.

## Byzantine fault tolerance (`SwarmConfig(bft_mode=True)`)

Everything above trusts message content at face value — fine against clean failures (a drone going offline), not against a rogue transmitter actively lying. `bft_mode` hardens the swarm against a specific, honestly-scoped threat model, using real Ed25519 signatures (the `cryptography` library — the same modern signature scheme behind SSH, TLS 1.3, and Signal — not hand-rolled crypto):

- **Impersonation** — a rogue transmitter claims to be a real drone. Defeated: every message is signed by the sender's own key, and verified against that drone's known public key before it's trusted at all. No valid signature, no effect on the swarm.
- **Priority forgery** — a compromised drone lies about its own priority to win elections it shouldn't. Defeated: every claimed priority must be backed by a `Credential` signed once, at swarm setup, by a `SwarmAuthority` key (standing in for a fleet operator/manufacturer). A drone has its own signing key but not the authority's, so it can forge its own messages freely but can't unilaterally claim a higher priority than it was actually issued.
- **Term inflation** — a forged heartbeat claims a huge term number to hijack the whole swarm's loyalty in one shot. Defeated: a heartbeat claiming a term more than one step ahead of what a receiver already knows must carry a `quorum_certificate` — real, independently-verified candidacies from a strict majority of the swarm, proving an election actually happened. Small, routine increments (ordinary cascading failover) never need one, so this doesn't change day-to-day behavior — it only closes the term-inflation attack specifically.
- **Replay / repurposed evidence** — resending an old legitimate message, or bundling a real captured candidacy into a certificate for a different term than it was actually signed for. Defeated: certificates only count entries whose term matches exactly.

**Honest limitation, stated plainly, not glossed over:** the quorum threshold is a majority of the swarm's *original* member count, fixed at setup. That's a deliberate simplification — it means a connected component smaller than a majority of the original swarm can't *cryptographically certify* a big term jump for itself under `bft_mode`, even though the underlying election mechanism would otherwise let it operate independently (see Partition tolerance above). This project defends against a **minority** of malicious/rogue nodes within a partition, not an unbounded majority, and doesn't solve fully dynamic BFT membership — both genuinely open problems in real distributed systems, not shortcuts unique to this toy version.

### The antagonist

`antagonist/` is a separate, deliberately loosely-coupled package that plays the adversary — it only ever talks to a target swarm through the same message-injection channel a real rogue transmitter would use (`mesh.broadcast()`), never by reaching into election internals or reading real private keys out of the swarm to cheat. It runs each attack above against a live `bft_mode` swarm and reports, attack by attack, whether the swarm's real elected nexus was affected. See `antagonist/README.md` for details — it's scoped so it could be extracted into its own general-purpose adversarial mesh-network testing tool later with minimal rework.

```bash
source .venv/bin/activate
python3 -m antagonist.cli
```

It's also live in the browser demo, in **Security mode** (see Scaling below) — the **☠ Launch Attack** button throws a random attack from `antagonist/attacks.py` at the actual running swarm. Watch it play out on the canvas (a projectile flies in, impacts, a vignette pulse), the Security stat climb, and the elected nexus stay completely unaffected.

## Scaling to 100 drones

Two real, distinct bottlenecks turned up profiling this, worth documenting because the second one very nearly got papered over with a config tweak instead of actually fixed:

1. **O(n) or O(n²) neighbor search.** Both `MeshNetwork.neighbors_of` and `topology.build_adjacency` originally checked every drone against every other drone to find who's in range — fine at 14 drones, measured at ~193ms/tick at 100 (a live demo needs to comfortably clear its tick budget many times over). Fixed with `drone_swarm/spatial_grid.py`, a uniform grid that buckets drones by cell so a query only checks nearby cells instead of the whole population — O(n) per tick instead of O(n²).
2. **The real bottleneck, found by profiling *through* that fix and seeing barely any improvement:** `MeshNetwork`'s relay flood tracked `seen_by` per relay *path*, not globally per message. In a densely-connected mesh, the same broadcast reaches the same recipient via multiple different paths, and each redundant copy keeps relaying further — combinatorial blowup, not linear. Measured: one broadcast in a 14-drone fully-connected mesh produced ~5.8 million redundant sends over 50 ticks. Fixed by sharing one mutable `seen_by` set across every path for a given broadcast, marking a recipient "seen" only once a delivery to them actually succeeds (so a loss-dropped attempt doesn't block a *different* path from still reaching them — redundant paths keep their real benefit, resilience against packet loss, without the explosion). `tests/test_mesh_network.py` has a regression suite for this, independently re-verified by temporarily reintroducing the bug and confirming it reproduces the blowup before restoring the fix.

With both fixed, 100 drones runs at single-digit milliseconds per tick in plain mode. `bft_mode` at 100 drones remains a real, harder frontier — Ed25519 verification cost scales with message *volume*, which still scales with connectivity density even after the fixes above, and a mesh sparse enough to keep that cheap looks fragmented and unimpressive, not sophisticated. Rather than fake it with a degraded mesh or a misleadingly slow cadence, the live demo has two honest, distinct modes instead of one compromise:

- **⚡ Scale** (default) — 100 drones, plain mode, fast, richly connected (~6.5 average neighbors).
- **🛡 Security** — 14 drones, `bft_mode` on, tuned to run comfortably, antagonist-ready.

### Movement

Drones aren't pinned in place — each one drifts at a constant velocity and bounces off the arena's edges (`Swarm._move`, run at the start of every tick, before positions sync to the mesh). This is what makes partitions and merges an ongoing, organic part of watching the demo rather than something you can only trigger by clicking — swarms split and reconverge on their own as drones wander in and out of range.

The mesh's actual geometric adjacency (`build_adjacency`/`connected_components`/`bfs_reachable` in `drone_swarm/topology.py`, written from scratch, no graph library) is kept separate from message delivery — it's used purely to draw the right edges in the visualization and to classify each drone's on-screen role (relay vs. leaf), not for election correctness. A drone can be geometrically "in range" of another without their messages having actually gotten through yet, or at all.

## Project structure

```
drone_swarm/
  model.py          # Drone dataclass (pure data — no election bookkeeping)
  spatial_grid.py    # uniform grid for O(n) proximity queries — see Scaling below
  topology.py        # adjacency / connected components / BFS — for visualization only
  protocol.py        # message schemas: NexusHeartbeat, ElectionMessage
  identity.py         # Ed25519 signing keys, credentials, SwarmAuthority — bft_mode only
  mesh_network.py   # range-limited mesh: latency, packet loss, multi-hop relay
  election.py        # NexusElection — per-drone heartbeat/term-based Bully algorithm
  swarm.py            # Swarm: owns drones, the mesh, and every drone's election state
  metrics.py          # recovery time, election/message counters — see Metrics below
  mission.py          # zone-coverage resource allocation — see Resource allocation below
  simulation.py      # random swarm generator
  cli.py               # headless terminal demo, no server needed
antagonist/          # adversarial testing tool — see the BFT section below
server.py            # FastAPI + WebSocket server for the live browser demo (Scale / Security modes)
frontend/            # canvas visualization (vanilla HTML/CSS/JS, no build step)
tests/                 # pytest suite covering topology, mesh, election, swarm, BFT, and metrics behavior
```

## Running it

### Quickest: headless terminal demo (no browser, no server)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m drone_swarm.cli --drones 14 --kills 3 --seed 1
```

This boots a random swarm, lets it elect its first nexus, then shoots down whoever is currently nexus three times in a row, printing the event log as it happens.

### Live visual demo

```bash
source .venv/bin/activate    # if not already active
uvicorn server:app --reload
```

Then open **http://localhost:8000** in a browser. Click any drone to shoot it down and watch the swarm re-elect a nexus in real time. "Reset Swarm" spawns a fresh random layout in the current mode; the **⚡ Scale / 🛡 Security** toggle switches between the 100-drone and BFT-hardened demos (see Scaling below).

## Metrics

The swarm is something you can measure, not just watch — `drone_swarm/metrics.py`, surfaced live in the browser demo's stats strip and printed at the end of every CLI run:

- **Recovery time** — mean/p50/p95/max of, per drone, how long it went without a known nexus. Deliberately a *per-drone* quantity rather than a single swarm-wide "time to recovery": with multiple drones, partitions, and concurrent campaigns there's no single unambiguous way to match a specific nexus death to the specific election that "recovered" from it, but each drone's own gap between losing and regaining a known nexus is precisely and honestly computable from its own state transitions.
- **Elections started / won, and merges** — cross-checked in tests against the event log's own tallies as an independent consistency check, not just trusted from the counting logic alone.
- **Message volume** — sent / delivered / dropped-to-loss, counted per (message, recipient) transmission attempt including relay hops, since that's what actually reflects radio traffic.
- **Security rejections** (`bft_mode`) — how many incoming messages a drone has dropped for failing signature/credential/quorum verification. Ties directly into the antagonist: run it against a swarm and watch this counter climb in real time as each attack is thrown and blocked.

### Running the tests

```bash
source .venv/bin/activate
pytest
```

## Resource allocation: zone coverage under threat

`drone_swarm/mission.py` is a genuinely new subsystem, not a bolt-on: the arena gets designated **zones** drones need to reach and hold, each requiring a minimum headcount to count as "secured." Every drone has a finite `battery` that drains with movement and drains *faster* while camped in an undersupported, contested zone (`threat_level > 0`) — running low doesn't destroy a drone (that's kill()/antagonist's job), it just makes it ineligible for further zone assignments, keeping this layer additive on top of the core election/mesh mechanics, the same principle `bft_mode` used.

Allocation decisions are made by whichever drone is currently the elected nexus, on a fixed interval — a real tie back into the failover story: lose the nexus mid-mission and the newly-elected one has to pick up allocation duties. The baseline is `HeuristicAllocator`, a fully explainable, weighted greedy assignment (battery, distance, and a penalty for pulling a structurally load-bearing `relay` off its post in favor of a less-critical `leaf`) — the honest bar a future learned policy needs to clear to be worth using at all, not a strawman.

**Honest simplification, stated plainly:** allocation currently runs with full, instantaneous knowledge of every drone's position and battery — an omniscient call made by the simulation (mirroring how role assignment already works), not routed through the mesh's own realistic, partial-information message-passing the way election is. That's a deliberate scope cut for this phase, and a natural next refinement — real uncertainty about drone state is exactly the kind of problem a learned policy would have a genuine reason to exist for, rather than just re-deriving the same heuristic weights.

It's live in Scale mode's demo (three zones of varying threat) — watch drones peel off, travel, and hold position as zones fill and lock in `secured`.

## Roadmap / possible extensions

This project is deliberately scoped to a working, well-tested core first. Natural next steps if extended further:
- **A learned allocation policy**: train a model (RL from simulated rollouts, or supervised imitation of/improvement on the heuristic) against the mission environment above, evaluated honestly against `HeuristicAllocator` — if it doesn't beat the baseline, that's a real finding to report, not something to bury.
- **Obstacle course**: scripted "turret" hazards and barriers layered onto zone navigation, picking off whichever drone is currently nexus mid-mission.
- **Route allocation through the real mesh**: replace the omniscient allocation call above with actual status-report messages, including the possibility of stale/missing data.
- **`bft_mode` at real scale**: closing the gap documented above — likely needs batching/amortizing verification rather than one Ed25519 op per message, not just more connectivity tuning.
- **Rate limiting**: the antagonist's one documented, honest gap (flood/spam isn't currently mitigated at the resource level).
- **Extract `antagonist/` into its own project**: it's already scoped for this (see above) — a general-purpose adversarial mesh-network testing tool, not drone-specific in its core attack logic.
- **Physical demo**: porting the coordination logic onto real hardware (e.g., an ESP-NOW mesh across a few ESP32 boards) or a software-in-the-loop simulator like ArduPilot/Gazebo.

## License

MIT — see [LICENSE](LICENSE).
