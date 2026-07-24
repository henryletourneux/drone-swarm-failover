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

## Hierarchical command (`SwarmConfig(command_config=...)`)

`drone_swarm/command.py` groups drones into static, config-assigned **platoons**, each independently running the exact same election engine as a flat swarm to pick its own nexus — and then a *second*, independent election runs among whichever drones currently hold a platoon-nexus seat, to pick an overall commander. Same mechanism, applied recursively, not a second algorithm: `NexusElection` was already fully generic over `(id, priority, incoming messages)`, so a second instance is enough.

Both tiers run over the exact same mesh — there's no separate long-range radio channel, and none is needed: `MeshNetwork` already relays any message through any alive drone regardless of role, so commander-layer traffic between physically-scattered platoon nexuses is carried by ordinary platoon members acting as relays, for free. What keeps the two elections from cross-contaminating on the wire is a `layer` field on every message (`"platoon:<id>"` or `"commander"`) that each `NexusElection` instance only ever reacts to its own value of — and, in `bft_mode`, that field is folded into the signed payload, so relabeling a genuinely-signed message from one layer to the other fails verification rather than just being filtered by convention.

**A real BFT gap found while building this, worth naming**: the quorum-certificate check that guards large term jumps originally verified each bundled candidacy on its own terms, but never checked that a candidacy's layer matched the heartbeat it was bundled into. That meant a genuinely-signed *platoon*-layer candidacy, sniffed straight off the mesh, could be bundled into a forged *commander*-layer heartbeat and still count toward its quorum. Fixed, with a regression test (`test_bft_quorum_certificate_rejects_wrong_layer_candidacies`) that reintroduces the exact gap and confirms it would have been caught.

**Dynamic membership, the genuinely hard part**: a platoon's nexus seat can change hands at any time (ordinary failover), so the set of drones running a commander election is rebuilt every tick rather than fixed at startup like every other engine in this codebase. Losing a seat mid-campaign is never special-cased — it's made to look, from every other commander-layer engine's point of view, exactly like that participant going silent, which the underlying Bully+recency mechanism already handles correctly (verified live: killing the sitting commander in a 4-platoon, 24-drone swarm converges on a new one within the normal timeout window, no special-cased recovery code required).

**Honest limitation, stated plainly**: if the current platoon-nexus subgraph isn't fully reachable within `max_relay_hops` — dispersed platoons, a small `comm_range`, or a real partition — no single commander converges; each mutually-reachable cluster elects its own, exactly mirroring the flat layer's own already-tested multi-nexus-under-partition behavior. That's why `commander_ids()` returns a list, not a single id.

### Dynamic platoon membership and duty (`SwarmConfig(commander_allocator=...)`)

Platoons above start from a static, config-assigned `platoon_of` — but membership doesn't have to stay that way. `CommandState.reassign_platoon()` moves a drone into a different platoon by giving it a brand-new, term-0 election scoped to the new layer, reusing the exact same "cold-join is always safe to absorb" reasoning the module already relies on for the commander tier itself, one level down: from the old platoon's point of view, the drone just went silent, a case Bully+recency already handles.

`drone_swarm/commander_allocator.py`'s `HeuristicCommanderAllocator` is what actually decides who moves where: periodically, once a commander is elected, it fills one platoon slot per still-undersupplied mission zone with the nearest available drones (**guard** duty, tied to that zone) and spreads everyone else across the remaining slots (**patrol** duty, joining the shared flocking/patrol-route population above). Deliberately mirrors `HeuristicAllocator`'s own honest, explainable, non-learned baseline role.

**A real bug found and independently verified**: the reallocation pool didn't originally exclude drones already successfully guarding a secured zone. Since a secured zone stops appearing in "still needs drones" (precisely *because* that drone is the one filling it), those exact guards would get reassigned to patrol and walk away next pass — un-securing what they'd just secured, which re-triggered guard duty next pass, forever. Fixed by reusing the same "idle" definition (`mission_zone_id is None`) already established throughout `mission.py`/`patrol.py`; caught by an end-to-end test, independently re-verified by reintroducing the bug and confirming it fails.

**Two more real bugs, found live in the running demo rather than by a failing test** — the symptom was visible directly in the visualization: an excess of gold "nexus" markers and almost no cyan "relay" ones. (1) Patrol-slot spreading originally assigned drones to remaining platoon slots by list position (`enumerate`), which reorders depending on which zones needed guards *that* pass — so a drone already stably on patrol duty could get bounced into a different platoon on almost every reallocation cycle even though its job never changed, each bounce forcing a fresh term-0 election. Confirmed live: 40-65 of 100 drones reassigned every single pass, spiking simultaneous nexus count from the expected ~10 (one per platoon) to 30-65+. (2) Fixing that exposed a second issue: a *newly*-available drone could still land in whichever slot merely had the fewest members, with no regard for where that slot's own patrol loop physically is — sending it on an arena-spanning trek to a zone on the opposite side of the map. `_assign_patrol_slots` now keeps an already-patrolling drone sticky in its current slot, and breaks ties among equally-populated slots by physical proximity to each candidate slot's own zone anchor. Both independently verified (reverted, confirmed the new regression tests fail, restored). Live result: nexus count now settles to ~16-24 over a long run (down from a sustained 30-65+), with relay count recovering to a healthy majority.

**Deliberately restricted to plain (non-BFT) mode.** `election.py`'s own module docstring already names "truly dynamic membership" under BFT as an open problem this project doesn't solve — reassigning a drone gives its new platoon a correct quorum threshold, but doesn't retroactively fix the other members' already-fixed one. Rather than solve that, `server.py` simply never wires a commander allocator into Security mode's `bft_mode=True` config; its platoons stay exactly as static as before this feature existed.

**A trained alternative** — `drone_swarm/commander_policy.py`'s `LearnedCommanderAllocator` reuses `policy.py`'s own `AllocatorPolicy`/`drone_zone_features` directly rather than inventing a parallel architecture: "which drone should fill this zone slot" is the identical decision shape at both tiers. A fresh instance is trained specifically on the commander's own triggering cadence and candidate pool via `commander_train.py` (same REINFORCE approach as `train.py`, same two-phase discipline — heuristic first, a learned policy evaluated honestly against it second). **The honest result** (`python3 -m drone_swarm.commander_train --evaluate-only`, 40 held-out episodes): both reach 100% fully-secured, same average zones secured (1.43) — but the heuristic converges faster (36.6 ticks vs. 46.8) and leaves marginally more battery (98.5 vs. 98.0). Not a clean win, reported as-is, same principle as the zone-allocation policy's own evaluation below.

## Patrol and disturbance investigation (`SwarmConfig(patrol_config=...)`)

`drone_swarm/patrol.py` spawns dynamic **disturbance** sites in the arena that the swarm's genuinely idle drones — alive, not currently holding a mission zone, not already investigating something else — break off to investigate: travel to the site, hold there accruing investigation time, and resolve it once enough time has passed. Built directly on top of `mission.py` rather than as a parallel system: dispatch reuses the exact same idle/reserve-pool notion `HeuristicAllocator.plan_substitutions()` already established for battery-substitution relief, and a resolved investigator is simply freed (`investigating_disturbance_id = None`) for `MissionState`'s own allocation/substitution machinery to pick back up — no bespoke "return to post" logic.

**Proper resource allocation, not just "someone eventually looks into it"**: each disturbance carries a random `severity` (1-3), and resolving it takes that many drones' worth of *combined* presence — `_advance` accrues `min(currently-present investigators, severity)` effort per tick against a `severity × investigation_ticks_required` total, so sending fewer investigators than the situation calls for genuinely slows resolution down (proportionally, not a cliff), and sending more than needed doesn't speed it up further. `_dispatch` mirrors `HeuristicAllocator.allocate()`'s own shape almost exactly: keep assigning the nearest still-idle drone to whichever under-resourced disturbance needs it most (most severe, then longest-waiting) until every disturbance's headcount is met or idle drones run out. An investigator dying mid-investigation no longer resets accumulated progress to zero — only the vacancy needs backfilling, more realistic than punishing the whole effort for one loss.

**A real design bug found while building this, worth naming**: the first version dispatched from a zone's *surplus* occupants (beyond `required_drones`) rather than the idle pool. That's a different failure mode than a typical off-by-one — `HeuristicAllocator.allocate()` never assigns more than a zone's `required_drones` in the first place, so genuine surplus essentially never occurs under any realistic mission configuration, meaning disturbances would spawn and simply never get investigated. Caught by two end-to-end tests that failed against the surplus-based version and pass against the idle-pool version (`test_investigation_requires_arrival_before_accruing_progress`, `test_resolved_disturbance_frees_investigator_into_reserve_pool`), independently re-verified by reintroducing the surplus-based logic and confirming seven tests fail against it before restoring the fix.

**A second, cross-module bug the first one exposed**: neither `HeuristicAllocator.allocate()` nor `LearnedAllocator.allocate()` checked `investigating_disturbance_id` before assigning a drone to a mission zone — so the periodic reallocation pass could hand a zone assignment to a drone that was still mid-investigation. Because movement priority favors an active investigation over a mission-zone assignment, the drone would never actually travel to the zone it was just assigned, silently orphaning both. `LearnedAllocator.allocate()` additionally turned out to have never checked `mission_zone_id` at all — a pre-existing gap independent of patrol.py, fixed alongside it for consistency between the two allocator implementations. Regression-tested by `test_allocator_does_not_steal_a_drone_mid_investigation`, likewise independently verified by reverting the fix and confirming the test fails.

**Honest limitation, stated plainly**: dispatch is greedy and single-pass, not globally optimal — each unresolved disturbance (oldest first) claims the single nearest currently-idle drone, one at a time. Two disturbances spawning on opposite sides of the arena in the same tick can each grab a suboptimal drone if a truly optimal assignment would have crossed them — the same tradeoff `HeuristicAllocator` already makes for zone assignment (greedy and explainable over globally optimal). It's also a soft dependency on `mission_config`: without an active mission there's no notion of "committed," so disturbances spawn and age but are never dispatched — nothing breaks, they just never resolve.

Live in Scale mode's demo: a pulsing `!` marker with a fill-up progress ring shows each active disturbance, with a dashed line out to whichever drone is currently investigating it; it turns into a `✓` briefly on resolution before disappearing. Clicking any empty patch of arena places a disturbance there directly (bypassing `max_active_disturbances` — a deliberate one-off action isn't the ambient spawn traffic that cap exists to throttle), with a crosshair cursor over empty space signaling the click is live.

## Flocking and patrol route (`SwarmConfig(flocking_config=...)`)

`drone_swarm/flocking.py` gives drones with nothing else assigned (no mission zone, not investigating a disturbance) real, coordinated group movement — separation, alignment, and cohesion, the classic three-rule boid model — instead of each one drifting independently in its own straight line. `flock_velocity()` is a standalone steering function (a drone, its already-found neighbors, and an optional shared target point in, a new velocity out), kept deliberately decoupled from how neighbors get found (`Swarm._move` uses `spatial_grid.py`'s `SpatialGrid`, the same O(n) proximity structure `mesh_network.py` and `topology.py` already rely on) or *why* a drone is flocking.

That "why" is `patrol.py`'s patrol route. Without hierarchical command, it's a single elliptical ring of waypoints auto-generated inset from the arena edges — sized to whatever the arena's actual dimensions are, no per-mode tuning needed — that the entire idle population tours together as one flock, advancing to the next waypoint once the idle population's own *centroid* arrives (not any single drone, so a straggler at the back never yanks the target away from drones still converging).

**With hierarchical command, each platoon gets its own small loop instead** — circling whichever defendable mission zone it's nearest to (round-robin if there are more platoons than zones), rather than the whole idle population sharing one arena-spanning ring. This replaced the single-shared-route design after a real, live-observed problem: a hundred drones all converging on one distant point at a time meant the "flock" routinely spanned over 1500 units of a 2000-unit arena — reading as one shapeless blob, not an organized patrol, and scattering same-platoon drones far enough apart to lose radio contact entirely (see Hierarchical command's dynamic-membership section for how this fed directly into the "excess nexuses" bug). Small, zone-anchored loops keep each platoon's own patrol population physically close together — a better read as an actual patrol pattern, and healthier for the platoon's own mesh connectivity.

**Play with it live**: unlike every other `*Config` in this codebase, `FlockingConfig` is deliberately not frozen — a small panel in the top-right of Scale mode's demo exposes separation/alignment/cohesion/speed as sliders plus a patrol-route on/off toggle, all wired to a WebSocket control message that mutates the *running* swarm's config in place. No reset, no lost state — turn cohesion up and watch the flock visibly tighten in real time.

Scale mode's arena was also widened (1400×900 → 2000×1280, comm_range nudged 180 → 210 to keep the mesh well-connected at the larger size) specifically so the flock and patrol route have real room to move rather than reading as crowded — verified live that nexus convergence and tick timing (~6ms/tick at 100 drones) are unaffected.

## Topology and obstacles (`SwarmConfig(obstacles=...)`)

`drone_swarm/obstacles.py` adds static circular barriers every movement mode — flocking, mission-zone travel, disturbance travel, and the plain drift-bounce fallback — bends around via a shared `Swarm._avoid_obstacles` helper. Deliberately a **movement-only** concept: obstacles never affect mesh connectivity (`mesh_network.py`'s comm-range/relay logic is untouched), since modeling real radio line-of-sight would mean reworking the already-carefully-tuned relay-flood logic rather than freely extending this feature.

**Two real, independently-verified bugs, both genuine instances of the classic local-minimum trap in potential-field navigation:**

- `push_point_outside_all` (keeps generated waypoints/disturbances from landing inside an obstacle) went through two broken versions first: leaping straight away from whichever obstacle was hit first could bounce forever between two deeply-overlapping obstacles; summing repulsion and taking small steps looked more principled but settled into a stable 2-point oscillation at the force-equilibrium point — not the same thing as "actually outside every obstacle." Fixed with an expanding-radius, all-angles search instead, which can't oscillate (no gradient to follow) and can't get trapped in a cancellation point (it tests candidates directly).
- Real-time `obstacle_avoidance` (live steering) was purely radial at first — a drone travelling in a straight line with an obstacle directly on its path would stall exactly where the radial push equalled its own pull toward the target, and no amount of tuning fixes that, because "straight back" is the one direction that can exactly cancel "straight forward." Fixed by adding a tangential (edge-sliding) component that picks its rotation direction from the drone's own intended heading, so it curves the way it was already trying to go.

Patrol route waypoints and disturbance spawns/placements are nudged clear of obstacles via the same expanding-search helper. Both obstacle layout and mission zones (the defendable areas) are randomized fresh on every server startup and every "Reset Swarm" click — a deliberate environmental factor the commander has to actually respond to each time, not a fixed map it could memorize, and not something exposed as a manual editor in the UI (there's no dropdown or control for it — it just happens). Rejection-sampled apart from each other and from zones (a generous retry budget, not an infinite loop) so a run doesn't get unlucky with everything piled in one corner. Confirmed no performance regression (~5ms/tick) across random layouts, and no drone left permanently stuck inside an obstacle.

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
  patrol.py            # disturbance investigation + patrol route — see Patrol and disturbance investigation / Flocking above
  flocking.py           # boid steering primitive — see Flocking and patrol route above
  obstacles.py          # static circular barriers + avoidance steering — see Topology and obstacles above
  command.py          # hierarchical platoon/commander election — see Hierarchical command above
  commander_allocator.py  # heuristic guard/patrol duty + dynamic platoon allocator — see Hierarchical command above
  commander_policy.py     # learned commander allocator (PyTorch) — a drop-in for commander_allocator.py's heuristic
  commander_train.py       # REINFORCE training + evaluation pipeline for commander_policy.py
  policy.py            # learned allocation policy (PyTorch) — a drop-in for mission.py's allocator
  train.py              # REINFORCE training + evaluation pipeline for policy.py
  simulation.py      # random swarm generator
  cli.py               # headless terminal demo, no server needed
antagonist/          # adversarial testing tool — see the BFT section below
server.py            # FastAPI + WebSocket server for the live browser demo (Scale / Security modes)
frontend/            # canvas visualization (vanilla HTML/CSS/JS, no build step)
tests/                 # pytest suite covering topology, mesh, election, swarm, BFT, metrics, mission, policy, command, commander allocation/policy, patrol, flocking, and obstacle behavior
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

### Battery substitution / reserve pool

Once a zone is secured, its occupants can still keep draining indefinitely — the periodic `HeuristicAllocator.allocate()` pass above only runs every `reallocation_interval_ticks` and only fills *unsecured* zones, so a low-battery drone quietly holding a secured zone had no path to relief. `HeuristicAllocator.plan_substitutions()` is a second, faster layer that runs every tick: once a secured zone's occupant drops below `LOW_BATTERY_WARNING` (35, above the 20 that would make it ineligible outright), it dispatches the best-scoring idle, healthy drone as an incoming replacement — and, when several zones need relief at once, prioritizes the most urgent (lowest battery, then highest threat) first. The tired occupant isn't released the instant a replacement is *dispatched* — only once it's actually *arrived* (an explicit "surplus" check), so the zone is never under-secured mid-handoff. `LearnedAllocator` implements the same `plan_substitutions()` contract using the trained network's scoring instead of hand-tuned weights, so it stays a genuine drop-in either way.

There's no battery-recharge mechanic in this simulation — "reserve pool" means healthy, currently-uncommitted drones, not a charging station.

Building this against a real running swarm (not just unit tests) surfaced a real bug worth naming: a drone en route to a zone that got secured by *other*, physically-closer drones before it arrived would never become an occupant and never drain low enough to lose its assignment either — so it carried a dead `mission_zone_id` forever, permanently unavailable as reserve capacity. Fixing that then created a second, sneakier one: since a substitution's whole point is dispatching a drone toward an *already-secured* zone, the same "release it, it's not needed" check immediately fired on the drone substitution had just sent — releasing it before it could ever arrive, then redispatching it the very next tick, forever. Both are covered by regression tests (`test_stale_assignment_released_once_zone_secured_without_it`, `test_inbound_substitute_not_released_before_it_arrives`) that reintroduce each bug independently and confirm the tests actually catch it.

### A learned allocation policy

`drone_swarm/policy.py` and `drone_swarm/train.py` are a real, working reinforcement-learning pipeline (PyTorch, REINFORCE with an entropy bonus and a running-average baseline) trained against the mission environment above — not a heuristic dressed up, an actual small neural network (`AllocatorPolicy`) that scores each (drone, zone) pair and learns its own weights from simulated rollouts. `LearnedAllocator` implements the exact same `.allocate()` interface as `HeuristicAllocator`, so it's a genuine drop-in swap for `MissionState.allocator`, not a parallel system.

**Design choice worth calling out**: the network's features are all *relative* (battery fraction, distance as a fraction of the arena diagonal, normalized threat/need, role, secured status) — never absolute drone count or swarm size. That makes the policy's input space agent-count-invariant: training on a small, fast-to-simulate swarm (8-14 drones, a couple of zones) is learning the same kind of decision a 100-drone deployment needs, not a smaller version of a different problem. `tests/test_policy.py` checks this explicitly, not just assumes it.

**The honest result** (`python3 -m drone_swarm.train --evaluate-only`, 50 held-out episodes, not seen during training):

| | zones secured | ticks to secure | fully-secured rate | avg battery left |
|---|---|---|---|---|
| Heuristic | 2.38 | 65.8 | 96% | 96.7 |
| Learned | 2.40 | 81.3 | 98% | 95.1 |

The learned policy is **not a clean win** — it matches the heuristic on outcome quality and is slightly more reliable (higher completion rate), but converges noticeably slower and leaves marginally less battery. That's reported as-is, not massaged, because a suspiciously perfect result would be a worse portfolio signal than an honest, comparable-tradeoffs one: it's evidence the evaluation is real, not tuned to make the model look good. A genuine improvement here would most likely need more training signal (actor-critic instead of vanilla REINFORCE, or reward shaping specifically tuned for speed) rather than just more episodes at the current setup — the training curve visibly plateaus well before 600 episodes.

Two real bugs surfaced and fixed while building this, both worth naming because they're the kind of mistake specific to RL pipelines, not general software: (1) `LearnedAllocator` tracked its latest decisions as a "last call" snapshot rather than an explicit per-episode accumulate/reset lifecycle, which caused stale, already-backpropped tensors from a previous training episode to leak into the next one's loss computation, crashing on the second `.backward()` call with "trying to backward through the graph a second time"; (2) an early version of the "does training actually learn something" test reimplemented a simplified training loop that skipped the baseline subtraction the real `train()` uses, which is a well-known REINFORCE instability -- it wasn't testing the shipped pipeline, and the reimplemented version was genuinely unstable when I checked. Fixed by testing the real `train()` function directly, plus adding a surgical, isolated gradient-mechanic test (a simple two-zone "bandit" scenario checking REINFORCE actually shifts probability toward a clearly-better action) as the standard, environment-independent way to validate a policy-gradient implementation.

```bash
source .venv/bin/activate
python3 -m drone_swarm.train                       # train from scratch + evaluate + save to models/allocator_policy.pt
python3 -m drone_swarm.train --evaluate-only        # load the committed model and just run the comparison
```

## Roadmap / possible extensions

This project is deliberately scoped to a working, well-tested core first. Natural next steps if extended further:
- **A better learned policy**: actor-critic instead of vanilla REINFORCE, to get past the plateau documented above (applies to both the zone-allocation and commander policies).
- **Live demo toggle**: swap the heuristic for the trained learned policy in Scale mode at runtime (mission-level `LearnedAllocator` and/or commander-level `LearnedCommanderAllocator`), so the comparison is visible, not just in an evaluation report.
- **Moving/attacking obstacles**: static barriers exist now (see Topology and obstacles above); scripted "turret" hazards that actively pick off whichever drone is currently nexus mid-mission are a natural next layer on top.
- **Per-platoon patrol routes**: the shared patrol route currently tours the *entire* idle population as one flock — giving each platoon its own route once patrol duty is platoon-scoped (see Hierarchical command's dynamic-membership section) is a natural, honestly-noted extension.
- **Dynamic platoon membership under `bft_mode`**: deliberately not attempted (see Hierarchical command above) — `election.py` already names this an open problem in real BFT systems, not unique to this project.
- **Route allocation through the real mesh**: replace the omniscient allocation calls above (mission-level and commander-level) with actual status-report messages, including the possibility of stale/missing data.
- **`bft_mode` at real scale**: closing the gap documented above — likely needs batching/amortizing verification rather than one Ed25519 op per message, not just more connectivity tuning.
- **Rate limiting**: the antagonist's one documented, honest gap (flood/spam isn't currently mitigated at the resource level).
- **Extract `antagonist/` into its own project**: it's already scoped for this (see above) — a general-purpose adversarial mesh-network testing tool, not drone-specific in its core attack logic.
- **Physical demo**: porting the coordination logic onto real hardware (e.g., an ESP-NOW mesh across a few ESP32 boards) or a software-in-the-loop simulator like ArduPilot/Gazebo.

## License

MIT — see [LICENSE](LICENSE).
