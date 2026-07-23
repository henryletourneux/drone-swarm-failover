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

### Movement

Drones aren't pinned in place — each one drifts at a constant velocity and bounces off the arena's edges (`Swarm._move`, run at the start of every tick, before positions sync to the mesh). This is what makes partitions and merges an ongoing, organic part of watching the demo rather than something you can only trigger by clicking — swarms split and reconverge on their own as drones wander in and out of range.

The mesh's actual geometric adjacency (`build_adjacency`/`connected_components`/`bfs_reachable` in `drone_swarm/topology.py`, written from scratch, no graph library) is kept separate from message delivery — it's used purely to draw the right edges in the visualization and to classify each drone's on-screen role (relay vs. leaf), not for election correctness. A drone can be geometrically "in range" of another without their messages having actually gotten through yet, or at all.

## Project structure

```
drone_swarm/
  model.py          # Drone dataclass (pure data — no election bookkeeping)
  topology.py        # adjacency / connected components / BFS — for visualization only
  protocol.py        # message schemas: NexusHeartbeat, ElectionMessage
  mesh_network.py   # range-limited mesh: latency, packet loss, multi-hop relay
  election.py        # NexusElection — per-drone heartbeat/term-based Bully algorithm
  swarm.py            # Swarm: owns drones, the mesh, and every drone's election state
  simulation.py      # random swarm generator
  cli.py               # headless terminal demo, no server needed
server.py            # FastAPI + WebSocket server for the live browser demo
frontend/            # canvas visualization (vanilla HTML/CSS/JS, no build step)
tests/                 # pytest suite covering topology, election, and swarm behavior
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

Then open **http://localhost:8000** in a browser. Click any drone to shoot it down and watch the swarm re-elect a nexus in real time. "Reset Swarm" spawns a fresh random layout.

### Running the tests

```bash
source .venv/bin/activate
pytest
```

## Roadmap / possible extensions

This project is deliberately scoped to a working, well-tested core first. Natural next steps if extended further:
- **Obstacle course / objectives**: give the swarm a goal beyond just staying coordinated — navigate from start to end through obstacles and scripted "laser" hazards that pick off whichever drone is currently nexus, forcing a real-time reassessment and re-election mid-navigation.
- **Byzantine fault tolerance**: handling a drone that's alive but sending bad/malicious election data (a false candidacy, a forged heartbeat), not just drones that go fully offline.
- **Physical demo**: porting the coordination logic onto real hardware (e.g., an ESP-NOW mesh across a few ESP32 boards) or a software-in-the-loop simulator like ArduPilot/Gazebo.

## License

MIT — see [LICENSE](LICENSE).
