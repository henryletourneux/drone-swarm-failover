# Drone Swarm Failover

A mesh network of simulated drones that elects a new coordinator ("nexus") whenever the current one goes down — and keeps doing it, no matter how many times you take out the new one. There's no central authority: the election spreads outward through the mesh one hop per tick, like ripples across a web, until every drone independently converges on the same winner.

Kill the nexus. Watch the swarm heal itself. Kill the replacement. Watch it happen again.

## Why this exists

This is a simplified version of a real, hard distributed-systems problem: how do you keep a decentralized network coordinated when the node everyone was relying on disappears without warning? It's a small relative of leader-election protocols that run in real production infrastructure (etcd, CockroachDB, and friends all solve versions of this). Here it's applied to something visual and intuitive — a swarm of drones — so the behavior is easy to watch and reason about.

## How the algorithm works

1. Every tick, the simulation rebuilds the mesh from scratch: two drones are connected if they're both alive and within radio (`comm_range`) of each other. This is pure geometry, recomputed fresh every tick — no persistent wiring.
2. The mesh naturally splits into **connected components** (drones that can't currently reach each other end up in separate groups — this happens automatically if a kill severs the only bridge between two clusters).
3. Each component checks: is there a drone here that's alive and confirmed as its own nexus? If yes, everyone else in the component just falls in line behind it.
4. If not, an election starts: every drone in the component nominates itself. Then, once per tick, each drone adopts the best candidate (highest `priority`, drone id as a tiebreaker) it's heard from its immediate neighbors so far. That "best so far" value spreads outward one hop per tick.
5. Once every drone in the component has converged on the same candidate, that candidate becomes the confirmed nexus — and the flood stops.

Because this is just geometry + a synchronous flood recomputed every tick, cascading failures and network partitions aren't special-cased — they fall directly out of the same mechanism:
- **Cascading handoff**: kill the new nexus, and next tick nobody in its component has a confirmed nexus anymore, so a new election kicks off automatically.
- **Partition tolerance**: if a kill splits the mesh into two physically disconnected groups, each group independently notices it has no nexus (or keeps the one it already had) and resolves on its own — you can end up with two nexuses at once, one per island, until they're reconnected.

All the graph algorithms (`build_adjacency`, `connected_components`, `bfs_reachable` in `drone_swarm/topology.py`) are written from scratch rather than pulled from a library, so the whole thing is readable end to end.

## Project structure

```
drone_swarm/
  model.py        # Drone dataclass
  topology.py      # adjacency / connected components / BFS — pure graph functions
  election.py      # the flood-based leader election algorithm
  swarm.py          # Swarm: owns drones, advances one tick at a time
  simulation.py    # random swarm generator
  cli.py             # headless terminal demo, no server needed
server.py          # FastAPI + WebSocket server for the live browser demo
frontend/          # canvas visualization (vanilla HTML/CSS/JS, no build step)
tests/               # pytest suite covering topology, election, and swarm behavior
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
- **Movement**: drones currently hold static positions; adding motion would require handling partitions merging back together (two nexuses meeting and yielding to one).
- **Realistic networking**: message latency, packet loss, and bandwidth limits between drones instead of instant, perfect delivery each tick.
- **Byzantine fault tolerance**: handling a drone that's alive but sending bad/malicious election data, not just drones that go fully offline.
- **Physical demo**: porting the coordination logic onto real hardware (e.g., an ESP-NOW mesh across a few ESP32 boards) or a software-in-the-loop simulator like ArduPilot/Gazebo.

## License

MIT — see [LICENSE](LICENSE).
