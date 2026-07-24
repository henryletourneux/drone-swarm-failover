# Product angle: coordination layer as infrastructure

This project started as a portfolio piece (mesh-network leader-election/failover simulator). This doc captures a separate thread: whether the underlying tech is the seed of an actual product/company, and what that would look like. Nothing here is a commitment — it's a working note.

## The pitch

Not "autonomous security drones." That market is crowded and well-funded (Anduril, Dedrone, Fortem, others), and fielding drones means hardware, FAA/BVLOS regulation, and liability — a fundamentally different, far more capital-intensive business than software.

Instead: **the coordination/resilience layer itself, sold as infrastructure** to companies that already have drones, robots, or distributed IoT hardware and need their fleet to keep coordinating correctly when individual nodes fail, get jammed, or are actively attacked — not just the happy path.

## Why this repo is the right foundation

What's actually differentiated here isn't "self-organizing security drones" (that's coverage-path-planning / patrol-scheduling, a real field with academic and commercial precedent — e.g. USC TEAMCORE's Stackelberg security game work). It's the resilience layer underneath:

- A genuine asynchronous, message-passing leader-election/failover mesh — real per-hop latency, packet loss, multi-hop relay, no instant global knowledge.
- Byzantine fault tolerance: real Ed25519-signed messages, quorum certificates, and a red-team package (`antagonist/`) that attacks it through the same public channel a real adversary would use — not a simulated "trust me it's secure."
- Profiled and scaled to 100 nodes, with two real combinatorial-blowup bugs found and fixed along the way, not just claimed to scale.
- A resource-allocation layer on top (heuristic + a real trained RL policy) proving the coordination layer can carry real application logic, not just survive failures in isolation.

Most published/commercial autonomous-patrol systems assume the coordination layer just works and don't rigorously address adversarial resilience of the network itself. That gap is the pitch.

## Open questions (not yet decided)

- **Who's the actual buyer?** Robotics/drone fleet operators who'd rather not build their own coordination layer? Industrial IoT / critical infrastructure? Needs real validation, not assumption.
- **Product shape**: embeddable library/SDK? A hosted coordination service? Something else?
- **License/IP strategy**: this repo is MIT (portfolio piece, meant to be public). If a coordination-core product ever gets extracted from it, that extraction is where a real licensing decision would need to be made — not this repo by default.
- **Naming**: no product name has been chosen.

## Status

Vision-stage only, no active build. See the drone-swarm-failover roadmap notes (patrol/formation/hierarchical-command ideas) for adjacent product-application thinking — most of that is drone-specific application logic sitting on top of this coordination core, useful context but not this doc's scope.
