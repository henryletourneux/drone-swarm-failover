"""Headless demo: no server or browser needed.

Boots a random swarm, lets it elect its first nexus, then repeatedly
shoots down whichever drone is currently nexus and watches the swarm
re-elect a replacement — run this to see the cascading handoff in your
terminal.

Usage: python -m drone_swarm.cli [--drones 14] [--kills 3] [--seed 1]
"""
from __future__ import annotations

import argparse

from .simulation import create_random_swarm


def _current_nexus(swarm) -> str | None:
    for drone in swarm.drones.values():
        if drone.alive and drone.nexus_id == drone.id:
            return drone.id
    return None


def run(n_drones: int, n_kills: int, seed: int | None) -> None:
    swarm = create_random_swarm(n=n_drones, seed=seed)
    print(f"Booting a {n_drones}-drone swarm (comm range {swarm.comm_range:.0f})...\n")

    for _ in range(50):
        swarm.tick()
        if _current_nexus(swarm) is not None:
            break

    for round_num in range(1, n_kills + 1):
        nexus_id = _current_nexus(swarm)
        alive = sum(d.alive for d in swarm.drones.values())
        print(f"[tick {swarm.tick_count}] nexus={nexus_id}  alive={alive}/{n_drones}")

        if nexus_id is None or alive <= 1:
            print("Swarm has no more nexus to shoot down. Stopping.")
            break

        print(f"  --> shooting down {nexus_id} ...")
        swarm.kill(nexus_id)

        for _ in range(50):
            swarm.tick()
            if _current_nexus(swarm) is not None or sum(d.alive for d in swarm.drones.values()) <= 1:
                break

        new_nexus = _current_nexus(swarm)
        print(f"  --> new nexus after {swarm.tick_count - 1} ticks of failover: {new_nexus}\n")

    print("Event log:")
    for event in swarm.event_log:
        print(f"  t={event['tick']:>3}  {event['type']:<16} {event['detail']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone swarm failover demo")
    parser.add_argument("--drones", type=int, default=14)
    parser.add_argument("--kills", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args.drones, args.kills, args.seed)


if __name__ == "__main__":
    main()
