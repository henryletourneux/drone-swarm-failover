from __future__ import annotations

import math
import random

from .model import Drone
from .swarm import Swarm, SwarmConfig

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 500


def create_random_swarm(
    n: int = 14,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    comm_range: float = 180.0,
    speed: float = 0.0,
    config: SwarmConfig | None = None,
    seed: int | None = None,
) -> Swarm:
    """speed=0.0 (default) gives a static swarm, same as before movement
    existed. Pass speed>0 for drones that drift in a random direction at
    that many world-units/tick, bouncing off the boundaries."""
    rng = random.Random(seed)
    drones = []
    for i in range(n):
        angle = rng.uniform(0, 2 * math.pi)
        drones.append(Drone(
            id=f"D{i}",
            x=rng.uniform(40, width - 40),
            y=rng.uniform(40, height - 40),
            priority=round(rng.uniform(1, 100), 1),
            vx=speed * math.cos(angle),
            vy=speed * math.sin(angle),
        ))
    return Swarm(drones, comm_range=comm_range, width=width, height=height, config=config, seed=seed)
