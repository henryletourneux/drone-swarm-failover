from __future__ import annotations

import random

from .model import Drone
from .swarm import Swarm

DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 500


def create_random_swarm(
    n: int = 14,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    comm_range: float = 180.0,
    seed: int | None = None,
) -> Swarm:
    rng = random.Random(seed)
    drones = [
        Drone(
            id=f"D{i}",
            x=rng.uniform(40, width - 40),
            y=rng.uniform(40, height - 40),
            priority=round(rng.uniform(1, 100), 1),
        )
        for i in range(n)
    ]
    return Swarm(drones, comm_range=comm_range)
