"""Flocking: boid-style coordinated group movement (Reynolds, 1987 --
separation, alignment, cohesion) for drones that aren't currently doing
anything else -- no mission zone, no disturbance investigation.

Built as a standalone steering primitive (a function of one drone, its
already-identified nearby flockmates, and an optional target point) rather
than folded directly into `Swarm._move`, so `patrol.py`'s patrol-route
steering can reuse it without patrol.py needing to know anything about
boid mechanics -- it just supplies a target point to steer the flock
toward. Finding *which* drones count as "nearby flockmates" is deliberately
left to the caller (`Swarm._move`, using `spatial_grid.py`'s `SpatialGrid`
the same way `mesh_network.py` and `topology.py` already do for O(n)
proximity queries) -- this module only does the steering math once the
neighbor list is in hand.

`FlockingConfig` is deliberately NOT a frozen dataclass, unlike every
other `*Config` in this codebase (`MissionConfig`, `PatrolConfig`,
`CommandConfig`). Those are fixed for a swarm's lifetime by design; this
one is meant to be tuned live from the running demo (see server.py's
"set_flocking" control message) so a person watching can actually feel
what turning cohesion up or separation down does to the flock, in real
time, without resetting the whole swarm and losing its state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FlockingConfig:
    neighbor_radius: float = 90.0
    separation_radius: float = 28.0
    separation_weight: float = 1.4
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.9
    target_weight: float = 1.6
    max_speed: float = 4.5


def flock_velocity(drone, neighbors: list, config: FlockingConfig, target=None) -> tuple:
    """Returns a new (vx, vy) for `drone`.

    `neighbors`: other drones already known to be within
    `config.neighbor_radius` (the caller's job to find -- see module
    docstring). `target`: an optional (x, y) world point the whole flock
    should be heading toward (patrol.py's current patrol waypoint); its
    pull is direction-only (normalized) so it acts like a steady compass
    bearing regardless of distance, rather than a force that would
    overwhelm everything else when the target is far away and fade to
    nothing right before arrival.
    """
    sep_x = sep_y = 0.0
    align_x = align_y = 0.0
    coh_x = coh_y = 0.0

    if neighbors:
        for n in neighbors:
            dx, dy = drone.x - n.x, drone.y - n.y
            dist = math.hypot(dx, dy) or 0.01
            if dist < config.separation_radius:
                sep_x += dx / dist
                sep_y += dy / dist
            align_x += n.vx
            align_y += n.vy
            coh_x += n.x
            coh_y += n.y

        count = len(neighbors)
        align_x /= count
        align_y /= count
        # Cohesion is the raw vector toward the flockmates' centroid,
        # damped by a small constant rather than normalized -- a straggler
        # far from the group gets pulled back proportionally harder than
        # one that's already close, without needing a separate "how far is
        # too far" threshold.
        coh_x = (coh_x / count - drone.x) * 0.02
        coh_y = (coh_y / count - drone.y) * 0.02

    tx = ty = 0.0
    if target is not None:
        dx, dy = target[0] - drone.x, target[1] - drone.y
        tdist = math.hypot(dx, dy)
        if tdist > 0:
            tx, ty = dx / tdist, dy / tdist

    vx = (
        sep_x * config.separation_weight
        + align_x * config.alignment_weight
        + coh_x * config.cohesion_weight
        + tx * config.target_weight
    )
    vy = (
        sep_y * config.separation_weight
        + align_y * config.alignment_weight
        + coh_y * config.cohesion_weight
        + ty * config.target_weight
    )

    speed = math.hypot(vx, vy)
    if speed > config.max_speed:
        vx, vy = vx / speed * config.max_speed, vy / speed * config.max_speed
    elif speed < 0.4 and target is None:
        # Nothing meaningfully pulling this drone anywhere (alone, no
        # target) -- keep its previous heading rather than snapping to a
        # dead stop, which would read as the boid model breaking rather
        # than "the flock is calmly drifting."
        vx, vy = drone.vx, drone.vy
    return vx, vy
