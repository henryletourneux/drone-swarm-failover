"""Topology / obstacles: static circular barriers in the arena that every
movement mode has to bend around -- flocking, travelling to a mission
zone, travelling to investigate a disturbance, and the patrol route
itself, not just one of them. Kept deliberately narrow in scope: this is
a MOVEMENT/pathing concept only, not a radio-propagation one -- an
obstacle blocks a drone's flight path but does not block mesh messages
(`mesh_network.py`'s comm_range/relay logic is untouched). Modeling radio
line-of-sight would mean reworking the already-carefully-tuned relay
flood logic (see mesh_network.py's own module docstring for the
combinatorial-blowup bug that logic was hard-won against); that's a real,
separate feature, not a free extension of this one.

Circles, not arbitrary polygons -- the same shape convention `mission.py`
already uses for zones, and the only shape `obstacle_avoidance`'s steering
math needs to handle. `Obstacle` is frozen/static: nothing in this
codebase moves or spawns one at runtime (a "turret" that repositions
itself is a real, separate future extension, not this one).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    id: str
    x: float
    y: float
    radius: float


def obstacle_avoidance(x: float, y: float, obstacles, margin: float = 30.0, heading: tuple | None = None, tangent_weight: float = 0.8) -> tuple:
    """A raw (not speed-normalized -- callers blend this into their own
    steering and re-clamp) push vector, summed across every obstacle whose
    surface is within `margin` of (x, y). Each obstacle contributes both a
    radial component (straight away from its center, stronger the deeper
    the intrusion -- the same "proportional to how close/deep" shape
    `flocking.py`'s separation force already uses) AND a tangential one
    (perpendicular to the radial direction, i.e. along the obstacle's
    edge), so a drone can actually slide around an obstacle instead of
    just being pushed straight back from it.

    The tangential-component omission was a real bug, found live rather
    than by a failing unit test: a drone travelling in a straight line
    with an obstacle directly on its path would approach, feel the radial
    push straight back at it, and stall exactly where that push equals its
    own straight-line pull toward the target -- the classic local-minimum
    trap in potential-field navigation (same category of bug as
    `push_point_outside_all`'s two earlier versions, different
    manifestation: that one oscillated between two competing pulls, this
    one froze at their exact cancellation point). A pure radial force can
    never resolve that on its own, no matter how it's tuned, because
    "straight back" is the one direction that can exactly oppose "straight
    forward." Regression-tested by
    `test_drone_does_not_stall_head_on_into_an_obstacle` in test_swarm
    obstacle-integration tests, independently verified by reverting to a
    radial-only version and confirming that test fails (the drone gets
    permanently stuck) before restoring this fix.

    `heading` (the caller's already-intended direction, e.g. straight
    toward its mission zone) picks WHICH of the two perpendicular
    directions the tangential push uses, so the drone curves the way it
    was already trying to go rather than an arbitrary fixed side -- falls
    back to a fixed counter-clockwise rotation with no heading given."""
    push_x = push_y = 0.0
    for obstacle in obstacles:
        dx, dy = x - obstacle.x, y - obstacle.y
        dist = math.hypot(dx, dy)
        surface_dist = dist - obstacle.radius
        if surface_dist >= margin:
            continue
        depth = margin - surface_dist
        if dist > 1e-6:
            nx, ny = dx / dist, dy / dist
            tx, ty = -ny, nx  # one of the two perpendicular directions
            if heading is not None:
                hx, hy = heading
                cross = nx * hy - ny * hx
                if cross < 0:
                    tx, ty = ny, -nx  # the other perpendicular direction
            push_x += (nx + tx * tangent_weight) * depth
            push_y += (ny + ty * tangent_weight) * depth
        else:
            # Dead-center of the obstacle (degenerate, but must not divide
            # by zero) -- push in an arbitrary fixed direction rather than
            # leaving the drone stuck with a zero vector.
            push_x += depth
    return push_x, push_y


def point_inside_any(x: float, y: float, obstacles, extra_margin: float = 0.0) -> bool:
    """Whether (x, y) is inside (or within extra_margin of) any obstacle --
    used to keep generated content (patrol waypoints, a user's placed
    disturbance) from landing somewhere physically inaccessible."""
    return any(math.hypot(x - o.x, y - o.y) <= o.radius + extra_margin for o in obstacles)


def push_point_outside_all(x: float, y: float, obstacles, extra_margin: float = 10.0) -> tuple:
    """Finds the nearest clear point to (x, y) via an expanding-radius,
    all-angles search, rather than following the local repulsion gradient
    (`obstacle_avoidance`) step by step toward one.

    Two versions of this were tried and both had a real, independently
    confirmed bug before this one, worth naming since they're two
    different flavors of the same underlying mistake -- iterative local
    steering isn't a safe way to guarantee "find *a* point that's clear,"
    only to guarantee "keep nudging away from what's closest right now":
    (1) leaping straight away from whichever single obstacle was checked
    first could bounce forever between two deeply-overlapping obstacles
    (clear A's margin by leaping into B's, clear B's by leaping back into
    A's); (2) taking small fixed steps along the *summed* repulsion from
    all obstacles at once looked more principled, but for a
    near-symmetric overlap it settled into a stable 2-point oscillation
    straddling the force equilibrium -- summed repulsion finds where
    opposing pushes cancel, which is not the same point as "actually
    outside every obstacle." Both are real instances of the classic
    local-minimum trap in potential-field navigation. An expanding search
    sidesteps the whole category: it can't oscillate because it never
    follows a gradient, and it can't get trapped in a cancellation point
    because it tests candidates directly rather than integrating a force.
    Caught by `test_push_point_outside_all_clears_overlapping_obstacles`,
    which failed against both earlier versions and passes against this
    one (independently verified for each by reintroducing it and
    confirming the failure, then restoring this fix)."""
    if not point_inside_any(x, y, obstacles, extra_margin):
        return x, y
    step = (max((o.radius for o in obstacles), default=10.0) or 10.0) / 4.0
    angle_steps = 16
    for radius_step in range(1, 60):
        r = step * radius_step
        for a in range(angle_steps):
            angle = 2 * math.pi * a / angle_steps
            cx, cy = x + r * math.cos(angle), y + r * math.sin(angle)
            if not point_inside_any(cx, cy, obstacles, extra_margin):
                return cx, cy
    return x, y  # degenerate: no clear point found within a generous search radius
