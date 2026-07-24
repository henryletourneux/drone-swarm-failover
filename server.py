"""Web server for the live drone-swarm demo.

Serves the visualization frontend and streams simulation state over a
WebSocket. One shared swarm simulation runs in the background and ticks
on a timer; every connected browser sees the same live state and can
click a drone to shoot it down.

Run with: uvicorn server:app --reload
"""
from __future__ import annotations

import asyncio
import random
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from antagonist.attacks import Antagonist
from drone_swarm.command import CommandConfig
from drone_swarm.flocking import FlockingConfig
from drone_swarm.mission import MissionConfig, Zone
from drone_swarm.patrol import PatrolConfig
from drone_swarm.simulation import create_random_swarm
from drone_swarm.swarm import SwarmConfig

FRONTEND_DIR = Path(__file__).parent / "frontend"
TICK_SECONDS = 0.4  # matches SwarmConfig.tick_dt_s default, so 1 real second ~= 1 simulated second

# Scale mode's resource-allocation mission -- see drone_swarm/mission.py.
# Sized for the 2000x1280 scale-mode arena (zoomed out from the original
# 1400x900 to give the swarm more room to actually move/flock/patrol in --
# coordinates below are the original zone layout scaled by the same
# ~1.43x the arena grew by, so they sit in the same relative spots);
# three zones of varying threat so the heuristic allocator's
# threat-priority ordering is actually exercised, not just distance/battery.
SCALE_MISSION = MissionConfig(
    zones=(
        Zone(id="Z1", x=280, y=280, radius=80, required_drones=8, threat_level=1.0),
        Zone(id="Z2", x=1700, y=1000, radius=80, required_drones=6, threat_level=2.0),
        Zone(id="Z3", x=1000, y=640, radius=80, required_drones=10, threat_level=0.5),
    ),
    reallocation_interval_ticks=10,
    # base_drain_per_tick's default (0.01) applies to EVERY drone
    # unconditionally forever, with no recharge mechanic -- bumping it
    # broadly was tried first and nearly broke the whole demo (verified
    # live: the entire 100-drone fleet drained to empty within ~5 real
    # minutes regardless of what any drone was doing, so zones never even
    # finished securing). secured_occupancy_drain_per_tick is a separate,
    # targeted knob that only speeds up drones actively holding an
    # already-secured zone, so idle/reserve/en-route drones keep the safe
    # default lifespan while a relief dispatch still becomes observable
    # within roughly 2-3 real minutes of a zone locking in "secured".
    secured_occupancy_drain_per_tick=0.17,
)

# Scale mode's mission only ever commits 24 of 100 drones to zones (8+6+10),
# so there's always a large, genuinely idle pool for patrol.py's dispatch to
# draw from (see patrol.py's module docstring for why "idle pool" and not
# "zone surplus" is the real dispatch source) -- not wired into Security
# mode, which has no mission_config at all, so patrol would just spawn
# disturbances nobody ever investigates.
SCALE_PATROL = PatrolConfig(
    spawn_interval_ticks=60,             # ~24 real seconds at TICK_SECONDS=0.4
    max_active_disturbances=2,
    investigation_range=35.0,
    investigation_ticks_required=20,     # ~8 real seconds once a drone arrives
    spawn_margin=60.0,
)

def _platoon_of(n: int, platoon_size: int) -> dict:
    """Static platoon assignment for hierarchical command (drone_swarm/
    command.py) -- drone ids from create_random_swarm are always D0..D(n-1)
    in order, so this can be computed from n alone before any Drone object
    exists, same as the mode's other config. Simple contiguous chunking,
    not derived from position (command.py's CommandConfig docstring notes
    that's a deliberately separate, bigger feature)."""
    return {f"D{i}": f"P{i // platoon_size}" for i in range(n)}


# Two distinct live-demo modes, not one compromise config. Profiling found a
# real tension: Ed25519 verification cost scales with message *volume*
# (roughly messages_delivered), which scales combinatorially with
# connectivity density x drone count. At 100 drones, keeping bft_mode fast
# required sparsifying the mesh down to ~1.6 average neighbors (many
# isolated pairs -- a fragmented, unimpressive mesh), while a genuinely
# richly-connected mesh (avg degree ~5) under bft_mode measured ~700ms/tick,
# far too slow for smooth interactivity at any reasonable cadence. Rather
# than fake it with a degraded mesh, each mode gets the config it's
# actually good at:
#   scale    -- 100 drones, plain mode, dense/rich connectivity, fast,
#               running the zone-coverage mission above.
#   security -- 14 drones, bft_mode on, the antagonist has real defenses to
#               demonstrate, already tuned to run comfortably.
MODE_SPECS = {
    # width/height zoomed out from the original 1400x900 -- see SCALE_MISSION's
    # comment above -- so 100 drones (plus flocking/patrolling ones) have real
    # room to move rather than the arena reading as crowded. comm_range bumped
    # alongside it (180 -> 210) to keep the mesh reasonably connected at the
    # larger scale; verified live that the swarm still reliably converges on a
    # single nexus rather than fragmenting into permanent islands.
    "scale": dict(n=100, width=2000, height=1280, comm_range=210,
                  config=SwarmConfig(max_relay_hops=2, mission_config=SCALE_MISSION,
                                      command_config=CommandConfig(platoon_of=_platoon_of(100, 10)),
                                      patrol_config=SCALE_PATROL,
                                      flocking_config=FlockingConfig())),
    "security": dict(n=14, width=800, height=500, comm_range=180,
                      config=SwarmConfig(bft_mode=True, max_relay_hops=2,
                                          command_config=CommandConfig(platoon_of=_platoon_of(14, 4)))),
}
DEFAULT_MODE = "scale"

ATTACK_CHOICES = (
    "impersonation",
    "garbage_signature",
    "priority_forgery",
    "term_inflation",
    "replay",
    "repurposed_certificate",
    "flood",
)


def _build_swarm(selected_mode: str):
    spec = MODE_SPECS[selected_mode]
    return create_random_swarm(
        n=spec["n"], width=spec["width"], height=spec["height"],
        comm_range=spec["comm_range"], speed=6.0, config=spec["config"], seed=None,
    )


mode = DEFAULT_MODE
swarm = _build_swarm(mode)
adversary = Antagonist(swarm)  # installs a bounded passive wiretap on swarm.mesh; harmless if bft_mode is off
connections: set = set()


async def simulation_loop() -> None:
    while True:
        await asyncio.sleep(TICK_SECONDS)
        swarm.tick()
        await _broadcast_state()


async def _broadcast_state() -> None:
    if not connections:
        return
    state = _state_payload()
    dead = set()
    for ws in connections:
        try:
            await ws.send_json(state)
        except Exception:
            dead.add(ws)
    connections.difference_update(dead)


def _state_payload() -> dict:
    state = swarm.to_state_dict()
    state["mode"] = mode
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(simulation_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.middleware("http")
async def no_cache(request, call_next):
    # This is a small local dev project — a stale cached app.js after an
    # edit is a much worse failure mode than the tiny perf cost of always
    # refetching a few KB of static files.
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connections.add(websocket)
    try:
        await websocket.send_json(_state_payload())
        while True:
            message = await websocket.receive_json()
            _handle_control_message(message)
    except WebSocketDisconnect:
        connections.discard(websocket)


def _handle_control_message(message: dict) -> None:
    global swarm, adversary, mode
    msg_type = message.get("type")
    if msg_type == "kill":
        swarm.kill(message.get("id", ""))
    elif msg_type == "reset":
        requested_mode = message.get("mode")
        if requested_mode in MODE_SPECS:
            mode = requested_mode
        swarm = _build_swarm(mode)
        adversary = Antagonist(swarm)
    elif msg_type == "attack":
        _launch_random_attack()
    elif msg_type == "add_disturbance":
        _add_user_disturbance(message.get("x"), message.get("y"))
    elif msg_type == "set_flocking":
        _update_flocking_params(message)
    elif msg_type == "set_patrol_route":
        _set_patrol_route_enabled(message.get("enabled"))


def _add_user_disturbance(x, y) -> None:
    """User-placed disturbance (clicking empty arena space in the
    frontend, see app.js). A no-op if patrol isn't active in the current
    mode (Security has no patrol_config -- see server.py's MODE_SPECS) or
    the click payload is malformed; the frontend already gates the click
    handler on `state.patrol` being present, this is defense-in-depth,
    same principle as _launch_random_attack's bft_mode check above."""
    if swarm.patrol is None:
        return
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return
    clamped_x = max(0.0, min(swarm.width, float(x)))
    clamped_y = max(0.0, min(swarm.height, float(y)))
    disturbance_id = swarm.patrol.add_disturbance(clamped_x, clamped_y, swarm.tick_count)
    swarm.event_log.append({
        "tick": swarm.tick_count,
        "type": "disturbance_spawned",
        "detail": f"disturbance {disturbance_id} placed by operator",
        "disturbance": disturbance_id,
    })


FLOCKING_PARAM_BOUNDS = {
    "separation": (0.0, 5.0, "separation_weight"),
    "alignment": (0.0, 5.0, "alignment_weight"),
    "cohesion": (0.0, 5.0, "cohesion_weight"),
    "speed": (0.5, 12.0, "max_speed"),
}


def _update_flocking_params(message: dict) -> None:
    """Live-tunes the running swarm's FlockingConfig in place (see
    flocking.py's module docstring for why that config is mutable, unlike
    every other one in this codebase) -- the whole point is a person
    watching the demo can feel the effect of a slider change immediately,
    without resetting the swarm and losing its current state. A no-op if
    flocking isn't active in the current mode, or a value is missing/not
    a number; each value is independently clamped to a sane range so a
    malformed payload can't push the simulation into instability."""
    fc = swarm.config.flocking_config
    if fc is None:
        return
    for key, (lo, hi, attr) in FLOCKING_PARAM_BOUNDS.items():
        value = message.get(key)
        if isinstance(value, (int, float)):
            setattr(fc, attr, max(lo, min(hi, float(value))))


def _set_patrol_route_enabled(enabled) -> None:
    if swarm.patrol is not None and isinstance(enabled, bool):
        swarm.patrol.route_enabled = enabled


def _current_nexus_id() -> str | None:
    for drone in swarm.drones.values():
        if drone.alive and drone.role == "nexus":
            return drone.id
    return None


def _launch_random_attack() -> None:
    """Picks a random attack from antagonist/attacks.py and throws it at the
    live swarm (the current nexus if one exists, else any alive drone), then
    logs it to the event log so it's visible in the demo. Whether it actually
    lands is left entirely to the swarm's real bft_mode defenses -- this
    function makes no assumption about the outcome.

    A no-op outside "security" mode: without bft_mode, verification always
    passes (see election.py), so a forged message would actually be
    ADOPTED rather than blocked -- attacking an undefended swarm doesn't
    demonstrate anything, it's just misleading. The frontend hides the
    attack control outside security mode too; this is the defense-in-depth
    backstop.
    """
    if not swarm.config.bft_mode:
        return
    alive = [d for d in swarm.drones.values() if d.alive]
    if not alive:
        return
    victim_id = _current_nexus_id() or alive[0].id
    term = max((e.term for e in swarm.elections.values()), default=0) + 1
    choice = random.choice(ATTACK_CHOICES)

    if choice == "impersonation":
        injection = adversary.impersonate_nexus(victim_id, term)
    elif choice == "garbage_signature":
        injection = adversary.inject_garbage_heartbeat(victim_id, term)
    elif choice == "priority_forgery":
        injection = adversary.forge_priority(victim_id, term, inflated_priority=999.0)
    elif choice == "term_inflation":
        injection = adversary.inflate_term(victim_id, term + 500)
    elif choice == "replay":
        injection = adversary.replay_captured_heartbeat()
    elif choice == "repurposed_certificate":
        injection = adversary.repurpose_certificate(victim_id, term + 500)
    else:
        injection = adversary.flood(victim_id, count=20)

    target_drone = victim_id
    if choice == "replay" and injection.messages:
        target_drone = injection.messages[0].sender_id

    swarm.event_log.append({
        "tick": swarm.tick_count,
        "type": "attack",
        "detail": f"ANTAGONIST: {injection.description}",
        "attack": injection.name,
        "drone": target_drone,
    })
