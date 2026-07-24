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
from drone_swarm.mission import MissionConfig, Zone
from drone_swarm.simulation import create_random_swarm
from drone_swarm.swarm import SwarmConfig

FRONTEND_DIR = Path(__file__).parent / "frontend"
TICK_SECONDS = 0.4  # matches SwarmConfig.tick_dt_s default, so 1 real second ~= 1 simulated second

# Scale mode's resource-allocation mission -- see drone_swarm/mission.py.
# Sized for the 1400x900 scale-mode arena; three zones of varying threat
# so the heuristic allocator's threat-priority ordering is actually
# exercised, not just distance/battery.
SCALE_MISSION = MissionConfig(
    zones=(
        Zone(id="Z1", x=200, y=200, radius=80, required_drones=8, threat_level=1.0),
        Zone(id="Z2", x=1200, y=700, radius=80, required_drones=6, threat_level=2.0),
        Zone(id="Z3", x=700, y=450, radius=80, required_drones=10, threat_level=0.5),
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
    "scale": dict(n=100, width=1400, height=900, comm_range=180,
                  config=SwarmConfig(max_relay_hops=2, mission_config=SCALE_MISSION)),
    "security": dict(n=14, width=800, height=500, comm_range=180,
                      config=SwarmConfig(bft_mode=True, max_relay_hops=2)),
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
