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
from drone_swarm.simulation import create_random_swarm
from drone_swarm.swarm import SwarmConfig

FRONTEND_DIR = Path(__file__).parent / "frontend"
TICK_SECONDS = 0.4  # matches SwarmConfig.tick_dt_s default, so 1 real second ~= 1 simulated second

# bft_mode is on for the live demo specifically so the antagonist has real
# defenses to demonstrate -- with it off, "Launch Attack" would have nothing
# to visibly block. max_relay_hops is lowered from the default of 4: in a
# 14-drone swarm this size, almost everyone is already directly reachable,
# so the extra hops were mostly redundant re-verification of the same
# messages -- measured at ~113x more per-tick cost than plain mode (up to
# ~230ms/tick, enough to peg a full CPU core against the 400ms tick budget).
# hops=2 keeps real multi-hop relay behavior exercised while cutting that to
# a small fraction of the tick budget even during heavy election activity.
LIVE_CONFIG = SwarmConfig(bft_mode=True, max_relay_hops=2)

swarm = create_random_swarm(speed=6.0, config=LIVE_CONFIG, seed=None)
adversary = Antagonist(swarm)  # installs a bounded passive wiretap on swarm.mesh
connections: set = set()

ATTACK_CHOICES = (
    "impersonation",
    "garbage_signature",
    "priority_forgery",
    "term_inflation",
    "replay",
    "repurposed_certificate",
    "flood",
)


async def simulation_loop() -> None:
    while True:
        await asyncio.sleep(TICK_SECONDS)
        swarm.tick()
        await _broadcast_state()


async def _broadcast_state() -> None:
    if not connections:
        return
    state = swarm.to_state_dict()
    dead = set()
    for ws in connections:
        try:
            await ws.send_json(state)
        except Exception:
            dead.add(ws)
    connections.difference_update(dead)


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
        await websocket.send_json(swarm.to_state_dict())
        while True:
            message = await websocket.receive_json()
            _handle_control_message(message)
    except WebSocketDisconnect:
        connections.discard(websocket)


def _handle_control_message(message: dict) -> None:
    global swarm, adversary
    msg_type = message.get("type")
    if msg_type == "kill":
        swarm.kill(message.get("id", ""))
    elif msg_type == "reset":
        swarm = create_random_swarm(speed=6.0, config=LIVE_CONFIG, seed=None)
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
    function makes no assumption about the outcome."""
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
