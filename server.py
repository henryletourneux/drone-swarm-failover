"""Web server for the live drone-swarm demo.

Serves the visualization frontend and streams simulation state over a
WebSocket. One shared swarm simulation runs in the background and ticks
on a timer; every connected browser sees the same live state and can
click a drone to shoot it down.

Run with: uvicorn server:app --reload
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from drone_swarm.simulation import create_random_swarm

FRONTEND_DIR = Path(__file__).parent / "frontend"
TICK_SECONDS = 0.4  # matches SwarmConfig.tick_dt_s default, so 1 real second ~= 1 simulated second

swarm = create_random_swarm(speed=6.0, seed=None)
connections: set = set()


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
    global swarm
    msg_type = message.get("type")
    if msg_type == "kill":
        swarm.kill(message.get("id", ""))
    elif msg_type == "reset":
        swarm = create_random_swarm(speed=6.0, seed=None)
