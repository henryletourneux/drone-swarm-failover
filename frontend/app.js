"use strict";

// Fixed backend coordinate space (see drone_swarm/simulation.py).
const WORLD_W = 800;
const WORLD_H = 500;

const ROLE_STYLE = {
  nexus: { fill: "#ffcb2b", stroke: "#fff1b8", radius: 16, glow: "#ffcb2b" },
  relay: { fill: "#38bdf8", stroke: "#bae6fd", radius: 10, glow: "#38bdf8" },
  leaf: { fill: "#6ee7b7", stroke: "#d1fae5", radius: 8, glow: "#6ee7b7" },
  unassigned: { fill: "#46586e", stroke: "#5b6f88", radius: 7, glow: null },
};

const canvas = document.getElementById("swarm-canvas");
const ctx = canvas.getContext("2d");
const tickValue = document.getElementById("tick-value");
const aliveValue = document.getElementById("alive-value");
const logEl = document.getElementById("event-log");
const resetBtn = document.getElementById("reset-btn");
const connEl = document.getElementById("conn-indicator");
const connLabel = document.getElementById("conn-label");

let state = { tick: 0, drones: [], edges: [], event_log: [] };
let socket = null;
let hoverId = null;
let reconnectTimer = null;

// --- Layout / scaling -------------------------------------------------------

let scale = 1;
let offsetX = 0;
let offsetY = 0;

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pad = 30;
  const availW = rect.width - pad * 2;
  const availH = rect.height - pad * 2;
  scale = Math.min(availW / WORLD_W, availH / WORLD_H);
  offsetX = (rect.width - WORLD_W * scale) / 2;
  offsetY = (rect.height - WORLD_H * scale) / 2;
  draw();
}

function worldToScreen(x, y) {
  return { x: offsetX + x * scale, y: offsetY + y * scale };
}

// --- Drawing ----------------------------------------------------------------

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  const byId = {};
  for (const d of state.drones) byId[d.id] = d;

  // Edges
  ctx.lineWidth = 1.2;
  for (const [a, b] of state.edges) {
    const da = byId[a];
    const db = byId[b];
    if (!da || !db) continue;
    const pa = worldToScreen(da.x, da.y);
    const pb = worldToScreen(db.x, db.y);
    const nexusLink = da.role === "nexus" || db.role === "nexus";
    ctx.strokeStyle = nexusLink ? "rgba(255, 203, 43, 0.35)" : "rgba(56, 189, 248, 0.18)";
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }

  // Drones
  for (const d of state.drones) {
    drawDrone(d);
  }
}

function drawDrone(d) {
  const p = worldToScreen(d.x, d.y);
  const style = ROLE_STYLE[d.role] || ROLE_STYLE.unassigned;
  const r = style.radius;

  if (!d.alive) {
    // Faded circle with an X marker.
    ctx.globalAlpha = 0.55;
    ctx.fillStyle = "#1c2635";
    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "#ff6b5e";
    ctx.lineWidth = 1.8;
    const q = r * 0.55;
    ctx.beginPath();
    ctx.moveTo(p.x - q, p.y - q);
    ctx.lineTo(p.x + q, p.y + q);
    ctx.moveTo(p.x + q, p.y - q);
    ctx.lineTo(p.x - q, p.y + q);
    ctx.stroke();
    ctx.globalAlpha = 1;
    drawLabel(d, p, r, true);
    return;
  }

  // Static ring calling out the nexus (previously an animated pulse, which
  // needed a continuously-running timer for no real benefit, and whose
  // outer radius reached beyond the actual clickable hit-test area).
  if (d.role === "nexus") {
    ctx.strokeStyle = "rgba(255, 203, 43, 0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + 8, 0, Math.PI * 2);
    ctx.stroke();
  }

  // shadowBlur is expensive to run every frame for every drone, so only
  // the nexus (the one node worth calling out) gets it — everyone else
  // is already visually distinct by size/color alone.
  if (d.role === "nexus") {
    ctx.shadowColor = style.glow;
    ctx.shadowBlur = 16;
  }
  ctx.fillStyle = style.fill;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Hover highlight ring.
  if (d.id === hoverId) {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r + 3, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.stroke();

  drawLabel(d, p, r, false);
}

function drawLabel(d, p, r, dead) {
  ctx.font = "600 10px SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = dead ? "#5b6f88" : "#0a0e14";
  if (d.role === "nexus") {
    ctx.fillText(d.id, p.x, p.y);
  } else {
    ctx.fillStyle = dead ? "#5b6f88" : "#9fb4cc";
    ctx.fillText(d.id, p.x, p.y + r + 9);
  }
}

// --- Hit testing ------------------------------------------------------------

function droneAt(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;
  // Iterate in reverse so top-drawn drones win.
  for (let i = state.drones.length - 1; i >= 0; i--) {
    const d = state.drones[i];
    if (!d.alive) continue;
    const p = worldToScreen(d.x, d.y);
    // +10 padding (rather than the visual fill radius alone) so the
    // clickable area comfortably covers the nexus's outer ring too,
    // and gives every drone a slightly more forgiving click target.
    const r = (ROLE_STYLE[d.role] || ROLE_STYLE.leaf).radius + 10;
    const dx = mx - p.x;
    const dy = my - p.y;
    if (dx * dx + dy * dy <= r * r) return d;
  }
  return null;
}

canvas.addEventListener("mousemove", (e) => {
  const d = droneAt(e.clientX, e.clientY);
  const id = d ? d.id : null;
  if (id !== hoverId) {
    hoverId = id;
    canvas.style.cursor = id ? "pointer" : "default";
    draw();
  }
});

canvas.addEventListener("mouseleave", () => {
  if (hoverId !== null) {
    hoverId = null;
    canvas.style.cursor = "default";
    draw();
  }
});

canvas.addEventListener("click", (e) => {
  const d = droneAt(e.clientX, e.clientY);
  if (d) send({ type: "kill", id: d.id });
});

resetBtn.addEventListener("click", () => send({ type: "reset" }));

// --- Event log --------------------------------------------------------------

const EV_LABEL = {
  drone_down: "DOWN",
  election_started: "ELECT",
  election_won: "WON",
  swarms_merged: "MERGE",
};

const MAX_LOG_ROWS = 18;
const seenEventKeys = new Set();

function eventKey(ev) {
  return `${ev.tick}:${ev.type}:${ev.detail}`;
}

// Only ever inserts genuinely new events (newest at top, slide-in
// animation plays once for that row) and silently trims old ones off the
// bottom — previously this cleared and rebuilt all 18 rows on every
// single update, replaying the animation on unchanged rows too, which is
// what made it look like constant flashing.
function renderLog() {
  const events = state.event_log || [];
  const emptyPlaceholder = logEl.querySelector(".ev-empty");

  for (const ev of events) {
    const key = eventKey(ev);
    if (seenEventKeys.has(key)) continue;
    seenEventKeys.add(key);

    if (emptyPlaceholder) emptyPlaceholder.remove();

    const li = document.createElement("li");
    li.className = "ev--" + ev.type;
    const tag = EV_LABEL[ev.type] || ev.type;
    li.innerHTML =
      `<span class="ev-tick">t${ev.tick}</span>` +
      `<span class="ev-detail">[${tag}] ${escapeHtml(ev.detail)}</span>`;
    logEl.insertBefore(li, logEl.firstChild);
  }

  while (logEl.children.length > MAX_LOG_ROWS) {
    logEl.removeChild(logEl.lastChild);
  }

  if (logEl.children.length === 0) {
    const li = document.createElement("li");
    li.className = "ev-empty";
    li.textContent = "Swarm nominal — no events yet.";
    logEl.appendChild(li);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

// --- WebSocket --------------------------------------------------------------

function setConn(status, label) {
  connEl.className = "conn conn--" + status;
  connLabel.textContent = label;
}

function connect() {
  clearTimeout(reconnectTimer);
  setConn("connecting", "connecting");
  socket = new WebSocket(`ws://${location.host}/ws`);

  socket.addEventListener("open", () => setConn("connected", "live"));

  socket.addEventListener("message", (e) => {
    const previousTick = state.tick;
    try {
      state = JSON.parse(e.data);
    } catch (err) {
      return;
    }
    if (state.tick < previousTick) {
      // Tick count went backwards: the swarm was reset. Old event keys
      // are no longer valid (tick numbers restarted from 1), so drop
      // them and clear the log rather than risk skipping new events
      // that happen to collide with pre-reset ones.
      seenEventKeys.clear();
      logEl.innerHTML = "";
    }
    tickValue.textContent = state.tick;
    aliveValue.textContent = state.drones.filter((d) => d.alive).length;
    renderLog();
    draw();
  });

  socket.addEventListener("close", () => {
    setConn("disconnected", "disconnected");
    reconnectTimer = setTimeout(connect, 2000);
  });

  socket.addEventListener("error", () => {
    setConn("disconnected", "disconnected");
    socket.close();
  });
}

function send(obj) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(obj));
  }
}

// --- Boot -------------------------------------------------------------------

// No continuously-running render loop at all: the canvas only redraws in
// response to an actual event (a new WebSocket message, a resize, or a
// hover change), so there's zero JS work happening between real updates.
window.addEventListener("resize", resizeCanvas);
resizeCanvas();
connect();
