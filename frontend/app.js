"use strict";

// Backend coordinate space — differs by demo mode (100-drone "scale" mode
// uses a larger arena than the 14-drone "security" mode), so these are read
// from the first field of each state message ("world": {width, height})
// rather than hardcoded. Defaults here only matter before the first message.
let WORLD_W = 800;
let WORLD_H = 500;

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
const attackBtn = document.getElementById("attack-btn");
const modeScaleBtn = document.getElementById("mode-scale-btn");
const modeSecurityBtn = document.getElementById("mode-security-btn");
const connEl = document.getElementById("conn-indicator");
const connLabel = document.getElementById("conn-label");

const stat = {
  recoveryMean: document.getElementById("stat-recovery-mean"),
  recoveryP95: document.getElementById("stat-recovery-p95"),
  recoveryN: document.getElementById("stat-recovery-n"),
  electWon: document.getElementById("stat-elect-won"),
  electStarted: document.getElementById("stat-elect-started"),
  merges: document.getElementById("stat-merges"),
  msgSent: document.getElementById("stat-msg-sent"),
  msgDelivered: document.getElementById("stat-msg-delivered"),
  msgDropped: document.getElementById("stat-msg-dropped"),
  security: document.getElementById("stat-security"),
  securitySub: document.getElementById("stat-security-sub"),
  securityTile: document.getElementById("stat-security-tile"),
};

let state = { tick: 0, drones: [], edges: [], event_log: [] };
let socket = null;
let hoverId = null;
let reconnectTimer = null;

// An antagonist attack plays out on the canvas itself in two stages: a
// glowing projectile flies in from off-screen toward the target drone
// (the forged packet arriving), then a bigger impact flash + a brief
// full-canvas red vignette mark the moment it's evaluated (and rejected).
// Purely cosmetic client state — the actual outcome is already decided
// server-side by the time any of this plays.
const ATTACK_PROJECTILE_MS = 550;
const ATTACK_FLASH_MS = 1300;
let activeAttackProjectile = null; // {droneId, startTime, angle}
let activeAttackFlash = null; // {droneId, startTime}

// --- Position interpolation --------------------------------------------------
//
// The server only broadcasts a new position ~2.5 times/sec (TICK_SECONDS in
// server.py). Drawing raw positions straight from each message makes motion
// look like a slideshow rather than smooth flight, no matter how fast the
// canvas itself can render. So drawing reads from a *continuously* advancing
// interpolation between the last two known positions instead of the raw
// server snapshot directly — the standard fix for smooth motion over a
// low-frequency network feed.
const UPDATE_INTERVAL_MS = 400; // should match server.py's TICK_SECONDS
let prevPositions = new Map(); // id -> {x, y}, interpolating FROM
let nextPositions = new Map(); // id -> {x, y}, interpolating TO
let updateStartTime = performance.now();

function interpolatedDrones() {
  const t = Math.min(1, (performance.now() - updateStartTime) / UPDATE_INTERVAL_MS);
  return state.drones.map((d) => {
    const from = prevPositions.get(d.id) || { x: d.x, y: d.y };
    const to = nextPositions.get(d.id) || { x: d.x, y: d.y };
    return { ...d, x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
  });
}

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

  const drones = interpolatedDrones();
  const byId = {};
  for (const d of drones) byId[d.id] = d;

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
  for (const d of drones) {
    drawDrone(d);
  }

  // Antagonist attack: an incoming projectile (the forged packet arriving),
  // then a bigger impact + vignette pulse the instant it lands, so the
  // whole thing plays out visibly in the arena, not just as a log entry.
  if (activeAttackProjectile) {
    const elapsed = performance.now() - activeAttackProjectile.startTime;
    if (elapsed > ATTACK_PROJECTILE_MS) {
      activeAttackFlash = { droneId: activeAttackProjectile.droneId, startTime: performance.now() };
      activeAttackProjectile = null;
    } else {
      const target = byId[activeAttackProjectile.droneId];
      if (target) {
        drawAttackProjectile(rect, target, elapsed / ATTACK_PROJECTILE_MS, activeAttackProjectile.angle);
      } else {
        activeAttackProjectile = null;
      }
    }
  }

  if (activeAttackFlash) {
    const elapsed = performance.now() - activeAttackFlash.startTime;
    if (elapsed > ATTACK_FLASH_MS) {
      activeAttackFlash = null;
    } else {
      const t = elapsed / ATTACK_FLASH_MS;
      const target = byId[activeAttackFlash.droneId];
      if (elapsed < 220) drawAttackVignette(rect, 1 - elapsed / 220);
      if (target) drawAttackFlash(target, t);
    }
  }
}

function drawAttackVignette(rect, alpha) {
  // A brief red glow around the edges of the whole canvas at the moment
  // of impact -- visible even if you weren't looking straight at the
  // target drone when the projectile landed.
  const grad = ctx.createRadialGradient(
    rect.width / 2, rect.height / 2, Math.min(rect.width, rect.height) * 0.25,
    rect.width / 2, rect.height / 2, Math.max(rect.width, rect.height) * 0.7
  );
  grad.addColorStop(0, "rgba(255, 59, 59, 0)");
  grad.addColorStop(1, `rgba(255, 59, 59, ${alpha * 0.35})`);
  ctx.save();
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.restore();
}

function drawAttackProjectile(rect, d, t, angle) {
  const targetP = worldToScreen(d.x, d.y);
  const launchDist = Math.max(rect.width, rect.height) * 0.9;
  const fromX = targetP.x + Math.cos(angle) * launchDist;
  const fromY = targetP.y + Math.sin(angle) * launchDist;
  // Ease in, so it starts fast (far away) and visibly decelerates into the hit.
  const eased = 1 - (1 - t) * (1 - t);
  const curX = fromX + (targetP.x - fromX) * eased;
  const curY = fromY + (targetP.y - fromY) * eased;
  const trailX = fromX + (targetP.x - fromX) * Math.max(0, eased - 0.12);
  const trailY = fromY + (targetP.y - fromY) * Math.max(0, eased - 0.12);

  ctx.save();
  const trail = ctx.createLinearGradient(trailX, trailY, curX, curY);
  trail.addColorStop(0, "rgba(255, 59, 59, 0)");
  trail.addColorStop(1, "rgba(255, 90, 80, 0.9)");
  ctx.strokeStyle = trail;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(trailX, trailY);
  ctx.lineTo(curX, curY);
  ctx.stroke();

  ctx.shadowColor = "#ff3b3b";
  ctx.shadowBlur = 14;
  ctx.fillStyle = "#ff5a50";
  ctx.beginPath();
  ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawAttackFlash(d, t) {
  const p = worldToScreen(d.x, d.y);
  const alpha = 1 - t;
  const radius = 18 + t * 55;

  ctx.save();
  ctx.strokeStyle = `rgba(255, 59, 59, ${alpha * 0.9})`;
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.stroke();

  ctx.strokeStyle = `rgba(255, 140, 130, ${alpha * 0.65})`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius * 0.6, 0, Math.PI * 2);
  ctx.stroke();

  ctx.globalAlpha = alpha;
  ctx.font = "18px system-ui, -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("☠", p.x, p.y - 24 - t * 12);
  ctx.restore();
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
  // Use the same interpolated positions draw() just rendered, so hit
  // testing always matches what's actually on screen right now.
  const drones = interpolatedDrones();
  // Iterate in reverse so top-drawn drones win.
  for (let i = drones.length - 1; i >= 0; i--) {
    const d = drones[i];
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
attackBtn.addEventListener("click", () => send({ type: "attack" }));
modeScaleBtn.addEventListener("click", () => send({ type: "reset", mode: "scale" }));
modeSecurityBtn.addEventListener("click", () => send({ type: "reset", mode: "security" }));

// Reflects the active demo mode: highlights the current mode button, and
// hides the attack control outside "security" mode -- without bft_mode,
// a forged message would actually be ADOPTED rather than blocked (see
// server.py's _launch_random_attack), so showing "Launch Attack" there
// would be actively misleading, not just inert.
let lastRenderedMode = null;
function renderModeUI() {
  if (state.mode === lastRenderedMode) return;
  lastRenderedMode = state.mode;
  modeScaleBtn.classList.toggle("mode-btn--active", state.mode === "scale");
  modeSecurityBtn.classList.toggle("mode-btn--active", state.mode === "security");
  attackBtn.style.display = state.mode === "security" ? "" : "none";
}

// --- Event log --------------------------------------------------------------

const EV_LABEL = {
  drone_down: "DOWN",
  election_started: "ELECT",
  election_won: "WON",
  swarms_merged: "MERGE",
  attack: "ATTACK",
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

    if (ev.type === "attack" && ev.drone) {
      activeAttackProjectile = { droneId: ev.drone, startTime: performance.now(), angle: Math.random() * Math.PI * 2 };
    }

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

// --- Metrics ----------------------------------------------------------------

const DASH = "—"; // em dash, for null/absent values

function fmtInt(n) {
  return typeof n === "number" ? n.toLocaleString("en-US") : "0";
}

function fmtSeconds(s) {
  return typeof s === "number" ? s.toFixed(2) + "s" : DASH;
}

function renderMetrics() {
  const m = state.metrics;
  if (!m) return;

  const rec = m.recovery || {};
  stat.recoveryMean.textContent = fmtSeconds(rec.mean_s);
  stat.recoveryP95.textContent = fmtSeconds(rec.p95_s);
  stat.recoveryN.textContent = "n=" + (rec.count || 0);

  stat.electWon.textContent = fmtInt(m.elections_won);
  stat.electStarted.textContent = fmtInt(m.elections_started);
  stat.merges.textContent = fmtInt(m.merges);

  stat.msgSent.textContent = fmtInt(m.messages_sent);
  stat.msgDelivered.textContent = fmtInt(m.messages_delivered);
  stat.msgDropped.textContent = fmtInt(m.messages_dropped_loss);

  const rej = m.security_rejections || 0;
  stat.security.textContent = fmtInt(rej);
  const attacked = rej > 0;
  stat.securityTile.classList.toggle("is-alert", attacked);
  stat.securitySub.textContent = attacked ? "rejected · under attack" : "no BFT activity";
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

    if (state.world && (state.world.width !== WORLD_W || state.world.height !== WORLD_H)) {
      // Different demo mode -> different arena size. Recompute the
      // canvas scale/offset for the new world immediately.
      WORLD_W = state.world.width;
      WORLD_H = state.world.height;
      resizeCanvas();
    }

    renderModeUI();

    // Wherever the interpolation currently is (not necessarily fully
    // caught up yet) becomes the new starting point, so a message that
    // arrives a little early or late never causes a visible jump.
    prevPositions = new Map(interpolatedDrones().map((d) => [d.id, { x: d.x, y: d.y }]));
    nextPositions = new Map(state.drones.map((d) => [d.id, { x: d.x, y: d.y }]));
    updateStartTime = performance.now();

    tickValue.textContent = state.tick;
    aliveValue.textContent = state.drones.filter((d) => d.alive).length;
    renderLog();
    renderMetrics();
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

// A continuously-running requestAnimationFrame loop, so drone motion reads
// as smooth flight instead of a ~2.5fps slideshow (see the interpolation
// section up top). This is safe to run every frame now — the earlier
// perf work already stripped out the actually-expensive parts (shadowBlur
// on every drone, backdrop-filter), and the isolated Canvas test proved
// this machine renders far more circles than this at a real 60fps with
// no dependency on our app code at all.
function renderLoop() {
  draw();
  requestAnimationFrame(renderLoop);
}
requestAnimationFrame(renderLoop);

// Deliberately NOT using window.addEventListener("resize", ...) here: the
// canvas's actual on-screen size is set by CSS (100% of its container),
// which can change for reasons that never fire a window resize event at
// all — a scrollbar toggling, any layout shift elsewhere on the page. When
// that happens, the canvas's drawing buffer silently goes stale relative
// to its real displayed size, and draw positions stop agreeing with click
// positions (worse the further a drone is from wherever the drift happens
// to be zero — which is exactly the "clicking is a bit off, and gets worse
// over time" symptom this was causing). ResizeObserver watches the actual
// element and fires on any real size change, for any reason, closing that
// gap entirely instead of only reacting to one specific cause of it.
const resizeObserver = new ResizeObserver(() => resizeCanvas());
resizeObserver.observe(canvas.parentElement);
resizeCanvas();
connect();
