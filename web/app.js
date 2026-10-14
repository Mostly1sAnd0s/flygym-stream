/* FlyGym Stream client (PLAN.md Phase 5).
 *
 * Wire protocol v1 is mirrored from src/flygym_stream/protocol.py:
 * ALL outgoing JSON is built through the single MSG map below (DRY);
 * incoming binary frames become MJPEG <img> swaps; text messages are
 * {"type":"status", ...} or {"type":"error", ...}.
 */
"use strict";

const PROTO = 1;

/* --- single source of truth for outgoing messages (protocol.py) --------- */
const MSG = {
  hello: (proto) => JSON.stringify({ type: "hello", proto }),
  orbit: (azimuth, elevation, distance) =>
    JSON.stringify({ type: "orbit", azimuth, elevation, distance }),
  actuator: (index, value) =>
    JSON.stringify({ type: "actuator", index, value }),
  actuators: (values) => JSON.stringify({ type: "actuators", values }),
  neutral: () => JSON.stringify({ type: "neutral" }),
  pause: () => JSON.stringify({ type: "pause" }),
  resume: () => JSON.stringify({ type: "resume" }),
  step: (n) => JSON.stringify({ type: "step", n }),
  speed: (value) => JSON.stringify({ type: "speed", value }),
  reset: () => JSON.stringify({ type: "reset" }),
};

/* --- state --------------------------------------------------------------- */
const state = {
  ws: null,
  orbit: { azimuth: 30, elevation: -20, distance: 40 }, // deg/deg/mm
  paused: false,
  nActuators: null,
  reconnectDelay: 1000,
  orbitThrottle: 0,
  lastFrameUrl: null,
};

/* --- DOM ------------------------------------------------------------------ */
const $ = (id) => document.getElementById(id);
const video = $("video");
const hud = $("hud");
const orbitState = $("orbit-state");
const slidersEl = $("sliders");

/* --- websocket ------------------------------------------------------------ */
function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws`;
}

function connect() {
  const ws = new WebSocket(wsUrl());
  state.ws = ws;
  setHud("connecting\u2026", "warn");

  ws.onopen = () => {
    state.reconnectDelay = 1000;
    ws.send(MSG.hello(PROTO));
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      onText(JSON.parse(ev.data));
    } else {
      onFrame(ev.data);
    }
  };

  ws.onclose = () => {
    setHud("disconnected \u2014 reconnecting\u2026", "err");
    const delay = state.reconnectDelay;
    state.reconnectDelay = Math.min(delay * 2, 15000);
    setTimeout(connect, delay);
  };

  ws.onerror = () => ws.close();
}

function onText(msg) {
  if (msg.type === "status") {
    onStatus(msg);
  } else if (msg.type === "error") {
    setHud(`error: ${msg.message}`, "err");
  }
  // unknown types: ignore (server keeps going)
}

function onStatus(s) {
  state.paused = s.paused;
  state.nActuators = s.n_actuators;
  setHud(
    `sim ${s.sim_time.toFixed(3)} s  \u00b7  ${s.fps.toFixed(1)} fps  \u00b7  ` +
      `${s.backend}  \u00b7  ${s.model}  ${s.paused ? "\u00b7  PAUSED" : ""}`
  );
  ensureSliders(s.n_actuators);
}

function onFrame(blob) {
  const url = URL.createObjectURL(blob);
  if (state.lastFrameUrl) URL.revokeObjectURL(state.lastFrameUrl);
  state.lastFrameUrl = url;
  video.src = url;
}

function setHud(text, cls) {
  hud.textContent = text;
  hud.className = "hud" + (cls ? ` ${cls}` : "");
}

/* --- actuators ------------------------------------------------------------- */
function ensureSliders(n) {
  if (n === null || n === state.slidersEl.childElementCount) return;
  slidersEl.replaceChildren();
  for (let i = 0; i < n; i++) {
    const row = document.createElement("div");
    row.className = "slider";

    const idx = document.createElement("span");
    idx.className = "idx";
    idx.textContent = i;

    const input = document.createElement("input");
    input.type = "range";
    input.min = "-3.14";
    input.max = "3.14";
    input.step = "0.01";
    input.value = "0";

    const val = document.createElement("span");
    val.className = "val";
    val.textContent = "0.00";

    input.addEventListener("input", () => {
      val.textContent = Number(input.value).toFixed(2);
      send(MSG.actuator(i, Number(input.value)));
    });

    row.append(idx, input, val);
    slidersEl.append(row);
  }
}

/* --- orbit camera ---------------------------------------------------------- */
let dragging = false;
let dragLast = null;

video.addEventListener("pointerdown", (ev) => {
  dragging = true;
  dragLast = [ev.clientX, ev.clientY];
  video.setPointerCapture(ev.pointerId);
});

video.addEventListener("pointermove", (ev) => {
  if (!dragging) return;
  const dx = ev.clientX - dragLast[0];
  const dy = ev.clientY - dragLast[1];
  dragLast = [ev.clientX, ev.clientY];
  state.orbit.azimuth -= dx * 0.4;
  state.orbit.elevation = clamp(state.orbit.elevation + dy * 0.4, -89, 89);
  sendOrbitThrottled();
});

video.addEventListener("pointerup", (ev) => {
  dragging = false;
  video.releasePointerCapture(ev.pointerId);
});

video.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  state.orbit.distance = clamp(
    state.orbit.distance * (1 + ev.deltaY * 0.001), 5, 500
  );
  sendOrbitThrottled();
}, { passive: false });

function sendOrbitThrottled() {
  orbitState.textContent =
    `az ${Math.round(state.orbit.azimuth)}\u00b0 \u00b7 ` +
    `el ${Math.round(state.orbit.elevation)}\u00b0 \u00b7 ` +
    `${state.orbit.distance.toFixed(0)} mm`;
  const now = performance.now();
  if (now - state.orbitThrottle < 33) return; // ~30 Hz cap
  state.orbitThrottle = now;
  send(MSG.orbit(state.orbit.azimuth, state.orbit.elevation, state.orbit.distance));
}

/* --- controls ---------------------------------------------------------------- */
function wire(id, handler) {
  $(id).addEventListener("click", handler);
}

wire("btn-neutral", () => send(MSG.neutral()));
wire("btn-pause", () => send(MSG.pause()));
wire("btn-resume", () => send(MSG.resume()));
wire("btn-step", () => send(MSG.step(Number($("step-n").value))));
wire("btn-reset", () => send(MSG.reset()));

$("speed").addEventListener("input", (ev) => {
  $("speed-val").textContent = `${Number(ev.target.value).toFixed(1)}\u00d7`;
  send(MSG.speed(Number(ev.target.value)));
});

/* --- helpers ----------------------------------------------------------------- */
function send(text) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(text);
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

connect();
