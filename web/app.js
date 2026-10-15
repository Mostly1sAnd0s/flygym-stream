/* FlyGym Stream client (PLAN.md Phase 5).
 *
 * Wire protocol v1 is mirrored from src/flygym_stream/protocol.py:
 * ALL outgoing JSON is built through the single MSG map below (DRY);
 * incoming binary frames become MJPEG <img> swaps; text messages are
 * {"type":"status", ...} or {"type":"error", ...}.
 */
"use strict";

const PROTO = 1;

/* WebRTC transport mode (PLAN.md Phase 6): enable via http://host/?webrtc.
 * The server is the WebRTC offerer; frames arrive on <video> instead of
 * the MJPEG <img>. The JSON control channel is identical either way. */
const WEBCRTC = new URLSearchParams(location.search).has("webrtc");

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
  /* Phase 7 research-sandbox controls (M7.3/M7.4): */
  mode: (value) => JSON.stringify({ type: "mode", value }),
  loco: (fields) => JSON.stringify(Object.assign({ type: "loco" }, fields)),
  spawn: (pos) => JSON.stringify(
    { type: "spawn", pos, quat: [1.0, 0.0, 0.0, 0.0] }),
  force: (body, force, duration) => JSON.stringify(
    { type: "force", body, force, duration }),
  param: (name, value) => JSON.stringify({ type: "param", name, value }),
  log: (on) => JSON.stringify({ type: "log", on }),
  info: () => JSON.stringify({ type: "info" }),
  /* app-layer WebRTC signaling (not part of protocol.py) */
  webrtc_start: () => JSON.stringify({ type: "webrtc-start" }),
  webrtc_answer: (sdp) => JSON.stringify({ type: "webrtc-answer", sdp }),
};

/* --- state --------------------------------------------------------------- */
/* The orbit placeholder is only used until the first status message
 * carries the server's true orbit state (the scale-aware default for the
 * ~3 m full-size fly); after that the server is the source of truth until
 * the user drags/zooms. */
const state = {
  ws: null,
  rtc: null,
  orbit: { azimuth: 30, elevation: -20, distance: 6400 }, // deg/deg/mm
  orbitTouched: false,
  paused: false,
  nActuators: null,
  reconnectDelay: 1000,
  orbitThrottle: 0,
  lastFrameUrl: null,
  lastVisionUrl: null,
};

/* --- DOM ------------------------------------------------------------------ */
const $ = (id) => document.getElementById(id);
const video = $("video");
const videoWebrtc = $("video-webrtc");
const visionImg = $("vision");  // M7.6: fly-eye overlay (serve --vision)
const hud = $("hud");
const orbitState = $("orbit-state");
const slidersEl = $("sliders");

if (WEBCRTC) {  // WebRTC transport: swap <img> for <video>
  video.style.display = "none";
  videoWebrtc.style.display = "block";
}

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
    if (WEBCRTC) webrtcBegin();
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      onText(JSON.parse(ev.data));
    } else if (!WEBCRTC) {
      onBinary(ev.data);
    }
  };

  ws.onclose = () => {
    if (state.rtc) {
      state.rtc.close();
      state.rtc = null;
    }
    setHud("disconnected \u2014 reconnecting\u2026", "err");
    const delay = state.reconnectDelay;
    state.reconnectDelay = Math.min(delay * 2, 15000);
    setTimeout(connect, delay);
  };

  ws.onerror = () => ws.close();
}

/* --- WebRTC transport (server is the offerer) ------------------------------ */
function webrtcBegin() {
  state.rtc = new RTCPeerConnection();
  state.rtc.addTransceiver("video", { direction: "recvonly" });
  state.rtc.ontrack = (ev) => {
    videoWebrtc.srcObject = ev.streams[0];
  };
  send(MSG.webrtc_start());
}

async function onWebrtcOffer(sdp) {
  try {
    await state.rtc.setRemoteDescription({ type: "offer", sdp });
    const answer = await state.rtc.createAnswer();
    await state.rtc.setLocalDescription(answer);
    send(MSG.webrtc_answer(state.rtc.localDescription.sdp));
  } catch (err) {
    setHud(`webrtc failed: ${err}`, "err");
  }
}

function onText(msg) {
  if (msg.type === "status") {
    onStatus(msg);
  } else if (msg.type === "error") {
    setHud(`error: ${msg.message}`, "err");
  } else if (msg.type === "webrtc-offer") {
    onWebrtcOffer(msg.sdp);
  } else if (msg.type === "model-info" && window.Sandbox) {
    window.Sandbox.init(msg);  // M7.5: sandbox UI activates
  } else if (msg.type === "telemetry" && window.Sandbox) {
    window.Sandbox.onTelemetry(msg);
  }
  // unknown types: ignore (server keeps going)
}

function onStatus(s) {
  state.paused = s.paused;
  state.nActuators = s.n_actuators;
  if (s.orbit && !state.orbitTouched) {
    // adopt the server's true orbit (scale-aware default; a hardcoded JS
    // default would snap the view back inside the fly on first drag)
    state.orbit = { azimuth: s.orbit.azimuth, elevation: s.orbit.elevation,
                    distance: s.orbit.distance };
    updateOrbitHud();
  }
  setHud(
    `sim ${s.sim_time.toFixed(3)} s  \u00b7  ${s.fps.toFixed(1)} fps  \u00b7  ` +
      `${s.backend}  \u00b7  ${s.model}  ${s.paused ? "\u00b7  PAUSED" : ""}`
  );
  ensureSliders(s.n_actuators);
  if (window.Sandbox) window.Sandbox.onStatus(s);  // M7.5 mode highlight
}

function onFrame(blob) {
  const url = URL.createObjectURL(blob);
  if (state.lastFrameUrl) URL.revokeObjectURL(state.lastFrameUrl);
  state.lastFrameUrl = url;
  video.src = url;
}

async function onBinary(data) {
  // Binary frames (M7.6): V1-prefixed (0x56 0x31) = fly-eye vision frame
  // (serve --vision); anything else is the bare main MJPEG frame — a
  // real JPEG never starts with the bytes "V1".
  const head = new Uint8Array(await data.slice(0, 2).arrayBuffer());
  if (head.length === 2 && head[0] === 0x56 && head[1] === 0x31) {
    visionImg.classList.remove("hidden");
    onVisionFrame(new Blob([data.slice(2)], { type: "image/jpeg" }));
  } else {
    onFrame(data);
  }
}

function onVisionFrame(blob) {
  const url = URL.createObjectURL(blob);
  if (state.lastVisionUrl) URL.revokeObjectURL(state.lastVisionUrl);
  state.lastVisionUrl = url;
  visionImg.src = url;
}

function setHud(text, cls) {
  hud.textContent = text;
  hud.className = "hud" + (cls ? ` ${cls}` : "");
}

/* --- actuators ------------------------------------------------------------- */
function ensureSliders(n) {
  if (n === null || n === slidersEl.childElementCount) return;
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
const videoPane = $("video-pane");  // drag/zoom work over either <img> or <video>

videoPane.addEventListener("pointerdown", (ev) => {
  ev.preventDefault(); // stop the native <img> drag ghost + text selection
  dragging = true;
  state.orbitTouched = true;
  dragLast = [ev.clientX, ev.clientY];
  videoPane.setPointerCapture(ev.pointerId);
});

videoPane.addEventListener("pointermove", (ev) => {
  if (!dragging) return;
  const dx = ev.clientX - dragLast[0];
  const dy = ev.clientY - dragLast[1];
  dragLast = [ev.clientX, ev.clientY];
  state.orbit.azimuth -= dx * 0.4;
  // camera follows the mouse (MuJoCo viewer convention): drag up -> camera
  // rises (elevation up); note dy is positive for a downward mouse move
  state.orbit.elevation = clamp(state.orbit.elevation - dy * 0.4, -89, 89);
  sendOrbitThrottled();
});

videoPane.addEventListener("pointerup", (ev) => {
  dragging = false;
  try { videoPane.releasePointerCapture(ev.pointerId); } catch (_) { /* ok */ }
});

videoPane.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  state.orbitTouched = true;
  state.orbit.distance = clamp(
    state.orbit.distance * (1 + ev.deltaY * 0.001), 5, 50000
  );
  sendOrbitThrottled();
}, { passive: false });

function updateOrbitHud() {
  const d = state.orbit.distance;
  orbitState.textContent =
    `az ${Math.round(state.orbit.azimuth)}\u00b0 \u00b7 ` +
    `el ${Math.round(state.orbit.elevation)}\u00b0 \u00b7 ` +
    (d >= 1000 ? `${(d / 1000).toFixed(1)} m` : `${d.toFixed(0)} mm`);
}

function sendOrbitThrottled() {
  updateOrbitHud();
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
