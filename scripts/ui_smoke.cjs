#!/usr/bin/env node
/* Headless smoke test for the web UI (PLAN.md M7.5) — no browser needed.
 *
 * Loads web/app.js + web/app_sandbox.js into a fake-DOM context and drives
 * the sandbox: model-info init -> virtualized DOF table, telemetry
 * resync, coalesced actuators send, mode/spawn/param/loco message shapes.
 * Run:  node scripts/ui_smoke.mjs   (exits non-zero on any failure)
 */
"use strict";

const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const appJs = fs.readFileSync(path.join(root, "web", "app.js"), "utf8");
const sandboxJs = fs.readFileSync(path.join(root, "web", "app_sandbox.js"), "utf8");
const indexHtml = fs.readFileSync(path.join(root, "web", "index.html"), "utf8");
const styleCss = fs.readFileSync(path.join(root, "web", "style.css"), "utf8");

/* --- minimal fake DOM ------------------------------------------------------ */
class El {
  constructor(id) {
    this.id = id;
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.listeners = {};
    this.textContent = "";
    this.value = "0";
    this.title = "";
    this.type = "";
    this.min = this.max = this.step = "";
    const self = this;
    this.classList = {
      _s: new Set(),
      add(c) { self._cls().add(c); },
      remove(c) { self._cls().delete(c); },
      contains(c) { return self._cls().has(c); },
      toggle(c, force) {
        const has = force !== undefined ? force : !self._cls().has(c);
        has ? self._cls().add(c) : self._cls().delete(c);
      },
    };
  }
  _cls() { return this.classList._s; }
  addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); }
  append(...els) { this.children.push(...els); }
  replaceChildren(...els) { this.children = els; }
  fire(t, ev) {
    ev = Object.assign({ preventDefault() {}, stopPropagation() {} }, ev);
    for (const fn of this.listeners[t] || []) fn(ev);
  }
  set clientHeight(v) { this._ch = v; }
  get clientHeight() { return this._ch !== undefined ? this._ch : 280; }
  set scrollTop(v) { this._st = v; }
  get scrollTop() { return this._st || 0; }
  setPointerCapture() {}  // no-ops in the fake DOM
  releasePointerCapture() {}
}

const els = new Map();
let anon = 0;
function el(id) {
  if (!els.has(id)) els.set(id, new El(id));
  return els.get(id);
}

global.document = {
  getElementById: (id) => el(id),
  createElement: (tag) => {
    const id = `anon-${anon++}`;
    const e = new El(id);
    els.set(id, e);
    return e;
  },
  querySelectorAll: (sel) => {
    const cls = sel.replace(".", "");
    return [...els.values()].filter((e) => e.classList.contains(cls));
  },
};
global.location = { search: "", host: "localhost", protocol: "http:" };
global.__sent = [];
global.WebSocket = class {
  static get OPEN() { return 1; }
  constructor() { this.readyState = 1; }
  send(t) { global.__sent.push(JSON.parse(t)); }
  close() {}
};
let rafQueue = [];
global.requestAnimationFrame = (fn) => { rafQueue.push(fn); return rafQueue.length; };
function flushRaf() { const q = rafQueue; rafQueue = []; for (const fn of q) fn(); }

/* Blob/URL fakes for the MJPEG <img> swap path (M7.6 onBinary checks). */
let blobCounter = 0;
global.Blob = class {
  constructor(parts, opts) { this.parts = parts; this.type = (opts || {}).type; }
};
global.URL = {
  createObjectURL: () => `blob:fake-${++blobCounter}`,
  revokeObjectURL: () => {},
};

/* The classic <script> files share the page's global object. In the
   browser a top-level `const Sandbox` is NOT a window property, so
   app.js's `window.Sandbox` activation gate must be tested against the
   real contract: here `window` IS the global scope. */
global.window = globalThis;

/* --- model-info fixture (shape of telemetry.gather_model_info) -------------- */
const N = 132;
const modelInfo = {
  type: "model-info", units: "mm", fly_name: "nmf", world: "flat",
  nq: 133, nv: 132, nu: N, timestep: 0.0001, gravity: [0, 0, -9810],
  joints: Array.from({ length: 127 }, (_, i) =>
    ({ id: i, name: `j${i}`, type: "hinge", qadr: i === 0 ? 0 : i })),
  actuators: Array.from({ length: N }, (_, i) =>
    ({ id: i, name: `nmf/act-${i}-position`, gaintype: 0, biastype: 1,
       dyntype: 0, joint: i % 127, tendon: null, range: [-3.14, 3.14] })),
  sites: [], cameras: [{ id: 0, name: "nmf/trackingcam" }],
  keyframes: ["neutral"],
  sensors: ["lf", "lm", "lh", "rf", "rm", "rh"].map((l, i) =>
    ({ id: i, name: `ground_contact_${l}_leg`, type: "force" })),
};

let failures = 0;
function check(cond, what) {
  if (cond) { console.log(`  ok  ${what}`); }
  else { failures += 1; console.error(` FAIL ${what}`); }
}

/* Static markup: the mode row is the SINGLE neutral (the duplicate
   top-row #btn-neutral button was the 2026-07-21 user bug). */
check(!indexHtml.includes('btn-neutral'),
      "index.html has no duplicate top-row neutral button (mode row is the single neutral)");

/* M7.7 static markup: the 6-face viewcube + the camera mode row. */
check(["front", "back", "left", "right", "top", "bottom"].every(
        (v) => indexHtml.includes(`data-view="${v}"`))
      && indexHtml.includes('id="cam-btn-first"')
      && indexHtml.includes('id="cube"'),
      "index.html has the 6-face viewcube + the orbit/1st/3rd camera row");

/* --- load both UI files in one scope, then run the harness ------------------- */
const harness = `
const __harness = (async function () {
  const $ = (id) => document.getElementById(id);
  /* mode buttons need the class before Sandbox wires them */
  ["neutral", "walk", "flap", "manual"].forEach((m) => {
    const b = document.createElement("button");
    b.classList.add("mode-btn");
    b.dataset.mode = m;
    b._mode = m;
  });

  /* Classic slider list: the first status builds n_actuators sliders
     (ensureSliders in app.js) — the first UI a user sees. */
  let statusErr = null;
  try {
    onStatus({ type: "status", sim_time: 1.0, fps: 30.0, paused: false,
               n_actuators: 42, backend: "cpu", model: "fly",
               orbit: { azimuth: 30, elevation: -20, distance: 6400 },
               mode: "neutral", n_dof: 133 });
  } catch (err) { statusErr = err; }
  check(statusErr === null,
        "onStatus/ensureSliders does not throw" +
        (statusErr ? \` (got \${statusErr.message})\` : ""));
  check(els.get("sliders").children.length === 42,
        \`status builds the 42 classic sliders (got \${els.get("sliders").children.length})\`);

  /* --- M7.7 camera: mode row + viewcube ----------------------------------- */
  /* The cube must be CONGRUENT with the world (not a mirror): with the
     CSS screen frame left-handed, the left/right face transforms are
     swapped vs the standard CSS cube and the azimuth sign is flipped,
     else the cube rotates opposite to the fly (v4 fix). */
  check(/#vc-left\\s*\\{[^}]*rotateY\\(90deg\\)/.test(styleCss)
        && /#vc-right\\s*\\{[^}]*rotateY\\(-90deg\\)/.test(styleCss),
        "style.css viewcube faces are the world-congruent (unmirrored) layout");
  check(els.get("cube").style.transform
        === "rotateX(20deg) rotateY(-30deg)",
        "viewcube tracks the camera (az 30, el -20, azimuth sign flipped); got "
        + els.get("cube").style.transform);
  check(els.get("orbit-state").textContent.indexOf("cam orbit") === 0,
        "orbit HUD reports the camera mode");
  els.get("cam-btn-first").fire("click");
  check(JSON.stringify(__sent.filter((m) => m.type === "camera").pop())
        === JSON.stringify({ type: "camera", mode: "first" }),
        "cam 1st button -> camera mode message shape");
  els.get("vc-back").fire("click");
  check(JSON.stringify(__sent.filter((m) => m.type === "camera").pop())
        === JSON.stringify({ type: "camera", view: "back" }),
        "viewcube back face -> camera view message shape");
  onStatus({ type: "status", sim_time: 1.1, fps: 30.0, paused: false,
             n_actuators: 42, backend: "cpu", model: "fly",
             orbit: { azimuth: 30, elevation: -20, distance: 6400 },
             mode: "neutral", n_dof: 133, cam_mode: "third" });
  check(els.get("cam-btn-third").classList.contains("active"),
        "cam_mode third highlights the 3rd button");
  check(!els.get("cam-btn-orbit").classList.contains("active"),
        "orbit button not highlighted in third mode");
  // Dragging the video in a body view switches back to orbit first.
  els.get("video-pane").fire("pointerdown",
    { preventDefault() {}, clientX: 100, clientY: 100, pointerId: 1 });
  check(JSON.stringify(__sent.filter((m) => m.type === "camera").pop())
        === JSON.stringify({ type: "camera", mode: "orbit" }),
        "video drag in a body view sends camera mode orbit first");
  els.get("video-pane").fire("pointerup", { pointerId: 1 });

  /* The sandbox activates through app.js's window.Sandbox gate: a
     top-level const in a classic <script> is NOT a window property. */
  check(window.Sandbox && typeof window.Sandbox.init === "function",
        "sandbox UI reachable as window.Sandbox (app.js activation gate)");
  if (!window.Sandbox) return;  // remaining checks need the sandbox wired

  onText({ type: "model-info", ...modelInfo });
  check(els.get("dof-table").classList.contains("hidden") === false,
        "DOF table shown for nu>42");
  check(els.get("sliders").style.display === "none",
        "classic slider list hidden in table mode");
  check(els.get("dof-content").style.height === 132 * 30 + "px",
        "virtualized content height = n*ROW_H");
  const nRows0 = els.get("dof-content").children.length;
  check(nRows0 > 0 && nRows0 < 40,
        \`only the visible window rendered (\${nRows0} rows)\`);

  /* telemetry: resync shadow, strip, LEDs */
  const ctrl = new Array(N).fill(0.123);
  const qpos = new Array(133).fill(0);
  qpos[0] = 1; qpos[1] = 2; qpos[2] = 0.5;
  window.Sandbox.onTelemetry({
    t: 0.1, qpos, qvel: [0.01, 0.02, 0.03], ctrl,
    root: { pos: [1, 2, 0.5], quat: [1, 0, 0, 0] },
    contact: { lf: [0.0], lm: [0.0], lh: [0.0], rf: [0.0], rm: [0.0],
               rh: [5.2] },
    paused: false, mode: "neutral",
  });
  check(!els.get("telemetry").classList.contains("hidden"),
        "telemetry strip revealed");
  check(/z 500 mm/.test(els.get("tele-z").textContent),
        "root z readout (" + els.get("tele-z").textContent + ")");
  check(els.get("led-rh").classList.contains("on"),
        "rh contact LED on (force 5.2)");
  check(!els.get("led-lf").classList.contains("on"),
        "lf contact LED off (force 0)");

  /* slider input -> coalesced full-array actuators send */
  const row0 = els.get("dof-content").children[0];
  row0._input.value = "0.55";
  row0._input.fire("input");
  flushRaf();
  const actMsg = __sent.filter((m) => m.type === "actuators").pop();
  check(actMsg && actMsg.values.length === N,
        "coalesced actuators message carries the full array");
  check(actMsg && Math.abs(actMsg.values[0] - 0.55) < 1e-9,
        "edited index carries the new value");
  check(actMsg && Math.abs(actMsg.values[1] - 0.123) < 1e-9,
        "untouched indices keep the server (telemetry) value");

  /* a second telemetry resyncs untouched sliders to the server value */
  const ctrl2 = new Array(N).fill(0.123);
  ctrl2[1] = -0.77;
  window.Sandbox.onTelemetry({
    t: 0.2, qpos, qvel: [0, 0, 0], ctrl: ctrl2,
    root: { pos: [1, 2, 0.5], quat: [1, 0, 0, 0] },
    contact: {}, paused: false, mode: "neutral",
  });
  const row1 = els.get("dof-content").children[1];
  check(row1 && row1._val.textContent === "-0.77",
        "untouched slider resynced from telemetry (got " +
        (row1 && row1._val.textContent) + ")");

  /* M7.7: the viewcube tracks the CAMERA, not the fly — a fly heading
     change in telemetry leaves the cube transform untouched (the old
     fly-relative cube shook and moved while the fly moved). */
  const cubeBefore = els.get("cube").style.transform;
  window.Sandbox.onTelemetry({
    t: 0.3, qpos, qvel: [0, 0, 0], ctrl: new Array(N).fill(0.123),
    root: { pos: [1, 2, 0.5],
            quat: [Math.SQRT1_2, 0, 0, Math.SQRT1_2] },  // yaw +90 deg
    contact: {}, paused: false, mode: "neutral",
  });
  check(els.get("cube").style.transform === cubeBefore,
        "fly heading change leaves the viewcube untouched (camera-tracked); "
        + "got " + els.get("cube").style.transform);

  /* mode button -> mode message */
  const walkBtn = document.querySelectorAll(".mode-btn")
    .find((b) => b._mode === "walk");
  walkBtn.fire("click");
  check(JSON.stringify(__sent.filter((m) => m.type === "mode").pop())
        === JSON.stringify({ type: "mode", value: "walk" }),
        "mode walk message shape");

  /* flap gating: the model-info above has no wing-pitch actuators, so
     the flap mode button must be disabled (server would reject it). */
  const flapBtn0 = document.querySelectorAll(".mode-btn")
    .find((b) => b.dataset.mode === "flap");
  check(flapBtn0 && flapBtn0.disabled === true,
        "flap mode button disabled without wing actuators");
  check(flapBtn0 && /wing/i.test(flapBtn0.title || ""),
        "disabled flap button explains why (title mentions wings)");

  /* spawn / param / loco / force / log message shapes */
  $("spawn-x").value = "100"; $("spawn-y").value = "-20";
  $("spawn-z").value = "900";
  $("btn-spawn").fire("click");
  const spawnMsg = __sent.filter((m) => m.type === "spawn").pop();
  check(spawnMsg && JSON.stringify(spawnMsg.pos) === "[100,-20,900]"
        && JSON.stringify(spawnMsg.quat) === "[1,0,0,0]",
        "spawn message shape (pos mm + identity quat)");

  $("grav-z").value = "-4905";
  $("grav-z").fire("change");
  check(JSON.stringify(__sent.filter((m) => m.type === "param").pop())
        === JSON.stringify({ type: "param", name: "gravity",
                             value: [0, 0, -4905] }),
        "param gravity message shape");

  $("timestep").value = "0.0002";
  $("btn-timestep").fire("click");
  const paramMsgs = __sent.filter((m) => m.type === "param");
  check(paramMsgs.length === 2 && paramMsgs[1].name === "timestep"
        && paramMsgs[1].value === 0.0002, "param timestep message shape");

  $("loco-freq").value = "4.5";
  $("loco-freq").fire("change");
  check(JSON.stringify(__sent.filter((m) => m.type === "loco").pop())
        === JSON.stringify({ type: "loco", freq: 4.5 }),
        "loco freq message shape");

  $("force-mag").value = "7000";
  $("btn-force-up").fire("click");
  const forceMsg = __sent.filter((m) => m.type === "force").pop();
  check(forceMsg && forceMsg.body === "nmf/c_thorax"
        && JSON.stringify(forceMsg.force) === "[0,0,7000]"
        && forceMsg.duration === 0.1,
        "force pulse message shape");

  $("btn-log").fire("click");
  check(JSON.stringify(__sent.filter((m) => m.type === "log").pop())
        === JSON.stringify({ type: "log", on: 1 }), "log on message shape");

  /* V1 vision routing (M7.6): a V1-prefixed binary frame goes to the
     vision <img> (prefix stripped); a bare JPEG goes to the main <img> */
  const frame = (bytes) => ({
    slice: (start, end) => ({
      arrayBuffer: async () =>
        new Uint8Array(bytes.slice(start, end)).buffer,
    }),
  });
  await onBinary(frame([0x56, 0x31, 0xff, 0xd8, 0x0a, 0x0b]));
  check(els.get("vision").classList.contains("hidden") === false,
        "V1 frame reveals the vision pane");
  const visionSrc = els.get("vision").src;
  check(typeof visionSrc === "string" && visionSrc.startsWith("blob:"),
        "V1 frame (prefix stripped) routed to the vision <img>");
  const mainSrcBefore = els.get("video").src;
  await onBinary(frame([0xff, 0xd8, 0x09, 0x00, 0x42]));
  check(els.get("video").src !== mainSrcBefore
        && els.get("video").src.startsWith("blob:"),
        "bare JPEG frame goes to the main <img>");
  check(els.get("vision").src === visionSrc,
        "bare JPEG leaves the vision pane untouched");

  /* wing actuators present (--dof all) -> flap enabled again: re-init
     the sandbox with a model-info that has the two wing-pitch DOFs. */
  onText({ type: "model-info", ...modelInfo, nu: N + 2,
           actuators: modelInfo.actuators.concat([
             { id: N, name: "nmf/c_thorax-l_wing-pitch-position",
               gaintype: 0, biastype: 1, dyntype: 0, joint: 0, tendon: null,
               range: [-3.14, 3.14] },
             { id: N + 1, name: "nmf/c_thorax-r_wing-pitch-position",
               gaintype: 0, biastype: 1, dyntype: 0, joint: 1, tendon: null,
               range: [-3.14, 3.14] },
           ]) });
  const flapBtn1 = document.querySelectorAll(".mode-btn")
    .find((b) => b.dataset.mode === "flap");
  check(flapBtn1 && flapBtn1.disabled === false,
        "flap mode button enabled when wing actuators are present");
})();
return __harness;
`;

/* Inject the harness helpers (check/els/modelInfo) into the scope. */
const src = [
  `const __sent = globalThis.__sent;`,
  `const N = globalThis.__N;`,
  `const flushRaf = globalThis.__flushRaf;`,
  `const els = globalThis.__els;`,
  `const modelInfo = globalThis.__modelInfo;`,
  `const check = globalThis.__check;`,
  `const styleCss = globalThis.__styleCss;`,
  appJs,
  sandboxJs,
  harness,
].join("\n");

globalThis.__els = els;
globalThis.__modelInfo = modelInfo;
globalThis.__check = check;
globalThis.__N = N;
globalThis.__flushRaf = flushRaf;
globalThis.__styleCss = styleCss;

let pending;
try {
  pending = new Function(src)();
} catch (err) {
  failures += 1;
  console.error(" UI smoke crashed:", err.stack);
}
Promise.resolve(pending).then(() => {
  if (failures) {
    console.error(`\n${failures} UI smoke check(s) FAILED`);
    process.exit(1);
  }
  console.log("\nUI smoke: all checks passed");
}).catch((err) => {
  failures += 1;
  console.error(" UI smoke crashed:", err.stack);
  process.exit(1);
});
