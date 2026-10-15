/* FlyGym Stream research-sandbox UI (PLAN.md Phase 7, M7.5).
 *
 * Activated by the server's model-info message (M7.4). Adds: a
 * virtualized DOF table (replaces the 42-slider list when
 * n_actuators > 42), the locomotion mode row + live tuning, the
 * perturbation panel (spawn / gravity / force pulse / timestep), and
 * the telemetry strip (root z, |v|, per-leg contact LEDs, mode).
 * Outgoing messages go through the MSG map in app.js (single source of
 * truth, mirrored from src/flygym_stream/protocol.py).
 */
"use strict";

/* Assigned to `window` (NOT a top-level const): app.js gates the sandbox
 * on `window.Sandbox`, and a top-level const in a classic <script> does
 * not create a window property — the UI would never activate. */
window.Sandbox = (() => {
  const $ = (id) => document.getElementById(id);
  const ROW_H = 30;       // px per virtualized row
  const OVERSCAN = 4;     // rows rendered beyond the viewport
  const LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"];

  const state = {
    info: null,          // model-info payload
    shadow: null,        // Float64Array ctrl mirror (server is truth)
    dirty: new Set(),    // actuator indices pending a coalesced send
    dragIndex: -1,       // slider being actively dragged
    flushQueued: false,
    windowStart: -1,
    windowEnd: -1,
    n: 0,                // actuator count
    tableMode: false,    // true when the DOF table replaces the sliders
    wingsPresent: false, // true when the model has wing-pitch actuators
  };

  /* --- init ---------------------------------------------------------------- */
  function init(info) {
    state.info = info;
    state.n = info.nu;
    state.shadow = new Float64Array(state.n);
    // Actuator i -> joint qadr (for the live qpos readout column).
    state.qadr = info.actuators.map((a) => {
      if (a.joint == null || a.joint >= info.joints.length) return null;
      return info.joints[a.joint].qadr;
    });
    state.names = info.actuators.map(
      (a) => a.name.replace(/^.*\//, "").replace(/-position$/, ""));
    state.ranges = info.actuators.map((a) => a.range);
    // flap mode needs wing-pitch actuators (--dof all); without them the
    // button is disabled (the server rejects the mode anyway).
    state.wingsPresent = info.actuators.some((a) => /wing-pitch/.test(a.name));

    // Large DOF builds (126+) get the virtualized table; the classic
    // 42-slider list is left untouched for the default scene (PLAN M7.5).
    state.tableMode = state.n > 42;
    $("sliders").style.display = state.tableMode ? "none" : "";
    $("dof-table").classList.toggle("hidden", !state.tableMode);
    if (state.tableMode) {
      $("dof-content").style.height = state.n * ROW_H + "px";
      $("dof-viewport").addEventListener("scroll",
        () => requestAnimationFrame(renderWindow));
      renderWindow();
    }
    wireControls();
  }

  /* --- virtualized DOF table ------------------------------------------------ */
  function renderWindow() {
    const vp = $("dof-viewport");
    const start = Math.max(0, Math.floor(vp.scrollTop / ROW_H) - OVERSCAN);
    const end = Math.min(state.n,
      start + Math.ceil(vp.clientHeight / ROW_H) + 2 * OVERSCAN);
    if (start === state.windowStart && end === state.windowEnd) return;
    state.windowStart = start;
    state.windowEnd = end;
    const content = $("dof-content");
    content.replaceChildren();
    for (let i = start; i < end; i++) content.append(makeRow(i));
  }

  function makeRow(i) {
    const row = document.createElement("div");
    row.className = "dofrow";
    row.style.top = i * ROW_H + "px";

    const name = document.createElement("span");
    name.className = "dofname";
    name.textContent = state.names[i];
    name.title = state.names[i];

    const input = document.createElement("input");
    input.type = "range";
    input.min = String(state.ranges[i][0]);
    input.max = String(state.ranges[i][1]);
    input.step = "0.01";
    input.value = state.shadow[i] != null ? state.shadow[i].toFixed(3) : "0";

    const val = document.createElement("span");
    val.className = "dofval";
    val.textContent = Number(input.value).toFixed(2);

    const q = document.createElement("span");
    q.className = "doqpos";
    if (state.qadr[i] != null) {
      q.textContent = "\u2014";
      q.dataset.qadr = state.qadr[i];
      q.title = `qpos[${state.qadr[i]}] (live, from telemetry)`;
    } else {
      q.textContent = "no-joint";
      q.title = "tendon/force target (no qpos readout)";
    }

    input.addEventListener("pointerdown", () => { state.dragIndex = i; });
    input.addEventListener("pointerup", () => { state.dragIndex = -1; });
    input.addEventListener("input", () => {
      state.shadow[i] = Number(input.value);
      val.textContent = Number(input.value).toFixed(2);
      state.dirty.add(i);
      queueFlush();
    });

    row.append(name, input, val, q);
    row._val = val;
    row._q = q;
    row._input = input;
    row._index = i;
    return row;
  }

  /* --- coalesced actuator sends --------------------------------------------- */
  function queueFlush() {
    if (state.flushQueued) return;
    state.flushQueued = true;
    requestAnimationFrame(() => {
      state.flushQueued = false;
      if (state.dirty.size === 0) return;
      state.dirty.clear();
      send(MSG.actuators(Array.from(state.shadow)));
    });
  }

  /* --- telemetry (10 Hz server truth) --------------------------------------- */
  function onTelemetry(t) {
    if (!state.info) return;
    // Resync the ctrl shadow from the server (skips what the user is
    // mid-edit on); in pattern modes this makes the sliders follow the
    // gait live.
    if (t.ctrl && t.ctrl.length === state.n) {
      for (let i = 0; i < state.n; i++) {
        if (state.dirty.has(i) || i === state.dragIndex) continue;
        state.shadow[i] = t.ctrl[i];
      }
    }
    refreshVisibleRows(t);
    updateTelemetryStrip(t);
  }

  function refreshVisibleRows(t) {
    const rows = $("dof-content").children;
    for (const row of rows) {
      const i = row._index;
      if (!state.dirty.has(i) && i !== state.dragIndex && t.ctrl
          && t.ctrl.length === state.n) {
        row._input.value = state.shadow[i].toFixed(3);
        row._val.textContent = state.shadow[i].toFixed(2);
      }
      if (row._q.dataset.qadr != null && t.qpos) {
        row._q.textContent = Number(t.qpos[row._q.dataset.qadr]).toFixed(2);
      }
    }
  }

  function updateTelemetryStrip(t) {
    $("telemetry").classList.remove("hidden");
    $("tele-z").textContent = `z ${Math.round(t.root.pos[2] * 1000)} mm`;
    const v = Math.hypot(t.qvel[0], t.qvel[1], t.qvel[2]);
    $("tele-v").textContent = `|v| ${Math.round(v * 1000)} mm/s`;
    $("tele-mode").textContent = `mode ${t.mode}`;
    for (const leg of LEGS) {
      const led = $(`led-${leg}`);
      if (!led) continue;
      const f = t.contact[leg] ? t.contact[leg][0] : 0;
      led.classList.toggle("on", f > 0.01);
    }
  }

  function onStatus(s) {
    if (s.mode != null) {
      document.querySelectorAll(".mode-btn").forEach((b) => {
        b.classList.toggle("active", b.dataset.mode === s.mode);
      });
    }
  }

  /* --- controls -------------------------------------------------------------- */
  function wireControls() {
    document.querySelectorAll(".mode-btn").forEach((b) => {
      b.addEventListener("click", () => {
        send(MSG.mode(b.dataset.mode));
        // Pattern modes take ctrl authority: drop client edits so the
        // telemetry resync can redraw the sliders.
        state.dirty.clear();
      });
    });

    // Gate flap on wing-actuator presence (legs-only builds: the server
    // rejects the mode; the button should say so up front).
    const flapBtn = Array.from(document.querySelectorAll(".mode-btn"))
      .find((b) => b.dataset.mode === "flap");
    if (flapBtn) {
      flapBtn.disabled = !state.wingsPresent;
      flapBtn.title = state.wingsPresent
        ? "Flap both wings in phase (needs --dof all)"
        : "Unavailable on this build: no wing actuators (serve with --dof all)";
    }

    const locoFields = { "loco-freq": "freq", "loco-amp": "amplitude",
                         "loco-phase": "phase" };
    for (const [id, field] of Object.entries(locoFields)) {
      const el = $(id);
      el.addEventListener("change", () => {
        $(id + "-val").textContent = Number(el.value).toFixed(2);
        send(MSG.loco({ [field]: Number(el.value) }));
      });
    }

    $("btn-spawn").addEventListener("click", () => {
      const pos = [Number($("spawn-x").value), Number($("spawn-y").value),
                   Number($("spawn-z").value)];
      send(MSG.spawn(pos));
    });

    const grav = $("grav-z");
    grav.addEventListener("change", () => {
      $("grav-z-val").textContent = `${Number(grav.value)}`;
      send(MSG.param("gravity", [0, 0, Number(grav.value)]));
    });

    const pulse = (dir) => {
      const mag = Number($("force-mag").value);
      send(MSG.force("nmf/c_thorax", [0, 0, dir * mag], 0.1));
    };
    $("btn-force-up").addEventListener("click", () => pulse(1));
    $("btn-force-dn").addEventListener("click", () => pulse(-1));

    $("btn-timestep").addEventListener("click", () => {
      const dt = Number($("timestep").value);
      if (dt > 0) send(MSG.param("timestep", dt));
    });

    let logging = false;
    $("btn-log").addEventListener("click", () => {
      logging = !logging;
      $("btn-log").classList.toggle("active", logging);
      $("btn-log").title = logging
        ? "Stop JSONL logging (file served at /log)"
        : "Start JSONL logging (telemetry + control events)";
      send(MSG.log(logging ? 1 : 0));
    });

    // A neutral/reset drops client ctrl authority: let the server (via
    // telemetry) be the source of truth again. (The mode-row neutral
    // button already clears dirty in its click handler above.)
    $("btn-reset").addEventListener("click", () => state.dirty.clear());
  }

  return { init, onTelemetry, onStatus };
})();
