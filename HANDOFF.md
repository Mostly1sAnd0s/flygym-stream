# HANDOFF — flygym-stream (as of 2026-07-21, EOD)

**Status**: Phase 7 (M7.1–M7.6) COMPLETE and committed (`226f674` vision,
`21a8e76` docs). A follow-up user bug report (4 items) is DIAGNOSED; 3 of 4
are quick fixes not yet implemented, 1 (default standing pose) needs a
research pass first. All probes below are verified — do not re-derive them.

Working tree: docs-only edits pending commit at EOD (this file, PLAN.md
standing-pose correction). Service 8765 restarted at EOD (sim thread was
dead from the flap crash — see Bug 4). 8793 (vision test instance) killed.

---

## The user's bug report (2026-07-21) — status per item

> 1. The fly should start standing upright and stationary, not lying
>    upside down and rotating.
> 2. There are two "neutral" buttons.
> 3. Pressing walk makes the fly walk, but it's upside down so it can't
>    walk correctly.
> 4. Pressing flap seems to crash the simulation; the fly window goes
>    black/broken.

### Bug 4 — flap crash: DIAGNOSED, fix pending (DO THIS FIRST)
Root cause in `src/flygym_stream/simulation.py`, `SimThread._dispatch`:
the exception handlers call **`log.warning(...)`** but the module's logger
is named **`logger`** → `NameError` inside the `except` block kills the
`flygym-sim` thread. Two affected sites:
- ~line 417: state-message branch (`apply_state` rejection / apply error)
- ~line 453: loco-branch `ValueError` (e.g. `flap` without wing actuators —
  the exact case in `/tmp/flygym-stream.log`: `ValueError: flap requires
  wing actuators (serve with --dof all)` then `NameError: name 'log' is not
  defined` in Thread `flygym-sim`)

Result: process stays alive, `/healthz` still returns ok (healthz does NOT
reflect sim-thread liveness — it's a pure ASGI endpoint), but no frames
arrive → black window. This is the "crash".

TDD spec:
- `tests/test_simulation.py`: SimThread with a fake loco whose `set_mode`
  raises (like real `flap` on a leg-only scene) → `tick()` must return
  `False`/gracefully, the thread must keep stepping on the next tick, and
  a subsequent valid message (e.g. `neutral` mode) must be dispatched.
  Also the state-message twin: a fake backend whose `apply_state` raises →
  same survival expectation.
- `tests/test_app.py`: `/healthz` gains `"sim_running": bool`
  (`app_env.sim_thread.is_alive()`) — regression: a dead sim thread must be
  visible to the operator. (Cheap, big operational value.)

Fix: `log.warning` → `logger.warning` (2 sites). The `ValueError` rejection
itself is CORRECT protocol behavior (flap needs `--dof all`) — keep it,
make it non-fatal.

### Bug 2 — two "neutral" buttons: DIAGNOSED, fix pending
`web/index.html` top row has `#btn-neutral` (sends `{type:"neutral"}`,
restores the neutral keyframe ctrl). The sandbox mode row
(`#mode-row`, rendered by `web/app_sandbox.js`) has a **mode** button
`data-mode="neutral"` (sends `{type:"mode",mode:"neutral"}`, which also
restores the neutral keyframe ctrl — `locomotion.py:135-139`). Both do the
same thing.

Fix (UI only, no protocol change — `neutral` message stays in protocol.py):
- Remove `#btn-neutral` from `index.html` and its `wire("btn-neutral", ...)`
  in `app.js`; remove the `btn-neutral` binding in `app_sandbox.js`
  `wireControls` (the mode-row buttons already clear dirty sliders).
- Keep the mode row's neutral as the single neutral.
- `scripts/ui_smoke.cjs`: add a check that `index.html` source no longer
  contains `btn-neutral` (injected text is in the smoke's harness).
- README §Web UI button list ("neutral / pause / resume / step / reset /
  log") — drop "neutral" from that sentence after the change.

### Bug 1 + 3 — fly starts prone/slowly-rotating; walks "upside down": DIAGNOSED, needs research
The default scene's rest state is the neutral keyframe + 0.8 m spawn
overwrite. Verified (probes, both meshes): the keyframe root quat is
**identity (upright)**; the fly FALLS from 0.8 m and lands with the body
on the ground, legs folded, then creep-rotates indefinitely (NoSlip +
position actuators). So "upside down and rotating" = the landed pose.
Walk then performs its offsets on that baseline (M7.3: pattern = offsets
from baseline captured at mode entry), so the gait is as good as the
baseline allows. Fixing the baseline to a true standing pose fixes both.

**Standing-pose research (the blocker)** — verified probe facts
(`/home/adam/neurofly/flygym-stream`, offline, `flygym` compose API — see
the probe scripts in this file's "Probe recipes" section):

- Neutral keyframe: root pos `[0.496, 0, 2.1]` m, quat `[1,0,0,0]`.
  Neutral ctrl (lf leg): coxa_pitch **0.332**, femur_pitch **−2.426**,
  tibia_pitch **1.361** (folded legs).
- 3 s settle from spawn (0,0,0.8), neutral ctrl: simplified root z 0.500 /
  fullsize 0.489 m; body geoms in contact; still slow-moving (the user's
  "slowly rotating"; fullsize |qvel| ~0.9, |v_root| ~0.29 at 3 s).
- The documented M7.3 standing offsets (coxa +0.4, femur +1.4, tibia −0.4,
  all 6 legs, applied to neutral ctrl) do **NOT** lift the fly from ground
  contact: `neutral+OFF` → root 0.519 (simp) / 0.527 (full) m, no body
  contact, but joints still flailing at 3 s (|qvel| 0.24–0.36).
  `zero+OFF` (simplified): **fully pinned** — |qvel| = 0.000, body on
  ground (NoSlip truss: feet + body + ground lock).
- Standing-INITIALIZED fullsize probe (18 leg joints' qpos set to
  neutral+offset, ctrl = neutral+offset, root placed at "standing height"):
  root FELL to 0.504 m and kept moving (|v_root| 0.13–0.33, |qvel| ~0.5 at
  3 s). CAVEAT: that probe's standing height was computed as min-z over
  **ALL** `nmf/` geoms → contaminated by the wingtips (wings reach −1.23 m
  below the root in the neutral pose), so the fly started 1+ m too high and
  fell. **Foot-geoms-only height is the correct measure — this probe is
  INVALID and must be redone.**

Open questions for the research pass:
1. Foot-geom-only standing height (which geoms? tibia/tarsus of each leg —
   find names via `mj_id2name(OBJ_GEOM)`; the tarsus/trochanter-femur/tibia
   geoms, NOT wings/head/thorax/abdomen).
2. Does an initialized-standing state (feet on ground, correct height) HOLD
   (≥10 s settle, fullsize): root z stable within mm, |v_root| < 0.05,
   |qvel| small, body not in contact?
3. If not: re-grid-search the offsets with the CORRECT metric (settle from
   initialized-standing, not from the 0.8 m drop). Iterate on simplified
   (fast), confirm on fullsize.
4. If a holding standing pose exists: it is a **full state snapshot**
   (qpos + qvel=0 + ctrl), NOT a ctrl-only fix — the legs cannot reach the
   pose from ground contact (verified pinning).

**Implementation design (for when research succeeds)**:
- `Scene` gains a `rest_state` (numpy qpos, qvel, ctrl arrays) replacing the
  `rest_ctrl`-only mechanism (or alongside — `rest_ctrl` for legacy
  builds; the default POSITION build gets the full snapshot). `reset()`
  restores qpos+qvel+ctrl; `neutral_actuators()` returns the snapshot's
  ctrl (loco neutral mode keeps working).
- `build_scene` computes it for POSITION builds: ctrl = neutral + standing
  offsets; the 18 leg joints' qpos += same offsets (joint qpos addresses:
  `m.actuator_trnid[aid,0]` → `m.jnt_qposadr[jid]`, hinge size 1); root
  x/y from `SceneConfig.spawn`, root z = foot-geom-only standing height;
  qvel = 0. Tethered (pinned root): snapshot qpos root ignored on restore
  (Scene already skips root restore when not free); MSK: unchanged.
- Test: default scene (network mark) → `scene.reset()`, step 3 sim s →
  root z > 1.0 m, |v_root| < 0.1, body geoms NOT in contact. (Thresholds
  from the research results.)
- UI: the mode-row `neutral` and the `neutral` message restore the
  keyframe ctrl only — decide whether "neutral" should restore the STANDING
  rest state instead (lean: yes, it's the home pose; check
  `Scene.neutral_actuators()` call sites: `app.py` neutral message,
  `locomotion.py` neutral mode, `simulation.py`? — grep before changing).
- After the pose change: the M7.3 walk gait baseline improves
  automatically (offsets captured from the standing baseline).

If research FAILS (no holding standing pose): document in PLAN.md, keep
the default, and tell the user the honest answer (the model + NoSlip
contact doesn't support a passive standing posture from this keyframe;
walking/righting-kick is the v1 behavior).

### Bug 4 (UI side) — flap button on a leg-only build: fix pending
With `--dof legs` (the default), `flap` is rejected by the server (correct,
non-fatal after Bug 4's fix) — but the UI should hide/disable the flap
mode button when the model has no wing actuators:
- `app_sandbox.js` `init`: `window.Sandbox.wingsPresent = actuators.some(a =>
  /wing-pitch/.test(a.name))` (sandbox init receives model-info with
  `actuators`); `buildModeRow` skips the flap button (or renders it
  disabled with `title` explanation) when `!wingsPresent`.
- `ui_smoke.cjs`: model-info WITHOUT wing actuators → no flap mode button
  (and WITH them → present).
- Server already logs the rejection; no protocol change.

---

## Ordered worklist for the next session

1. **Bug 4 server fix** (TDD per spec above): `log`→`logger` (2 sites) +
   healthz `sim_running`. Verify: default suite; then live serve: send
   `{"type":"flap"}` (WS) → rejection logged, frames KEEP flowing.
2. **Bug 2 UI fix**: single neutral button + ui_smoke check + README line.
3. **Bug 4 UI**: flap button gated on wing actuators + ui_smoke checks.
4. **Standing-pose research** (probe recipes below; iterate simplified,
   confirm fullsize; ≥10 s settles). Only if a holding pose is found:
   `Scene.rest_state` snapshot + `build_scene` + network test + neutral
   semantics decision + UI/README updates.
5. Re-verify end-to-end: full pytest matrix, ui_smoke, Playwright pass
   (recipe below), live serve 8765.
6. Docs: HANDOFF/PLAN/README updates; commit per milestone (TDD: red
   first, green, commit).

## Ops (current state)

- 8765 running (restarted at EOD, current code, NO `--overlay depth`):
  pid in `/tmp/flygym-stream.pid`, log `/tmp/flygym-stream.log`.
  Restart recipe:
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl nohup .venv/bin/python -m flygym_stream.cli serve --bind 0.0.0.0 --insecure --port 8765 >>/tmp/flygym-stream.log 2>&1 &`
  (env vars MUST be set before python starts; static `web/` files serve
  from disk per request — UI changes don't need a restart; code changes
  do).
- `--overlay depth` costs ~4× frame time (6–7 fps vs 30; sim well below
  real time). Restart without it for normal use.
- Single-client WS: second client rejected (close 1013).
- **pkill gotcha**: `pkill -f "port 8765"` kills the CALLING shell too
  (its cmdline contains the pattern). Use the pid file or
  `pgrep -af "flygym_stream.cli serve" | grep -v "bash -c"`.
- MuJoCo 3.9 binding API notes (bitten this session): no
  `m.jnt_posadr` (use `m.jnt_qposadr`); `MjContactList` has no `.body`
  attr — contacts are field arrays: `d.contact.geom[i]` = [geom0, geom1],
  body via `m.geom_bodyid[g]`; `m.nsensor` not `nsens`; `o_friction`/
  `o_solref`/`o_solimp`/`o_margin` are the MjOpt field names;
  `jnt_type` HINGE=3.
- `encode_jpeg(frame, quality)` REQUIRES the quality arg.

## Verification playbook (all green at `a280311`)

- `uv run pytest -q` → 145; `uv run pytest -m gpu -q` → 18 (needs
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` exported BEFORE pytest starts);
  `uv run pytest -m network -q` → 18 (~2 min, S3 fetch on cold cache);
  `node scripts/ui_smoke.cjs` → 26 checks.
- Real-browser (the ONLY way to catch browser-scoping bugs — the fake-DOM
  ui_smoke concatenates scripts into one scope, which masked the
  `const Sandbox` bug): `cd /tmp/uibug && npx playwright install chromium`
  (~120 MB, cached in `~/.cache/ms-playwright`), then a node script:
  `chromium.launch({headless:true})` → `page.on("pageerror")` →
  `page.goto("http://127.0.0.1:8765")` → wait for `window.Sandbox` →
  drive buttons/sliders, read HUD + telemetry strip. (Throwaway scripts
  lived in `/tmp/uibug/` + `/tmp/uibug2/` — /tmp may be wiped; the pattern
  is: capture pageerrors, wait for `#telemetry .val` non-empty, click
  buttons, assert HUD text.)
- Live serve check: `curl -s localhost:8765/healthz`; WS frames = bare
  JPEG (main) + `V1`-prefixed JPEG (vision builds); JSON text messages per
  protocol v1.

## Probe recipes (standing pose — run from repo root)

Build (both meshes, legs-only, exactly what the service runs):
```python
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, ContactBodiesPreset, JointPreset, Skeleton
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, MeshType, NeuroMechFly
from flygym.utils.math import Rotation3D
import mujoco as mj
fly = NeuroMechFly(mesh_type=MeshType.FULLSIZE)          # or SIMPLIFIED_MAX2000FACES (fast)
sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.YAW_PITCH_ROLL)
fly.add_joints(sk, KinematicPosePreset.NEUTRAL)
dofs = sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
fly.add_actuators(dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL,
                  kp=50.0, ctrlrange=(-3.14, 3.14), forcelimited=True, forcerange=(-30.0, 30.0))
fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
fly.colorize()
world = FlatGroundWorld()
world.add_fly(fly, (0.0, 0.0, 0.8), Rotation3D("quat", (1.0, 0, 0, 0)),
              bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
m, d = world.compile()
key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
scratch = mj.MjData(m); mj.mj_resetDataKeyframe(m, scratch, key)
neutral = scratch.ctrl.copy()
```
Actuator names (`nmf/` prefix):
- coxa_pitch: `c_thorax-{leg}_coxa-pitch-position`
- femur_pitch: `{leg}_coxa-{leg}_trochanterfemur-pitch-position`
- tibia_pitch: `{leg}_trochanterfemur-{leg}_tibia-pitch-position`
- legs: lf lm lh rf rm rh (same pattern list as `locomotion.LEG_ACTUATOR_NAMES`)
- joint qpos adr: `j = int(m.actuator_trnid[aid, 0]); adr = int(m.jnt_qposadr[j])`

Standing-height (CORRECT version — foot geoms only, NOT all nmf/ geoms):
```python
q = scratch.qpos.copy(); ctrl = neutral.copy()
# ... set 18 leg joints' qpos += offset, ctrl += offset ...
q[2] = 1.0
d2 = mj.MjData(m); d2.qpos[:] = q; d2.qvel[:] = 0; d2.ctrl[:] = ctrl
mj.mj_forward(m, d2)
# min-z over FOOT geoms only (tarsus/tibia; list names first:
#   for i in range(m.ngeom): print(mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i))
# and pick the foot ones) → q[2] = 1.0 - min_foot_z; re-forward; step ≥10 s.
```
Success metric: root z stable (mm), |v_root| < 0.05, |qvel| small, body
geoms (c_thorax/abdomen) NOT in contact (check via `d2.contact.geom[i]`
→ `m.geom_bodyid` → body name).

## Key file map

`protocol.py` (wire truth) · `scenes.py` (SceneConfig/build_scene, MSK +
vision paths) · `scene.py` (state-write surface, `reset()`,
`neutral_actuators()`, `vision` attr) · `simulation.py` (Backend/SimThread
— **the `log`/`logger` bug is here**) · `locomotion.py` (patterns;
`LEG_ACTUATOR_NAMES`; `flap` requires wing actuators) · `telemetry.py` ·
`vision.py` (V1 framing) · `stream.py` (FrameQueue) · `app.py` (WS +
healthz — add `sim_running`) · `cli.py` (all flags) · `web/index.html`
(duplicate `#btn-neutral`), `web/app.js` (wire + onBinary V1 routing +
ensureSliders), `web/app_sandbox.js` (`window.Sandbox`; mode row;
wireControls) · `scripts/ui_smoke.cjs` (26 checks).

## Git (newest first)

`a280311` UI browser-scoping fix (Sandbox window prop + slidersEl; ui_smoke
models real page contract; 26 checks) · `21a8e76` docs (Phase 7 done) ·
`226f674` M7.6 vision · `4d837cc` WIP stop · `e8dbe48` M7.6 msk ·
`4fa78cc` M7.5 · `eab746b` M7.4 · `1493b9a` M7.3 · `af27734` M7.2 ·
`a185829` M7.1. Commit identity:
`git -c user.email=dev@neurofly.local -c user.name=dev commit`.
