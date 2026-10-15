# HANDOFF — flygym-stream (as of 2026-07-21, after bug-fix session)

**Status**: The 4-bug user report is RESOLVED to the extent the model
allows: 3 of 4 fully fixed and committed; bug 1/3 (default standing pose)
concluded as **passively infeasible** in this model after a full research
pass — documented with the complete evidence chain; the honest fix
(closed-loop righting controller) is scoped below as the next research
milestone. All probe data verified — do not re-derive.

Working tree at this stop point: docs-only edits pending commit (this
file, PLAN.md M7.3 note, README standing-pose note).

## The user's bug report (2026-07-21) — final status

> 1. Fly should start standing upright and stationary, not upside down / rotating.
> 2. Two "neutral" buttons.
> 3. Walk works but the fly is upside down so it can't walk correctly.
> 4. Flap crashes the simulation (black window).

### Bug 4 — flap crash: FIXED `6088f19`
Root cause: `simulation.py` `_dispatch` exception handlers called
`log.warning` (module logger is named `logger`) → `NameError` inside the
handler killed the `flygym-sim` thread on the first rejection (flap on a
legs-only build); process stayed alive and `/healthz` stayed `ok` (it never
reflected sim-thread liveness) → black window. Fix: `logger.warning` at all
3 sites + `/healthz` gains `sim_running`. TDD: 3 new tests (loco-rejection
survival, state-rejection survival, healthz liveness) RED→GREEN.
Live-verified: WS `mode=flap` → rejection logged, 152 frames in the next
3 s, `sim_running` true. **Note:** the protocol mode message field is
`value` (`{"type":"mode","value":"flap"}`), not `mode`; unknown types
(e.g. `{"type":"flap"}`) are silently ignored, not rejected.

### Bug 2 — two neutral buttons: FIXED `497f694`
Top-row `#btn-neutral` duplicated the sandbox mode-row `neutral` mode
(same effect: restore neutral keyframe ctrl; the mode button additionally
drives the telemetry-strip mode). Removed the top-row button (index.html),
its `wire()` (app.js), its dirty-clear listener (app_sandbox.js). The
mode-row neutral is the single neutral (gained a title). `MSG.neutral`
stays in the map (the protocol message remains valid for other clients).
ui_smoke now checks the index.html source for no `btn-neutral`.

### Bug 4 (UI) — flap button on a leg-only build: FIXED `497f694`
The flap mode button is disabled when model-info has no wing-pitch
actuators (`state.wingsPresent`, regex `/wing-pitch/` on actuator names;
real names `nmf/c_thorax-{l,r}_wing-pitch-position`), with a title
explaining why; enabled when present (server-side non-fatal rejection from
`6088f19` remains the backstop). ui_smoke: 4 new checks (26→30).
**Also fixed a latent ui_smoke harness bug found while adding them:** the
async harness IIFE was an expression statement, so `new Function(src)`
returned `undefined` and the verdict ran BEFORE the awaited V1 checks
settled — the 4 V1 checks printed after "all checks passed" and never
counted (a red V1 check could not fail the smoke). The harness promise is
now captured in `__harness` and `return`ed, so the verdict awaits the full
async flow. **And a real-browser-only bug the fake DOM missed:**
`document.querySelectorAll(...).find` — `NodeList` has no `.find` in a real
browser (the fake returns an Array); it threw in `wireControls` and left
the later wiring unwired. Now `Array.from(...)`. Both caught by the
Playwright pass (see Verification).

### Bug 1 + 3 — default standing pose: CONCLUDED (passively infeasible)
The fix would have been a `Scene.rest_state` snapshot (qpos+qvel+ctrl)
holding a standing pose. Research (full-size, corrected probe recipe)
concluded **no passive standing pose holds in this model**:

Verified facts (all re-runnable via the probe recipes below):
- Neutral keyframe: root `[0.496, 0, 2.1]` m, quat identity, **dorsal-up**
  (wing geoms z 2.229 > thorax 2.085). The "prone" default is the
  belly-down landing from the 0.8 m spawn — anatomically the right side up,
  but the fly rests on the ground and creep-rotates indefinitely
  (|v_root| 0.1–0.33 m/s at 10 s; does not decay). A 180°-about-x "supine"
  spawn settles quieter (|v_root| ~0.04) but is the truly upside-down pose
  (wings on the ground) — NOT shippable.
- Model mass is **0.001 kg total** (biological scale). Actuator force
  limits (±30) are not the constraint (kp=50 position actuators).
- Standing geometry: the 18 pitch-DOF offsets (coxa +0.4, femur +1.4,
  tibia −0.4) extend the legs so the feet land 1.23 m below the root at
  world x −0.35…−1.36 — all head-side of the root (the COM sits root-side,
  x ≈ +0.5 before centering) → the COM is **behind the support polygon** →
  the fly tips head-down. Even after centering the COM over the stance
  (root x shift −0.505 m): tumbles (root drifts 2.7 m, body rolls 58°
  head-down, hind legs flung to z 1.17 m). The unactuated coxa-yaw/roll
  DOFs give ZERO roll stiffness (laterally neutral stance — no restoring
  force), so any perturbation rolls the body onto the ground.
- Actuating ALL 48 leg DOFs (`ActuatedDOFPreset.LEGS_ONLY`, 66 actuators
  total incl. body) changes nothing: same 0.5273 m endpoint, worse
  flailing (|qvel| 1.6–4.5). A ghost-pinned-root static test still shows
  the pitch joints drifting off target (error ≈ 0.167 × femur_offset rad
  after 3 s) — but treat the pinned test as suggestive (per-step teleport
  pinning can inject artifacts).
- Knife-edge contact: initializing with feet EXACTLY at ground contact
  (z=0.000) explodes (kN-scale contact impulse on a 1 g body, solref
  2e-4 s) — spawn with a few cm clearance, like the real service does.
- The M7.3 "root ~1.52 m, stable at rest" grid-search claim is not
  reproducible under any reasonable reading (spawn drop, initialized-
  standing with correct foot height, COM centered, 18 vs 48 DOF, forcerange
  30 vs 300). Treat the original note as an artifact (likely a different
  measurement setup that no longer exists).
- The simplified mesh is **not valid for stance research**: its tarsus
  geoms don't extend (foot min-z 1.23 m below root full-size vs 0.13 m
  simplified → standing height 2.23 m vs 0.13 m). Use FULLSIZE.

Next milestone (research, NOT service plumbing): a **closed-loop righting /
stance controller** — the 6 built-in contact sensors (`ground_contact_*`,
force type, already in telemetry) + per-DOF ctrl to keep the COM over the
stance: (1) detect the roll/yaw drift via root quat + contact asymmetry;
(2) apply corrective ctrl on the 18 pitch DOFs (and, if needed, build with
`LEGS_ONLY` to get the 48 leg DOFs incl. coxa yaw/roll); (3) start from the
standing-initialized snapshot with a clearance drop. The sandbox
(`set_state` + per-DOF ctrl + telemetry + force + vision) is the toolset.
A quasi-static gait (the M7.3 open task) builds on the same controller —
the walk gait's baseline improves automatically once the fly can stand.

## Ordered worklist for the next session

1. Commit the docs stop point (this file + PLAN M7.3 + README note).
2. (Research) Righting/stance controller per the recipe above — TDD style:
   a pure controller function (state → ctrl deltas) unit-tested first,
   then wired as a new mode or a `--righting` flag (CLI + SimThread hook
   after `_loco_apply`). Keep it off by default until it holds a 30 s
   standing settle (root z stable within cm, |v_root| < 0.05, no body
   contact, FULLSIZE).
3. When a standing rest state exists: wire `Scene.rest_state` (qpos/qvel/
   ctrl snapshot; `reset()` restores it; `neutral_actuators()` returns its
   ctrl), computed in `build_scene` for POSITION builds; the mode-row
   `neutral` should restore the standing rest state (grep call sites of
   `neutral_actuators` in app.py / simulation.py / locomotion.py first).
4. Full verification (below) after any change.

## Ops (current state)

- 8765 running (pid file `/tmp/flygym-stream.pid`, log
  `/tmp/flygym-stream.log`), current code, NO `--overlay depth` (~30 fps;
  the overlay costs ~4× frame time). Restart recipe:
  `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl nohup .venv/bin/python -m flygym_stream.cli serve --bind 0.0.0.0 --insecure --port 8765 >>/tmp/flygym-stream.log 2>&1 &`
  (env vars MUST precede python start; static `web/` serves from disk per
  request — UI changes don't need a restart).
- **Single-client WS: the user's browser tab holds the slot** and
  auto-reconnects every ~1 s — a second client gets close 1013. For
  verification, run a dedicated instance on another port
  (`--bind 127.0.0.1 --port 8790`), never contend with the user's tab.
- **pkill gotcha**: `pkill -f "port 8765"` kills the calling shell too
  (its cmdline contains the pattern). Use the pid file /
  `pgrep -af "flygym_stream.cli serve" | grep -v "bash -c"`.
- MuJoCo 3.9 binding notes (bitten this session): no `m.jnt_posadr`
  (use `m.jnt_qposadr`); `MjContactList` = field arrays —
  `d.contact.geom[i]` = [geom0, geom1], body via `m.geom_bodyid[g]`;
  no `d.body_com_pos` / `d.body_xmat` (use `d.subtree_com[0]` for the
  whole-fly world COM — a 3-vector, don't `float()` it directly);
  `m.body_mass` is the mass array (fly total 0.001 kg).
- `encode_jpeg(frame, quality)` REQUIRES the quality arg.

## Verification playbook

- `uv run pytest -q` → 148; `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run pytest -m gpu -q`
  → 18; `uv run pytest -m network -q` → 18 (~2 min); `node scripts/ui_smoke.cjs`
  → 30 checks.
- Real-browser (the ONLY way to catch browser-scoping bugs — the fake-DOM
  ui_smoke concatenates scripts into one Array-based scope; a real
  browser's `NodeList`/`window`/scope semantics differ):
  `cd /tmp/uibug && npx playwright install chromium` (cached in
  `~/.cache/ms-playwright` if present) then a node script:
  `chromium.launch({headless:true})` → `page.on("pageerror")` → goto a
  DEDICATED port → wait for `window.Sandbox` → drive buttons/sliders,
  assert HUD/telemetry strip, `flap.isDisabled()` etc. Scripts
  `verify_ui2_8790.cjs` (legs-only: no #btn-neutral, 42 sliders, mode-row
  neutral round-trip, disabled flap + title) and `verify_dofall.cjs`
  (126-DOF: flap enabled, virtualized table, hidden classic sliders) both
  passed 10/10 and 6/6.
- Live serve check: `curl -s localhost:8765/healthz` (now carries
  `sim_running`); WS frames = bare JPEG (main) + `V1`-prefixed JPEG
  (vision builds); JSON text per protocol v1; mode message field is
  `value` (`{"type":"mode","value":"walk"}`).

## Probe recipes (standing pose — run from repo root)

Build (FULLSIZE only; legs-only or `LEGS_ONLY` as needed):
```python
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, ContactBodiesPreset, JointPreset, Skeleton
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, MeshType, NeuroMechFly
from flygym.utils.math import Rotation3D
import mujoco as mj
fly = NeuroMechFly(mesh_type=MeshType.FULLSIZE)
sk = Skeleton(joint_preset=JointPreset.ALL_BIOLOGICAL, axis_order=AxisOrder.YAW_PITCH_ROLL)
fly.add_joints(sk, KinematicPosePreset.NEUTRAL)
fly.add_actuators(sk.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY),
                  ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL,
                  kp=50.0, ctrlrange=(-3.14, 3.14), forcelimited=True, forcerange=(-30.0, 30.0))
fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
fly.colorize()
world = FlatGroundWorld()
world.add_fly(fly, (0.0, 0.0, 0.8), Rotation3D("quat", (1.0, 0, 0, 0)),
              bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)
m, d = world.compile()
key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
scratch = mj.MjData(m); mj.mj_resetDataKeyframe(m, scratch, key)
neutral = scratch.ctrl.copy()   # == keyframe joint angles exactly (delta 0.0)
```
- Actuator names (`nmf/` prefix): coxa_pitch
  `c_thorax-{leg}_coxa-pitch-position`; femur_pitch
  `{leg}_coxa-{leg}_trochanterfemur-pitch-position`; tibia_pitch
  `{leg}_trochanterfemur-{leg}_tibia-pitch-position`; legs lf lm lh rf rm rh.
  Joint qpos adr: `j=int(m.actuator_trnid[aid,0]); adr=int(m.jnt_qposadr[j])`.
- Foot geoms: `nmf/{leg}_tarsus1..5` (30 total) — the stance/foot
  reference. Standing height: forward with the standing joints,
  `minz = min(geom_xpos[i][2] + min(geom_size[i][2]) over foot geoms)`,
  then `root_z = current_root_z - minz + clearance` (clearance ≥ 0.05 m —
  exact zero-contact inits explode: solref 2e-4 s impulse on a 1 g body).
- COM (world): `d.subtree_com[0]` (3-vector). Stance centroid: mean of the
  6 `nmf/{leg}_tarsus3` geom x's. Centering: `root_x += centroid_x - com_x`
  (necessary but not sufficient — the roll is laterally neutral).
- Settle metric: sample every 1 s for ≥10 s: root z, |v_root|
  (qvel[0:3]), |qvel|, body contact (contact geom → `m.geom_bodyid` →
  body name contains `c_thorax`/`abdomen`). Full-size 10 s ≈ 1–2 min wall.

## Key file map

`protocol.py` (wire truth; mode field = `value`) · `scenes.py`
(SceneConfig/build_scene; MSK + vision paths) · `scene.py` (state-write
surface, `reset()`, `neutral_actuators()`, `vision` attr) ·
`simulation.py` (Backend/SimThread; rejections now logged non-fatally) ·
`locomotion.py` (patterns; `LEG_ACTUATOR_NAMES`; flap requires wing
actuators) · `telemetry.py` (contact sensors `ground_contact_{leg}_leg`) ·
`vision.py` (V1 framing) · `stream.py` · `app.py` (WS + healthz with
`sim_running`) · `cli.py` · `web/index.html` (single neutral in the mode
row) · `web/app.js` (MSG map — `MSG.neutral` kept as protocol mirror) ·
`web/app_sandbox.js` (`window.Sandbox`; `state.wingsPresent` flap gate) ·
`scripts/ui_smoke.cjs` (30 checks; async harness promise fix).

## Git (newest first)

`497f694` UI: single neutral + flap gating + ui_smoke async-harness fix
(30 checks; Playwright both directions) · `6088f19` Sim: rejection
survival + healthz `sim_running` (flap crash) · `0cbc73c` docs EOD stop ·
`a280311` UI browser-scoping fix · `21a8e76` docs · `226f674` M7.6 vision ·
`4d837cc` WIP stop · `e8dbe48` M7.6 msk · `4fa78cc` M7.5 · `eab746b` M7.4 ·
`1493b9a` M7.3 · `af27734` M7.2 · `a185829` M7.1. Commit identity:
`git -c user.email=dev@neurofly.local -c user.name=dev commit`.
