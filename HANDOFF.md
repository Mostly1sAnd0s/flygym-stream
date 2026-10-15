# HANDOFF — flygym-stream (as of 2026-07-21, after the default-rest-state fix)

**Status**: The user's 2nd bug report (default fly lying on its back and
slowly rotating) is **FIXED and committed**: the fly now starts in the
neutral keyframe's upright pose, dips ~1 m in ~0.5 sim s and **stands on
its six feet, dorsal-up, stationary** (sub-mm). The earlier "no passive
standing pose exists" conclusion was refuted — it was based on the wrong
stance geometry (see below). The remaining open item is gait QUALITY
(walking works — ~0.5 m in 2 sim s from the standing rest — but is not
quasi-static), which stays a research task. All suites green; 8765
serving the fixed code.

## The fix (this session)

### What the user saw, and why
The old default overwrote the neutral keyframe root with a spawn at
z=0.8 m. The fly fell, landed belly-down, and the leg actuators' reaction
torque (±30 force on a 1 g thorax, zero passive yaw stiffness) drove a
~1 Hz yaw limit cycle — "lying on its back and slowly rotating". (Probe:
default spawn ends at `wing_z − thorax_z = −0.25 m` = on its back, +1 m
x-drift in 8 s.)

### The correct stance (the refuted conclusion)
The model's OWN neutral-ctrl pose is a self-stable stance: from the
keyframe (root 2.1 m, feet 0.84 m up, no contacts) the fly lands in ~0.5 s
on all six feet (tarsus5 geoms at ground level), root z = 1.081 m, COM
x = −0.20 INSIDE the support polygon (feet x −1.53…+1.38), dorsal up
(wing z − thorax z = +0.215), z/x/y constant to the millimetre over 10 s,
body oscillation 0.01–0.2 mm p-p. The earlier "infeasible" evidence used
manually extended joint offsets whose feet ALL sit head-side (COM behind
the support polygon) — a different, genuinely unstable pose. Do not
re-derive; the corrected evidence chain is in PLAN.md M7.3 notes.

### Mechanism: the rest anchor
- `Scene.rest_anchor` (default True on free-root scenes): while active,
  `step()` zeroes the root's angular velocity (`qvel[3:6]`) after every
  `mj_step` — the root holds its pose, appendages stay fully dynamic.
  Kills the yaw limit cycle (≤0.01° over 10 s; position held by the foot
  contacts). No tuning, no force units, no frame math.
- Gating: ON by default and in `neutral` mode; OFF after `spawn` /
  `set_state` (explicit state writes take over the physics); `reset()` /
  `mode neutral` / `neutral` re-enable + full restore.
- `Scene.neutral_restore()`: full qpos/qvel/ctrl from the neutral keyframe
  (plain default state when the model has no such keyframe) + home root +
  anchor on, sim clock untouched. Both the `Neutral` message and
  `mode neutral` use it — the UI's neutral button returns the fly to its
  standing reference pose, wherever it is.
- `SceneConfig.spawn` default → None (CLI `--spawn` default → None): the
  root keeps the keyframe pose; an explicit spawn still overrides. The
  tethered world (pinned mocap root) falls back to (0, 0, 0.8) when no
  spawn is given.
- `Backend.set_rest_anchor(active)`: MuJoCoBackend routes to the Scene;
  base class no-op (GPU v1 leaves the anchor inert — the CPU shadow scene
  is not stepped).
- SimThread: `Mode` dispatch calls `set_rest_anchor(mode == "neutral")`
  and, for neutral, `apply_input(Neutral())` (full restore).
- **Pacing re-base on reset** (found live): `reset` rewinds sim time to
  the keyframe's 0; the telemetry `_next_telemetry` and vision `_last`
  pacing states must re-base with it, else telemetry/vision go silent
  until sim time crawls back up (observed live: telemetry dead for the
  whole pre-reset duration after a WS `reset`). `SimThread.reset()` now
  re-bases both (`VisionFeed.reset()` added).

### Rejected alternative (do not retry)
Force-based "hover" (xfrc_applied gravity compensation + damping): the 42
position actuators fling the 1 g thorax regardless (per-step root impulses
up to ±30 force vs 0.01 N weight); the fly landed on its back or launched
metres. Also learned empirically (minimal-model test): MuJoCo 3.x
`xfrc_applied` spatial forces act in the **world frame**, not the body
frame (a rotated body with f_body=(0,0,W) cancelled world gravity).
Also: mesh geoms invalidate `geom_size`-based clearance math (a 6 cm
penetration was invisible to `geom_xpos + geom_size` min-z) — use
iterative `ncon == 0` lifts if ever needed.

## Verification (all green at this stop point)

- `uv run pytest -q` → **157** (was 148; +9 new tests: rest-anchor
  holds-root / state-write-disables / neutral_restore full-state,
  backend routing, mode→anchor driving, mode-neutral full restore,
  default-spawn-None keyframe root, default-rest-stands (network),
  reset rebases telemetry pacing, vision pacing reset)
- `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl uv run pytest -m gpu -q` → 18
  (walk headline test now walks FROM the standing rest and its `at_rest`
  gate is position-stability based — a standing fly has brief per-leg-
  strike velocity spikes (|v| up to 0.7 m/s) but its position is held to
  the mm)
- `uv run pytest -m network -q` → 20 (~2 min)
- `node scripts/ui_smoke.cjs` → 30 checks
- Live (dedicated instance, full WS protocol): REST standing (0.625, 0,
  1.081) stable · WALK >50 mm (reached ~0.6 m) · NEUTRAL returns to the
  reference stance (0.624, 0, 1.081) · SPAWN (0,0,1.0) honored + free
  fall + re-stands · RESET returns to the standing rest with telemetry
  alive immediately (re-base fix).

## Ops (current state)

- 8765 running the fixed code (pid `/tmp/flygym-stream.pid`, log
  `/tmp/flygym-stream.log`), no `--overlay depth`. **The user just needs
  to reload their browser tab** (their tab held the single-client WS slot
  on the old process; it auto-reconnects to the new one).
- 8792 was the dedicated verification instance (pid
  `/tmp/flygym-8792.pid`) — kill it when done
  (`kill $(cat /tmp/flygym-8792.pid)`).
- **pkill gotcha (bitten again this session)**: `pkill -f "flygym_stream.cli
  serve"` kills the calling shell AND the user's 8765 (the shell's own
  cmdline contains the pattern). Use the pid files, or the bracket
  trick: `pkill -f "flygym_stream[.]cli serve"`.
- Single-client WS: second concurrent client gets close 1013; a crashed
  verification script leaves its socket held until the process dies — if
  a fresh client gets status but no telemetry/frames, check for an
  orphaned client first.

## MuJoCo 3.9 binding notes (bitten across sessions)

- No `m.jnt_posadr` (use `m.jnt_qposadr`); `d.contact.geom[i]` = [g0, g1],
  body via `m.geom_bodyid[g]`; no `d.body_com_pos`/`d.body_xmat`
  (`d.subtree_com[0]` = whole-body world COM, a 3-vector); `m.body_mass`
  is the mass array (fly total 0.001 kg).
- `xfrc_applied` (nbody×6) forces act in the **world frame** (empirical);
  it IS writable through the Python bindings.
- MJCF `<keyframe>` elements do NOT accept `name`/`qpos` attributes in the
  compiled schema — flygym creates named keyframes programmatically
  (`mjcf_root.add_key(name="neutral")` then `.qpos/.ctrl`). Tiny test
  models cannot have named keyframes; `Scene` falls back to the plain
  default state when the keyframe is absent.
- Model units: the fly is ~3000 (m) wide; gravity −9810 (m/s²) — i.e.
  1000×-scaled lengths with 1000× gravity (biological time scaling);
  mass in kg (total 0.001). Root body = `nmf/c_thorax` (body 1), free
  joint 0, qveladr 0:6.

## Next milestone (research, NOT service plumbing)

**Gait quality from the standing rest** (the fly now walks ~0.5 m / 2 sim
s right-side-up, but not quasi-statically). TDD style: a pure controller
function (state → ctrl deltas) unit-tested first, then wired as a new
mode or a `--gait` flag (CLI + SimThread hook after `_loco_apply`).
Tools: the 6 built-in contact sensors (`ground_contact_*`, force type, in
telemetry), root quat, per-DOF ctrl, `set_state`, `force`, vision.
Success bar: a ≥10 s walk at ~0.2 m/s with no body contact, root z stable
within cm, and step-like (not kick-like) translation, FULLSIZE mesh.

## Key file map (updated)

`protocol.py` (wire truth; mode field = `value`) · `scenes.py`
(SceneConfig; **spawn default None**; tethered spawn fallback) ·
`scene.py` (**rest anchor** in `step()`, `set_rest_anchor`,
`neutral_restore()`, state writes disable the anchor, `reset()`
re-enables) · `simulation.py` (Backend/SimThread; `set_rest_anchor`
routing; `Neutral` → full restore; `Mode(neutral)` → anchor + full
restore; `reset()` re-bases telemetry/vision pacing) · `locomotion.py`
(patterns; flap requires wing actuators) · `telemetry.py` (contact
sensors; `TelemetryQueue` single-slot) · `vision.py` (V1 framing;
`reset()` pacing re-base) · `stream.py` · `app.py` (WS + healthz with
`sim_running`) · `cli.py` (`--spawn` default None) · `web/*` (unchanged
this session) · `scripts/ui_smoke.cjs` (30 checks) ·
`tests/conftest.py` (`FREE_TINY_XML` free-root fixture; FakeBackend
`set_rest_anchor`).

## Git (newest first)

(pre-commit of this stop: rest-state fix — scene anchor + spawn default +
mode/neutral full restore + pacing re-base; suites 157/18/20/30) ·
`e8e9b89` docs: standing-pose research concluded (SUPERSEDED by this
session's corrected stance finding) · `497f694` UI: single neutral + flap
gating · `6088f19` Sim: rejection survival + healthz `sim_running` ·
`0cbc73c` docs EOD stop · `a280311` UI browser-scoping fix · `21a8e76`
docs · `226f674` M7.6 vision. Commit identity:
`git -c user.email=dev@neurofly.local -c user.name=dev commit`.
