# HANDOFF — flygym-stream, Phase 7 research sandbox (2026-10-15)

New session: read this first, then PLAN.md §Phase 7 (all milestones
M7.1–M7.6 have ✅ status + impl notes), then the last ~8 git log entries
(every commit message is a verification record).

**Phase 7 is complete.** Every NeuroMechFly/flygym capability is
reachable from the browser (see README "Phase 7"). The open *research*
task (not service work): building a quasi-static gait — the v1 open-loop
tripod gait righting-kicks from the prone neutral pose and collapses from
the known standing pose; the sandbox (set_state + per-DOF ctrl +
telemetry + force + vision) exists to build the next one.

## Repo / environment (unchanged house rules)

- Project: `/home/adam/neurofly/flygym-stream` (venv `.venv`, Python 3.12,
  `uv`); flygym at `/home/adam/neurofly/flygym` (editable dep — do NOT
  modify flygym).
- GPU tests need `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` exported BEFORE
  pytest starts. Sudo password: `rocknrollmcds` (not needed normally).
- Test suites: `uv run pytest -q` (default: 145 passed, 1 skipped),
  `-m gpu` (18), `-m network` (18, slow ~2 min, mesh cache),
  `node scripts/ui_smoke.cjs` (23 UI checks). House rule: strict TDD
  (failing test first), every file < 500 lines (meta-test
  `tests/test_line_limits.py`), DRY (protocol.py = wire truth, one scene
  builder, one stepping interface), green suite + commit per milestone.
- Git identity: `git -c user.email=dev@neurofly.local -c user.name=dev commit`.

## Status: M7.1–M7.6 all done (Phase 7 complete)

Committed & verified: M7.1 scenes/CLI variants (`a185829`), M7.2 state
control (`af27734`), M7.3 locomotion (`1493b9a`), M7.4 telemetry/
model-info/logging (`eab746b`), M7.5 sandbox UI (`4fa78cc`), M7.6
musculoskeletal + telemetry-starvation fix (`e8dbe48`), M7.6 vision
`--vision` (`226f674`).

Vision (M7.6, `226f674`): `--vision` builds through
`flygym.Simulation(world)` (it compiles the world; the Scene steps the
SAME MjData `get_raw_vision` renders — ncam==3: `nmf/trackingcam` +
`nmf/l_eye_cam_camera` + `nmf/r_eye_cam_camera`); `VisionFeed` (vision.py)
paces at half output fps on the sim thread (GL context binds to it);
V1-prefixed (0x56 0x31) stacked-eye JPEGs (450x1024) flow on the same WS
binary channel as the bare main JPEGs (no protocol.py change; the send
loop drains the vision queue every iteration, same starvation rule as
telemetry); UI `#vision` overlay revealed on the first V1 frame. The WIP
commit `4d837cc` (TDD stop point with the 3 red spec tests) is its
parent's context — all red tests are green now.

## Known facts that save time (verified 2026-10-15)

- `flygym.Simulation(world)` compiles the world itself, resets to the
  `neutral` keyframe, and exposes `sim.mj_model` / `sim.mj_data`;
  `get_raw_vision(fly_name)` returns `(2, h, w, 3)` [left, right]
  fisheye-corrected uint8 frames (512x450 per eye for the default NMf);
  eye cameras are named `nmf/l_eye_cam_camera` / `nmf/r_eye_cam_camera`
  (the MJCF compiler appends `_camera`); `Simulation` import is
  `from flygym import Simulation`. MSK fly also has `add_vision` but
  MSK+vision is experimental — `build_scene` rejects it with ValueError.
- MuJoCo 3.9 binding quirks: `jnt_type` HINGE=3; no `actuator_acttype`
  (report gaintype/biastype/dyntype; muscle dyntype=4 there);
  `xfrc_applied` is (nbody,6) 2-D; `qfrc_applied` is nv-length
  joint-space; `o_friction`/`o_solref`/`o_solimp`/`o_margin` are the MjOpt
  names; `m.nsensor` not `nsens`; enum names via `mjtJoint(v).name`.
- Fly model (default scene): 133 q / 132 v / 42 actuators; faces **−x**
  (head x < thorax x); neutral keyframe is **prone** (body on ground);
  standing-pose ctrl offsets (coxa-pitch +0.4, femur-pitch +1.4,
  tibia-pitch −0.4, all legs) are statically stable but the v1 open-loop
  gait can't hold them — walking is the open research task.
- Telemetry and vision flow only while unpaused; the WS send loop checks
  the telemetry AND vision queues EVERY iteration (a continuously-full
  frame queue otherwise starves them — regression tests
  `test_ws_telemetry_not_starved_by_frames`,
  `test_ws_vision_frames_pass_through`).
- The running service (if any) was restarted 2026-10-15 with the
  Phase-7 code; restart it whenever server code changes: `MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl .venv/bin/python -m flygym_stream.cli serve
  --bind 0.0.0.0 --insecure --port 8765` (log to /tmp/flygym-stream.log).
  Static web/ files are served from disk per request (no restart for UI
  changes); server-code changes need a restart.
- MJWarp (Phase 6) refuses NoSlip models (incl. every NMf preset) —
  `--backend cpu` is the default; MSK model is NoSlip-free so
  `--backend auto` may pick GPU for it (untested).
- `encode_jpeg(frame, quality)` REQUIRES the quality arg (VisionFeed
  stores `config.jpeg_quality`; the WIP feed once forgot it — a
  `missing 1 required positional argument` TypeError at render time).

## File map (Phase 7 additions)

`scenes.py` (SceneConfig/build_scene, MSK + vision paths), `scene.py`
(state-write surface + `vision` attr), `protocol.py` (v1 additive
messages: set_state/spawn/force/param/keyframe/adhesion/mode/loco/log/
info; WIRE_TYPES), `simulation.py` (apply_state dispatch, loco +
telemetry + vision hooks, Backend.data), `locomotion.py` (pattern
engine), `telemetry.py` (queue/model-info/gatherer/JsonlLogger),
`vision.py` (VisionFeed: V1 + stacked-eye JPEG), `stream.py`
(FrameQueue.get_nowait for the vision queue), `app.py` (model-info,
telemetry, vision send, logging, /log), `cli.py` (all flags),
`web/app_sandbox.js` + `web/app.js` (onBinary V1 routing) +
`scripts/ui_smoke.cjs` (23 checks incl. V1 routing).
