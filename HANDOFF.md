# HANDOFF — flygym-stream, Phase 7 research sandbox (2026-10-15)

New session: read this first, then PLAN.md §Phase 7 (milestones M7.1–M7.6
have ✅ status + impl notes), then the last ~8 git log entries (every
commit message is a verification record).

## Repo / environment (unchanged house rules)

- Project: `/home/adam/neurofly/flygym-stream` (venv `.venv`, Python 3.12,
  `uv`); flygym at `/home/adam/neurofly/flygym` (editable dep — do NOT
  modify flygym).
- GPU tests need `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` exported BEFORE
  pytest starts. Sudo password: `rocknrollmcds` (not needed normally).
- Test suites: `uv run pytest -q` (default, ~145), `-m gpu` (17),
  `-m network` (17, slow ~2 min, mesh cache), `node scripts/ui_smoke.cjs`
  (19 UI checks). House rule: strict TDD (failing test first), every file
  < 500 lines (meta-test `tests/test_line_limits.py`), DRY (protocol.py =
  wire truth, one scene builder, one stepping interface), green suite +
  commit per milestone.
- Git identity: `git -c user.email=dev@neurofly.local -c user.name=dev commit`.

## Status: M7.1–M7.5 done, M7.6 musculoskeletal done, M7.6 vision IN PROGRESS

Committed & verified: M7.1 scenes/CLI variants (`a185829`), M7.2 state
control (`af27734`), M7.3 locomotion (`1493b9a`), M7.4 telemetry/
model-info/logging (`eab746b`), M7.5 sandbox UI (`4fa78cc`), M7.6
musculoskeletal + telemetry-starvation fix (`e8dbe48`).

**Current working tree = TDD stop point for vision (`--vision`)** —
committed as WIP so the tree is clean; the suite has exactly ONE
expected failure:

```
uv run pytest -q   -> 1 failed, 144 passed   (test_ws_vision_frames_pass_through)
```

DONE (in tree): `src/flygym_stream/vision.py` (VisionFeed: sim-time paced
at half output fps; `render_frame()` = `b"V1"` + JPEG of both fisheye eyes
stacked via `flygym.Simulation.get_raw_vision(fly_name)`);
`Scene.__init__(..., vision=None)` attribute.

RED (written, failing — these are the spec):
- `tests/test_scenes.py::test_vision_scene_builds_with_eye_cameras`
  [network]: `SceneConfig(vision=True)` -> `m.ncam == 3` (tracking +
  l_eye + r_eye), `scene.vision is not None`, stepping works.
- `tests/test_scenes.py::test_vision_frame_renders_eye_images` [gpu]:
  VisionFeed on a real vision scene -> bytes start `V1` + FFD8, PIL
  decodes, not black.
- `tests/test_app.py::test_ws_vision_frames_pass_through`: V1-prefixed
  binary flows over the WS alongside main frames.

REMAINING (exact order, details in PLAN.md M7.6 🚧 block):
1. `scenes.py` build path: `config.vision` (and not musculoskeletal) ->
   `sim_wrapper = flygym.Simulation(world)` (`from flygym import
   Simulation as FlygymSimulation`; its `__init__` calls
   `world.compile()`), use `sim_wrapper.mj_model/.mj_data`, pass
   `vision=sim_wrapper` to `Scene(...)`. Reject musculoskeletal+vision
   with ValueError. (Everything else in build_scene stays; the
   non-vision path still compiles via `world.compile()` as today.)
2. `simulation.py` SimThread: kwargs `vision=None, vision_queue=None`;
   in `tick()` after `_maybe_telemetry()`: if `self._vision is not None
   and self._vision.render_as_needed(self._backend.data):
   self._vision_queue.put(self._vision.render_frame())`. Sim thread only
   (GL context binds to creating thread — same reason OffscreenRenderer
   is lazy).
3. `app.py`: `Producer.vision_queue: object | None = None`; in the send
   loop, right after the telemetry check: `vframe = await
   run_sync(producer.vision_queue.get_nowait)` -> `await
   ws.send_bytes(vframe)` (V1 prefix marks it; main frames stay bare
   JPEG — no protocol.py change).
4. `cli.py` serve: when `scene.vision is not None`: `vision_feed =
   VisionFeed(scene, config)`, `vision_frames = FrameQueue()`, pass
   `vision=vision_feed, vision_queue=vision_frames` to SimThread and
   `vision_queue=vision_frames` to Producer.
5. UI: `web/index.html` `<img id="vision" ...>` overlay in `#video-pane`
   (hidden by default); `web/style.css` style it (~180 px, top-right);
   `web/app.js`: binary handling — read first 2 bytes (`new
   Uint8Array(await ev.data.slice(0,2).arrayBuffer())`), if `V1`
   (0x56 0x31) route `new Blob([ev.data.slice(2)])` to the vision img,
   else existing `onFrame`. Keep `node scripts/ui_smoke.cjs` green.
6. Verify: full suites (default must be all green again), GPU test,
   live `serve --vision --bind 127.0.0.1 --port <free>` smoke (status +
   model-info + V1 vision frames + main frames), update HANDOFF/PLAN
   (mark vision ✅), commit M7.6.

## Known facts that save time (verified 2026-10-15)

- `flygym.Simulation(world)` compiles the world itself and exposes
  `sim.mj_model` / `sim.mj_data`; `get_raw_vision(fly_name)` returns
  `(2, h, w, 3)` [left, right] fisheye-corrected frames; eye cameras are
  added by `NeuroMechFly.add_vision()` (already called in build_scene
  when `config.vision`); MSK fly also has `add_vision` but MSK+vision is
  experimental — rejected by design for now.
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
- Telemetry flows only while unpaused; the WS send loop must check the
  telemetry/vision queues EVERY iteration (a continuously-full frame
  queue otherwise starves them — regression test
  `test_ws_telemetry_not_starved_by_frames`).
- The running service (if any) is from BEFORE Phase 7 — restart it to
  get model-info/telemetry/sandbox UI: `MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl .venv/bin/python -m flygym_stream.cli serve
  --bind 0.0.0.0 --insecure --port 8765` (log to /tmp/flygym-stream.log).
  Static web/ files are served from disk per request (no restart for UI
  changes); server-code changes need a restart.
- MJWarp (Phase 6) refuses NoSlip models (incl. every NMf preset) —
  `--backend cpu` is the default; MSK model is NoSlip-free so
  `--backend auto` may pick GPU for it (untested).

## File map (Phase 7 additions)

`scenes.py` (SceneConfig/build_scene, MSK path), `scene.py` (state-write
surface + `vision` attr), `protocol.py` (v1 additive messages:
set_state/spawn/force/param/keyframe/adhesion/mode/loco/log/info;
WIRE_TYPES), `simulation.py` (apply_state dispatch, loco + telemetry
hooks, Backend.data), `locomotion.py` (pattern engine), `telemetry.py`
(queue/model-info/gatherer/JsonlLogger), `vision.py` (WIP feed), `app.py`
(model-info, telemetry, logging, /log), `cli.py` (all flags),
`web/app_sandbox.js` + `scripts/ui_smoke.cjs`.
