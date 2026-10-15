"""GL-dependent end-to-end tests for flygym_stream.app (PLAN.md Phase 5).

Same harness as test_app.py, but with the real scene + OffscreenRenderer:
these are marked ``gpu`` and skip unless MUJOCO_GL=egl /
PYOPENGL_PLATFORM=egl are exported before pytest starts (mujoco reads the
env at import time). Shared helpers are imported from test_app (DRY).
"""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from flygym_stream.app import Producer, create_app
from flygym_stream.simulation import SimThread
from flygym_stream.stream import FrameQueue, InputQueue
from test_app import HELLO, recv_frame, recv_status, wait_for


@pytest.mark.gpu
def test_full_stack_real_scene(tmp_config) -> None:
    """Whole real pipeline through the app: scene -> backend -> renderer ->
    sim thread -> WebSocket -> status + real JPEG (GL required)."""
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import make_backend

    scene = build_default_scene()
    backend = make_backend(scene, tmp_config)
    renderer = OffscreenRenderer(scene, tmp_config)
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(backend, renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, backend, sim))
    try:
        with TestClient(app) as client:
            assert client.get("/healthz").json()["ok"]
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                status = recv_status(ws)
                assert status["n_actuators"] == 42
                assert status["backend"] == "cpu"
                frame = recv_frame(ws)
                assert frame[:2] == b"\xff\xd8"  # a real JPEG
                assert len(frame) > 1000
                print(f"\nfull stack: status={status} frame={len(frame)} bytes")
    finally:
        sim.stop()
        renderer.close()


@pytest.mark.gpu
def test_full_stack_orbit_and_actuators_change_pixels(tmp_config) -> None:
    """Control -> pixels: orbit + actuators over the WS visibly change the
    real rendered frames (GL required)."""
    import numpy as np
    from PIL import Image

    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import make_backend

    def mean(frame: bytes) -> float:
        with Image.open(io.BytesIO(frame)) as img:
            return float(np.asarray(img).mean())

    def changed_frac(a: bytes, b: bytes, thresh: int = 10) -> float:
        """Fraction of pixels differing by > thresh (whole-fly view makes
        frame *means* insensitive to pose changes; the fly is ~1/3 of the
        frame and the rest is background)."""
        with Image.open(io.BytesIO(a)) as ia, Image.open(io.BytesIO(b)) as ib:
            da, db = np.asarray(ia).astype(int), np.asarray(ib).astype(int)
        return float((np.abs(da - db).max(axis=2) > thresh).mean())

    scene = build_default_scene()
    backend = make_backend(scene, tmp_config)
    renderer = OffscreenRenderer(scene, tmp_config)
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(backend, renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, backend, sim))
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                recv_status(ws)
                first = mean(recv_frame(ws))
                ws.send_text(json.dumps(
                    {"type": "orbit", "azimuth": 200.0, "elevation": 10.0,
                     "distance": 6000.0}))
                wait_for(lambda: any(
                    type(m).__name__ == "Orbit" for m in backend.inputs))
                recv_frame(ws)  # discard a frame possibly rendered pre-orbit
                second_img = recv_frame(ws)
                second = mean(second_img)
                assert abs(first - second) > 1.0, (first, second)
                ws.send_text(json.dumps({"type": "pause"}))
                wait_for(lambda: sim.paused)
                ws.send_text(json.dumps(
                    {"type": "actuators", "values": [1.0] * 42}))
                t0 = backend.time
                ws.send_text(json.dumps({"type": "step", "n": 300}))
                # the step control drains inputs first, then advances exactly
                # 300 substeps and pushes one forced (post-pose) frame; the
                # send loop may still be delivering an in-flight pre-step
                # frame, so take the first frame showing the pose change
                wait_for(lambda: backend.time >= t0 + 299 * backend.dt)
                post = None
                for _ in range(10):
                    cand = recv_frame(ws)
                    if changed_frac(second_img, cand, thresh=20) > 0.05:
                        post = cand
                        break
                assert post is not None, "no post-pose frame arrived"
                print(f"\ncontrol->pixels: {first:.1f} -> {second:.1f} (orbit)"
                      f" -> {changed_frac(second_img, post, thresh=20):.3f} changed (actuators)")
    finally:
        sim.stop()
        renderer.close()


@pytest.mark.gpu
def test_gpu_state_messages_change_real_physics(tmp_config) -> None:
    """M7.2 full stack: the six research-sandbox state messages over the WS
    change the *real* physics of the real fly scene (paused so assertions
    race no stepping; GL required)."""
    from flygym_stream.protocol import Adhesion, Force, Keyframe, Param, SetState, Spawn
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scenes import SceneConfig, build_scene
    from flygym_stream.simulation import make_backend

    scene = build_scene(SceneConfig(adhesion=True))  # 42 + 6 = 48 actuators
    backend = make_backend(scene, tmp_config)
    renderer = OffscreenRenderer(scene, tmp_config)
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(backend, renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, backend, sim))
    d = scene.mj_data
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                status = recv_status(ws)
                assert status["n_actuators"] == 48
                ws.send_text(json.dumps({"type": "pause"}))
                wait_for(lambda: sim.paused, timeout=5.0)

                def applied(cls: type) -> None:
                    wait_for(lambda: any(isinstance(m, cls) for m in backend.inputs),
                             timeout=5.0)

                nu = scene.mj_model.nu
                ws.send_text(json.dumps({"type": "set_state",
                                         "ctrl": [0.31] * nu}))
                applied(SetState)
                assert d.ctrl[0] == 0.31

                ws.send_text(json.dumps(
                    {"type": "spawn", "pos": [0.0, 0.0, 1.5],
                     "quat": [1.0, 0.0, 0.0, 0.0]}))
                applied(Spawn)
                assert abs(d.qpos[2] - 1.5) < 1e-6

                ws.send_text(json.dumps(
                    {"type": "force", "body": "nmf/c_thorax",
                     "force": [0.0, 0.0, 5.0]}))
                applied(Force)
                import mujoco as mj
                body = mj.mj_name2id(scene.mj_model, mj.mjtObj.mjOBJ_BODY,
                                     "nmf/c_thorax")
                assert d.xfrc_applied.reshape(-1)[6 * body + 2] == 5.0

                ws.send_text(json.dumps(
                    {"type": "param", "name": "gravity",
                     "value": [0.0, 0.0, 0.0]}))
                applied(Param)
                assert list(scene.mj_model.opt.gravity) == [0.0, 0.0, 0.0]

                ws.send_text(json.dumps(
                    {"type": "adhesion", "leg": "lf", "state": 1}))
                applied(Adhesion)
                assert d.ctrl[nu - 6] == 1.0

                ws.send_text(json.dumps(
                    {"type": "keyframe", "name": "neutral"}))
                applied(Keyframe)
                scratch = mj.MjData(scene.mj_model)
                mj.mj_resetDataKeyframe(scene.mj_model, scratch,
                                        mj.mj_name2id(scene.mj_model,
                                                       mj.mjtObj.mjOBJ_KEY,
                                                       "neutral"))
                assert abs(d.qpos[2] - float(scratch.qpos[2])) < 1e-6
                # the keyframe's own root is restored (no spawn override)
    finally:
        sim.stop()
        renderer.close()


@pytest.mark.gpu
def test_gpu_mode_walk_moves_root_more_than_50mm_in_2sim_s(tmp_config) -> None:
    """M7.3 headline test: `mode walk` over the WS drives the built-in
    tripod gait and the fly's free root translates > 50 mm within
    2 sim seconds (GL required)."""
    from flygym_stream.locomotion import LocController
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import make_backend

    scene = build_default_scene()
    backend = make_backend(scene, tmp_config)
    renderer = OffscreenRenderer(scene, tmp_config)
    frames, inputs = FrameQueue(), InputQueue()
    loco = LocController(scene.mj_model, scene.neutral_actuators())
    sim = SimThread(backend, renderer, frames, inputs, tmp_config, loco=loco)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, backend, sim))
    d = scene.mj_data
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                recv_status(ws)
                # Default rest: the fly starts at the neutral keyframe
                # (~2.1 m up), lands on its feet (~0.5 sim s) and stands
                # (the rest anchor holds the orientation). A standing fly
                # has brief per-leg-strike velocity spikes, so "at rest"
                # means POSITION stability, not |qvel| < eps.
                import numpy as np
                rest_t, rest_p = None, None

                def at_rest() -> bool:
                    nonlocal rest_t, rest_p
                    now = time.monotonic()
                    p = d.qpos[0:3]
                    if rest_t is None or now - rest_t >= 0.4:
                        rest_t, rest_p = now, p.copy()
                        return False
                    return (float(np.abs(p - rest_p).max()) < 0.02
                            and float(d.qpos[2]) < 1.6)
                deadline = time.monotonic() + 20.0
                while not at_rest() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert at_rest(), "fly did not settle before walking"
                x0, z0 = d.qpos[0], d.qpos[2]
                ws.send_text(json.dumps({"type": "mode", "value": "walk"}))
                wait_for(lambda: loco.mode == "walk", timeout=5.0)
                t0 = d.time
                moved = False
                deadline = time.monotonic() + 15.0  # 2 sim s + margin
                while time.monotonic() < deadline:
                    if d.time - t0 >= 2.0:
                        break
                    if abs(d.qpos[0] - x0) > 0.05:
                        moved = True
                        break
                    time.sleep(0.02)
                dx, dz = d.qpos[0] - x0, d.qpos[2] - z0
                sim_s = d.time - t0
                print(f"\nwalk: dx={dx*1000:.1f} mm dz={dz*1000:.1f} mm "
                      f"in {sim_s:.2f} sim s (from rest)")
                assert moved, (f"root moved {dx*1000:.1f} mm x, "
                               f"{dz*1000:.1f} mm z in {sim_s:.2f} sim s")
    finally:
        sim.stop()
        renderer.close()


@pytest.mark.gpu
def test_gpu_telemetry_model_info_and_status_fields(tmp_config) -> None:
    """M7.4 full stack: hello -> status (mode/n_dof) + model-info;
    telemetry messages stream at the configured rate with real MjData
    values (GL required)."""
    from flygym_stream.locomotion import LocController
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import make_backend
    from flygym_stream.telemetry import (
        TelemetryGatherer, TelemetryQueue, TelemetrySink, gather_model_info,
    )

    scene = build_default_scene()
    backend = make_backend(scene, tmp_config)
    renderer = OffscreenRenderer(scene, tmp_config)
    frames, inputs = FrameQueue(), InputQueue()
    loco = LocController(scene.mj_model, scene.neutral_actuators())
    model_info = gather_model_info(scene.mj_model, fly_name=scene.fly_name,
                                   world="flat")
    assert model_info["nq"] == 133 and model_info["nu"] == 42
    tqueue = TelemetryQueue()
    sink = TelemetrySink(
        gatherer=TelemetryGatherer(scene.mj_model, scene.mj_data,
                                   model_info=model_info),
        queue=tqueue, hz=20.0)
    sim = SimThread(backend, renderer, frames, inputs, tmp_config, loco=loco,
                    telemetry=sink)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, backend, sim,
                                          telemetry_queue=tqueue,
                                          n_dof=scene.mj_model.nq,
                                          model_info=model_info))
    d = scene.mj_data
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                status = recv_status(ws)
                assert status["n_dof"] == 133
                assert status["mode"] == "neutral"
                # model-info follows the initial status (binary frames and
                # telemetry may interleave)
                info = None
                deadline = time.monotonic() + 5.0
                while info is None and time.monotonic() < deadline:
                    msg = ws.receive()
                    if msg.get("text"):
                        data = json.loads(msg["text"])
                        if data.get("type") == "model-info":
                            info = data
                assert info is not None, "no model-info after hello"
                assert info["units"] == "mm" and info["nq"] == 133
                assert len(info["joints"]) == 127  # 126 hinges + free root
                assert len(info["actuators"]) == 42
                assert len(info["sensors"]) == 6
                assert "neutral" in info["keyframes"]
                # telemetry: real MjData at 20 Hz
                tel = None
                deadline = time.monotonic() + 5.0
                while tel is None and time.monotonic() < deadline:
                    msg = ws.receive()
                    if msg.get("text"):
                        data = json.loads(msg["text"])
                        if data.get("type") == "telemetry":
                            tel = data
                assert tel is not None, "no telemetry within 5 s"
                assert len(tel["qpos"]) == 133 and len(tel["qvel"]) == 132
                assert len(tel["ctrl"]) == 42
                assert set(tel["contact"]) == {"lf", "lm", "lh",
                                               "rf", "rm", "rh"}
                assert tel["paused"] is False and tel["mode"] == "neutral"
                # matches the live MjData root (free root: qpos[0:3])
                assert tel["root"]["pos"] == pytest.approx(
                    list(d.qpos[0:3]), abs=1e-3)
    finally:
        sim.stop()
        renderer.close()


@pytest.mark.gpu
def test_camera_views_change_rendered_pixels(tmp_config) -> None:
    """M7.7 control -> pixels: 1st-person / 3rd-person / viewcube-front each
    visibly change the real rendered frame, and status reports the active
    cam_mode (GL required). The fly rests at the anchored standing pose, so
    after it settles any big frame change is the camera, not motion."""
    import numpy as np
    from PIL import Image

    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import make_backend

    def changed_frac(a: bytes, b: bytes, thresh: int = 12) -> float:
        with Image.open(io.BytesIO(a)) as ia, Image.open(io.BytesIO(b)) as ib:
            da, db = np.asarray(ia).astype(int), np.asarray(ib).astype(int)
        return float((np.abs(da - db).max(axis=2) > thresh).mean())

    def frame_after_change(ws, before: bytes) -> bytes:
        """The first frame (within ~1 s) that visibly differs from ``before``
        (the fly is at rest, so only a camera change can do this)."""
        for _ in range(30):
            cand = recv_frame(ws)
            if changed_frac(before, cand) > 0.05:
                return cand
        raise AssertionError("no camera-changed frame within 30 frames")

    scene = build_default_scene()
    backend = make_backend(scene, tmp_config)
    renderer = OffscreenRenderer(scene, tmp_config)
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(backend, renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, backend, sim))
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                assert recv_status(ws)["cam_mode"] == "orbit"
                orbit_img = recv_frame(ws)
                # Let the fly settle onto its feet (rest state), so the
                # baseline frame is stable.
                deadline = time.monotonic() + 5.0
                while (time.monotonic() < deadline
                       and changed_frac(orbit_img, recv_frame(ws)) > 0.005):
                    orbit_img = recv_frame(ws)
                # 1st person: camera on top of the head, looking forward
                ws.send_text(json.dumps({"type": "camera", "mode": "first"}))
                wait_for(lambda: scene.camera_mode == "first")
                first_img = frame_after_change(ws, orbit_img)
                status = recv_status(ws)
                assert status["cam_mode"] == "first"
                # 3rd person: trailing behind and above the fly
                ws.send_text(json.dumps({"type": "camera", "mode": "third"}))
                wait_for(lambda: scene.camera_mode == "third")
                third_img = frame_after_change(ws, first_img)
                # Viewcube front: orbit camera snaps to the fly's heading
                ws.send_text(json.dumps({"type": "camera", "view": "front"}))
                wait_for(lambda: scene.camera_mode == "orbit")
                assert scene.orbit[0] == pytest.approx(
                    scene.fly_heading_deg(), abs=1.0)
                front_img = frame_after_change(ws, third_img)
                print(f"\ncamera views: orbit vs first "
                      f"{changed_frac(orbit_img, first_img):.3f}, "
                      f"first vs third {changed_frac(first_img, third_img):.3f}, "
                      f"third vs front {changed_frac(third_img, front_img):.3f}")
    finally:
        sim.stop()
        renderer.close()
