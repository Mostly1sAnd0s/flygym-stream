"""GL-dependent end-to-end tests for flygym_stream.app (PLAN.md Phase 5).

Same harness as test_app.py, but with the real scene + OffscreenRenderer:
these are marked ``gpu`` and skip unless MUJOCO_GL=egl /
PYOPENGL_PLATFORM=egl are exported before pytest starts (mujoco reads the
env at import time). Shared helpers are imported from test_app (DRY).
"""

from __future__ import annotations

import io
import json

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
                assert abs(d.qpos[2] - 0.8) < 1e-3  # spawn height restored
    finally:
        sim.stop()
        renderer.close()
