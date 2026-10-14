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
                     "distance": 25.0}))
                wait_for(lambda: any(
                    type(m).__name__ == "Orbit" for m in backend.inputs))
                recv_frame(ws)  # discard a frame possibly rendered pre-orbit
                second = mean(recv_frame(ws))
                ws.send_text(json.dumps(
                    {"type": "actuators", "values": [1.0] * 42}))
                wait_for(lambda: any(
                    type(m).__name__ == "ActuatorSet" for m in backend.inputs))
                recv_frame(ws)  # discard a frame possibly rendered pre-set
                third = mean(recv_frame(ws))
                assert abs(first - second) > 1.0, (first, second)
                assert abs(second - third) > 1.0, (second, third)
                print(f"\ncontrol->pixels: {first:.1f} -> {second:.1f} (orbit)"
                      f" -> {third:.1f} (actuators)")
    finally:
        sim.stop()
        renderer.close()
