"""End-to-end tests for flygym_stream.app (PLAN.md Phase 5 DoD).

FakeBackend/FakeRenderer + a real SimThread on a real clock; FastAPI
TestClient drives the WebSocket. Covers: / and /healthz, hello handshake
-> status, binary frames, control messages reaching the sim, pause
reflected in status, second client rejected (1013 single-client), bad
input closes 1008 without killing the server, unknown type tolerated,
disconnect leaves the sim thread running.
"""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from flygym_stream.app import Producer, create_app
from flygym_stream.simulation import SimThread
from flygym_stream.stream import FrameQueue, InputQueue

HELLO = json.dumps({"type": "hello", "proto": 1})


def wait_for(pred, timeout: float = 5.0) -> None:
    """Poll ``pred`` until true (sim thread applies inputs async)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


def recv_status(ws) -> dict:
    """Receive the next status text message, discarding binary frames."""
    while True:
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            raise WebSocketDisconnect(msg.get("code", 1000))
        if "text" in msg:
            data = json.loads(msg["text"])
            if data.get("type") == "status":
                return data


def recv_frame(ws) -> bytes:
    """Receive the next binary frame, discarding text messages."""
    while True:
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            raise WebSocketDisconnect(msg.get("code", 1000))
        if "bytes" in msg:
            return msg["bytes"]


def recv_close(ws) -> WebSocketDisconnect:
    """Receive the server's close, discarding any frames sent first."""
    msg = ws.receive()
    while msg["type"] != "websocket.close":
        msg = ws.receive()
    return WebSocketDisconnect(msg.get("code", 1000), msg.get("reason", ""))


@pytest.fixture
def app_env(tmp_config, fake_backend, fake_renderer):
    """App wired to a running SimThread over the fakes."""
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, fake_backend, sim))
    yield app, sim, fake_backend
    sim.stop()


def test_healthz_and_index(app_env) -> None:
    app = app_env[0]
    with TestClient(app) as client:
        assert client.get("/healthz").json()["ok"] is True
        index = client.get("/")
        assert index.status_code == 200
        assert "FlyGym Stream" in index.text


def test_ws_handshake_status_and_frames(app_env) -> None:
    app, sim, backend = app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            status = recv_status(ws)
            assert status["type"] == "status"
            assert status["n_actuators"] == backend.n_actuators
            assert status["backend"] == "fake"
            assert status["paused"] is False
            frame = recv_frame(ws)  # fake JPEG bytes arrive at ~30 fps
            assert frame.startswith(b"JPEG-")


def test_ws_orbit_and_pause_reflected_in_status(app_env) -> None:
    app, sim, backend = app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)  # initial status
            ws.send_text(json.dumps(
                {"type": "orbit", "azimuth": 45.0, "elevation": -10.0,
                 "distance": 30.0}))
            wait_for(lambda: any(
                type(m).__name__ == "Orbit" and m.azimuth == 45.0
                for m in backend.inputs))
            ws.send_text(json.dumps({"type": "pause"}))
            wait_for(lambda: sim.paused)
            # sim time is frozen, so no more frames are due; the periodic
            # status keeps flowing and must report the paused state.
            status = recv_status(ws)
            assert status["paused"] is True


def test_ws_status_reports_live_orbit_state(app_env) -> None:
    """After an orbit control message, the periodic status reports the new
    orbit state so the browser can adopt the server's true camera."""
    app, sim, backend = app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)  # initial status
            ws.send_text(json.dumps(
                {"type": "orbit", "azimuth": 45.0, "elevation": -10.0,
                 "distance": 6000.0}))
            wait_for(lambda: any(
                type(m).__name__ == "Orbit" and m.azimuth == 45.0
                for m in backend.inputs))
            status = recv_status(ws)
            assert status["orbit"] == {"azimuth": 45.0, "elevation": -10.0,
                                       "distance": 6000.0}


def test_second_client_rejected_single_client(app_env) -> None:
    app = app_env[0]
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws1:
            ws1.send_text(HELLO)
            recv_status(ws1)
            with client.websocket_connect("/ws") as ws2:
                err = json.loads(ws2.receive_text())
                assert err == {"type": "error", "message": "single-client"}
                with pytest.raises(WebSocketDisconnect) as exc_info:
                    ws2.receive_text()
                assert exc_info.value.code == 1013
            recv_frame(ws1)  # first client keeps streaming


def test_bad_input_closes_1008_server_survives(app_env) -> None:
    app, sim = app_env[0], app_env[1]
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)
            ws.send_text("this is not json")
            exc = recv_close(ws)
            assert exc.code == 1008
        # server still healthy: healthz + a fresh client works
        assert client.get("/healthz").status_code == 200
        with client.websocket_connect("/ws") as ws2:
            ws2.send_text(HELLO)
            assert recv_status(ws2)["type"] == "status"
    assert sim.running  # disconnect left the sim thread running


def test_unknown_type_ignored(app_env) -> None:
    app, sim = app_env[0], app_env[1]
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)
            ws.send_text(json.dumps({"type": "teleport", "x": 1}))
            recv_frame(ws)  # still streaming, no crash
    assert sim.running


def test_hello_must_come_first(app_env) -> None:
    app = app_env[0]
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "pause"}))
            exc = recv_close(ws)
            assert exc.code == 1008


def test_static_assets_served(app_env) -> None:
    app = app_env[0]
    with TestClient(app) as client:
        css = client.get("/style.css")
        js = client.get("/app.js")
        assert css.status_code == 200 and "body" in css.text
        assert js.status_code == 200 and "MSG" in js.text
        assert client.get("/nope.js").status_code == 404




def test_ws_state_messages_reach_backend(app_env) -> None:
    """M7.2: set_state/force/param/keyframe/spawn/adhesion decode and
    reach the backend's apply_state (fake backend records them)."""
    app, sim, backend = app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)
            for payload in (
                {"type": "set_state", "ctrl": [0.25]},
                {"type": "force", "body": 1, "force": [0, 0, 1.0]},
                {"type": "param", "name": "timestep", "value": 0.001},
                {"type": "keyframe", "name": "neutral"},
                {"type": "spawn", "pos": [0, 0, 1.0],
                 "quat": [1.0, 0.0, 0.0, 0.0]},
                {"type": "adhesion", "leg": "lf", "state": 1},
            ):
                ws.send_text(json.dumps(payload))
            # State messages drain through the (async) sim thread: poll
            # until all six are applied rather than racing it.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                types = {type(m).__name__ for m in backend.inputs}
                if len(backend.inputs) >= 6:
                    break
                time.sleep(0.01)
            types = {type(m).__name__ for m in backend.inputs}
            for expected in ("SetState", "Force", "Param", "Keyframe",
                             "Spawn", "Adhesion"):
                assert expected in types, (expected, types)


def test_ws_bad_param_name_closes_1008(app_env) -> None:
    app, sim, backend = app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)
            ws.send_text(json.dumps(
                {"type": "param", "name": "nuclear", "value": 1.0}))
            exc = recv_close(ws)
            assert exc.code == 1008
