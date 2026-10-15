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


# --- Phase 7 (M7.4): logging + status.mode -----------------------------------

def test_ws_status_has_mode_and_info_is_tolerated(app_env) -> None:
    """status carries the locomotion mode; an `info` request on a
    producer without model-info is a no-op that must not break the
    connection."""
    app, sim, backend = app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            status = recv_status(ws)
            assert status["mode"] == "neutral"  # no pattern engine on fakes
            ws.send_text(json.dumps({"type": "info"}))
            frame = recv_frame(ws)  # still streaming
            assert frame.startswith(b"JPEG-")


@pytest.fixture
def log_app_env(tmp_config, fake_backend, fake_renderer, tmp_path):
    """app_env with the JSONL log dir pointed at tmp_path."""
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, fake_backend, sim,
                                          log_dir=str(tmp_path)))
    yield app, sim, fake_backend
    sim.stop()


def test_log_on_off_and_get_log(log_app_env, tmp_path) -> None:
    """`log on` -> control events are JSONL-logged -> `log off` ->
    GET /log serves the latest log file."""
    import json as _json
    from pathlib import Path

    app, sim, backend = log_app_env
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(HELLO)
            recv_status(ws)
            ws.send_text(json.dumps({"type": "log", "on": 1}))
            ws.send_text(json.dumps(
                {"type": "orbit", "azimuth": 12.0, "elevation": -3.0,
                 "distance": 50.0}))
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not any(
                    p.suffix == ".jsonl" for p in Path(tmp_path).iterdir()):
                time.sleep(0.01)
            ws.send_text(json.dumps({"type": "log", "on": 0}))
        resp = client.get("/log")
        assert resp.status_code == 200
        lines = [json.loads(l) for l in resp.text.splitlines() if l.strip()]
        orbits = [l for l in lines if l.get("msg", {}).get("type") == "orbit"]
        assert orbits, lines
        assert orbits[0]["msg"]["azimuth"] == 12.0


def test_ws_telemetry_not_starved_by_frames(
        tmp_config, fake_backend, fake_renderer) -> None:
    """M7.4 regression: telemetry text must keep flowing while the
    (always-full) frame queue streams continuously — the send loop
    checks the telemetry queue every iteration, not only when a frame
    poll times out."""
    import asyncio
    from flygym_stream.telemetry import TelemetryQueue

    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(fake_backend, fake_renderer, frames, inputs, tmp_config)
    sim.start()
    tqueue = TelemetryQueue()
    app = create_app(tmp_config, Producer(frames, inputs, fake_backend, sim,
                                          telemetry_queue=tqueue))
    received: list[dict] = []
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                recv_status(ws)
                # wait a couple of real frames so the frame queue is
                # demonstrably streaming, then push telemetry
                recv_frame(ws)
                recv_frame(ws)
                tqueue.put({"t": 0.5, "qpos": [0.0]})
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    msg = ws.receive()
                    if msg.get("text"):
                        data = json.loads(msg["text"])
                        if data.get("type") == "telemetry":
                            received.append(data)
                            break
                assert received, "telemetry starved by continuous frames"
                assert received[0]["t"] == 0.5
    finally:
        sim.stop()


def test_ws_vision_frames_pass_through(
        tmp_config, fake_backend, fake_renderer) -> None:
    """V1-prefixed binary vision frames flow to the client alongside
    main frames (same WS, binary channel, prefix framing)."""
    from flygym_stream.stream import FrameQueue as FQ

    frames, inputs = FQ(), InputQueue()
    vqueue = FQ()
    sim = SimThread(fake_backend, fake_renderer, frames, inputs, tmp_config)
    sim.start()
    app = create_app(tmp_config, Producer(frames, inputs, fake_backend, sim,
                                          vision_queue=vqueue))
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                recv_status(ws)
                vqueue.put(b"V1" + b"\xff\xd8fakejpeg")
                # receive until the vision frame (frames interleave)
                deadline = time.monotonic() + 5.0
                got = None
                while time.monotonic() < deadline:
                    msg = ws.receive()
                    if msg.get("bytes"):
                        if msg["bytes"][:2] == b"V1":
                            got = msg["bytes"]
                            break
                assert got == b"V1\xff\xd8fakejpeg"
    finally:
        sim.stop()
