"""FastAPI app: web UI, /healthz, and the single-client WebSocket (Phase 5).

``create_app(config, producer)`` takes everything it needs via dependency
injection so tests can wire FakeBackend/FakeRenderer (PLAN.md Phase 5).
Protocol: hello handshake -> status; binary JPEG frames out (~config fps);
control JSON in (decoded by protocol.decode_control, bad input closes with
1008 without killing the server); a second concurrent client is rejected
with close 1013 / "single-client"; disconnect leaves the sim thread up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from anyio.to_thread import run_sync
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from flygym_stream.config import Config
from flygym_stream.protocol import (
    Hello, ProtocolError, Status, decode_control, encode_status,
)
from flygym_stream.simulation import Backend, SimThread
from flygym_stream.stream import FrameQueue, InputQueue
from flygym_stream.webrtc import WebRtcSession

WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
log = logging.getLogger(__name__)
STATUS_INTERVAL_S = 1.0  # plan: status ~1 Hz
FRAME_POLL_S = 0.1       # frame-queue poll cadence in the send loop


@dataclass
class Producer:
    """Everything the WebSocket handler needs (DI for tests).

    Attributes:
        frames: JPEG frame queue the send loop drains.
        inputs: control queue the recv loop feeds.
        backend: stepping backend (status: time/name/n_actuators).
        sim: SimThread (status: paused; control dispatch).
        model_name: model field of the status message.
    """

    frames: FrameQueue
    inputs: InputQueue
    backend: Backend
    sim: SimThread
    model_name: str = "fly"


def create_app(config: Config, producer: Producer) -> FastAPI:
    """Build the FastAPI app wired to ``producer`` (PLAN.md §4)."""
    app = FastAPI(title="flygym-stream")
    active: list[WebSocket] = []          # single-client slot
    fps_box: list[float] = [0.0]          # measured frame rate (send loop)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "model": producer.model_name,
                "backend": producer.backend.name,
                "paused": producer.sim.paused}

    def _status(fps: float) -> str:
        return encode_status(Status(
            sim_time=producer.backend.time, fps=fps,
            paused=producer.sim.paused,
            n_actuators=producer.backend.n_actuators,
            backend=producer.backend.name, model=producer.model_name,
            orbit=producer.backend.orbit,
        ))

    async def _send_loop(ws: WebSocket) -> None:
        """Pop frames (threaded, non-blocking server) and send binary."""
        last = None
        while True:
            frame = await run_sync(producer.frames.get, FRAME_POLL_S)
            if frame is None:
                continue
            now = time.monotonic()
            if last is not None and now > last:
                inst = 1.0 / (now - last)
                fps_box[0] = inst if fps_box[0] == 0.0 else 0.9 * fps_box[0] + 0.1 * inst
            last = now
            try:
                await ws.send_bytes(frame)
            except Exception:
                log.exception("frame send loop ended")
                return

    async def _status_loop(ws: WebSocket) -> None:
        while True:
            await asyncio.sleep(STATUS_INTERVAL_S)
            try:
                await ws.send_text(_status(fps_box[0]))
            except Exception:
                log.exception("status loop ended")
                return

    async def _recv_loop(ws: WebSocket) -> None:
        """Control JSON in: decode, queue, or close on ProtocolError."""
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            raw = msg.get("text")
            if raw is None:
                continue  # binary input is not part of the protocol
            try:
                decoded = decode_control(raw)
            except ProtocolError as err:
                await ws.send_text(str(err))
                await ws.close(err.close_code, str(err)[:120])
                return
            if decoded is not None:
                producer.inputs.put(decoded)

    async def _webrtc_handshake(ws: WebSocket) -> "WebRtcSession | None":
        """App-layer signaling: webrtc-start in; offer/answer out (server
        is the offerer, see webrtc.py). The JSON control protocol
        (protocol.py) is untouched; only the video transport changes.
        """
        try:
            msg = json.loads(await ws.receive_text())
            ok = isinstance(msg, dict) and msg.get("type") == "webrtc-start"
        except Exception:
            ok = False
        if not ok:
            await ws.close(1008, "webrtc-start expected as first message")
            return None
        try:
            session = WebRtcSession(config, producer.frames)
            offer_sdp = await session.create_offer()
            await ws.send_text(
                json.dumps({"type": "webrtc-offer", "sdp": offer_sdp}))
            msg = json.loads(await ws.receive_text())
            ok = (isinstance(msg, dict)
                  and msg.get("type") == "webrtc-answer"
                  and isinstance(msg.get("sdp"), str))
        except Exception:
            log.exception("webrtc handshake failed")
            await ws.close(1011, "webrtc unavailable")
            return None
        if not ok:
            await ws.close(1008, "webrtc-answer expected")
            return None
        try:
            await session.handle_answer(msg["sdp"])
        except Exception:
            log.exception("webrtc answer rejected")
            await ws.close(1011, "webrtc unavailable")
            return None
        return session

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        if active:  # one client by design (PLAN.md §3)
            await websocket.send_text(
                json.dumps({"type": "error", "message": "single-client"})
            )
            await websocket.close(code=1013, reason="single-client")
            return
        active.append(websocket)
        session: WebRtcSession | None = None
        try:
            raw = await websocket.receive_text()
            try:
                hello = decode_control(raw)
            except ProtocolError as err:
                await websocket.send_text(str(err))
                await websocket.close(err.close_code, str(err)[:120])
                return
            if not isinstance(hello, Hello):
                await websocket.close(1008, "hello expected as first message")
                return
            await websocket.send_text(_status(fps_box[0]))
            sender: asyncio.Task | None = None
            if config.transport == "webrtc":
                session = await _webrtc_handshake(websocket)
                if session is None:
                    return  # closed in the handshake
            else:
                sender = asyncio.create_task(_send_loop(websocket))
            statuser = asyncio.create_task(_status_loop(websocket))
            try:
                await _recv_loop(websocket)
            finally:
                if sender is not None:
                    sender.cancel()
                statuser.cancel()
        finally:
            if session is not None:
                await session.close()
            active.clear()

    # Registered last: / and /healthz above take precedence, everything else
    # (style.css, app.js, ...) resolves against the web/ directory.
    app.mount("/", StaticFiles(directory=WEB_DIR), name="static")

    return app
