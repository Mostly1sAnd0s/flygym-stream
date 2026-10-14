"""Tests for the WebRTC transport (PLAN.md Phase 6, 'webrtc' extra).

The JSON control channel stays on the WebSocket unchanged; only the video
moves to WebRTC. ``test_frame_track_yields_frames`` is pure (no GL);
``test_webrtc_e2e`` is gpu-marked because it renders real frames.
"""

from __future__ import annotations

import asyncio
import io
import json
import queue
import threading

import numpy as np
import pytest

pytest.importorskip("aiortc")  # the 'webrtc' extra

from av import VideoFrame  # noqa: E402

from flygym_stream.config import Config  # noqa: E402
from flygym_stream.rendering import encode_jpeg  # noqa: E402
from flygym_stream.stream import FrameQueue  # noqa: E402
from flygym_stream.webrtc import FrameTrack  # noqa: E402
from test_app import HELLO, recv_status  # noqa: E402


def test_frame_track_yields_frames() -> None:
    """JPEG bytes from a FrameQueue become correctly sized av VideoFrames."""
    frames = FrameQueue()

    async def run() -> list[VideoFrame]:
        track = FrameTrack(frames, width=64, height=48, fps=30)
        out = []
        for i in range(4):  # sequential: the queue is drop-oldest maxsize=1
            arr = np.full((48, 64, 3), (i * 60) % 255, dtype=np.uint8)
            frames.put(encode_jpeg(arr, quality=95))
            out.append(await track.recv())
        return out

    out = asyncio.run(run())
    assert all(f.width == 64 and f.height == 48 for f in out)
    assert all(f.format.name == "rgb24" for f in out)
    # content follows the source (lossy JPEG: allow slack)
    for i, f in enumerate(out):
        mean = float(np.asarray(f.to_ndarray(format="rgb24")).mean())
        assert abs(mean - (i * 60) % 255) < 30, (i, mean)


@pytest.mark.gpu
def test_webrtc_e2e_real_frames() -> None:
    """Full app (real scene + renderer) -> real WebRTC -> decoded frames."""
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCSessionDescription

    from fastapi.testclient import TestClient

    from flygym_stream.app import Producer, create_app
    from flygym_stream.rendering import OffscreenRenderer
    from flygym_stream.scene import build_default_scene
    from flygym_stream.simulation import SimThread, make_backend
    from flygym_stream.stream import InputQueue

    config = Config(width=320, height=240, output_fps=30, transport="webrtc")
    scene = build_default_scene()
    backend = make_backend(scene, config)
    renderer = OffscreenRenderer(scene, config)
    frames, inputs = FrameQueue(), InputQueue()
    sim = SimThread(backend, renderer, frames, inputs, config)
    sim.start()
    app = create_app(config, Producer(frames, inputs, backend, sim))

    offer_q: queue.Queue[str] = queue.Queue()
    answer_q: queue.Queue[str] = queue.Queue()
    result_q: queue.Queue[int] = queue.Queue()

    def client() -> None:
        async def main() -> None:
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            pc.addTransceiver("video")  # answerer: receive-only
            frames_rx: list[VideoFrame] = []

            @pc.on("track")
            def on_track(track) -> None:
                async def pump() -> None:
                    while len(frames_rx) < 8:
                        frames_rx.append(await track.recv())

                asyncio.ensure_future(pump())

            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=offer_q.get(timeout=120),
                                      type="offer"))
            await pc.setLocalDescription(await pc.createAnswer())
            answer_q.put(pc.localDescription.sdp)
            while len(frames_rx) < 8:
                await asyncio.sleep(0.1)
            result_q.put(len(frames_rx))
            await pc.close()

        asyncio.run(main())

    client_thread = threading.Thread(target=client, name="webrtc-client")
    client_thread.start()
    try:
        with TestClient(app) as client_:
            with client_.websocket_connect("/ws") as ws:
                ws.send_text(HELLO)
                recv_status(ws)
                ws.send_text(json.dumps({"type": "webrtc-start"}))
                offer = _recv_ws_json(ws)
                assert offer["type"] == "webrtc-offer"
                offer_q.put(offer["sdp"])
                # relay the client's answer back to the server over the WS
                ws.send_text(json.dumps(
                    {"type": "webrtc-answer",
                     "sdp": answer_q.get(timeout=120)}))
                recv_status(ws)  # the WS keeps serving status + control
        n = result_q.get(timeout=60)
        assert n >= 8
        print(f"\nwebrtc e2e: {n} decoded 320x240 frames")
    finally:
        client_thread.join(timeout=15)
        sim.stop()
        renderer.close()


def _recv_ws_json(ws) -> dict:
    """Next text JSON message on the WS (webrtc mode: no binary frames)."""
    msg = ws.receive()
    while "text" not in msg:
        msg = ws.receive()
    return json.loads(msg["text"])
