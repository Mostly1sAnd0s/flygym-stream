"""WebRTC transport for the frame stream (PLAN.md Phase 6, 'webrtc' extra).

The JSON control channel (protocol.py) stays on the WebSocket unchanged;
only the video moves: a server-side RTCPeerConnection sends a
VideoStreamTrack fed from the FrameQueue. Signaling rides the same WS as
app-layer messages (``webrtc-start`` / ``webrtc-offer`` /
``webrtc-answer``) with the server as offerer. Requires the ``webrtc``
extra (aiortc); importing this module without it is safe, only
constructing raises an actionable error.
"""

from __future__ import annotations

import fractions
import io
import logging

import numpy as np
from PIL import Image

try:  # the 'webrtc' extra (optional)
    from aiortc import (
        RTCPeerConnection,
        RTCConfiguration,
        RTCSessionDescription,
        VideoStreamTrack,
    )
    from av import VideoFrame

    _AIORTC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    VideoStreamTrack = object  # keeps the class definition below valid
    VideoFrame = None
    _AIORTC_AVAILABLE = False

from anyio.to_thread import run_sync

from flygym_stream.config import Config
from flygym_stream.stream import FrameQueue

log = logging.getLogger(__name__)
_EXTRA_HINT = "install the 'webrtc' extra (uv sync --extra webrtc)"
FRAME_TIMEOUT_S = 0.5  # queue poll cadence inside FrameTrack.recv


def _require_aiortc() -> None:
    if not _AIORTC_AVAILABLE:
        raise RuntimeError(
            "transport 'webrtc' needs the aiortc package; " + _EXTRA_HINT
        )


class FrameTrack(VideoStreamTrack):
    """A VideoStreamTrack serving JPEG frames from a FrameQueue.

    Decodes each JPEG to RGB and wraps it in an ``av.VideoFrame`` at the
    configured fps. When the queue is empty (e.g. sim paused) the last
    frame is repeated so the RTP stream keeps flowing.
    """

    def __init__(self, frames: FrameQueue, *, width: int, height: int,
                 fps: int) -> None:
        _require_aiortc()
        super().__init__()
        self._frames = frames
        self._width = width
        self._height = height
        self._time_base = fractions.Fraction(1, fps)
        self._pts = 0
        self._last = np.zeros((height, width, 3), dtype=np.uint8)

    async def recv(self) -> "VideoFrame":
        """Next frame: fresh JPEG from the queue, or the last one repeated."""
        jpeg = await run_sync(self._frames.get, FRAME_TIMEOUT_S)
        if jpeg is not None:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
            if img.size != (self._width, self._height):
                img = img.resize((self._width, self._height))
            self._last = np.asarray(img)
        frame = VideoFrame(self._width, self._height).reformat(format="rgb24")
        frame.to_ndarray(format="rgb24")[:] = self._last
        frame.pts = self._pts
        frame.time_base = self._time_base
        self._pts += 1
        return frame


class WebRtcSession:
    """One WebRTC video session: server-side OFFERER for a WS client.

    The server is the offerer because it owns the video track (aiortc 1.15
    omits track-less recvonly transceivers from offers, so a client-side
    offerer would produce an empty offer). Signaling on the control WS:
    client sends ``{"type": "webrtc-start"}`` -> server replies with
    ``{"type": "webrtc-offer", "sdp"}`` -> client replies with
    ``{"type": "webrtc-answer", "sdp"}``.
    """

    def __init__(self, config: Config, frames: FrameQueue) -> None:
        _require_aiortc()
        self._pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self._pc.addTrack(
            FrameTrack(frames, width=config.width, height=config.height,
                       fps=config.output_fps)
        )

    async def create_offer(self) -> str:
        """Create + store the offer; return its SDP (incl. candidates)."""
        await self._pc.setLocalDescription(await self._pc.createOffer())
        return self._pc.localDescription.sdp

    async def handle_answer(self, sdp: str) -> None:
        """Store the client's answer SDP; video flows afterwards."""
        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="answer")
        )

    async def close(self) -> None:
        await self._pc.close()
