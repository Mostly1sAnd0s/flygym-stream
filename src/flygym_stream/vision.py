"""Fly vision feed (PLAN.md Phase 7, M7.6): the fly's own eyes.

When a scene is built with ``vision=True`` (``SceneConfig(vision=True)`` /
``serve --vision``), ``scenes.build_scene`` constructs the model through
``flygym.Simulation`` (whose ``get_raw_vision`` helper renders the two
eye cameras with fisheye correction and self-occlusion masking) and the
Scene holds that wrapper. ``VisionFeed`` paces the extra render pass on
sim time (same semantics as OffscreenRenderer) and emits a JPEG with a
2-byte ``V1`` prefix on the WS binary channel (main frames are
unprefixed JPEGs; a real JPEG never starts with ``V1`` = 0x56 0x31).

The vision pass renders the SAME MjData the sim thread steps — a single
source of truth; the frame is simply (re)rendered from the current pose.
"""

from __future__ import annotations

import numpy as np

from flygym_stream.config import Config
from flygym_stream.rendering import encode_jpeg

#: Binary prefix marking a vision frame on the WS (main frames: none).
VISION_PREFIX = b"V1"


class VisionFeed:
    """Latest-frame, sim-time-paced fly-eye renderer (M7.6).

    Args:
        scene: A scene with ``scene.vision`` set (flygym.Simulation).
        config: Resolution/fps config (paces at half the output fps:
            two fisheye renders per main frame is more than the pane
            needs, and each pass is a full MuJoCo scene render).

    Raises:
        ValueError: If the scene has no vision wrapper (build with
            ``SceneConfig(vision=True)``).
    """

    def __init__(self, scene, config: Config) -> None:
        if scene.vision is None:
            raise ValueError(
                "VisionFeed needs a vision scene (SceneConfig(vision=True))")
        self._scene = scene
        self._interval = 2.0 / max(1, int(config.output_fps))
        self._quality = int(config.jpeg_quality)
        self._last = -np.inf

    def render_as_needed(self, data) -> bool:
        """Sim-time pacing: True when a vision frame is due."""
        t = float(data.time)
        if t < self._last + self._interval - 1e-9:
            return False
        self._last = t
        return True

    def reset(self) -> None:
        """Re-base the pacing clock (a sim reset rewinds time to 0; a
        stale ``_last`` would freeze the vision pane until sim time
        crawled back up)."""
        self._last = -np.inf

    def render_frame(self) -> bytes:
        """Render both eyes (left on top) -> ``V1`` + JPEG bytes.

        Must be called on the sim thread (the eye renderer's GL context
        binds to its creating thread, like the main OffscreenRenderer).
        """
        raw = self._scene.vision.get_raw_vision(self._scene.fly_name)
        # raw: (2, h, w, 3) float or uint8, [left, right]
        left, right = raw[0], raw[1]
        if left.dtype != np.uint8:
            left = np.clip(left, 0, 255).astype(np.uint8)
            right = np.clip(right, 0, 255).astype(np.uint8)
        return VISION_PREFIX + encode_jpeg(np.vstack((left, right)),
                                           self._quality)
